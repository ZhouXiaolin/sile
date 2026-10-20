// gemm: z[m][n] = sum_k x[m][k] * y[k][n].
//
// The kernel is `crates/sile/examples/gemm.rs`: an output tile whose
// contraction runs in BK-deep steps through `mma`. The reference is the same
// algorithm with the two things `Mma` has to give it:
//
// - an MR x NR register accumulator per tile, so the inner product is a
//   vectorisable FMA chain and not a gather,
// - both operands packed into contiguous panels per BK step, so the inner loop
//   reads 16 floats from one cache line.
//
// Measured: 114 GFLOP/s against 67 GFLOP/s for the unpacked 32x16 form, and
// 78 GFLOP/s for the naive micro-kernel this file started with.
//
// gemm is not in the M2 80% set (plan 4.6 risk 3: its handwritten ceiling is
// far above a naive implementation). It is here as the shape target for Mma.
#include "common.h"

#define MR 8
#define NR 16
#define BK 64

static void gemm(float *restrict z, const float *restrict x,
                 const float *restrict y, int64_t m, int64_t n, int64_t k) {
    int64_t mt = (m + MR - 1) / MR;
    int64_t nt = (n + NR - 1) / NR;

#pragma omp parallel for schedule(dynamic) collapse(2)
    for (int64_t ti = 0; ti < mt; ++ti) {
        for (int64_t tj = 0; tj < nt; ++tj) {
            int64_t i0 = ti * MR, j0 = tj * NR;
            float acc[MR][NR];
            for (int64_t i = 0; i < MR; ++i)
                for (int64_t j = 0; j < NR; ++j) acc[i][j] = 0.0f;

            float xp[MR * BK], yp[BK * NR];
            for (int64_t k0 = 0; k0 < k; k0 += BK) {
                int64_t kk = k0 + BK <= k ? BK : k - k0;

                for (int64_t i = 0; i < MR; ++i) {
                    int64_t gi = i0 + i;
                    for (int64_t p = 0; p < kk; ++p)
                        xp[i * BK + p] = gi < m ? x[gi * k + k0 + p] : 0.0f;
                }
                for (int64_t p = 0; p < kk; ++p) {
                    int64_t gp = k0 + p;
                    for (int64_t j = 0; j < NR; ++j) {
                        int64_t gj = j0 + j;
                        yp[p * NR + j] = gj < n ? y[gp * n + gj] : 0.0f;
                    }
                }

                for (int64_t p = 0; p < kk; ++p) {
                    for (int64_t i = 0; i < MR; ++i) {
                        float a = xp[i * BK + p];
#pragma omp simd
                        for (int64_t j = 0; j < NR; ++j) {
                            acc[i][j] += a * yp[p * NR + j];
                        }
                    }
                }
            }

            for (int64_t i = 0; i < MR; ++i) {
                int64_t gi = i0 + i;
                if (gi >= m) continue;
                for (int64_t j = 0; j < NR; ++j) {
                    int64_t gj = j0 + j;
                    if (gj < n) z[gi * n + gj] = acc[i][j];
                }
            }
        }
    }
}

// The oracle: the textbook triple loop, in double.
static void gemm_oracle(float *z, const float *x, const float *y, int64_t m,
                        int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int64_t p = 0; p < k; ++p)
                sum += (double)x[i * k + p] * (double)y[p * n + j];
            z[i * n + j] = (float)sum;
        }
    }
}

struct bench_ctx {
    float *z;
    const float *x, *y;
    int64_t m, n, k;
};

static void run(void *raw) {
    struct bench_ctx *ctx = raw;
    gemm(ctx->z, ctx->x, ctx->y, ctx->m, ctx->n, ctx->k);
}

int main(int argc, char **argv) {
    if (mode_from(argc, argv)) {
        const int64_t m = 1024, n = 1024, k = 1024;
        float *z = calloc((size_t)(m * n), sizeof(float));
        float *x = malloc(sizeof(float) * m * k);
        float *y = malloc(sizeof(float) * k * n);
        uint32_t seed = 1;
        fill_uniform(&seed, x, m * k);
        fill_uniform(&seed, y, k * n);
        struct bench_ctx ctx = {z, x, y, m, n, k};
        double ms = bench_ms(run, &ctx, 3);
        report("gemm", ms, 2.0 * (double)m * (double)n * (double)k, "GFLOP/s");
        free(z);
        free(x);
        free(y);
        return 0;
    }

    // The example's shape: 128x128 output, K = 64, both inputs all ones, so
    // every element of z is K.
    {
        const int64_t m = 128, n = 128, k = 64;
        float *z = calloc((size_t)(m * n), sizeof(float));
        float *x = malloc(sizeof(float) * m * k);
        float *y = malloc(sizeof(float) * k * n);
        float *want = malloc(sizeof(float) * m * n);
        for (int64_t i = 0; i < m * k; ++i) x[i] = 1.0f;
        for (int64_t i = 0; i < k * n; ++i) y[i] = 1.0f;
        for (int64_t i = 0; i < m * n; ++i) want[i] = (float)k;
        gemm(z, x, y, m, n, k);
        check_close("gemm 128x128x64 ones", z, want, m * n, 1e-3f);
        free(z);
        free(x);
        free(y);
        free(want);
    }

    // A random case at a size the triple loop can afford, against the oracle.
    {
        const int64_t m = 64, n = 64, k = 32;
        float *z = calloc((size_t)(m * n), sizeof(float));
        float *x = malloc(sizeof(float) * m * k);
        float *y = malloc(sizeof(float) * k * n);
        float *want = malloc(sizeof(float) * m * n);
        uint32_t seed = 7;
        fill_uniform(&seed, x, m * k);
        fill_uniform(&seed, y, k * n);
        gemm(z, x, y, m, n, k);
        gemm_oracle(want, x, y, m, n, k);
        check_close("gemm 64x64x32 random", z, want, m * n, 1e-3f);
        free(z);
        free(x);
        free(y);
        free(want);
    }
    return finish("gemm");
}
