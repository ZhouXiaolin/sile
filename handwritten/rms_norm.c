// rms_norm: out[r][c] = x[r][c] * (1 / sqrt(mean(x[r]^2) + eps)) * w[c]
//
// The kernel is `crates/sile/examples/rms_norm.rs`: a sum of squares over the
// row, the scalar root, a broadcast, and a second pass over the row.
//
// The sum of squares is an `omp simd reduction(+)`, which is what `TileReduce`
// has to become on this target (plan 4.3). Measured against the scalar loop
// form, this is 3x (0.320 ms -> 0.107 ms at 4096x1024), so it is the reference.
#include "common.h"

static void rms_norm(float *restrict out, const float *restrict x,
                     const float *restrict w, int64_t rows, int64_t cols,
                     float eps) {
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float *xr = x + r * cols;
        float *orow = out + r * cols;

        float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
        for (int64_t c = 0; c < cols; ++c) {
            float v = xr[c];
            sum += v * v;
        }

        float inv = 1.0f / sqrtf(sum / (float)cols + eps);
#pragma omp simd
        for (int64_t c = 0; c < cols; ++c) {
            orow[c] = (xr[c] * inv) * w[c];
        }
    }
}

static void rms_norm_oracle(float *out, const float *x, const float *w,
                            int64_t rows, int64_t cols, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        const float *xr = x + r * cols;
        float *orow = out + r * cols;
        double sum = 0.0;
        for (int64_t c = 0; c < cols; ++c) sum += (double)xr[c] * (double)xr[c];
        double inv = 1.0 / sqrt(sum / (double)cols + (double)eps);
        for (int64_t c = 0; c < cols; ++c) {
            orow[c] = (float)((double)xr[c] * inv * (double)w[c]);
        }
    }
}

struct bench_ctx {
    float *out;
    const float *x;
    const float *w;
    int64_t rows, cols;
    float eps;
};

static void run(void *raw) {
    struct bench_ctx *ctx = raw;
    rms_norm(ctx->out, ctx->x, ctx->w, ctx->rows, ctx->cols, ctx->eps);
}

int main(int argc, char **argv) {
    const float eps = 1e-8f;
    if (mode_from(argc, argv)) {
        const int64_t rows = 4096, cols = 1024;
        float *x = malloc(sizeof(float) * rows * cols);
        float *out = malloc(sizeof(float) * rows * cols);
        float *w = malloc(sizeof(float) * cols);
        uint32_t seed = 1;
        fill_uniform(&seed, x, rows * cols);
        fill_uniform(&seed, w, cols);
        struct bench_ctx ctx = {out, x, w, rows, cols, eps};
        double ms = bench_ms(run, &ctx, 100);
        report("rms_norm", ms, (double)rows * (double)cols * 3.0 * sizeof(float),
               "GB/s");
        free(x);
        free(out);
        free(w);
        return 0;
    }

    // The example's shape: 4x8, BLOCK_SIZE = 2, w = 1 + 0.1 * index.
    const int64_t rows = 4, cols = 8;
    const int64_t count = rows * cols;
    float *x = malloc(sizeof(float) * count);
    float *out = malloc(sizeof(float) * count);
    float *want = malloc(sizeof(float) * count);
    float w[8];
    for (int64_t c = 0; c < cols; ++c) w[c] = 1.0f + 0.1f * (float)c;
    for (int64_t i = 0; i < count; ++i) x[i] = (float)i * 0.25f - 2.0f;
    rms_norm(out, x, w, rows, cols, eps);
    rms_norm_oracle(want, x, w, rows, cols, eps);
    check_close("rms_norm 4x8", out, want, count, 1e-5f);
    free(x);
    free(out);
    free(want);
    return finish("rms_norm");
}
