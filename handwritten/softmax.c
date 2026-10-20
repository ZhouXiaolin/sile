// softmax over the last axis: y[r][c] = exp(x[r][c] - max_r) / sum_r exp(...)
//
// The kernel is `crates/sile/examples/softmax.rs`: reduce_max and reduce_sum on
// axis 1, broadcast back over the row, then a divide.
//
// The two reductions are written the way `TileReduce` has to materialise on
// this target (plan 4.3): one outer `parallel for` over rows, one
// `omp simd reduction(...)` per tile reduction. Measured against the scalar
// loop form, this is 5.5x (4.09 ms -> 0.74 ms at 4096x1024), so it is the
// reference and not an optimisation to reach later.
#include "common.h"

static void softmax(float *restrict y, const float *restrict x, int64_t rows,
                    int64_t cols) {
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const float *xr = x + r * cols;
        float *yr = y + r * cols;

        float max = -3.402823e38f;
#pragma omp simd reduction(max : max)
        for (int64_t c = 0; c < cols; ++c) {
            max = xr[c] > max ? xr[c] : max;
        }

        // The exponent is stored in one pass and summed in the next: a store
        // inside a reduction loop keeps gcc from vectorising the reduction.
        // The extra pass is over data that is already in L1.
#pragma omp simd
        for (int64_t c = 0; c < cols; ++c) {
            yr[c] = expf(xr[c] - max);
        }

        float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
        for (int64_t c = 0; c < cols; ++c) {
            sum += yr[c];
        }

        float inv = 1.0f / sum;
#pragma omp simd
        for (int64_t c = 0; c < cols; ++c) {
            yr[c] *= inv;
        }
    }
}

// The oracle: the same formula in double, so the f32 reference is checked
// against something better than itself.
static void softmax_oracle(float *y, const float *x, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        const float *xr = x + r * cols;
        float *yr = y + r * cols;
        double max = xr[0];
        for (int64_t c = 1; c < cols; ++c) {
            if (xr[c] > max) max = xr[c];
        }
        double sum = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            double e = exp((double)xr[c] - max);
            yr[c] = (float)e;
            sum += e;
        }
        for (int64_t c = 0; c < cols; ++c) {
            yr[c] = (float)((double)yr[c] / sum);
        }
    }
}

struct bench_ctx {
    float *y;
    const float *x;
    int64_t rows, cols;
};

static void run(void *raw) {
    struct bench_ctx *ctx = raw;
    softmax(ctx->y, ctx->x, ctx->rows, ctx->cols);
}

int main(int argc, char **argv) {
    if (mode_from(argc, argv)) {
        const int64_t rows = 4096, cols = 1024;
        float *x = malloc(sizeof(float) * rows * cols);
        float *y = malloc(sizeof(float) * rows * cols);
        uint32_t seed = 1;
        fill_uniform(&seed, x, rows * cols);
        struct bench_ctx ctx = {y, x, rows, cols};
        double ms = bench_ms(run, &ctx, 25);
        report("softmax", ms, (double)rows * (double)cols * 4.0 * sizeof(float),
               "GB/s");
        free(x);
        free(y);
        return 0;
    }

    // The example's shape: 4x8 input, tile (2, 8), so two rows per program.
    const int64_t rows = 4, cols = 8;
    const int64_t count = rows * cols;
    float *x = malloc(sizeof(float) * count);
    float *y = malloc(sizeof(float) * count);
    float *want = malloc(sizeof(float) * count);
    for (int64_t i = 0; i < count; ++i) {
        x[i] = (float)i * 0.5f - 2.0f;
    }
    softmax(y, x, rows, cols);
    softmax_oracle(want, x, rows, cols);
    check_close("softmax 4x8", y, want, count, 1e-5f);

    for (int64_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) sum += y[r * cols + c];
        if (fabsf(sum - 1.0f) > 1e-4f) {
            printf("FAIL row %lld sums to %.6f\n", (long long)r, (double)sum);
            ++failures;
        }
    }
    free(x);
    free(y);
    free(want);
    return finish("softmax");
}
