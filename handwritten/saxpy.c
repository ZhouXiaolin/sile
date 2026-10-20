// saxpy: y = a * x + y over a flat array.
//
// The kernel is `crates/sile/examples/saxpy.rs`: a tile-shaped load of x and y,
// a broadcast scalar, and a store back into y. Flattened, that is this loop.
#include "common.h"

static void saxpy(float *restrict y, const float *restrict x, float a, int64_t n) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

struct bench_ctx {
    float *y;
    const float *x;
    float a;
    int64_t n;
};

static void run(void *raw) {
    struct bench_ctx *ctx = raw;
    saxpy(ctx->y, ctx->x, ctx->a, ctx->n);
}

int main(int argc, char **argv) {
    if (mode_from(argc, argv)) {
        const int64_t n = 1 << 24;  // 16M elements: 64 MB per buffer, bandwidth bound
        float *x = malloc(sizeof(float) * n);
        float *y = malloc(sizeof(float) * n);
        uint32_t seed = 1;
        fill_uniform(&seed, x, n);
        fill_uniform(&seed, y, n);
        struct bench_ctx ctx = {y, x, 2.0f, n};
        double ms = bench_ms(run, &ctx, 25);
        report("saxpy", ms, (double)n * 3.0 * sizeof(float), "GB/s");
        free(x);
        free(y);
        return 0;
    }

    // The example's shape: 4x8, a = 2, x = y = arange(32).
    const int64_t m = 4, n = 8, count = m * n;
    float *x = malloc(sizeof(float) * count);
    float *y = malloc(sizeof(float) * count);
    float *want = malloc(sizeof(float) * count);
    for (int64_t i = 0; i < count; ++i) {
        x[i] = y[i] = (float)i;
        want[i] = 2.0f * (float)i + (float)i;
    }
    saxpy(y, x, 2.0f, count);
    check_close("saxpy 4x8", y, want, count, 1e-5f);
    free(x);
    free(y);
    free(want);
    return finish("saxpy");
}
