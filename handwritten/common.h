// Shared harness for the handwritten references: timing, input, checking.
//
// The references are compiled with the same flags the C backend uses
// (`cc -O2 -fopenmp`), so a comparison against generated code measures the
// lowering, not a flag difference. See README.md.
#ifndef SILE_HANDWRITTEN_COMMON_H
#define SILE_HANDWRITTEN_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

// Deterministic fill: xorshift32 mapped to [-1, 1). A fixed seed keeps two
// builds of a reference comparable.
static void fill_uniform(uint32_t *state, float *dst, int64_t n) {
    uint32_t s = *state ? *state : 0x1234567u;
    for (int64_t i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        dst[i] = (float)(int32_t)s / 2147483648.0f;
    }
    *state = s;
}

static int failures = 0;

static void check_close(const char *what, const float *got, const float *want,
                        int64_t n, float tol) {
    float worst = 0.0f;
    int64_t at = 0;
    for (int64_t i = 0; i < n; ++i) {
        float d = fabsf(got[i] - want[i]);
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    if (worst > tol) {
        printf("FAIL %-24s worst=%.3e at %lld (got %.6f want %.6f)\n", what,
               (double)worst, (long long)at, (double)got[at], (double)want[at]);
        ++failures;
    } else {
        printf("PASS %-24s worst=%.3e (tol %.0e)\n", what, (double)worst,
               (double)tol);
    }
}

// Milliseconds of the median of `reps` runs. The median carries two jobs on
// this box. The clock itself can read another CPU's counter after a thread
// migrates, and a fastest-run statistic promotes one such ~0.1 ms reading to
// the number. Worse is the cold-start transient: the first ~20 omp regions of
// a process each cost a flat ~6 ms whatever they run — a bare parallel-for
// over empty bodies measures 6.0 ms, then snaps to 0.005 ms and stays — so a
// short bench can sit entirely inside the transient. The rep count has to be
// high enough that fewer than half the launches land in it; 100 is measured
// safe, and the median then reports the steady state on both sides alike.
static int cmp_double(void const *a, void const *b) {
    double x = *(double const *)a;
    double y = *(double const *)b;
    return (x > y) - (x < y);
}

static double bench_ms(void (*run)(void *), void *ctx, int reps) {
    double samples[reps];
    for (int r = 0; r < reps; ++r) {
        double t0 = now_ms();
        run(ctx);
        samples[r] = now_ms() - t0;
    }
    qsort(samples, (size_t)reps, sizeof samples[0], cmp_double);
    return samples[reps / 2];
}

static void report(const char *kernel, double ms, double work, const char *unit) {
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("BENCH %-12s %10.3f ms  %8.2f %s  threads=%d\n", kernel, ms,
           work / (ms * 1e-3) / 1e9, unit, threads);
}

// Every reference takes `--self-check` (default) or `--bench`.
static int mode_from(int argc, char **argv) {
    return (argc > 1 && strcmp(argv[1], "--bench") == 0) ? 1 : 0;
}

static int finish(const char *kernel) {
    if (failures) {
        printf("%s: %d check(s) failed\n", kernel, failures);
        return 1;
    }
    printf("%s: all checks passed\n", kernel);
    return 0;
}

#endif
