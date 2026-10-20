# Handwritten references (M0)

`docs/BACKEND_PLAN.md` principle 5: every performance target gets a handwritten
best-effort implementation first. The reference is three things at once:

- the **oracle** a generated kernel is checked against,
- the **shape target** the lowering has to reach (workgroup size, shared
  memory, barrier, micro-kernel),
- the **100% line** of the 80% performance bar.

Without it, "similar performance" has no definition.

## Layout

| file | target | shape |
|---|---|---|
| `saxpy.c` / `.wgsl` | C / WGSL | flat `y = a*x + y`, one element per thread |
| `softmax.c` / `.wgsl` | C / WGSL | row-wise max then sum, WGSL tree reduction in `var<workgroup>` |
| `rms_norm.c` / `.wgsl` | C / WGSL | row-wise sum of squares, same reduction shape as softmax |
| `gemm.c` / `.wgsl` | C / WGSL | 32x16 tile, 8-deep C micro-kernel; WGSL 4x8 threads with shared panels |

Each C reference takes `--self-check` (default) or `--bench`; the bench size is
large enough to be bandwidth or FLOP bound, the check size is the example's own
shape (`crates/sile/examples/*.rs`), so a generated kernel and the reference
are fed the same problem.

The WGSL references share one uniform `Params { dims, scalars }` and bind their
storage buffers after it, so one runner covers all of them.

## Running

```
python handwritten/run.py            # self-check every reference, then time it
python handwritten/run.py --check    # self-checks only
python handwritten/run.py --only gemm
```

The C side compiles with the backend's own flags (`cc -O2 -fopenmp`), so a
comparison against generated C measures the lowering rather than a flag
difference. The WGSL side runs through `crates/handwritten-bench`, which uses
the same wgpu stack as the WGPU backend.

A reference that fails its own check fails the script. A reference is an oracle
only when it is right.

## The C references are written in the shape the lowering has to reach

`softmax` and `rms_norm` reduce with `#pragma omp simd reduction(...)` inside
one outer `parallel for` over rows, because that is exactly what plan 4.3 says
`TileReduce` becomes on this target. Written as plain scalar loops instead, the
same references measure 5.5x slower (softmax 4.09 -> 0.74 ms) and 3x slower
(rms_norm 0.320 -> 0.107 ms). A weak reference would make the 80% bar trivial
to clear, so the reference carries the target shape and not just the formula.

`gemm` packs both operands into `MR x NR`-per-tile contiguous panels and holds
the accumulator in registers: 114 GFLOP/s against 78 GFLOP/s for the naive
micro-kernel. That is the shape `Mma` has to give the backend.

Measured on an idle-ish Ryzen 9 8945HX / RTX 5070 Ti laptop, 32 OpenMP threads:

| kernel | C | WGSL |
|---|---|---|
| saxpy (16M elements) | 57-64 GB/s | 44 GB/s |
| softmax (4096x1024) | 0.65-0.71 ms, 95-103 GB/s | 43 GB/s |
| rms_norm (4096x1024) | 0.09-0.11 ms, 460-540 GB/s | 45 GB/s |
| gemm (1024^3) | 59-80 GFLOP/s | 31 GFLOP/s |

## How generated code gets compared

- **Numerics**: the generated kernel is fed the reference's self-check size and
  its output diffed against the reference's output, not against a third
  formula.
- **Performance**: the reference's `--bench` number is the 100% line; the bar is
  80% of it, per kernel, per target (plan 4.6 risk 3). Bench on an idle machine,
  and read the throughput column rather than wall-clock alone — a 4096x1024
  row kernel is a fraction of a millisecond on a discrete GPU, so wall-clock is
  dominated by launch overhead at that size.

## Known limits

- The C references are OpenMP over rows or tiles, not thread-pinned, and this
  machine runs other work: the fastest of 25 runs is reported because a single
  run otherwise swings by 6x (softmax measures 0.65-4.9 ms depending on the
  rep count and the load). Compare two labels back to back, on the same machine
  state, and read the throughput column rather than wall-clock alone.
- Bench sizes are chosen so the working set does not fit in cache, which is
  what makes the number a bandwidth number. `saxpy` at 4M elements reads
  252 GB/s because 48 MB fits in the 64 MB L3; at 16M it reports the honest
  57-64 GB/s of this machine's memory.
- The WGSL `gemm` reference reaches 391 GFLOP/s of f32 on an RTX 5070 Ti; it
  uses no tensor cores. The PTX reference in M4 is the one that will use
  `mma.sync`.
- `wgpu` needs every storage binding declared `read_write` here, because the
  runner builds one bind group layout for all of them. That is a runner
  constraint, not a reference one.
- The WGSL row kernels dispatch one workgroup per row (4096 workgroups of 256
  threads for 1024 columns), so each thread owns at most 4 elements. That is
  the shape the M3 lowering has to produce; it is not tuned further.
