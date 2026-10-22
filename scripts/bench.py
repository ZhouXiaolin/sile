#!/usr/bin/env python3
"""The three-target benchmark: what the compiler does to each real kernel.

One run of an example is one compilation of one kernel. The dump directory
records every stage of that compilation, so this script reads the compiler's
own timings and sizes out of it and adds the numbers the compiler cannot
report: how long the whole launch took, backend included, and how big each
target's source came out.

The rows are the comparison unit: two runs with different `--label` values
produce two `summary.tsv` files that `compare.py` diffs.

Usage:
    python scripts/bench.py --device C --label baseline
    python scripts/bench.py --device WGPU --label after
    python scripts/bench.py --device CUDA --only gemm,fmha
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = [
    "saxpy",
    "softmax",
    "gemm",
    "gemm_static",
    "batch_matmul",
    "rms_norm",
    "dropout",
    "flash_attention",
    "flash_attention_causal",
    "async_add",
    "async_gemm",
]


def read_tsv(path):
    """Read a two-or-more column TSV into a dict keyed by its first column."""
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return rows
        for row in reader:
            if row:
                rows[row[0]] = row[1:]
    return rows


def stage_sum(timing, prefix):
    return sum(int(value[0]) for key, value in timing.items() if key.startswith(prefix))


def median_timing(timing_runs):
    """Fold one timing.tsv per attempt into one dict: the median per stage."""
    runs = [timing for timing in timing_runs if timing]
    merged = {}
    for label in sorted({key for timing in runs for key in timing}):
        seen = sorted(int(timing[label][0]) for timing in runs if label in timing)
        merged[label] = [str(seen[len(seen) // 2])]
    return merged


def last_lsir_metrics(metrics):
    keys = sorted(key for key in metrics if key.startswith("lsir-"))
    if not keys:
        return ["0"] * 7
    return metrics[keys[-1]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="C")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--only", default="")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each kernel this many times; report the per-stage median")
    parser.add_argument("--profile", default="release", choices=["release", "debug"],
                        help="which built example binaries to run")
    args = parser.parse_args()

    chosen = [name.strip() for name in args.only.split(",") if name.strip()] or EXAMPLES
    out_dir = ROOT / "dumps" / f"bench-{args.label}-{args.device}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for example in chosen:
        exe = ROOT / "target" / args.profile / "examples" / example
        if not exe.exists():
            exe = exe.with_name(example + ".exe")
        if not exe.exists():
            print(f"warning: {example} is not built", file=sys.stderr)
            continue

        stage_dir = out_dir / example
        if stage_dir.exists():
            shutil.rmtree(stage_dir)

        env = dict(os.environ)
        env["SILE_DEVICE"] = args.device
        env["SILE_DUMP_DIR"] = str(stage_dir)

        attempt_ms, output, exit_code = [], "", 0
        for attempt in range(args.repeat):
            started = time.perf_counter()
            done = subprocess.run(
                [str(exe)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
            )
            attempt_ms.append((time.perf_counter() - started) * 1000)
            output, exit_code = done.stdout, done.returncode
        # A single run of a compile stage swings by several times on a loaded
        # machine; the median of a few runs is the number two labels can
        # compare honestly.
        elapsed_ms = sorted(attempt_ms)[len(attempt_ms) // 2]

        # The dump writes under a directory named after the kernel, which is
        # not always the example's name: flash_attention declares `fmha`. One
        # directory per attempt, because two instantiations never share one.
        kernel_dirs = sorted(path for path in stage_dir.iterdir() if path.is_dir()) \
            if stage_dir.exists() else []
        stage_dir = kernel_dirs[0] if kernel_dirs else stage_dir

        timing = median_timing(read_tsv(d / "timing.tsv") for d in kernel_dirs)
        metrics = read_tsv(stage_dir / "metrics.tsv")
        final = last_lsir_metrics(metrics)

        sizes = {}
        for target in ("c", "metal", "wgsl", "ptx"):
            path = stage_dir / f"source-{target}.txt"
            sizes[target] = len(path.read_text(encoding="utf-8", errors="replace").splitlines()) \
                if path.exists() else 0

        # Longest single pass: a compiler that is slow is slow in one of them.
        slowest = max(timing.items(), key=lambda item: int(item[1][0]), default=("-", ["0"]))
        last_line = [line for line in output.splitlines() if line.strip()]
        summary = last_line[-1] if last_line else ""

        rows.append(
            {
                "kernel": example,
                "device": args.device,
                "exit": exit_code,
                "e2e_ms": round(elapsed_ms),
                "hir_us": stage_sum(timing, "hir-"),
                "tileir_us": stage_sum(timing, "tileir-"),
                "lsir_us": stage_sum(timing, "lsir-"),
                "source_us": stage_sum(timing, "source-"),
                "slowest_stage": slowest[0],
                "slowest_us": int(slowest[1][0]),
                "ir_lines": int(final[1]) if len(final) > 1 else 0,
                "ir_blocks": int(final[2]) if len(final) > 2 else 0,
                "ir_insts": int(final[3]) if len(final) > 3 else 0,
                "ir_allocas": int(final[4]) if len(final) > 4 else 0,
                "ir_phis": int(final[5]) if len(final) > 5 else 0,
                "src_c": sizes["c"],
                "src_metal": sizes["metal"],
                "src_wgsl": sizes["wgsl"],
                "src_ptx": sizes["ptx"],
                "result": summary,
            }
        )
        print(
            f"{example:<24} exit={exit_code} {round(elapsed_ms):>7} ms  "
            f"compile={rows[-1]['hir_us'] + rows[-1]['tileir_us'] + rows[-1]['lsir_us'] + rows[-1]['source_us']:>9} us  "
            f"slowest={slowest[0]}({int(slowest[1][0])}us)  lsir lines={rows[-1]['ir_lines']:>5}"
        )

    summary_path = out_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["kernel"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
