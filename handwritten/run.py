#!/usr/bin/env python3
"""Build, self-check and time the handwritten references.

M0 of docs/BACKEND_PLAN.md: each performance target needs a handwritten
reference first, so that "similar performance" has a definition. This script is
the one entry point for them:

    python handwritten/run.py              # check every reference, then bench
    python handwritten/run.py --check      # self-checks only
    python handwritten/run.py --only saxpy,gemm

The C references are compiled with the same flags the C backend uses
(`cc -O2 -fopenmp`), so a comparison against generated code measures the
lowering. The WGSL references run through `crates/handwritten-bench`, which
uses the same wgpu stack as the WGPU backend. A reference that fails its own
self-check fails this script: a reference is an oracle only when it is right.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / ".build"

C_KERNELS = ["saxpy", "softmax", "rms_norm", "gemm"]
WGSL_KERNELS = ["saxpy", "softmax", "rms_norm", "gemm"]

# The headline number each reference reports, and its unit.
BENCH_RE = re.compile(r"BENCH\s+(\S+)\s+([\d.]+)\s+ms(?:\s+([\d.]+)\s+(\S+))?")
CHECK_RE = re.compile(r"^(PASS|FAIL)\s+(.*)$")


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# The C backend's own flags: Apple's clang has no `-fopenmp`, so macOS names
# libomp explicitly. A reference built with different flags would compare
# lowering against a compiler setting.
CFLAGS = ["-O2"]
if sys.platform == "darwin":
    CFLAGS += ["-Xpreprocessor", "-fopenmp", "-I/usr/local/opt/libomp/include",
               "-L/usr/local/opt/libomp/lib", "-lomp"]
else:
    CFLAGS += ["-fopenmp"]


def build_c(kernels):
    BUILD.mkdir(exist_ok=True)
    for name in kernels:
        exe = BUILD / name
        if exe.exists() and exe.stat().st_mtime > (HERE / f"{name}.c").stat().st_mtime:
            continue
        done = run(["cc", *CFLAGS, "-o", str(exe), str(HERE / f"{name}.c"), "-lm"])
        if done.returncode != 0:
            print(f"cc failed for {name}:\n{done.stderr}", file=sys.stderr)
            sys.exit(1)


def bench_c(kernels, check):
    rows = []
    for name in kernels:
        args = [str(BUILD / name)] + ([] if check else ["--bench"])
        done = run(args)
        if done.returncode != 0:
            print(done.stdout, done.stderr, file=sys.stderr)
            sys.exit(1)
        checks = [line for line in done.stdout.splitlines() if CHECK_RE.match(line)]
        for line in checks:
            print(line)
        if any(line.startswith("FAIL") for line in checks):
            print(f"{name}: reference is wrong, fix it before it is used as an oracle", file=sys.stderr)
            sys.exit(1)
        if not check:
            match = BENCH_RE.search(done.stdout)
            if not match:
                print(f"{name}: no BENCH line\n{done.stdout}", file=sys.stderr)
                sys.exit(1)
            rows.append((name, "C", float(match.group(2)), match.group(3) or "", match.group(4) or ""))
    return rows


def bench_wgsl(kernels, check):
    rows = []
    exe = ROOT / "target" / "release" / "handwritten-bench"
    if not exe.exists():
        done = run(["cargo", "build", "--release", "-p", "handwritten-bench"], cwd=ROOT)
        if done.returncode != 0:
            print(done.stderr, file=sys.stderr)
            sys.exit(1)
    for name in kernels:
        done = run([str(exe), name])
        if done.returncode != 0:
            print(done.stdout, done.stderr, file=sys.stderr)
            sys.exit(1)
        lines = done.stdout.splitlines()
        for index, line in enumerate(lines):
            if CHECK_RE.match(line):
                print(line)
        if check:
            continue
        # The bench prints the wall clock, then the derived throughput on the
        # next line.
        match = next((BENCH_RE.search(l) for l in lines if BENCH_RE.search(l)), None)
        if match is None:
            print(f"{name}: no BENCH line\n{done.stdout}", file=sys.stderr)
            sys.exit(1)
        derived = next((l.split() for l in lines if l.startswith("       ")), ["", ""])
        rows.append((name, "WGSL", float(match.group(2)), derived[0], derived[1]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--check", action="store_true", help="self-checks only, no timing")
    args = parser.parse_args()
    chosen = [name.strip() for name in args.only.split(",") if name.strip()]
    c_kernels = [k for k in C_KERNELS if not chosen or k in chosen]
    wgsl_kernels = [k for k in WGSL_KERNELS if not chosen or k in chosen]

    build_c(c_kernels)
    rows = bench_c(c_kernels, args.check)
    rows += bench_wgsl(wgsl_kernels, args.check)

    if args.check:
        print("\nall references pass their self-check")
        return 0

    print()
    print(f"{'kernel':<12} {'target':<6} {'ms':>10}  {'metric':>12}")
    for name, device, ms, value, unit in rows:
        ms_text = "-" if ms is None else f"{ms:.3f}"
        metric = f"{value} {unit}".strip()
        print(f"{name:<12} {device:<6} {ms_text:>10}  {metric:>12}")
    print("\nThese numbers are the 100% line for M2/M3. Bench them on an idle machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
