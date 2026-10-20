#!/usr/bin/env python3
"""Interleaved benchmark: sile's C backend against the handwritten reference.

Running the two sides one after the other measures the machine's load as much
as the kernels: on this box the same reference swings between 15 and 104 GB/s
depending on what else is running. This script alternates the two sides inside
one loop and reports each side's best time, so both see the same conditions.

Each side is a process, so the alternation is between processes: the reference
is the already-built `handwritten/<kernel>` binary and the sile side is
`target/release/examples/bench_m2`. Both print one `BENCH <name> <ms> ms <rate>`
line and both report their own median of 100 launches (the first ~20 omp
regions of a process run under a flat cold-start transient, so the median has
to outvote them).

Usage:
    python3 scripts/compare_golden.py --rounds 5
    python3 scripts/compare_golden.py --rounds 3 --only softmax
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ["saxpy", "softmax", "rms_norm"]
# The rate is reported in GB/s by both sides, so the ratio needs no conversion.
LINE = re.compile(r"^BENCH\s+(\S+)\s+([\d.]+)\s+ms\s+([\d.]+)\s+GB/s", re.MULTILINE)
REFERENCE = ROOT / "handwritten" / ".build"


def rates(output: str) -> dict[str, float]:
    return {name: float(rate) for name, _, rate in LINE.findall(output)}


def run(cmd: list[str], env: dict[str, str]) -> dict[str, float]:
    done = subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    if done.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{done.stderr[-2000:]}")
    return rates(done.stdout + done.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--only", default="")
    parser.add_argument("--threshold", type=float, default=80.0)
    args = parser.parse_args()

    chosen = [k.strip() for k in args.only.split(",") if k.strip()] or KERNELS
    missing = [name for name in chosen if not (REFERENCE / name).exists()]
    if missing:
        sys.exit(f"build the references first: python3 handwritten/run.py --check ({missing})")
    env = {"PATH": "/usr/bin:/bin", "SILE_DEVICE": "C"}

    best: dict[str, dict[str, float]] = {k: {"sile": 0.0, "hand": 0.0} for k in chosen}
    for round_index in range(args.rounds):
        # Reference first on even rounds and second on odd ones, so neither side
        # always runs on the colder machine.
        order = ["hand", "sile"] if round_index % 2 == 0 else ["sile", "hand"]
        for side in order:
            if side == "hand":
                # The reference is one binary per kernel; `handwritten/run.py`
                # builds them, so they are already there.
                found = {}
                for name in chosen:
                    found.update(run([str(REFERENCE / name), "--bench"], env))
            else:
                found = run(
                    ["target/release/examples/bench_m2", *chosen], env
                )
            for name, rate in found.items():
                if name in best and rate > best[name][side]:
                    best[name][side] = rate
        print(f"round {round_index + 1}/{args.rounds} done", file=sys.stderr)

    print(f"{'kernel':<12} {'sile GB/s':>10} {'hand GB/s':>10} {'ratio':>8}  verdict")
    failed = []
    for name in chosen:
        sile = best[name]["sile"]
        hand = best[name]["hand"]
        ratio = 100.0 * sile / hand if hand else 0.0
        verdict = "OK" if ratio >= args.threshold else "SHORT"
        if ratio < args.threshold:
            failed.append(name)
        print(f"{name:<12} {sile:>10.2f} {hand:>10.2f} {ratio:>7.1f}%  {verdict}")

    print(f"\nthreshold {args.threshold:.0f}%")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
