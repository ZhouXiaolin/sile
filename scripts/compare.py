#!/usr/bin/env python3
"""Diff two benchmark labels into one comparison table.

Reads the `summary.tsv` each `bench.py` label wrote and prints, per kernel
and device, what changed: the compile-time stage totals, the final IR shape,
the per-target source sizes and the end-to-end launch time. A column whose
value did not move prints as a dash, so the rows that matter are the ones
with numbers in them.

Usage:
    python scripts/compare.py --before baseline --after licm-cse-fusion
    python scripts/compare.py --before baseline --after licm --device CUDA
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COLUMNS = [
    "e2e_ms",
    "hir_us",
    "tileir_us",
    "lsir_us",
    "source_us",
    "ir_lines",
    "ir_blocks",
    "ir_insts",
    "ir_allocas",
    "ir_phis",
    "src_c",
    "src_metal",
    "src_wgsl",
    "src_ptx",
]
# A percent under this is noise from process and timer jitter.
QUIET_PERCENT = 2.0


def read_summary(label, device):
    path = ROOT / "dumps" / f"bench-{label}-{device}" / "summary.tsv"
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["kernel"]] = row
    return rows


def fmt_change(before, after):
    try:
        old, new = int(before), int(after)
    except (TypeError, ValueError):
        return f"{before}->{after}"
    if old == new:
        return "-"
    if old == 0:
        return f"0->{new}"
    percent = (new - old) * 100.0 / old
    if abs(percent) < QUIET_PERCENT:
        return "-"
    arrow = "+" if new > old else ""
    return f"{old}->{new} ({arrow}{percent:.0f}%)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--device", default="C")
    args = parser.parse_args()

    before = read_summary(args.before, args.device)
    after = read_summary(args.after, args.device)

    keys = sorted(set(before) & set(after))
    missing = sorted(set(before) ^ set(after))
    for kernel in missing:
        print(f"note: {kernel} is only in one label")

    header = ["kernel"] + COLUMNS
    print("\t".join(header))
    for kernel in keys:
        old_row, new_row = before[kernel], after[kernel]
        cells = [kernel]
        for column in COLUMNS:
            cells.append(fmt_change(old_row.get(column), new_row.get(column)))
        print("\t".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
