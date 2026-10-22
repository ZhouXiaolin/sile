#!/bin/sh
# Check every sile example's Tile IR against the cutile-rs golden fixture.
#
#   scripts/check_tileir_alignment.sh [out-dir]
#
# The fixtures under crates/sile/tests/fixtures/tileir/ are the printed Tile IR
# of the matching cutile-rs kernel (cutile/tests/tileir_alignment.rs, compiled
# with the same generics and strides). This script dumps sile's own Tile IR and
# diffs the two byte for byte. Exit status is 0 only when every fixture matches.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$(mktemp -d)}
FIXTURES="$ROOT/crates/sile/tests/fixtures/tileir"

"$ROOT/scripts/dump_tileir.sh" "$OUT" || true

passed=0
failed=0
for fixture in "$FIXTURES"/*.tileir; do
    name=$(basename "$fixture" .tileir)
    actual="$OUT/$name.tileir"
    if [ ! -f "$actual" ]; then
        printf '%-26s FAIL no output\n' "$name"
        failed=$((failed + 1))
        continue
    fi
    if python3 "$ROOT/scripts/tileir_diff.py" "$fixture" "$actual"; then
        printf '%-26s PASS\n' "$name"
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
    fi
done

printf '%d/%d PASS\n' "$passed" "$((passed + failed))"
[ "$failed" -eq 0 ]
