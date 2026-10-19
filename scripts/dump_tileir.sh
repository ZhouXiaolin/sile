#!/bin/sh
# Export the Tile IR of every sile example into one directory, one file each.
#
#   scripts/dump_tileir.sh <out-dir>
#
# Each example runs with a fresh SILE_DUMP_DIR, and the first Tile IR snapshot
# (tileir-00-lowered) is copied out under the example's name. The examples are
# run, not inspected, because the HIR is built by the macro at compile time and
# a launch is what dumps it. An example that fails still leaves its dump, so a
# failure is reported but does not stop the export.
set -eu

EXAMPLES="saxpy async_add async_gemm batch_matmul dropout flash_attention \
flash_attention_causal gemm gemm_static rms_norm softmax"

OUT=${1:?usage: dump_tileir.sh <out-dir>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
status=0

for example in $EXAMPLES; do
    raw=$(mktemp -d)
    # Only the Tile IR stage: the rest of the pipeline is a full compile of
    # every backend, which the fixtures never read. In this mode the launch
    # returns before it runs, so the example's own result check fails; that is
    # expected and the dump is what this script is for.
    SILE_DUMP_TILEIR_ONLY=1 SILE_DUMP_DIR="$raw" \
        cargo run --quiet --manifest-path "$ROOT/Cargo.toml" \
        --example "$example" >"$raw/run.log" 2>&1 || true
    # An example that launches the same kernel twice (flash_attention_causal runs
    # the masked and the unmasked instantiation) leaves one dump per launch; the
    # fixture is the `-1` one, which is the masked run (`fmha-1` sorts before
    # `fmha/` because `-` precedes `/`).
    dump=$(find "$raw" -name tileir-00-lowered.txt | sort | head -1)
    if [ -z "$dump" ]; then
        printf '%-26s no dump\n' "$example"
        status=1
        rm -rf "$raw"
        continue
    fi
    cp "$dump" "$OUT/$example.tileir"
    printf '%-26s %s\n' "$example" "$OUT/$example.tileir"
    rm -rf "$raw"
done

exit "$status"
