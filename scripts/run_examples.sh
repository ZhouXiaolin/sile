#!/bin/sh
# Run every example under a wall-clock and a memory limit.
#
#   scripts/run_examples.sh [--timeout SECONDS] [--mem MB] [--device NAME] [example...]
#
# A lowering that loops forever or allocates without bound has to die, not hang
# the terminal for an hour. Each example gets its own limit, and the run
# reports TIMEOUT or KILLED instead of blocking.
set -u

TIMEOUT=${TIMEOUT:-60}
MEM_MB=${MEM_MB:-8192}
DEVICE=${SILE_DEVICE:-C}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT/target/release/examples"

while [ $# -gt 0 ]; do
    case $1 in
        --timeout) TIMEOUT=$2; shift 2 ;;
        --mem) MEM_MB=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        *) break ;;
    esac
done

if [ $# -gt 0 ]; then
    EXAMPLES=$*
else
    EXAMPLES="saxpy softmax rms_norm gemm gemm_static batch_matmul \
flash_attention flash_attention_causal"
fi

status=0
for example in $EXAMPLES; do
    printf '%-24s ' "$example"
    log=$(mktemp)
    # `ulimit -v` is a virtual-memory ceiling: it stops an unbounded allocator
    # with a bad_alloc instead of letting the machine swap.
    (
        ulimit -v $((MEM_MB * 1024)) 2>/dev/null || true
        SILE_DEVICE="$DEVICE" exec timeout --kill-after=5 "$TIMEOUT" "$BIN/$example"
    ) >"$log" 2>&1
    code=$?
    case $code in
        0) echo "OK" ;;
        124|137) echo "TIMEOUT after ${TIMEOUT}s"; status=1 ;;
        134) echo "ABORT (assert/panic)"; grep -m1 -E 'panicked|assert' "$log"; status=1 ;;
        *) echo "FAIL ($code)"; grep -m2 -E 'Error|panicked' "$log"; status=1 ;;
    esac
    rm -f "$log"
done
exit $status
