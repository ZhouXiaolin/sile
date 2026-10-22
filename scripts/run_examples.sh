#!/bin/sh
# Run every example under a wall-clock and a memory limit.
#
#   scripts/run_examples.sh [--timeout SECONDS] [--mem MB] [--device NAME] [--all] [example...]
#
# A lowering that loops forever or allocates without bound has to die, not hang
# the terminal for an hour. Each example gets its own limit, and the run
# reports TIMEOUT or KILLED instead of blocking.
#
# --all runs the suite on every backend (C, WGPU, CUDA) in turn and fails if
# any of them fails. A backend the machine cannot reach (no CUDA driver, no
# wgpu adapter) is reported SKIPPED, not failed: the gate is about kernels
# that hang or balloon, not about hardware inventory.
#
# The all-mode default timeout is larger than the single-device one:
# flash_attention_causal takes ~80s on the wgpu path (two 40s variants) where
# the C path takes ~7s, and a 60s gate would flag a slow-but-correct kernel
# as a hang.
set -u

TIMEOUT=${TIMEOUT:-60}
MEM_MB=${MEM_MB:-8192}
DEVICE=${SILE_DEVICE:-C}
ALL=0
ALL_DEFAULT=0

while [ $# -gt 0 ]; do
    case $1 in
        --timeout) TIMEOUT=$2; shift 2 ;;
        --mem) MEM_MB=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        --all) ALL=1; ALL_DEFAULT=1; shift ;;
        *) break ;;
    esac
done

if [ "$TIMEOUT" = 60 ] && [ $ALL_DEFAULT = 1 ]; then
    TIMEOUT=180
fi

if [ $ALL -eq 1 ]; then
    status=0
    for dev in C WGPU CUDA; do
        echo "== $dev =="
        SILE_DEVICE="$dev" "$0" --timeout "$TIMEOUT" --mem "$MEM_MB" --device "$dev" "$@" || status=1
    done
    exit $status
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT/target/release/examples"

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
    # with a bad_alloc instead of letting the machine swap. 0 disables it — a
    # Vulkan driver reserves virtual address space far past any RSS budget, so
    # a cgroup-style ceiling (systemd-run MemoryMax) is the gate to use here,
    # not this one.
    (
        [ "$MEM_MB" -gt 0 ] && ulimit -v $((MEM_MB * 1024)) 2>/dev/null || true
        SILE_DEVICE="$DEVICE" exec timeout --kill-after=5 "$TIMEOUT" "$BIN/$example"
    ) >"$log" 2>&1
    code=$?
    if [ "$code" -eq 1 ] && grep -q 'InvalidDevice\|no CUDA-capable device\|no suitable adapter\|AdapterNotFound' "$log" 2>/dev/null; then
        echo "SKIP (backend unavailable)"
        continue
    fi
    case $code in
        0) echo "OK" ;;
        124|137) echo "TIMEOUT after ${TIMEOUT}s"; status=1 ;;
        134) echo "ABORT (assert/panic)"; grep -m1 -E 'panicked|assert' "$log"; status=1 ;;
        *) echo "FAIL ($code)"; grep -m2 -E 'Error|panicked' "$log"; status=1 ;;
    esac
    rm -f "$log"
done
exit $status
