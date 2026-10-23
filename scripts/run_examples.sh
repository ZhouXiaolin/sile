#!/bin/sh
# Run every example under a wall-clock and a memory limit.
#
#   scripts/run_examples.sh [--timeout SECONDS] [--mem MB] [--device NAME] [--all] [example...]
#
# A lowering that loops forever or allocates without bound has to die, not hang
# the terminal for an hour. Each example gets its own limit, and the run
# reports TIMEOUT or KILLED instead of blocking.
#
# --all runs the suite on every backend (C, WGPU, METAL, CUDA) in turn and
# fails if any of them fails. A backend the machine cannot reach (no CUDA
# driver, no wgpu adapter, no Metal device) is reported SKIPPED, not failed:
# the gate is about kernels that hang or balloon, not about hardware inventory.
#
# Ten seconds is the whole budget for one example. Every example is a small
# kernel — a few million elements or a 1024-cubed gemm — and the reference
# implementations finish in well under a second. An example that needs longer
# is not slow, it is wrong: a lowering that materialised a tile per thread, or
# a launch grid that walks a row one workgroup at a time.
#
# One known exception: `flash_attention_causal` on WGPU. The Metal driver
# compiles a pipeline state per distinct kernel, on its own XPC compiler
# service, and for the fmha kernels that costs about a minute per kernel the
# first time the exact source is seen. The driver caches persistently and the
# spelled source is deterministic (crates/sile-lsir/src/lower/module.rs keeps
# materialization order off HashMap iteration for exactly this reason), so
# every run after the first is seconds. A fresh machine — or a cache evicted
# by macOS — pays the one-time compile: run the suite as
# `scripts/run_examples.sh --all --timeout 500` once, then without.
set -u

TIMEOUT=${TIMEOUT:-10}
MEM_MB=${MEM_MB:-8192}
DEVICE=${SILE_DEVICE:-C}
ALL=0

while [ $# -gt 0 ]; do
    case $1 in
        --timeout) TIMEOUT=$2; shift 2 ;;
        --mem) MEM_MB=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        --all) ALL=1; shift ;;
        *) break ;;
    esac
done

if [ $ALL -eq 1 ]; then
    status=0
    for dev in C WGPU METAL CUDA; do
        echo "== $dev =="
        SILE_DEVICE="$dev" "$0" --timeout "$TIMEOUT" --mem "$MEM_MB" --device "$dev" "$@" || status=1
    done
    exit $status
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="$ROOT/target/release/examples"

# Two ways to hold a process to a memory budget. `ulimit -v` is the real
# ceiling: the allocator fails and the process dies. macOS refuses to lower it
# for a non-root user ("cannot modify limit: Invalid argument"), so there the
# only gate left is the peak the process reports, checked after it exits. Both
# are read here; `TIME_FLAGS` is empty when neither works.
TIME_BIN=""
TIME_FLAGS=""
if [ "$MEM_MB" -gt 0 ]; then
    if /usr/bin/time -l true >/dev/null 2>&1; then
        TIME_BIN=/usr/bin/time
        TIME_FLAGS=-l
    elif /usr/bin/time -v true >/dev/null 2>&1; then
        TIME_BIN=/usr/bin/time
        TIME_FLAGS=-v
    fi
fi

# The peak resident set the log's `time` report names, in MB. BSD `time -l`
# prints bytes, GNU `time -v` kilobytes and says so in the label. The peak is
# the resident set, not macOS's `peak memory footprint`, which leaves out the
# pages the process shares.
peak_mb() {
    awk '/maximum resident set size/ {
             if ($1 ~ /^[0-9]+$/) v = int($1 / 1048576)
             else v = int($NF / 1024)
             if (v > mb) mb = v
         }
         END {print mb + 0}' "$1"
}

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
    (
        [ -z "$TIME_FLAGS" ] && [ "$MEM_MB" -gt 0 ] && ulimit -v $((MEM_MB * 1024)) 2>/dev/null || true
        if [ -n "$TIME_FLAGS" ]; then
            SILE_DEVICE="$DEVICE" exec timeout --kill-after=5 "$TIMEOUT" \
                "$TIME_BIN" "$TIME_FLAGS" "$BIN/$example"
        fi
        SILE_DEVICE="$DEVICE" exec timeout --kill-after=5 "$TIMEOUT" "$BIN/$example"
    ) >"$log" 2>&1
    code=$?
    peak=$(peak_mb "$log")
    if [ "$code" -eq 1 ] && grep -q 'InvalidDevice\|no CUDA-capable device\|no suitable adapter\|AdapterNotFound\|cannot load libcuda' "$log" 2>/dev/null; then
        echo "SKIP (backend unavailable)"
        continue
    fi
    case $code in
        0)
            if [ "$MEM_MB" -gt 0 ] && [ "$peak" -gt "$MEM_MB" ]; then
                echo "OVER MEMORY (${peak} MB > ${MEM_MB} MB)"
                status=1
            else
                echo "OK (${peak} MB)"
            fi
            ;;
        124|137) echo "TIMEOUT after ${TIMEOUT}s"; status=1 ;;
        134) echo "ABORT (assert/panic)"; grep -m1 -E 'panicked|assert' "$log"; status=1 ;;
        *) echo "FAIL ($code)"; grep -m2 -E 'Error|panicked' "$log"; status=1 ;;
    esac
    rm -f "$log"
done
exit $status
