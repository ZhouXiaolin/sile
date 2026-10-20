#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

extract_llir() {
  awk '
    /^define void @/ { flag = 1 }
    flag { print }
    /^}$/ { if (flag) exit }
  ' "$1"
}

report_example() {
  local example="$1"
  local out_file="$TMP_DIR/${example}.txt"
  local llir_file="$TMP_DIR/${example}.llir"

  (
    cd "$ROOT_DIR"
    RUSTC_WRAPPER= SILE_PRINT_LLVMIR=1 cargo run --example "$example" >"$out_file" 2>&1
  )

  extract_llir "$out_file" >"$llir_file"

  local lines
  local allocas
  local phis

  lines="$(wc -l <"$llir_file" | tr -d ' ')"
  allocas="$(rg -c 'alloca' "$llir_file" || true)"
  # LOWER_DESIGN §5: the descent emits no phi; Mem2Reg is SROA's promote.
  phis="$(rg -c ' phi ' "$llir_file" || true)"

  printf '%-8s lines=%-6s allocas=%-4s phi=%-3s\n' \
    "$example" "$lines" "${allocas:-0}" "${phis:-0}"
}

report_example saxpy
report_example softmax
report_example gemm
