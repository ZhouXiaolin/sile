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
    RUSTC_WRAPPER= SILE_PRINT_LLIR=1 rtk cargo run --example "$example" >"$out_file" 2>&1
  )

  extract_llir "$out_file" >"$llir_file"

  local lines
  local allocas
  local tile_expr
  local tile_store_fused
  local tile_store

  lines="$(wc -l <"$llir_file" | tr -d ' ')"
  allocas="$(rg -c 'alloca' "$llir_file" || true)"
  tile_expr="$(rg -c 'tile_expr_loop' "$llir_file" || true)"
  tile_store_fused="$(rg -c 'tile_store_fused_loop' "$llir_file" || true)"
  tile_store="$(rg -c 'tile_store_loop' "$llir_file" || true)"

  printf '%-8s lines=%-4s allocas=%-2s tile_expr=%-2s tile_store_fused=%-2s tile_store=%-2s\n' \
    "$example" "$lines" "$allocas" "${tile_expr:-0}" "${tile_store_fused:-0}" "${tile_store:-0}"
}

report_example softmax
report_example matmul
