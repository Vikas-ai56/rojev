#!/usr/bin/env bash
# hook.sh: Xiaomi-Robotics-1's side of the GPU setup interface (RoboCasa365 checkpoint as a remote S1 service).
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to executors/robocasa-xr1 (on the Mac
# or on the box) and ROJEV_ROOT pointing at the repo root there.
#
# Xiaomi-Robotics-1 specifics
#   - Env: this component's uv env (torch 2.8.0 cu128, transformers 4.57.1, flash-attn 2.8.3 from its sdist, which
#     fetches the matching prebuilt wheel; see pyproject.toml); no vendor clone (the checkpoint carries its own model
#     code, loaded with trust_remote_code at the pinned revision). The model needs flash-attn's CUDA kernels (no SDPA
#     support), so readiness imports them: a kernel-less flash-attn build fails check_remote / prepare_remote.
#   - Weights: config/xr1_weights.txt (public, no token), ~10.2 GB; peak 11.0 GB measured on an A100 (BF16).
#   - serve.py binds 127.0.0.1 and needs `Authorization: Bearer $ROJEV_TOKEN` (our wire; never the upstream pickle
#     socket server).
#   - No Mac-side assets.

set -euo pipefail

readonly REMOTE_PORT=8004
readonly IDLE_MIN=600
readonly GPU_MEM_MIB=11776

weights() {
  uv run --frozen --no-sync python "$ROJEV_ROOT/scripts/hf_pins.py" snapshot \
    --pin-file "$ROJEV_ROOT/config/xr1_weights.txt" "$@"
}

kernels() {
  uv run --frozen --no-sync python -c 'import torch, flash_attn_2_cuda' 2>/dev/null
}

key() {
  echo xr1
}

auth() {
  echo token
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock serve.py xr1_inputs.py
}

vendors() {
  return 0
}

remote_port() {
  echo "$REMOTE_PORT"
}

check_local() {
  return 0
}

prepare_local() {
  return 0
}

assets() {
  return 0
}

client_hint() {
  echo "cd executors/robocasa-pi052 && uv run python realtime.py --policy remote --s1 xr1 --service $1"
}

check_remote() {
  weights --offline >/dev/null && kernels
}

prepare_remote() {
  weights >/dev/null
  kernels || { echo "flash-attn CUDA kernels missing in the xr1 env (prebuilt wheel not fetched at build?)" >&2; exit 1; }
}

serve() {
  exec uv run --frozen --no-sync python serve.py --model "$(weights --offline)" --port "$REMOTE_PORT" \
    --idle-min "$IDLE_MIN"
}

case "${1:-}" in
  key|auth|gpu_mem_mib|files|vendors|remote_port|check_local|prepare_local|assets|client_hint|check_remote|prepare_remote|serve)
    "$@"
    ;;
  *)
    echo "usage: hook.sh {key|auth|gpu_mem_mib|files|vendors|remote_port|check_local|prepare_local|assets|client_hint JSON URL|check_remote|prepare_remote|serve}" >&2
    exit 2
    ;;
esac
