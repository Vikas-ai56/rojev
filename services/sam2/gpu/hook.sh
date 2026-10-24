#!/usr/bin/env bash
# hook.sh: SAM 2.1's side of the GPU setup interface (System 2 object-mask service, serve_sam2.py).
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to services/sam2 (on the Mac or on
# the box) and ROJEV_ROOT pointing at the repo root there.
#
# SAM 2.1 specifics
#   - Env: services/sam2 (transformers' SAM 2 port; torch 2.9 cu128). No vendor clone.
#   - Weights: facebook/sam2.1-hiera-tiny at the revision pinned in config/sam2_weights.txt (public, no token).
#   - serve_sam2.py binds 127.0.0.1 and needs `Authorization: Bearer $ROJEV_TOKEN` (auth = token, its own token).
#   - No Mac-side assets.

set -euo pipefail

readonly REMOTE_PORT=8003
readonly IDLE_MIN=600
readonly GPU_MEM_MIB=3072

weights() {
  uv run --frozen --no-sync python "$ROJEV_ROOT/scripts/hf_pins.py" snapshot \
    --pin-file "$ROJEV_ROOT/config/sam2_weights.txt" "$@"
}

key() {
  echo sam2
}

auth() {
  echo token
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock serve_sam2.py
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
  echo "SAM 2.1 at $2 (entry \"sam2\" {url, token} in $1; client rojev.system2.models.RemoteSam2)"
}

check_remote() {
  weights --offline >/dev/null
}

prepare_remote() {
  weights >/dev/null
}

serve() {
  exec uv run --frozen --no-sync python serve_sam2.py --weights "$(weights --offline)" --device cuda:0 \
    --port "$REMOTE_PORT" --idle-min "$IDLE_MIN"
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
