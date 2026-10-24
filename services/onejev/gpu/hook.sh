#!/usr/bin/env bash
# hook.sh: OneJev's side of the GPU setup interface (System 2 decision model served by `qev serve`).
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to services/onejev (on the Mac or on
# the box) and ROJEV_ROOT pointing at the repo root there.
#
# OneJev specifics
#   - Env: services/onejev (qev[torch] from vendor/OneJev, CUDA torch from the cu128 index).
#   - Weights: the HF revision pinned in config/onejev_weights.txt (OmniJev/OneJev1.6-4B, ~9 GB), pre-downloaded on
#     the box and served from the snapshot directory.
#   - qev has no authentication and binds 0.0.0.0 by default, so `--host 127.0.0.1` is mandatory: the SSH tunnel is
#     the only way in, and the service entry carries a url and no token (auth = none).
#   - No Mac-side assets.

set -euo pipefail

readonly REMOTE_PORT=8002
readonly GPU_MEM_MIB=14336

pins() {
  uv run --frozen --no-sync python "$ROJEV_ROOT/scripts/hf_pins.py" "$@" --pin-file "$ROJEV_ROOT/config/onejev_weights.txt"
}

key() {
  echo onejev
}

auth() {
  echo none
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock
}

vendors() {
  printf '%s\t%s\t%s\n' OneJev https://github.com/OmniJev/OneJev.git onejev.commit
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
  echo "OneJev (qev, no auth, only through the tunnel) at $2: realtime.py reads its 'onejev' entry from $1"
}

check_remote() {
  pins snapshot --offline >/dev/null
}

prepare_remote() {
  pins snapshot
}

serve() {
  local model name
  model=$(pins snapshot --offline)
  name=$(pins repo)
  exec uv run --frozen --no-sync qev serve --model "$model" --name "${name##*/}" --multimodal --dtype bfloat16 \
    --device cuda:0 --host 127.0.0.1 --port "$REMOTE_PORT"
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
