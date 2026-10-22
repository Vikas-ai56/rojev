#!/usr/bin/env bash
# hook.sh: MJev-Spatial's side of the GPU setup interface (System 2 spatial evidence service, serve_mjev.py).
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to services/mjev (on the Mac or on
# the box) and ROJEV_ROOT pointing at the repo root there.
#
# MJev specifics
#   - Env: services/mjev (mjev-spatial from vendor/MJev-Spatial at config/mjev.commit; torch 2.9 cu128).
#   - Weights: the checkpoint pinned in config/mjev_weights.txt; its config.json names the base model and revision
#     (base_model, base_revision) whose vision tower it loads, so the base is read from there, never hard-coded.
#   - Precision: bf16 language trunk on an A100 (fp16 was the T4 choice; both match fp32 top-1, E11 / E12f).
#   - serve_mjev.py binds 127.0.0.1 and needs `Authorization: Bearer $ROJEV_TOKEN` (auth = token, its own token).
#   - No Mac-side assets.

set -euo pipefail

readonly REMOTE_PORT=8001
readonly IDLE_MIN=600
readonly GPU_MEM_MIB=9728

pins() {
  uv run --frozen --no-sync python "$ROJEV_ROOT/scripts/hf_pins.py" "$@"
}

checkpoint() {
  pins snapshot --pin-file "$ROJEV_ROOT/config/mjev_weights.txt" "$@"
}

base_ref() {
  uv run --frozen --no-sync python -c \
    'import json, sys; c = json.load(open(sys.argv[1] + "/config.json")); print(c["base_model"] + "@" + c["base_revision"])' "$1"
}

key() {
  echo mjev
}

auth() {
  echo token
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock serve_mjev.py
}

vendors() {
  printf '%s\t%s\t%s\n' MJev-Spatial https://github.com/Fr0zenCrane/MJev-Spatial.git mjev.commit
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
  echo "MJev-Spatial at $2 (entry \"mjev\" {url, token} in $1; client rojev.system2.models.RemoteMJev)"
}

check_remote() {
  local ckpt
  ckpt=$(checkpoint --offline) && pins snapshot "$(base_ref "$ckpt")" --offline >/dev/null
}

prepare_remote() {
  local ckpt
  ckpt=$(checkpoint)
  pins snapshot "$(base_ref "$ckpt")"
}

serve() {
  local ckpt base
  ckpt=$(checkpoint --offline)
  base=$(pins snapshot "$(base_ref "$ckpt")" --offline)
  exec uv run --frozen --no-sync python serve_mjev.py --checkpoint "$ckpt" --base-dir "$base" --device cuda:0 \
    --language-dtype bf16 --port "$REMOTE_PORT" --idle-min "$IDLE_MIN"
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
