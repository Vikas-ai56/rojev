#!/usr/bin/env bash
# hook.sh: the LeRobot policy service's side of the GPU setup interface (any LeRobot-format checkpoint as a remote S1).
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to services/lerobot-policy (on the Mac
# or on the box) and ROJEV_ROOT pointing at the repo root there.
#
# LeRobot policy specifics
#   - Env: this service's uv env; on Linux LeRobot 0.6.1 with its groot extra (torch / torchvision from PyTorch's
#     CUDA 12.8 index, transformers 5.5, SDPA attention: no flash-attn, no Isaac-GR00T). Readiness imports
#     LeRobot's GR00T modeling module, the adapter and the server.
#   - Models: every config/s1/*.yaml with `adapter: lerobot` (model_config.py) is prepared, so any of them can be
#     served without a download: each pinned snapshot it names (the checkpoint and pinned overrides such as GR00T's
#     base model nvidia/GR00T-N1.7-3B; public) is pre-downloaded by scripts/hf_pins.py, which sends no token, and
#     each gated repo it loads by repo id (its `loaded_by_repo_id` pin files, e.g. GR00T's backbone
#     nvidia/Cosmos-Reason2-2B in config/groot_backbone.txt) is downloaded at its pin by scripts/hf_gated.py with
#     the owner's login, which then points the cache's refs/main at the pin.
#   - Login: needed only while some config names a gated repo. The Mac's Hugging Face token file (mode 600, checked)
#     is an asset copied as a file over ssh to the box's ~/.cache/huggingface/token, kept at mode 600, never read,
#     printed or put on a command line here.
#   - Serving: serve.py --config $LEROBOT_POLICY_CONFIG (repo-relative, default config/s1/groot-libero-goal.yaml).
#     GR00T runs online (decision 2026-11-07: the model's processor loads its backbone repo by id), so serve first
#     requires every gated repo of that config to still be the Hub's main at its pin.
#   - Memory: the GR00T N1.7 LIBERO checkpoints hold 12.6 GB of fp32 weights run under bf16 autocast; 16 GiB free
#     is the floor asked for here, not a measurement.
#   - serve.py binds 127.0.0.1 and needs `Authorization: Bearer $ROJEV_TOKEN`.

set -euo pipefail

readonly REMOTE_PORT=8007
readonly IDLE_MIN=600
readonly GPU_MEM_MIB=16384
readonly DEFAULT_CONFIG=config/s1/groot-libero-goal.yaml
readonly BOX_LOGIN_FILE=.cache/huggingface/token

python_env() {
  uv run --frozen --no-sync python "$@"
}

model_configs() {
  printf '%s\n' "$ROJEV_ROOT"/config/s1/*.yaml
}

served_config() {
  echo "${LEROBOT_POLICY_CONFIG:-$DEFAULT_CONFIG}"
}

listing() {
  local what=$1 config configs=()
  shift
  while IFS= read -r config; do
    configs+=("$config")
  done < <("$@")
  python_env model_config.py "$what" "${configs[@]}"
}

gated_pins() {
  listing loaded-by-repo-id "$@"
}

login_needed() {
  local pins
  pins=$(gated_pins model_configs) || exit 1
  [[ -n "$pins" ]]
}

gated() {
  python_env "$ROJEV_ROOT/scripts/hf_gated.py" "$@"
}

gated_snapshots() {
  local source=$1 pins pin
  shift
  pins=$(gated_pins "$source") || return 1
  for pin in $pins; do
    gated snapshot --pin-file "$ROJEV_ROOT/$pin" "$@" || return 1
  done
}

public_snapshots() {
  local refs ref
  refs=$(listing pinned model_configs) || return 1
  for ref in $refs; do
    python_env "$ROJEV_ROOT/scripts/hf_pins.py" snapshot "$ref" "$@" >/dev/null || return 1
  done
}

served_config_file() {
  echo "$ROJEV_ROOT/$(served_config)"
}

imports() {
  python_env -c 'import lerobot.policies.groot.modeling_groot, adapter, serve'
}

key() {
  echo lerobot-policy
}

auth() {
  echo token
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock adapter.py model_config.py serve.py
}

vendors() {
  :
}

remote_port() {
  echo "$REMOTE_PORT"
}

check_local() {
  ! login_needed || gated login >/dev/null
}

prepare_local() {
  check_local || {
    echo "log in on this Mac first (hf auth login) and accept the gated repos the model configs name" >&2
    exit 1
  }
}

assets() {
  ! login_needed || gated login --asset
}

client_hint() {
  echo "service entry 'lerobot-policy' in $1 ($2): a rojev RemotePolicy serving $(served_config)"
}

check_remote() {
  gated_snapshots model_configs --check && public_snapshots --offline && imports >/dev/null 2>&1
}

prepare_remote() {
  ! login_needed || chmod 600 "$HOME/$BOX_LOGIN_FILE"
  gated_snapshots model_configs
  public_snapshots
  imports || { echo "the lerobot-policy env cannot import LeRobot's GR00T policy or the adapter" >&2; exit 1; }
}

serve() {
  gated_snapshots served_config_file --check
  exec uv run --frozen --no-sync python serve.py --config "$(served_config)" --port "$REMOTE_PORT" \
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
