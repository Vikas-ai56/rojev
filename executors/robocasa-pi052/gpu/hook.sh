#!/usr/bin/env bash
# hook.sh: robocasa-pi052's side of the GPU setup interface (π0.5² as a remote System 1 service).
#
# Called by scripts/gpu_setup.sh, always with the working directory set to this component's directory and
# ROJEV_ROOT set to the repo root: on the Mac for the Mac-side subcommands, and on the box (after the code mirror)
# for the box-side ones. Everything machine-level (driver, packages, uv, code mirror, vendor clones, `uv sync`,
# tokens, service JSON, systemd units, GPU memory check, tunnels) is owned by gpu_setup.sh; a hook owns only what
# is specific to its component.
#
# Interface (every component, executors/<name>/gpu/hook.sh and services/<name>/gpu/hook.sh, implements all of these)
#   Mac side
#     key                   entry name in the box's service JSON (e.g. pi052, mjev, onejev)
#     auth                  token (service checks `Authorization: Bearer $ROJEV_TOKEN`) or none (tunnel-only access)
#     gpu_mem_mib           GPU memory the running service needs; start is refused when less is free
#     files                 component files to mirror to the box, relative to its dir (gpu/ always is)
#     vendors               vendor repos its env needs: <dir>\t<git url>\t<pin file in config/>
#     remote_port           port the service binds on the box's 127.0.0.1
#     check_local           exit 0 when Mac-side assets are ready (read-only)
#     prepare_local         fetch Mac-side assets
#     assets                files to copy to the box: <local file>\t<box dir relative to $HOME>
#     client_hint JSON URL  print how to use the service from this Mac (service JSON path, tunnel URL)
#   Box side (uv on PATH, UV_PYTHON_PREFERENCE=only-managed, env already synced)
#     check_remote          exit 0 when box-side preparation is done (read-only)
#     prepare_remote        box-side preparation
#     serve                 exec the service in the foreground on 127.0.0.1:<remote_port> (reads $ROJEV_TOKEN if auth)
#
# robocasa-pi052 specifics
#   - Vendors: the LeRobot steerable branch, RoboCasa and robosuite (this env's path sources); OneJev belongs to the
#     rojev core and is declared by gpu_setup.sh.
#   - The PaliGemma tokenizer is gated: it is downloaded on the Mac with the Mac's own Hugging Face login and only
#     its files are copied to ~/paligemma-tokenizer on the box. No Hugging Face token ever reaches the box.
#   - The public weights (lerobot/pi052_robocasa, ~6.5 GB, and lerobot/fast-action-tokenizer) are pre-downloaded on
#     the box so the first service start does not spend minutes downloading.
#   - serve.py exits after IDLE_MIN minutes without requests; re-running gpu_setup.sh starts it again.
#   - PI052_SERVE_FLAGS (unit environment) passes extra serve.py options, e.g. --planner-memory for an A/B run.

set -euo pipefail

readonly REMOTE_PORT=8000
readonly IDLE_MIN=600
readonly GPU_MEM_MIB=10240
readonly TOKENIZER_BOX_DIR=paligemma-tokenizer

hub() {
  uv run --frozen --no-sync python -m gpu.hub "$@"
}

key() {
  echo pi052
}

auth() {
  echo token
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock drive.py embodiment.py serve.py
}

vendors() {
  printf '%s\t%s\t%s\n' \
    lerobot-steerable https://github.com/huggingface/lerobot.git lerobot-steerable.commit \
    robocasa https://github.com/robocasa/robocasa.git robocasa.commit \
    robosuite https://github.com/ARISE-Initiative/robosuite.git robosuite.commit
}

remote_port() {
  echo "$REMOTE_PORT"
}

check_local() {
  hub tokenizer --offline >/dev/null
}

prepare_local() {
  hub tokenizer >/dev/null
}

assets() {
  local path
  hub tokenizer --offline | while IFS= read -r path; do
    printf '%s\t%s\n' "$path" "$TOKENIZER_BOX_DIR"
  done
}

client_hint() {
  echo "cd executors/robocasa-pi052 && uv run python realtime.py --policy remote --service $1"
}

check_remote() {
  hub weights --offline >/dev/null
}

prepare_remote() {
  hub weights
}

serve() {
  exec uv run --frozen --no-sync python serve.py --tokenizer-dir "$HOME/$TOKENIZER_BOX_DIR" \
    --port "$REMOTE_PORT" --idle-min "$IDLE_MIN" ${PI052_SERVE_FLAGS:-}
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
