#!/usr/bin/env bash
# hook.sh: LIBERO's side of the GPU setup interface: a robot plug-in whose simulator runs on the box, next to the S1
# service it is evaluated against. It serves nothing.
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to executors/libero (on the Mac or on
# the box) and ROJEV_ROOT pointing at the repo root there.
#
# LIBERO specifics
#   - Env: this component's uv env; on Linux it installs lerobot[libero] 0.6.1 with LeRobot's pinned simulator set
#     (hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.3.1; see pyproject.toml) and CPU torch (no model runs here).
#   - Setup: `gpu_setup.sh HOST --executor libero --skip-serve` (with the S1 service in its own run): there is no
#     service, port or tunnel. key/auth/remote_port/gpu_mem_mib answer what the interface requires: auth none, port 0
#     (binds nothing), 0 MiB (EGL rendering needs no reserved GPU memory). `serve` refuses.
#   - Assets: nothing is copied from the Mac. Box-side preparation (gpu/box_setup.py) downloads LIBERO's assets
#     (dataset lerobot/libero-assets at LeRobot's pinned revision, public, no token), writes ~/.libero/config.yaml
#     (LIBERO prompts for input on first import without it) and links ~/.cache/libero/assets to the assets, as
#     LeRobot's LIBERO image does.
#   - Readiness: that preparation is in place (offline check) and LIBERO renders one reset with MUJOCO_GL=egl (LeRobot's
#     reset protocol, then one step; a blank frame fails). EGL needs the NVIDIA driver's libEGL on the box.
#   - Evaluation runs on the box through gpu/session.sh (scripts/gpu_session.sh), results under
#     $RESULTS/libero/<step>/.

set -euo pipefail

box_setup() {
  uv run --frozen --no-sync python -m gpu.box_setup "$@"
}

key() {
  echo libero
}

auth() {
  echo none
}

gpu_mem_mib() {
  echo 0
}

files() {
  printf '%s\n' pyproject.toml uv.lock backend.py embodiment.py evaluate.py suites.py
}

vendors() {
  return 0
}

remote_port() {
  echo 0
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
  echo "LIBERO runs on the box: scripts/gpu_session.sh HOST with the executors/libero steps (bash gpu/session.sh --list)"
}

check_remote() {
  box_setup check && MUJOCO_GL=egl box_setup render >/dev/null
}

prepare_remote() {
  box_setup prepare
  MUJOCO_GL=egl box_setup render
}

serve() {
  echo "executors/libero is a robot plug-in and serves nothing: set it up with gpu_setup.sh --skip-serve" >&2
  exit 1
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
