#!/usr/bin/env bash
# hook.sh: GR00T N1.7 (fine-tuned for the SO-101)'s side of the GPU setup interface, as a remote S1 service.
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to executors/so101-groot (on the Mac
# or on the box) and ROJEV_ROOT pointing at the repo root there.
#
# GR00T specifics
#   - Env: this component's uv env; on Linux it installs Isaac-GR00T pinned by commit (torch 2.9.0, transformers
#     4.57.3, flash-attn 2.8.3 from Isaac-GR00T's prebuilt-wheel source). Needs an Ampere-or-newer GPU (bf16; GR00T
#     casts the model to bf16). Without flash-attn's kernels GR00T silently falls back to SDPA, so readiness imports
#     them.
#   - Backbone: nvidia/Cosmos-Reason2-2B (config/groot_backbone.txt) is gated and loaded by repo id at start. The box
#     downloads it with the owner's Hugging Face login (decision 2026-11-07: uploading 4.9 GB from the Mac dominated
#     setup): the Mac's token file (mode 600, checked) is an asset copied as a file over ssh to the box's
#     ~/.cache/huggingface/token, kept at mode 600, never printed or put on a command line; `scripts/hf_gated.py
#     snapshot` downloads the pinned revision and names it in the cache ref. GR00T builds its backbone and processor
#     from this repo id (Isaac-GR00T @ 9fbe86a, gr00t/model/modules/qwen3_backbone.py:183,
#     processing_gr00t_n1d7.py:129) and runs online (transformers 4.57.3 queries the Hub while loading the tokenizer
#     even with HF_HUB_OFFLINE=1), so readiness also requires the Hub's main to still equal the pin
#     (`scripts/hf_gated.py snapshot --check`).
#   - Base model: nvidia/GR00T-N1.7-3B (config/groot_base_weights.txt), pre-downloaded on the box: zero-shot on its
#     pretraining embodiments (the DROID check) and the start of SO-101 fine-tunes.
#   - Vendor: Isaac-GR00T at config/isaac-groot.commit for its own tools (gr00t/eval/open_loop_eval.py,
#     scripts/deployment/benchmark_inference.py, gr00t/experiment/launch_finetune.py, examples/SO100); vendors clone
#     LFS files as pointers, so only the two sample datasets used here are pulled (demo_data/droid_sample,
#     demo_data/cube_to_bowl_5).
#   - Checkpoint to serve: $GROOT_CHECKPOINT (a directory on the box, e.g. a fine-tune output) or the SO-101
#     fine-tune pinned in config/groot_weights.txt (`repo <id> @ <revision>`). Neither exists until a fine-tune does;
#     every preparation step works without one, and `serve` stops with that message (run gpu_setup.sh --skip-serve).
#   - Video: GR00T decodes with torchcodec, which loads the system FFmpeg (4-7; gpu_box.sh installs ffmpeg);
#     readiness loads its decoders.
#   - Memory: Isaac-GR00T's README asks for a GPU with 16 GB+ for inference; not measured here yet.
#   - serve.py binds 127.0.0.1 and needs `Authorization: Bearer $ROJEV_TOKEN` (our wire; never GR00T's socket server).

set -euo pipefail

readonly REMOTE_PORT=8005
readonly IDLE_MIN=600
readonly GPU_MEM_MIB=16384
readonly CHECKPOINT_PIN=config/groot_weights.txt
readonly BASE_PIN=config/groot_base_weights.txt
readonly BACKBONE_PIN=config/groot_backbone.txt
readonly SAMPLES=("demo_data/droid_sample/**" "demo_data/cube_to_bowl_5/**")

gated() {
  uv run --frozen --no-sync python "$ROJEV_ROOT/scripts/hf_gated.py" "$@"
}

backbone() {
  gated snapshot --pin-file "$ROJEV_ROOT/$BACKBONE_PIN" "$@"
}

pinned() {
  uv run --frozen --no-sync python "$ROJEV_ROOT/scripts/hf_pins.py" snapshot --pin-file "$ROJEV_ROOT/$1" "${@:2}"
}

checkpoint() {
  if [[ -n "${GROOT_CHECKPOINT:-}" ]]; then
    [[ -f "$GROOT_CHECKPOINT/config.json" ]] || {
      echo "GROOT_CHECKPOINT=$GROOT_CHECKPOINT has no config.json" >&2
      return 1
    }
    echo "$GROOT_CHECKPOINT"
  elif [[ -f "$ROJEV_ROOT/$CHECKPOINT_PIN" ]]; then
    pinned "$CHECKPOINT_PIN" "$@"
  else
    echo "no GR00T checkpoint to serve: set GROOT_CHECKPOINT or pin a fine-tune in $CHECKPOINT_PIN" >&2
    return 1
  fi
}

isaac_groot() {
  echo "$ROJEV_ROOT/vendor/isaac-groot"
}

samples_ready() {
  local include
  for include in "${SAMPLES[@]}"; do
    [[ -n "$(git -C "$(isaac_groot)" lfs ls-files -I "$include")" ]] || return 1
    ! git -C "$(isaac_groot)" lfs ls-files -I "$include" | awk '$2 == "-"' | grep -q . || return 1
  done
}

samples_pull() {
  local include
  git -C "$(isaac_groot)" lfs install --local >/dev/null
  for include in "${SAMPLES[@]}"; do
    git -C "$(isaac_groot)" lfs pull -I "$include"
  done
}

kernels() {
  uv run --frozen --no-sync python -c 'import torch, flash_attn_2_cuda' 2>/dev/null
}

decoders() {
  uv run --frozen --no-sync python -c 'import torch, torchcodec.decoders' 2>/dev/null
}

key() {
  echo groot
}

auth() {
  echo token
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock serve.py groot_io.py modality.json parity.py probe.py
}

vendors() {
  printf '%s\t%s\t%s\n' isaac-groot https://github.com/NVIDIA/Isaac-GR00T.git isaac-groot.commit
}

remote_port() {
  echo "$REMOTE_PORT"
}

check_local() {
  gated login >/dev/null
}

prepare_local() {
  gated login >/dev/null || {
    echo "log in on this Mac first (hf auth login) and accept nvidia/Cosmos-Reason2-2B" >&2
    exit 1
  }
}

assets() {
  gated login --asset
}

client_hint() {
  echo "service entry 'groot' in $1 ($2): a rojev RemotePolicy for the SO-101 (front + wrist_cam, joint radians)"
}

check_remote() {
  backbone --check && pinned "$BASE_PIN" --offline >/dev/null && samples_ready && kernels && decoders || return 1
  [[ ! -f "$ROJEV_ROOT/$CHECKPOINT_PIN" ]] || pinned "$CHECKPOINT_PIN" --offline >/dev/null
}

prepare_remote() {
  chmod 600 "$HOME/.cache/huggingface/token"
  backbone
  pinned "$BASE_PIN" >/dev/null
  samples_pull
  [[ ! -f "$ROJEV_ROOT/$CHECKPOINT_PIN" ]] || pinned "$CHECKPOINT_PIN" >/dev/null
  kernels || { echo "flash-attn CUDA kernels missing in the groot env" >&2; exit 1; }
  decoders || { echo "torchcodec cannot load FFmpeg (needs the system ffmpeg, version 4-7)" >&2; exit 1; }
}

serve() {
  local model
  model=$(checkpoint --offline)
  backbone --check
  exec uv run --frozen --no-sync python serve.py --model "$model" --port "$REMOTE_PORT" \
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
