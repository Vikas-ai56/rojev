#!/usr/bin/env bash
# hook.sh: planner-vlm's side of the GPU setup interface (Embodied-R1.5, the S2 planner candidate).
#
# Same contract as every component hook; the full interface is documented in scripts/gpu_setup.sh and
# executors/robocasa-pi052/gpu/hook.sh. Called with the working directory set to services/planner-vlm (on the Mac or
# on the box) and ROJEV_ROOT pointing at the repo root there.
#
# planner-vlm specifics
#   - Serves R1.5 live with vLLM's own OpenAI-compatible server, exactly as R1.5's model card documents (`vllm serve`
#     with its flags), on 127.0.0.1:8006; the only changes are the pinned local snapshot instead of the repo id and a
#     GPU memory share of GPU_MEMORY_SHARE (0.45, card: 0.7) so an S1 model fits beside it on a 40 GB GPU. The
#     bearer token reaches vLLM through its VLLM_API_KEY environment variable, never the command line. Clients:
#     `rojev.system2.models.chat_vlm` (planner and done checker), prompts from config/s2/embodied-r1.5.yaml.
#   - The S2 bake-off (gpu/session.sh) runs on the same env.
#   - Env: vLLM 0.18.0 with transformers 4.57 (R1.5 was saved with 4.57.1) and qwen-vl-utils 0.0.14, Linux only.
#   - Weights: IffYuan/Embodied-R1.5 at the revision in config/embodied_r1_weights.txt (public, 17.5 GB bf16),
#     pre-downloaded on the box; no Hugging Face token is needed or sent.
#   - Assets: the case sets under data/s2-cases/ (built on the Mac by executors/so101-data) are copied to
#     ~/rojev-data/s2-cases/ on the box.
#   - Memory: serving takes GPU_MEMORY_SHARE of the GPU (about 18 GB on 40 GB); the batch bench (engine.py) takes 0.85
#     and runs alone.

set -euo pipefail

readonly REMOTE_PORT=8006
readonly GPU_MEM_MIB=18432
readonly GPU_MEMORY_SHARE=0.45
readonly SERVED_NAME=Embodied-R1.5
readonly WEIGHTS_PIN=config/embodied_r1_weights.txt
readonly CASE_SETS=(so101-steps armnetbench)
readonly BOX_CASES_DIR=rojev-data/s2-cases

pinned() {
  uv run --frozen --no-sync python "$ROJEV_ROOT/scripts/hf_pins.py" snapshot --pin-file "$ROJEV_ROOT/$WEIGHTS_PIN" "$@"
}

key() {
  echo planner
}

auth() {
  echo token
}

gpu_mem_mib() {
  echo "$GPU_MEM_MIB"
}

files() {
  printf '%s\n' pyproject.toml uv.lock prompts.py engine.py bench.py
}

vendors() {
  :
}

remote_port() {
  echo "$REMOTE_PORT"
}

check_local() {
  local set
  for set in "${CASE_SETS[@]}"; do
    [[ -f "$ROJEV_ROOT/data/s2-cases/$set/cases.jsonl" ]] || return 1
  done
}

prepare_local() {
  check_local || {
    echo "build the case sets first: executors/so101-data (cases.py build, rollouts.py build) into data/s2-cases/" >&2
    exit 1
  }
}

assets() {
  local set
  for set in "${CASE_SETS[@]}"; do
    printf '%s\t%s\n' "$ROJEV_ROOT/data/s2-cases/$set" "$BOX_CASES_DIR"
  done
}

client_hint() {
  echo "service entry 'planner' in $1 ($2): OpenAI-compatible /v1/chat/completions, model $SERVED_NAME; client rojev.system2.models.chat_vlm with config/s2/embodied-r1.5.yaml"
}

check_remote() {
  pinned --offline >/dev/null && uv run --frozen --no-sync python -c 'import vllm, qwen_vl_utils' 2>/dev/null \
    && [[ -x .venv/bin/vllm ]]
}

prepare_remote() {
  pinned >/dev/null
  uv run --frozen --no-sync python -c 'import vllm, qwen_vl_utils' || {
    echo "vllm / qwen_vl_utils do not import in the planner-vlm env" >&2
    exit 1
  }
}

serve() {
  local snapshot
  snapshot=$(pinned --offline)
  VLLM_API_KEY="$ROJEV_TOKEN" exec .venv/bin/vllm serve "$snapshot" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$REMOTE_PORT" --tensor-parallel-size 1 --mm-encoder-tp-mode data \
    --gpu-memory-utilization "$GPU_MEMORY_SHARE" --async-scheduling --max-model-len 20000 \
    --limit-mm-per-prompt '{"image": 8, "video": 1}' --media-io-kwargs '{"video": {"num_frames": 32}, "image": {"max_num": 32}}'
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
