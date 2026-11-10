#!/usr/bin/env bash
# session.sh: GR00T's steps for a GPU session, run on the box by scripts/gpu_session.sh, one step per call:
#   bash gpu/session.sh <step>      (working directory executors/so101-groot; ROJEV_ROOT and RESULTS set by the runner)
#   bash gpu/session.sh --list      the steps, in their dependency order
#
# Isaac-GR00T's own tools do the work wherever they exist; our probe only adds input / output variety on the same
# official checkpoint (no rojev code in its path). Session 1 runs the official base zero-shot on the DROID sample:
#   droid_reference  standalone_inference_script.py exactly as examples/DROID/README.md documents it (episodes 1 2,
#                    execution horizon 8; episode 0 has an empty instruction), so the action error can be checked
#                    against NVIDIA's published zero-shot numbers (MSE ~0.0149, MAE ~0.0753, ~262 ms / step on H100)
#   droid_latency    benchmark_inference.py, eager and torch.compile
#   droid_probe      probe.py: 8 time points per episode x the episode's instruction and its variants
#                    (gpu/droid_variants.json: paraphrase, other target, unrelated, empty) x 5 seeds x 4 and 8
#                    denoising steps; action error, spread across seeds, movement per instruction, latency
# The SO-101 steps (so101_finetune on NVIDIA's cube_to_bowl_5 sample, so101_openloop, so101_parity: our binding and
# service against GR00T called directly, plus a swapped-instruction probe) need a fine-tune and wait for our own
# demos; they stay available but are not in the session 1 plan. Inputs come from gpu/hook.sh's preparation: the base
# model pinned in config/groot_base_weights.txt, the gated backbone in the HF cache, the Isaac-GR00T vendor checkout
# with its two sample datasets. Each step writes to $RESULTS/groot/<step>/. FINETUNE_STEPS (default 300) sets the
# SO-101 fine-tune length: a valid checkpoint for plumbing, not a trained policy. Steps run online (GR00T loads its
# backbone by repo id) and first check that the Hub's main still equals the pinned backbone.

set -euo pipefail

readonly DROID_TAG=OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
readonly STEPS=(droid_reference droid_latency droid_probe so101_finetune so101_openloop so101_parity)
readonly PYTHON="$PWD/.venv/bin/python"

base() {
  "$PYTHON" "$ROJEV_ROOT/scripts/hf_pins.py" snapshot --pin-file "$ROJEV_ROOT/config/groot_base_weights.txt" --offline
}

finetuned() {
  local newest
  newest=$(ls -d "$FINETUNE_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [[ -n "$newest" ]] || { echo "no SO-101 fine-tune under $FINETUNE_DIR: run so101_finetune first" >&2; return 1; }
  echo "$newest"
}

out() {
  mkdir -p "$OUT_ROOT/$1"
  echo "$OUT_ROOT/$1"
}

in_isaac() {
  (cd "$ISAAC" && "$PYTHON" "$@")
}

droid_reference() {
  local dir model
  dir=$(out droid_reference)
  model=$(base)
  in_isaac scripts/deployment/standalone_inference_script.py --model-path "$model" \
    --dataset-path demo_data/droid_sample --embodiment-tag "$DROID_TAG" --traj-ids 1 2 --inference-mode pytorch \
    --execution-horizon 8 \
    2>&1 | tee "$dir/standalone_inference.log"
}

droid_latency() {
  local dir model
  dir=$(out droid_latency)
  model=$(base)
  in_isaac scripts/deployment/benchmark_inference.py --model-path "$model" --dataset-path demo_data/droid_sample \
    --embodiment-tag "$DROID_TAG" --num-iterations 50 --warmup 10 2>&1 | tee "$dir/benchmark_inference.log"
}

droid_probe() {
  local dir model
  dir=$(out droid_probe)
  model=$(base)
  (cd "$ISAAC" && "$PYTHON" "$OLDPWD/probe.py" --model "$model" --embodiment-tag "$DROID_TAG" \
    --dataset demo_data/droid_sample --variants "$OLDPWD/gpu/droid_variants.json" --points 8 --seeds 0 1 2 3 4 \
    --denoising-steps 4 8 --out "$dir") 2>&1 | tee "$dir/probe.log"
}

so101_finetune() {
  local dir model steps=${FINETUNE_STEPS:-300}
  dir=$(out so101_finetune)
  model=$(base)
  in_isaac gr00t/experiment/launch_finetune.py --base-model-path "$model" --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/SO100/so100_config.py --num-gpus 1 \
    --output-dir "$FINETUNE_DIR" --max-steps "$steps" --save-steps "$steps" --global-batch-size 16 \
    2>&1 | tee "$dir/launch_finetune.log"
}

so101_openloop() {
  local dir model
  dir=$(out so101_openloop)
  model=$(finetuned)
  in_isaac gr00t/eval/open_loop_eval.py --model-path "$model" --embodiment-tag NEW_EMBODIMENT \
    --dataset-path demo_data/cube_to_bowl_5 --traj-ids 0 1 --steps 300 --execution-horizon 16 \
    --save-plot-path "$dir/plots" 2>&1 | tee "$dir/open_loop_eval.log"
}

so101_parity() {
  local dir model
  dir=$(out so101_parity)
  model=$(finetuned)
  "$PYTHON" parity.py --model "$model" --dataset "$ISAAC/demo_data/cube_to_bowl_5" --swap-text "cube into green bowl" \
    --out "$dir" 2>&1 | tee "$dir/parity.log"
}

case "${1:-}" in
  --list)
    printf '%s\n' "${STEPS[@]}"
    ;;
  droid_reference|droid_latency|droid_probe|so101_finetune|so101_openloop|so101_parity)
    ISAAC="${ROJEV_ROOT:?ROJEV_ROOT is set by scripts/gpu_session.sh}/vendor/isaac-groot"
    OUT_ROOT="${RESULTS:?RESULTS is set by scripts/gpu_session.sh}/groot"
    FINETUNE_DIR="$OUT_ROOT/so101_finetune/checkpoints"
    "$PYTHON" "$ROJEV_ROOT/scripts/hf_gated.py" snapshot --pin-file "$ROJEV_ROOT/config/groot_backbone.txt" --check
    "$1"
    ;;
  *)
    echo "usage: session.sh {--list|$(IFS='|'; echo "${STEPS[*]}")}" >&2
    exit 2
    ;;
esac
