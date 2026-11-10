#!/usr/bin/env bash
# session.sh: planner-vlm's steps for a GPU session, run on the box by scripts/gpu_session.sh, one step per call:
#   bash gpu/session.sh <step>      (working directory services/planner-vlm; ROJEV_ROOT and RESULTS set by the runner)
#   bash gpu/session.sh --list      the steps, in their order
#
#   r15   bench.py: Embodied-R1.5 (one load) over every case set copied by gpu/hook.sh's assets (the SO-101 step cases
#         and the ArmNetBench attempts); answers and timing per set under $RESULTS/planner-vlm/r15/<set>/, scored on
#         the Mac by executors/so101-data/score.py.

set -euo pipefail

readonly STEPS=(r15)
readonly PYTHON="$PWD/.venv/bin/python"
readonly CASES_ROOT="$HOME/rojev-data/s2-cases"

r15() {
  local out sets
  out="$OUT_ROOT/r15"
  mkdir -p "$out"
  sets=("$CASES_ROOT"/*/)
  [[ -f "${sets[0]}cases.jsonl" ]] || { echo "no case sets under $CASES_ROOT (run gpu_setup.sh)" >&2; return 1; }
  "$PYTHON" bench.py --cases "${sets[@]%/}" --out "$out" 2>&1 | tee "$out/bench.log"
}

case "${1:-}" in
  --list)
    printf '%s\n' "${STEPS[@]}"
    ;;
  r15)
    OUT_ROOT="${RESULTS:?RESULTS is set by scripts/gpu_session.sh}/planner-vlm"
    "$1"
    ;;
  *)
    echo "usage: session.sh {--list|$(IFS='|'; echo "${STEPS[*]}")}" >&2
    exit 2
    ;;
esac
