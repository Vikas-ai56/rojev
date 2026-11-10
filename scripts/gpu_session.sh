#!/usr/bin/env bash
# gpu_session.sh: run a planned list of GPU-session steps on a box prepared by scripts/gpu_setup.sh, one at a time,
# and bring the results back to this Mac.
#
#   scripts/gpu_session.sh HOST [--plan FILE] [--only STEP[,STEP...]] [--dry-run]
#
# The plan (default config/gpu_session.txt) has one step per line, `<component dir> <step> <timeout seconds>`; lines
# starting with # are notes. Each step runs on the box as `bash gpu/session.sh <step>` in ~/rojev/<component dir>,
# with ROJEV_ROOT=~/rojev and RESULTS=~/rojev-results/<stamp>, under its own timeout; its output streams to
# results/gpu-session/<stamp>/<component>/<step>.log here. A failed step is recorded and the next one runs (a step
# that needs an earlier one stops with its own message). At the end the box's results are copied back and
# summary.tsv lists each step's exit code and seconds. --dry-run checks ssh and that every component's session.sh
# offers its planned steps, prints the plan and runs nothing.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
readonly REPO
readonly SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30)

usage() {
  echo "usage: scripts/gpu_session.sh HOST [--plan FILE] [--only STEP[,STEP...]] [--dry-run]" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
HOST=$1
shift
PLAN="$REPO/config/gpu_session.txt"
ONLY=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) PLAN=$2; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) usage ;;
  esac
done

COMPONENTS=()
STEPS=()
TIMEOUTS=()
while read -r component step timeout_s; do
  [[ -z "$component" || "$component" == \#* ]] && continue
  [[ -z "$ONLY" || "$ONLY" == *",$step,"* ]] || continue
  COMPONENTS+=("$component")
  STEPS+=("$step")
  TIMEOUTS+=("$timeout_s")
done < "$PLAN"
[[ ${#STEPS[@]} -gt 0 ]] || { echo "no steps selected from $PLAN" >&2; exit 2; }

remote() {
  ssh "${SSH_OPTS[@]}" "$HOST" "$@"
}

remote true || { echo "cannot reach $HOST over ssh" >&2; exit 1; }
for component in $(printf '%s\n' "${COMPONENTS[@]}" | sort -u); do
  offered=$(remote "cd ~/rojev/$component && bash gpu/session.sh --list") \
    || { echo "$component: gpu/session.sh --list failed on the box (run gpu_setup.sh first)" >&2; exit 1; }
  for i in "${!STEPS[@]}"; do
    [[ "${COMPONENTS[$i]}" == "$component" ]] || continue
    grep -qx "${STEPS[$i]}" <<<"$offered" || { echo "$component offers no step ${STEPS[$i]}" >&2; exit 1; }
  done
done

for i in "${!STEPS[@]}"; do
  printf '%-28s %-18s %6ss\n' "${COMPONENTS[$i]}" "${STEPS[$i]}" "${TIMEOUTS[$i]}"
done
[[ $DRY_RUN -eq 0 ]] || exit 0

STAMP=$(date +%Y%m%d-%H%M%S)
LOCAL="$REPO/results/gpu-session/$STAMP"
mkdir -p "$LOCAL"
printf 'component\tstep\texit\tseconds\n' > "$LOCAL/summary.tsv"
for i in "${!STEPS[@]}"; do
  component=${COMPONENTS[$i]}
  step=${STEPS[$i]}
  mkdir -p "$LOCAL/$(basename "$component")"
  echo "== $component $step (timeout ${TIMEOUTS[$i]}s)"
  start=$(date +%s)
  code=0
  remote "cd ~/rojev/$component && ROJEV_ROOT=\$HOME/rojev RESULTS=\$HOME/rojev-results/$STAMP timeout ${TIMEOUTS[$i]} bash gpu/session.sh $step" \
    > "$LOCAL/$(basename "$component")/$step.log" 2>&1 || code=$?
  seconds=$(( $(date +%s) - start ))
  printf '%s\t%s\t%s\t%s\n' "$component" "$step" "$code" "$seconds" >> "$LOCAL/summary.tsv"
  echo "   exit $code after ${seconds}s"
done
rsync -a -e "ssh ${SSH_OPTS[*]}" "$HOST:rojev-results/$STAMP/" "$LOCAL/box/" \
  || echo "copying the box's results back failed; they stay in ~/rojev-results/$STAMP on $HOST" >&2
echo "results: $LOCAL"
