#!/usr/bin/env bash
# session.sh: LIBERO's steps for a GPU session, run on the box by scripts/gpu_session.sh, one step per call:
#   bash gpu/session.sh <step>      (working directory executors/libero; ROJEV_ROOT and RESULTS set by the runner)
#   bash gpu/session.sh --list      the steps, in their order
#
# Stage 1 scores an S1 service alone through the whole runtime (evaluate.py: one rojev Session per task episode,
# NativePlanner, lockstep) on the simulator rendering here with EGL:
#   stage1_goal      libero_goal
#   stage1_spatial   libero_spatial
#   stage1_10        libero_10
# and the same in real time (lockstep off: the model's delay costs robot time, config/runs/libero-realtime.yaml):
#   realtime_goal    libero_goal
# Each writes $RESULTS/libero/<step>/ (results.json, one session directory per episode, evaluate.log).
# The S1 service is the box's component whose hook key is $LIBERO_ENTRY (default lerobot-policy); it must be running
# (gpu_setup.sh). Its address is the box's 127.0.0.1:<its remote_port>, its token the one gpu_setup.sh stored in
# ~/.rojev/<component>/token; both go into ~/.rojev/libero/service.json (mode 600, never into the results, which are
# copied back to the Mac). Overrides: LIBERO_TASKS (default "0 1 2 3 4 5 6 7 8 9"), LIBERO_EPISODES (default 2),
# LIBERO_SEED (default 1000), LIBERO_VIDEO=0 (no view.mp4 per episode; on by default), LIBERO_CONFIG (a YAML with a
# runtime section, replacing the step's own: config/runs/libero-stage1.yaml, or libero-realtime.yaml for realtime_*).

set -euo pipefail

readonly STEPS=(stage1_goal stage1_spatial stage1_10 realtime_goal)
readonly PYTHON="$PWD/.venv/bin/python"
readonly STATE_DIR="$HOME/.rojev/libero"
readonly SERVICE_JSON="$STATE_DIR/service.json"

component_hook() {
  (cd "$1" && ROJEV_ROOT="$ROJEV_ROOT" bash gpu/hook.sh "$2")
}

write_service_json() {
  local hook dir port auth token=""
  for hook in "$ROJEV_ROOT"/executors/*/gpu/hook.sh "$ROJEV_ROOT"/services/*/gpu/hook.sh; do
    [[ -f "$hook" ]] || continue
    dir=${hook%/gpu/hook.sh}
    [[ "$(component_hook "$dir" key 2>/dev/null)" == "$ENTRY" ]] || continue
    port=$(component_hook "$dir" remote_port)
    auth=$(component_hook "$dir" auth)
    if [[ "$auth" == token ]]; then
      token=$(< "$HOME/.rojev/${dir##*/}/token")
    fi
    mkdir -p "$STATE_DIR"
    (umask 077 && printf '{"%s": {"url": "http://127.0.0.1:%s", "token": "%s"}}\n' "$ENTRY" "$port" "$token" \
      > "$SERVICE_JSON")
    printf 'header = "Authorization: Bearer %s"\n' "$token" | curl -fsS -m 10 -K - "http://127.0.0.1:$port/health" \
      >/dev/null || { echo "service $ENTRY (${dir#"$ROJEV_ROOT"/}) does not answer on port $port" >&2; return 1; }
    return 0
  done
  echo "no component on this box has the service key $ENTRY" >&2
  return 1
}

evaluate() {
  local suite=$1 dir="$OUT_ROOT/$2" config=$3 tasks options=()
  read -ra tasks <<<"${LIBERO_TASKS:-0 1 2 3 4 5 6 7 8 9}"
  [[ "${LIBERO_VIDEO:-1}" == 0 ]] || options+=(--video)
  options+=(--config "${LIBERO_CONFIG:-$config}")
  mkdir -p "$dir"
  MUJOCO_GL=egl "$PYTHON" evaluate.py --suite "$suite" --tasks "${tasks[@]}" --episodes "${LIBERO_EPISODES:-2}" \
    --seed "${LIBERO_SEED:-1000}" --service "$SERVICE_JSON" --entry "$ENTRY" --out "$dir" "${options[@]}" \
    2>&1 | tee "$dir/evaluate.log"
}

stage1_goal() {
  evaluate libero_goal stage1_goal "$ROJEV_ROOT/config/runs/libero-stage1.yaml"
}

stage1_spatial() {
  evaluate libero_spatial stage1_spatial "$ROJEV_ROOT/config/runs/libero-stage1.yaml"
}

stage1_10() {
  evaluate libero_10 stage1_10 "$ROJEV_ROOT/config/runs/libero-stage1.yaml"
}

realtime_goal() {
  evaluate libero_goal realtime_goal "$ROJEV_ROOT/config/runs/libero-realtime.yaml"
}

case "${1:-}" in
  --list)
    printf '%s\n' "${STEPS[@]}"
    ;;
  stage1_goal|stage1_spatial|stage1_10|realtime_goal)
    OUT_ROOT="${RESULTS:?RESULTS is set by scripts/gpu_session.sh}/libero"
    ENTRY="${LIBERO_ENTRY:-lerobot-policy}"
    : "${ROJEV_ROOT:?ROJEV_ROOT is set by scripts/gpu_session.sh}"
    "$PYTHON" -m gpu.box_setup check
    write_service_json
    "$1"
    ;;
  *)
    echo "usage: session.sh {--list|$(IFS='|'; echo "${STEPS[*]}")}" >&2
    exit 2
    ;;
esac
