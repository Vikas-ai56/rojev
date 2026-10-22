#!/usr/bin/env bash
# gpu_setup.sh: one command from the Mac turns a fresh Ubuntu GPU box into running model services (S1 + S2).
#
# Usage
#   scripts/gpu_setup.sh HOST [--executor NAME] [--services a,b] [--local-port KEY=N]... [--driver-branch B]
#                        [--dry-run|--check] [--skip-serve] [--restart] [--tunnel]
#     HOST                an ssh host alias (key-based login, passwordless sudo on the box)
#     --executor NAME     component executors/NAME (S1, e.g. robocasa-pi052)
#     --services a,b      components services/<a>, services/<b> (S2, e.g. onejev,mjev); any executor uses them
#     --local-port KEY=N  Mac end of KEY's tunnel (default: the port already in the service JSON, else a free 8010-8099)
#     --driver-branch B   NVIDIA server driver branch to install when nvidia-smi does not work yet (default 580)
#     --dry-run           read-only: run every check and report what would be done; changes nothing anywhere
#     --skip-serve        set the box up but do not start services or tunnels
#     --restart           restart every selected service even if nothing changed
#     --tunnel            only reopen the tunnels and check health (e.g. after the Mac slept); needs a previous run
#   Example: scripts/gpu_setup.sh lambda --executor robocasa-pi052 --services onejev,mjev
#
# Components: an executor or a service, each <kind>/<name>/gpu/hook.sh (kind = executors | services). Each gets its own
# env, systemd unit rojev-<name>, tunnel and entry in the box's service JSON.
#
# Steps (each prints one [step] line: ok = already done, would = dry run, done = applied now)
#   ssh, gpu            reachable with BatchMode (accepts a new host key); an NVIDIA display/3D controller on PCI
#   local-assets:<n>    hook check_local / prepare_local on this Mac (first, so a missing login fails early)
#   driver              nvidia-smi works; else install the -server driver branch + utils matching the built module
#   packages, uv        build-essential git rsync curl (evdev builds from source); uv >= this Mac's version
#   code:core           rojev/ pyproject.toml uv.lock README.md (only with an executor: its env depends on the core)
#   code:shared         config/ (pins) and scripts/ (hf_pins.py)
#   code:<n>            the component's hook files + gpu/; trees mirrored with --delete, compared by checksum
#   vendor:<dir>        git clones at the commits pinned in config/*.commit (core + every hook's vendors, deduplicated)
#   env:<n>             uv sync --frozen with uv-managed Python (system Python lacks Python.h)
#   assets:<n>          hook assets copied to the box, compared by size + mtime: they are symlinks into the HF cache,
#                       and macOS openrsync with -c -L checksums the link rather than its target (every file reads as
#                       changed, exit 23)
#   prepare-remote:<n>  hook check_remote / prepare_remote on the box (weights pre-download)
#   torch:<n>           the component env's torch sees CUDA
#   service-json        ~/.config/rojev/services/<host>.json {"<key>": {"url", "token"?}}, mode 600; tokens made here;
#                       other entries in the file are kept
#   token:<key>         the box's copy of that token matches (sha256 compare; sent only when it differs)
#   service:<n>         unit rojev-<n> running hook serve; restarted when a step it depends on changed; start refused
#                       if free GPU memory < hook gpu_mem_mib; a port held by a process this script did not start
#                       is reported blocked and never touched; waits until GET /health answers
#   tunnel:<key>        ssh -L <local>:127.0.0.1:<remote_port> in the background; GET /health through it
#
# Component interface: <kind>/<name>/gpu/hook.sh <subcommand>, working dir = component dir, ROJEV_ROOT = repo root.
#   Mac:  key, auth, gpu_mem_mib, files, vendors, remote_port, check_local, prepare_local, assets, client_hint JSON URL
#   Box:  check_remote, prepare_remote, serve
# Full contract in executors/robocasa-pi052/gpu/hook.sh; box-side steps live in scripts/gpu_box.sh.
#
# Security: tokens are generated on this Mac, stored in the service JSON (mode 600) and on the box (mode 600), sent
# only over SSH inside the streamed script, never printed and never on a command line. No Hugging Face token leaves
# this Mac. Services bind 127.0.0.1 on the box; only the SSH tunnels reach them (the only access path for auth=none).
#
# Idempotent: re-running skips every step whose check passes; it never stops a process it did not start.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO
readonly BOX_LIB="$REPO/scripts/gpu_box.sh"
readonly CORE_TREES=(rojev)
readonly CORE_FILES=(pyproject.toml uv.lock README.md)
readonly SHARED_TREES=(config scripts)
readonly CORE_VENDORS=$'OneJev\thttps://github.com/OmniJev/OneJev.git\tonejev.commit'
readonly SYNC_EXCLUDES=(--exclude=__pycache__ --exclude='*.pyc' --exclude=.DS_Store --exclude=.venv --exclude=.pytest_cache)
readonly SERVICES_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/rojev/services"
readonly LOCAL_PORT_RANGE=(8010 8099)
readonly TUNNEL_HEALTH_ATTEMPTS=10

HOST=
EXECUTOR=
SERVICES=
LOCAL_PORT_FLAGS=
SERVICE_JSON=
DRIVER_BRANCH=580
DRY_RUN=0
SKIP_SERVE=0
FORCE_RESTART=0
TUNNEL_ONLY=0
CHANGED_ALL=0
CHANGED_NAMES=" "
PENDING=0
STEP_DETAIL=
STEP_ITEMS=
PROJECTED_FREE_MIB=
CONTROL_DIR=
SSH_OPTS=()
RSYNC_SSH=
TUNNEL_CMD=()
COMPONENTS=()
NAMES=()
KEYS=()
AUTHS=()
REMOTE_PORTS=()
GPU_MEMS=()
LOCAL_PORTS=()

usage() {
  sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

log() {
  printf '[%s] %s\n' "$1" "$2"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

parse_args() {
  [[ $# -gt 0 ]] || usage 2
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help) usage 0 ;;
      --executor) EXECUTOR=${2:?--executor needs a name}; shift ;;
      --services) SERVICES=${2:?--services needs a list}; shift ;;
      --local-port)
        [[ "${2:-}" =~ ^[A-Za-z0-9_-]+=[0-9]+$ ]] || die "--local-port needs KEY=PORT (e.g. pi052=8010)"
        LOCAL_PORT_FLAGS+="$2"$'\n'; shift ;;
      --driver-branch) DRIVER_BRANCH=${2:?--driver-branch needs a number}; shift ;;
      --dry-run|--check) DRY_RUN=1 ;;
      --skip-serve) SKIP_SERVE=1 ;;
      --restart) FORCE_RESTART=1 ;;
      --tunnel) TUNNEL_ONLY=1 ;;
      -*) die "unknown option $1" ;;
      *) [[ -z "$HOST" ]] || die "one host only (got $HOST and $1)"; HOST=$1 ;;
    esac
    shift
  done
  [[ -n "$HOST" ]] || die "missing HOST"
  [[ -n "$EXECUTOR" || -n "$SERVICES" ]] || die "nothing to set up: give --executor NAME and/or --services a,b"
  [[ "$DRIVER_BRANCH" =~ ^[0-9]+$ ]] || die "--driver-branch must be a number"
  local service
  [[ -z "$EXECUTOR" ]] || COMPONENTS+=("executors/$EXECUTOR")
  for service in $(printf '%s' "$SERVICES" | tr ',' ' '); do
    COMPONENTS+=("services/$service")
  done
  SERVICE_JSON="$SERVICES_DIR/$(printf '%s' "$HOST" | tr -c 'A-Za-z0-9._-' '_').json"
}

available_components() {
  local hook
  for hook in "$REPO"/executors/*/gpu/hook.sh "$REPO"/services/*/gpu/hook.sh; do
    [[ -f "$hook" ]] && printf '%s ' "${hook#"$REPO"/}" | sed 's|/gpu/hook.sh||'
  done
}

hook_local() {
  local component=$1
  shift
  (cd "$REPO/$component" && ROJEV_ROOT="$REPO" bash gpu/hook.sh "$@")
}

load_components() {
  local i component value
  for i in "${!COMPONENTS[@]}"; do
    component=${COMPONENTS[$i]}
    [[ -f "$REPO/$component/gpu/hook.sh" ]] ||
      die "$component/gpu/hook.sh not found; components with GPU setup: $(available_components)"
    NAMES[i]=${component##*/}
    KEYS[i]=$(hook_local "$component" key)
    AUTHS[i]=$(hook_local "$component" auth)
    REMOTE_PORTS[i]=$(hook_local "$component" remote_port)
    GPU_MEMS[i]=$(hook_local "$component" gpu_mem_mib)
    [[ "${AUTHS[$i]}" == token || "${AUTHS[$i]}" == none ]] || die "$component: hook auth must print token or none"
    for value in "${REMOTE_PORTS[$i]}" "${GPU_MEMS[$i]}"; do
      [[ "$value" =~ ^[0-9]+$ ]] || die "$component: hook remote_port / gpu_mem_mib must print a number"
    done
  done
  for value in "$(printf '%s\n' "${NAMES[@]}" | sort | uniq -d)" "$(printf '%s\n' "${KEYS[@]}" | sort | uniq -d)" \
               "$(printf '%s\n' "${REMOTE_PORTS[@]}" | sort | uniq -d)"; do
    [[ -z "$value" ]] || die "components clash: name, key or remote port '$value' is used twice"
  done
}

has_executor() {
  [[ -n "$EXECUTOR" ]]
}

require_mac_tools() {
  local tool
  for tool in ssh rsync curl python3 uv lsof shasum; do
    command -v "$tool" >/dev/null || die "$tool not found on this Mac"
  done
}

open_ssh_master() {
  CONTROL_DIR=$(mktemp -d /tmp/rojev-gpu.XXXXXX)
  SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30
            -o ControlMaster=auto -o "ControlPath=$CONTROL_DIR/%C" -o ControlPersist=300)
  RSYNC_SSH="ssh ${SSH_OPTS[*]}"
  trap close_ssh_master EXIT
}

close_ssh_master() {
  ssh "${SSH_OPTS[@]}" -O exit "$HOST" 2>/dev/null || true
  rm -rf "$CONTROL_DIR"
}

box() {
  { cat "$BOX_LIB"; printf '%q ' "$@"; printf '\n'; } | ssh "${SSH_OPTS[@]}" "$HOST" bash -s
}

box_with_secret() {
  local secret=$1
  shift
  { cat "$BOX_LIB"; printf '%q ' "$@"; printf "<<'ROJEV_SECRET_EOF'\n%s\nROJEV_SECRET_EOF\n" "$secret"; } |
    ssh "${SSH_OPTS[@]}" "$HOST" bash -s
}

mark_changed() {
  case "$1" in
    local) ;;
    all) CHANGED_ALL=1 ;;
    *) CHANGED_NAMES+="$1 " ;;
  esac
}

needs_restart() {
  (( FORCE_RESTART || CHANGED_ALL )) || [[ "$CHANGED_NAMES" == *" $1 "* ]]
}

run_step() {
  local scope=$1 name=$2 description=$3 check=$4 apply=$5
  shift 5
  STEP_DETAIL=
  STEP_ITEMS=
  if "$check" "$@"; then
    log "$name" "ok"
    return 0
  fi
  if (( DRY_RUN )); then
    log "$name" "would $description${STEP_DETAIL:+ ($STEP_DETAIL)}"
    [[ -z "$STEP_ITEMS" ]] || printf '%s\n' "$STEP_ITEMS" | sed 's/^/    /'
    PENDING=1
    return 0
  fi
  log "$name" "$description${STEP_DETAIL:+ ($STEP_DETAIL)}"
  "$apply" "$@"
  "$check" "$@" || die "$name: still not done after: $description"
  mark_changed "$scope"
  log "$name" "done"
}

run_verify() {
  local name=$1
  shift
  local output
  if output=$("$@" 2>&1); then
    log "$name" "ok: $(printf '%s' "$output" | tail -n 1)"
  elif (( DRY_RUN && PENDING )); then
    log "$name" "not yet (depends on the steps marked would)"
  else
    printf '%s\n' "$output" >&2
    die "$name failed"
  fi
}

step_ssh() {
  ssh "${SSH_OPTS[@]}" "$HOST" true || die "cannot ssh to $HOST non-interactively (check the alias, the key and ssh-agent)"
  log ssh "ok"
}

step_gpu() {
  box gpu_present || die "no NVIDIA GPU on $HOST's PCI bus"
  log gpu "ok"
}

local_assets_ready() {
  hook_local "$1" check_local >/dev/null 2>&1
}

local_assets_prepare() {
  hook_local "$1" prepare_local
}

driver_ready() {
  box driver_ready
}

driver_install() {
  box driver_install "$DRIVER_BRANCH" ||
    die "driver install failed; if nvidia-smi reports a driver/library version mismatch, reboot the box and re-run"
}

step_driver() {
  run_step all driver "install nvidia-driver-$DRIVER_BRANCH-server and the utils matching the built kernel module" \
    driver_ready driver_install
  if driver_ready; then
    log driver "$(box gpu_summary)"
  fi
}

packages_ready() {
  box packages_ready
}

packages_install() {
  box packages_install
}

uv_version_here() {
  uv --version | awk '{print $2}'
}

uv_ready() {
  box uv_ready "$(uv_version_here)"
}

uv_install() {
  box uv_install "$(uv_version_here)"
}

group_trees() {
  case "$1" in
    core) printf '%s\n' "${CORE_TREES[@]}" ;;
    shared) printf '%s\n' "${SHARED_TREES[@]}" ;;
    *) printf '%s\n' "$1/gpu" ;;
  esac
}

group_files() {
  case "$1" in
    core) printf '%s\n' "${CORE_FILES[@]}" ;;
    shared) ;;
    *) hook_local "$1" files | sed "s|^|$1/|" ;;
  esac
}

rsync_group() {
  local mode=$1 group=$2 tree
  local flags=(-a -c -i -e "$RSYNC_SSH")
  [[ "$mode" == check ]] && flags+=(-n)
  while IFS= read -r tree; do
    rsync "${flags[@]}" --delete "${SYNC_EXCLUDES[@]}" "$REPO/$tree/" "$HOST:rojev/$tree/" < /dev/null
  done < <(group_trees "$group")
  local files=()
  while IFS= read -r tree; do
    files+=("$tree")
  done < <(group_files "$group")
  if (( ${#files[@]} )); then
    (cd "$REPO" && rsync "${flags[@]}" -R "${files[@]}" "$HOST:rojev/" < /dev/null)
  fi
}

real_changes() {
  grep -E '^([<>ch]|\*deleting)' || true
}

group_ready() {
  local changes
  changes=$(rsync_group check "$1" 2>/dev/null | real_changes) || return 1
  [[ -z "$changes" ]] && return 0
  STEP_DETAIL="$(printf '%s\n' "$changes" | wc -l | tr -d ' ') entries differ"
  STEP_ITEMS=$changes
  return 1
}

group_install() {
  case "$1" in
    core|shared) box make_dirs rojev ;;
    *) box make_dirs "rojev/$1" ;;
  esac
  rsync_group apply "$1" | real_changes | sed 's/^/  /'
}

step_code() {
  local i
  if has_executor; then
    run_step "$EXECUTOR" code:core "rsync the rojev core to ~/rojev" group_ready group_install core
  fi
  run_step all code:shared "rsync config/ and scripts/ to ~/rojev" group_ready group_install shared
  for i in "${!COMPONENTS[@]}"; do
    run_step "${NAMES[$i]}" "code:${NAMES[$i]}" "rsync ${COMPONENTS[$i]} to ~/rojev" group_ready group_install \
      "${COMPONENTS[$i]}"
  done
}

vendor_lines() {
  local component
  {
    if has_executor; then
      printf '%s\n' "$CORE_VENDORS"
    fi
    for component in "${COMPONENTS[@]}"; do
      hook_local "$component" vendors
    done
  } | awk -F'\t' 'NF && !seen[$1]++'
}

pin_of() {
  local pin
  pin=$(tr -d '[:space:]' < "$REPO/config/$1")
  [[ "$pin" =~ ^[0-9a-f]{40}$ ]] || die "config/$1 does not hold a 40-hex commit"
  printf '%s' "$pin"
}

vendor_ready() {
  box vendor_ready "$1" "$3"
}

vendor_install() {
  box vendor_install "$1" "$2" "$3"
}

step_vendors() {
  local name url pin_file pin
  while IFS=$'\t' read -r name url pin_file; do
    pin=$(pin_of "$pin_file")
    run_step all "vendor:$name" "clone $url at ${pin:0:12}" vendor_ready vendor_install "$name" "$url" "$pin"
  done < <(vendor_lines)
}

env_ready() {
  box env_ready "$1"
}

env_install() {
  box env_install "$1"
}

rsync_assets() {
  local mode=$1 component=$2 lines dir
  local flags=(-a -L -i -e "$RSYNC_SSH")
  [[ "$mode" == check ]] && flags+=(-n)
  lines=$(hook_local "$component" assets 2>/dev/null) || return 1
  [[ -n "$lines" ]] || return 0
  while IFS= read -r dir; do
    local sources=()
    local source
    while IFS= read -r source; do
      sources+=("$source")
    done < <(printf '%s\n' "$lines" | awk -F'\t' -v d="$dir" '$2 == d {print $1}')
    rsync "${flags[@]}" "${sources[@]}" "$HOST:$dir/" < /dev/null
  done < <(printf '%s\n' "$lines" | cut -f2 | sort -u)
}

assets_ready() {
  local changes
  changes=$(rsync_assets check "$1" 2>/dev/null | real_changes) || return 1
  [[ -z "$changes" ]] && return 0
  STEP_DETAIL="$(printf '%s\n' "$changes" | wc -l | tr -d ' ') entries differ"
  STEP_ITEMS=$changes
  return 1
}

assets_install() {
  rsync_assets apply "$1" | real_changes | sed 's/^/  /'
}

remote_prepared() {
  box hook "$1" check_remote >/dev/null 2>&1
}

remote_prepare() {
  box hook "$1" prepare_remote
}

torch_cuda() {
  box torch_cuda "$1"
}

step_components_env() {
  local i name component
  for i in "${!COMPONENTS[@]}"; do
    name=${NAMES[$i]}
    component=${COMPONENTS[$i]}
    run_step "$name" "env:$name" "uv sync --frozen in $component" env_ready env_install "$component"
    run_step "$name" "assets:$name" "copy $name's assets to the box" assets_ready assets_install "$component"
    run_step "$name" "prepare-remote:$name" "run $name's box-side preparation" remote_prepared remote_prepare \
      "$component"
    run_verify "torch:$name" torch_cuda "$component"
  done
}

json_get() {
  [[ -f "$SERVICE_JSON" ]] || return 1
  python3 - "$SERVICE_JSON" "$1" "$2" <<'PY'
import json, sys
value = json.load(open(sys.argv[1])).get(sys.argv[2], {}).get(sys.argv[3])
if value is None:
    sys.exit(1)
print(value)
PY
}

local_port_flag() {
  printf '%s' "$LOCAL_PORT_FLAGS" | awk -F= -v k="$1" '$1 == k {print $2}' | tail -n 1
}

ports_claimed() {
  cat "$SERVICES_DIR"/*.json 2>/dev/null | grep -oE '127\.0\.0\.1:[0-9]+' | cut -d: -f2 || true
  printf '%s\n' "${LOCAL_PORTS[@]:-}"
}

pick_local_ports() {
  local i url port
  for i in "${!COMPONENTS[@]}"; do
    LOCAL_PORTS[i]=$(local_port_flag "${KEYS[$i]}")
    if [[ -z "${LOCAL_PORTS[$i]}" ]] && url=$(json_get "${KEYS[$i]}" url); then
      LOCAL_PORTS[i]=${url##*:}
    fi
  done
  for i in "${!COMPONENTS[@]}"; do
    [[ -z "${LOCAL_PORTS[$i]}" ]] || continue
    for (( port = LOCAL_PORT_RANGE[0]; port <= LOCAL_PORT_RANGE[1]; port++ )); do
      if ! ports_claimed | grep -qx "$port" && ! lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null; then
        LOCAL_PORTS[i]=$port
        break
      fi
    done
    [[ -n "${LOCAL_PORTS[$i]}" ]] ||
      die "no free local port in ${LOCAL_PORT_RANGE[0]}-${LOCAL_PORT_RANGE[1]}; pass --local-port ${KEYS[$i]}=N"
  done
  [[ $(printf '%s\n' "${LOCAL_PORTS[@]}" | sort | uniq -d | wc -l) -eq 0 ]] || die "two components got the same local port"
}

service_url() {
  printf 'http://127.0.0.1:%s' "${LOCAL_PORTS[$1]}"
}

service_json_entries() {
  local i
  for i in "${!COMPONENTS[@]}"; do
    printf '%s\n%s\n%s\n' "${KEYS[$i]}" "$(service_url "$i")" "${AUTHS[$i]}"
  done
}

service_json_ready() {
  [[ -f "$SERVICE_JSON" && "$(stat -f %Lp "$SERVICE_JSON")" == 600 ]] || return 1
  service_json_entries | python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
rows = sys.stdin.read().split("\n")
for key, url, auth in zip(rows[0::3], rows[1::3], rows[2::3]):
    entry = data.get(key, {})
    if entry.get("url") != url or ("token" in entry) != (auth == "token"):
        sys.exit(1)
' "$SERVICE_JSON"
}

service_json_write() {
  mkdir -p "$SERVICES_DIR"
  chmod 700 "$SERVICES_DIR"
  service_json_entries | python3 -c '
import json, os, secrets, sys
path = sys.argv[1]
data = json.load(open(path)) if os.path.exists(path) else {}
rows = sys.stdin.read().split("\n")
for key, url, auth in zip(rows[0::3], rows[1::3], rows[2::3]):
    entry = data.get(key, {})
    entry["url"] = url
    if auth == "token":
        entry["token"] = entry.get("token") or secrets.token_urlsafe(32)
    else:
        entry.pop("token", None)
    data[key] = entry
fd = os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    json.dump(data, f, indent=2)
os.replace(path + ".tmp", path)
' "$SERVICE_JSON"
}

local_token_digest() {
  json_get "$1" token | shasum -a 256 | awk '{print $1}'
}

token_ready() {
  local key=$1 name=$2
  json_get "$key" token >/dev/null 2>&1 && [[ "$(box token_digest "$name")" == "$(local_token_digest "$key")" ]]
}

token_install() {
  local key=$1 name=$2
  box_with_secret "$(json_get "$key" token)" token_store "$name"
}

step_tokens() {
  local i
  for i in "${!COMPONENTS[@]}"; do
    [[ "${AUTHS[$i]}" == token ]] || continue
    run_step "${NAMES[$i]}" "token:${KEYS[$i]}" "send ${KEYS[$i]}'s token to the box" token_ready token_install \
      "${KEYS[$i]}" "${NAMES[$i]}"
  done
}

gpu_memory_report() {
  local name=$1 need=$2 free=$3
  printf '%s needs %s MiB GPU memory, %s MiB free' "$name" "$need" "$free"
}

require_gpu_memory() {
  local name=$1 need=$2 free
  free=$(box gpu_free_mib)
  (( free >= need )) ||
    die "$(gpu_memory_report "rojev-$name" "$need" "$free"); GPU processes: $(box gpu_apps). Free memory or skip it."
}

step_service() {
  local i=$1 state action holder
  local name=${NAMES[$i]} component=${COMPONENTS[$i]} port=${REMOTE_PORTS[$i]} auth=${AUTHS[$i]} need=${GPU_MEMS[$i]}
  state=$(box service_state "$name" "$port")
  case "$state" in
    running)
      if needs_restart "$name"; then action="restart"; else action="wait"; fi ;;
    stopped)
      action="start" ;;
    port-busy*)
      holder=${state#port-busy }
      if (( DRY_RUN )); then
        log "service:$name" "blocked: foreign process on box port $port (not started by gpu_setup; left alone): $holder"
        PENDING=1
        return 0
      fi
      die "service:$name: box port $port is held by a process gpu_setup did not start: $holder (stop it, then re-run)" ;;
    *)
      die "unexpected service state from the box: $state" ;;
  esac
  if (( DRY_RUN )); then
    case "$action" in
      wait) log "service:$name" "ok (unit rojev-$name running; health checked on a real run)" ;;
      restart) log "service:$name" "would restart unit rojev-$name"; PENDING=1 ;;
      start)
        [[ -n "$PROJECTED_FREE_MIB" ]] || PROJECTED_FREE_MIB=$(box gpu_free_mib)
        if (( PROJECTED_FREE_MIB >= need )); then
          log "service:$name" "would start unit rojev-$name on 127.0.0.1:$port ($(gpu_memory_report "$name" "$need" "$PROJECTED_FREE_MIB"))"
          PROJECTED_FREE_MIB=$(( PROJECTED_FREE_MIB - need ))
        else
          log "service:$name" "blocked: $(gpu_memory_report "$name" "$need" "$PROJECTED_FREE_MIB")"
        fi
        PENDING=1 ;;
    esac
    return 0
  fi
  if [[ "$action" == restart ]]; then
    log "service:$name" "restarting unit rojev-$name (a step it depends on changed)"
    box service_stop "$name"
  fi
  if [[ "$action" != wait ]]; then
    require_gpu_memory "$name" "$need"
    log "service:$name" "starting unit rojev-$name"
    box service_start "$name" "$component" "$auth"
  fi
  log "service:$name" "ok: $(box service_wait "$name" "$port" "$auth")"
}

tunnel_command() {
  TUNNEL_CMD=(ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30
              -o ServerAliveCountMax=3 -o ControlPath=none -L "${LOCAL_PORTS[$1]}:127.0.0.1:${REMOTE_PORTS[$1]}" "$HOST")
}

local_health() {
  local i=$1 url
  url="$(service_url "$i")/health"
  if [[ "${AUTHS[$i]}" == token ]]; then
    json_get "${KEYS[$i]}" token | sed 's/.*/header = "Authorization: Bearer &"/' | curl -fsS -m 10 -K - "$url"
  else
    curl -fsS -m 10 "$url"
  fi
}

step_tunnel() {
  local i=$1 listener holder attempt
  local key=${KEYS[$i]} lport=${LOCAL_PORTS[$i]} rport=${REMOTE_PORTS[$i]}
  tunnel_command "$i"
  if local_health "$i" >/dev/null 2>&1; then
    log "tunnel:$key" "ok: localhost:$lport -> $HOST:$rport $(local_health "$i")"
    return 0
  fi
  listener=$(lsof -nP -iTCP:"$lport" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)
  if [[ -n "$listener" ]]; then
    holder=$(ps -o command= -p "$listener")
    [[ "$holder" == "${TUNNEL_CMD[*]}" ]] || die "local port $lport is in use by: $holder (pass --local-port $key=N)"
  fi
  if (( DRY_RUN )); then
    log "tunnel:$key" "would ${listener:+replace the stale tunnel and }open localhost:$lport -> $HOST:127.0.0.1:$rport"
    return 0
  fi
  [[ -z "$listener" ]] || kill "$listener"
  "${TUNNEL_CMD[@]}"
  for (( attempt = 1; attempt <= TUNNEL_HEALTH_ATTEMPTS; attempt++ )); do
    if local_health "$i" >/dev/null 2>&1; then
      log "tunnel:$key" "ok: localhost:$lport -> $HOST:$rport $(local_health "$i")"
      return 0
    fi
    sleep 2
  done
  die "tunnel:$key is up but GET /health through it fails"
}

print_summary() {
  local i
  if (( DRY_RUN )); then
    log summary "dry run: nothing was changed (service JSON would be $SERVICE_JSON)"
    for i in "${!COMPONENTS[@]}"; do
      log summary "${KEYS[$i]}: localhost:${LOCAL_PORTS[$i]} -> box 127.0.0.1:${REMOTE_PORTS[$i]}, unit rojev-${NAMES[$i]}, auth ${AUTHS[$i]}"
    done
    return
  fi
  log summary "service JSON (urls + tokens, mode 600): $SERVICE_JSON"
  for i in "${!COMPONENTS[@]}"; do
    log "next:${KEYS[$i]}" "$(hook_local "${COMPONENTS[$i]}" client_hint "$SERVICE_JSON" "$(service_url "$i")")"
  done
}

main() {
  local i
  parse_args "$@"
  require_mac_tools
  load_components
  open_ssh_master
  step_ssh
  if (( TUNNEL_ONLY )); then
    [[ -f "$SERVICE_JSON" ]] || die "no $SERVICE_JSON yet: run without --tunnel first"
    pick_local_ports
    for i in "${!COMPONENTS[@]}"; do
      step_tunnel "$i"
    done
    print_summary
    return
  fi
  step_gpu
  for i in "${!COMPONENTS[@]}"; do
    run_step local "local-assets:${NAMES[$i]}" "prepare ${NAMES[$i]}'s assets on this Mac" local_assets_ready \
      local_assets_prepare "${COMPONENTS[$i]}"
  done
  step_driver
  run_step all packages "apt-get install build tools" packages_ready packages_install
  run_step all uv "install uv $(uv_version_here)" uv_ready uv_install
  step_code
  step_vendors
  step_components_env
  if (( SKIP_SERVE )); then
    log summary "--skip-serve: box is set up; services and tunnels not touched"
    return
  fi
  pick_local_ports
  run_step local service-json "write the service JSON entries ($(printf '%s ' "${KEYS[@]}"))" service_json_ready \
    service_json_write
  step_tokens
  for i in "${!COMPONENTS[@]}"; do
    step_service "$i"
  done
  for i in "${!COMPONENTS[@]}"; do
    step_tunnel "$i"
  done
  print_summary
}

main "$@"
