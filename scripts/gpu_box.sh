# gpu_box.sh: box-side step library for scripts/gpu_setup.sh (runs on the GPU box, never on the Mac).
#
# Not executed directly. gpu_setup.sh streams this file over SSH followed by one call line, e.g.
#   { cat gpu_box.sh; echo "driver_ready"; } | ssh HOST bash -s
# so it works on a fresh box before rsync, uv or the repo exist there, and a dry run leaves nothing behind.
# Secrets (service tokens) arrive as a here-document inside that stream: never on a command line, never in `ps`.
#
# Conventions
#   <thing>_ready ...   exit 0 when the step is already done; quiet; read-only (used by --dry-run)
#   <thing>_install ... make it so; idempotent; loud
#   other functions     queries that print one value (token_digest, service_state, gpu_free_mib)
#   COMPONENT           a repo-relative component dir (executors/<name> or services/<name>); NAME = its last part
#
# Box layout (all under the login user's $HOME)
#   rojev/                     code mirror (core, config/, scripts/, one dir per component), vendor/ clones at pins
#   .rojev/<name>/token        bearer token for that component's service when its auth is token (mode 600)
#   .rojev/<name>/serve.sh     launcher: exports ROJEV_ROOT (+ ROJEV_TOKEN), cds into the component, runs hook serve
#   .rojev/<name>/serve.log    service stdout/stderr (appended across restarts)
#   systemd unit rojev-<name>  transient unit (systemd-run) that owns the service's process tree
#
# Why systemd-run for services: stopping a nohup'd server by pattern (`pkill -f 'python serve.py'`) also matches
# the `bash -c '...'` that SSH runs to issue the kill, so the kill takes down its own shell. A unit owns the whole
# process tree in a cgroup; stop/status are exact, with no pattern matching and no PID files.
#
# Why the driver step installs utils by the built module's version: Ubuntu's nvidia-driver-570-server is a
# transitional package that pulls the 580 kernel module, so naming one branch for driver and utils can still
# leave nvidia-smi (utils) on a different branch than the loaded module. The built module version is the truth.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_PREFERENCE=only-managed
export DEBIAN_FRONTEND=noninteractive

readonly ROJEV_DIR="$HOME/rojev"
readonly STATE_DIR="$HOME/.rojev"
readonly APT_PACKAGES=(build-essential git rsync curl)
readonly NVIDIA_VENDOR=0x10de
readonly SERVICE_READY_TIMEOUT_S=1200
export ROJEV_ROOT="$ROJEV_DIR"

apt_get() {
  sudo -n DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -q "$@"
}

require_sudo() {
  sudo -n true 2>/dev/null || { echo "passwordless sudo is required on the box" >&2; return 1; }
}

gpu_present() {
  local dev
  for dev in /sys/bus/pci/devices/*; do
    if [[ "$(cat "$dev/vendor")" == "$NVIDIA_VENDOR" && "$(cat "$dev/class")" == 0x03* ]]; then
      return 0
    fi
  done
  return 1
}

gpu_summary() {
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
}

gpu_free_mib() {
  nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

gpu_apps() {
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | paste -sd ';' -
}

driver_ready() {
  command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1
}

driver_install() {
  local branch=$1 module_version utils_branch
  require_sudo
  apt_get update
  apt_get install "nvidia-driver-${branch}-server" "linux-headers-$(uname -r)"
  module_version=$(modinfo -F version nvidia) || {
    echo "no nvidia kernel module was built for the running kernel $(uname -r)" >&2
    return 1
  }
  utils_branch=${module_version%%.*}
  apt_get install "nvidia-utils-${utils_branch}-server"
  if ! sudo -n modprobe nvidia; then
    echo "modprobe nvidia failed (module $module_version); a loaded nouveau or Secure Boot needs a reboot:" >&2
    echo "  ssh <host> sudo reboot   then re-run gpu_setup.sh" >&2
    return 1
  fi
  echo "kernel module $module_version, utils branch $utils_branch"
}

whole_gpu_ready() {
  local mode
  mode=$(nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | tr -d ' ')
  [[ "$mode" != "Enabled" ]]
}

whole_gpu_apply() {
  require_sudo
  if [[ -n "$(gpu_apps)" ]]; then
    echo "MIG is enabled and GPU processes are running ($(gpu_apps)); not changing the GPU mode" >&2
    return 1
  fi
  sudo -n nvidia-smi -i 0 -mig 0
}

packages_ready() {
  dpkg -s "${APT_PACKAGES[@]}" >/dev/null 2>&1
}

packages_install() {
  require_sudo
  apt_get update
  apt_get install "${APT_PACKAGES[@]}"
}

uv_ready() {
  local minimum=$1 installed
  command -v uv >/dev/null || return 1
  installed=$(uv --version | awk '{print $2}')
  [[ "$(printf '%s\n%s\n' "$minimum" "$installed" | sort -V | head -1)" == "$minimum" ]]
}

uv_install() {
  local version=$1
  curl -LsSf "https://astral.sh/uv/${version}/install.sh" | sh
}

make_dirs() {
  local dir
  for dir in "$@"; do
    mkdir -p "$HOME/$dir"
  done
}

vendor_ready() {
  local name=$1 pin=$2
  [[ "$(git -C "$ROJEV_DIR/vendor/$name" rev-parse HEAD 2>/dev/null)" == "$pin" ]]
}

vendor_install() {
  local name=$1 url=$2 pin=$3 dir="$ROJEV_DIR/vendor/$1"
  mkdir -p "$ROJEV_DIR/vendor"
  [[ -d "$dir/.git" ]] || git clone -q --filter=blob:none "$url" "$dir"
  git -C "$dir" cat-file -e "${pin}^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin "$pin"
  git -C "$dir" checkout -q --detach "$pin"
}

component_dir() {
  echo "$ROJEV_DIR/$1"
}

env_ready() {
  (cd "$(component_dir "$1")" && uv sync --frozen --check >/dev/null 2>&1)
}

env_install() {
  (cd "$(component_dir "$1")" && uv sync --frozen)
}

torch_cuda() {
  (cd "$(component_dir "$1")" && uv run --frozen --no-sync python -c \
    "import sys, torch; ok = torch.cuda.is_available(); print(torch.__version__, 'cuda', torch.version.cuda, 'available' if ok else 'NOT available'); sys.exit(0 if ok else 1)")
}

hook() {
  local component=$1 subcommand=$2
  (cd "$(component_dir "$component")" && bash gpu/hook.sh "$subcommand")
}

token_file() {
  echo "$STATE_DIR/$1/token"
}

token_digest() {
  local file
  file=$(token_file "$1")
  if [[ -f "$file" ]]; then
    sha256sum < "$file" | awk '{print $1}'
  else
    echo missing
  fi
}

token_store() {
  local name=$1 token file
  file=$(token_file "$name")
  IFS= read -r token
  (umask 077 && mkdir -p "$(dirname "$file")" && printf '%s\n' "$token" > "$file.tmp")
  mv "$file.tmp" "$file"
}

unit_name() {
  echo "rojev-$1"
}

service_health() {
  local name=$1 port=$2 auth=$3 token
  if [[ "$auth" == token ]]; then
    token=$(< "$(token_file "$name")")
    printf 'header = "Authorization: Bearer %s"\n' "$token" | curl -fsS -m 10 -K - "http://127.0.0.1:$port/health"
  else
    curl -fsS -m 10 "http://127.0.0.1:$port/health"
  fi
}

service_state() {
  local name=$1 port=$2 listener
  if systemctl is-active --quiet "$(unit_name "$name")"; then
    echo running
    return
  fi
  listener=$(ss -Hltnp "sport = :$port" 2>/dev/null || true)
  if [[ -n "$listener" ]]; then
    echo "port-busy $listener"
  else
    echo stopped
  fi
}

write_launcher() {
  local name=$1 component=$2 auth=$3 launcher=$4 token_lines=
  if [[ "$auth" == token ]]; then
    token_lines="ROJEV_TOKEN=\$(< \"$(token_file "$name")\")
export ROJEV_TOKEN"
  fi
  cat > "$launcher" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export UV_PYTHON_PREFERENCE=only-managed
export ROJEV_ROOT="$ROJEV_DIR"
$token_lines
cd "$(component_dir "$component")"
exec bash gpu/hook.sh serve
LAUNCHER
  chmod 700 "$launcher"
}

service_start() {
  local name=$1 component=$2 auth=$3 dir="$STATE_DIR/$1"
  require_sudo
  mkdir -p "$dir"
  touch "$dir/serve.log"
  write_launcher "$name" "$component" "$auth" "$dir/serve.sh"
  sudo -n systemctl reset-failed "$(unit_name "$name")" 2>/dev/null || true
  sudo -n systemd-run --quiet --unit="$(unit_name "$name")" --uid="$(id -u)" --gid="$(id -g)" \
    --setenv=HOME="$HOME" --working-directory="$(component_dir "$component")" \
    --property=CollectMode=inactive-or-failed \
    --property=StandardOutput="append:$dir/serve.log" --property=StandardError="append:$dir/serve.log" \
    /bin/bash "$dir/serve.sh"
}

service_stop() {
  require_sudo
  sudo -n systemctl stop "$(unit_name "$1")" 2>/dev/null || true
}

service_wait() {
  local name=$1 port=$2 auth=$3 waited=0
  while ! service_health "$name" "$port" "$auth" >/dev/null 2>&1; do
    if ! systemctl is-active --quiet "$(unit_name "$name")"; then
      echo "$name exited before it was ready; last log lines of $STATE_DIR/$name/serve.log:" >&2
      tail -n 40 "$STATE_DIR/$name/serve.log" >&2
      return 1
    fi
    if (( waited >= SERVICE_READY_TIMEOUT_S )); then
      echo "$name not ready after ${waited}s; see $STATE_DIR/$name/serve.log" >&2
      return 1
    fi
    (( waited % 60 == 0 )) && echo "waiting for $name to load (${waited}s)" >&2
    sleep 5
    waited=$((waited + 5))
  done
  service_health "$name" "$port" "$auth"
}
