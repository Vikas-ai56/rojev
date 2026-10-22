# GPU setup: one command from the Mac to serving S1 + S2 models (2026-10-21)

## Usage
- `scripts/gpu_setup.sh <ssh-host> --executor robocasa-pi052 --services onejev,mjev`: fresh Ubuntu GPU box → π0.5², OneJev and MJev running, one SSH tunnel each, one service JSON for the box.
- `--dry-run` (alias `--check`): every check runs, nothing changes on the box or the Mac; prints `would …` with the files that differ, and `blocked: …` for a foreign process or too little GPU memory.
- `--tunnel`: only reopen the tunnels and check health (e.g. after the Mac slept). `--skip-serve`: set up, leave services and tunnels alone. `--restart`: restart every selected service.
- `--local-port KEY=N` (repeatable; default: the port already in the JSON, else the first free 8010–8099), `--driver-branch B` (default 580).
- Service JSON `~/.config/rojev/services/<host>.json`: `{"pi052": {url, token}, "mjev": {url, token}, "onejev": {url}}`; other entries, and other keys inside an entry, are kept. `realtime.py` passes the whole `pi052` entry to `RemotePolicy(**entry)`, so `"codec": "webp90"` (the decided remote default; the code default is `raw`) goes there. `url` is always rewritten to the tunnel's `http://127.0.0.1:<port>`: an `h3://` entry needs its own JSON file ([`remote-models.md`](remote-models.md)).
- Then: `cd executors/robocasa-pi052 && uv run python realtime.py --policy remote --service ~/.config/rojev/services/<host>.json` (`service_addresses` reads the `pi052` entry; workflow arm D also reads `mjev`; OneJev: `--onejev <url>` / `QEV_BASE_URL`).
- Needs: an SSH alias with key login (BatchMode; ssh-agent for passphrases), passwordless sudo on the box, an HF login on the Mac with access to `google/paligemma-3b-pt-224`.

## Components
| Component | Key | Box port | Auth | GPU need | Weights |
|---|---|---|---|---|---|
| `executors/robocasa-pi052` (S1) | `pi052` | 8000 | token | 10240 MiB | `lerobot/pi052_robocasa` + FAST tokenizer; gated PaliGemma tokenizer from the Mac |
| `services/onejev` (S2) | `onejev` | 8002 | none (qev has no auth; `--host 127.0.0.1` mandatory) | 14336 MiB | `config/onejev_weights.txt` (OneJev1.6-4B) |
| `services/mjev` (S2) | `mjev` | 8001 | token | 9728 MiB | `config/mjev_weights.txt` + base model read from its `config.json` |
- Measured on one A100 40 GB with all three: 32.6 GB used (π0.5² ~9.6, MJev ~9.0, OneJev ~13.2 GB).
- π0.5² and MJev hooks start their services with `--idle-min 600` (exit after 10 h without requests; re-running the setup restarts them).
- HTTP only: the π0.5² hook runs `serve.py --tokenizer-dir … --port 8000 --idle-min 600`, without `--h3-port --cert --key`. No certificate is made, no UDP port opened, no `cert_pem` written; HTTP/3 is started by hand (`make_certificate`, then `serve.py --h3-port N --cert C --key K`, then a separate service JSON with `h3://host:N` and `cert_pem`).

## Files
| File | Role |
|---|---|
| `scripts/gpu_setup.sh` | Mac orchestrator: components, steps, rsync, tokens, service JSON, GPU memory check, tunnels (bash 3.2) |
| `scripts/gpu_box.sh` | box-side step library, streamed over SSH per call (`*_ready` checks, `*_install`, systemd units) |
| `scripts/hf_pins.py` | pinned HF snapshot fetch / offline lookup / path, run with the component's env |
| `<kind>/<name>/gpu/hook.sh` | component interface (below) |
| `executors/robocasa-pi052/gpu/hub.py` | gated tokenizer (Mac) and π0.5² weights (box) |
| `config/onejev_weights.txt`, `config/mjev_weights.txt` | weight pins (`repo <id> @ <revision>`) |

## Steps (each prints `[step] ok | would … | done`)
- `ssh` → `gpu` → `local-assets:<n>` (first: a missing HF login fails before a long install).
- `driver`: skipped if `nvidia-smi` works; else `nvidia-driver-<B>-server` + `linux-headers-$(uname -r)`, then `nvidia-utils-<major of the built module>-server`, `modprobe nvidia`, `nvidia-smi`.
- `packages` (build-essential git rsync curl) → `uv` (≥ the Mac's version).
- `code:core` (rojev/ + root pyproject/uv.lock/README; only with an executor) → `code:shared` (config/, scripts/) → `code:<n>` (hook files + gpu/). Trees mirrored with `--delete`, compared by checksum.
- `vendor:<dir>`: clone + checkout of `config/*.commit` pins (core OneJev + each hook's vendors, deduplicated).
- Per component: `env:<n>` (`uv sync --frozen`, uv-managed Python) → `assets:<n>` → `prepare-remote:<n>` → `torch:<n>` (CUDA visible).
- `service-json` → `token:<key>` (auth=token only) → `service:<n>` per component (start sequentially, each waits for `GET /health`) → `tunnel:<key>`.
- Restart scope: a component's service restarts only when its own steps changed (its code, env, assets, weights, token), or a shared step changed (driver, packages, uv, config/scripts, vendors); `code:core` restarts only the executor.

## Component interface: `<kind>/<name>/gpu/hook.sh <subcommand>` (working dir = component dir, `ROJEV_ROOT` = repo root)
| Side | Subcommand | Contract |
|---|---|---|
| Mac | `key` / `auth` | entry name in the service JSON / `token` or `none` |
| Mac | `gpu_mem_mib` | GPU memory the running service needs |
| Mac | `files` | component files to mirror (relative); `gpu/` always mirrored |
| Mac | `vendors` | `<dir>\t<git url>\t<pin file in config/>` per line |
| Mac | `remote_port` | port bound on the box's 127.0.0.1 |
| Mac | `check_local` / `prepare_local` / `assets` | Mac-side assets: ready? / fetch / `<local file>\t<box dir>` lines |
| Mac | `client_hint JSON URL` | how to use the service from the Mac |
| Box | `check_remote` / `prepare_remote` | box-side preparation done? / do it |
| Box | `serve` | exec the service in the foreground; reads `$ROJEV_TOKEN` when auth=token |

## Box layout
- `~/rojev/` code + `vendor/`; `~/.rojev/<name>/{token, serve.sh, serve.log}` (`serve.sh` exports `ROJEV_TOKEN` from the token file); transient unit `rojev-<name>` (`systemd-run`; `systemctl status rojev-<name>`; output in `serve.log`).

## Pitfalls (cause → fix)
- `nvidia-driver-570-server` is transitional to the 580 kernel module; `nvidia-utils-570-server` then mismatches → utils installed by the built module's version (`modinfo -F version nvidia`).
- System Python 3.12 has no `Python.h`; evdev (robosuite → pynput → evdev) builds from source → `UV_PYTHON_PREFERENCE=only-managed` + `build-essential`.
- `pkill -f 'python serve.py'` inside `ssh host '…'` kills its own shell → services run as systemd units; stop/status by unit name.
- qev binds 0.0.0.0 by default and has no auth → OneJev is always served with `--host 127.0.0.1`; the tunnel is the only access path.
- MJev loads its vision tower from a base model named in the checkpoint's `config.json` → the base is resolved from there, never hard-coded.
- macOS openrsync: `-c -L` checksums the symlink, not its target (every HF-cache file reads as changed, exit 23) → assets compare by size + mtime; `-R --delete` deletes inside implied parent dirs (removed `.venv` in a local test) → trees with `--delete` synced one by one.
- Fresh boxes hold the dpkg lock (unattended-upgrades) → apt waits (`DPkg::Lock::Timeout=600`).
- `modprobe nvidia` fails (nouveau, Secure Boot) or `nvidia-smi` version mismatch → `ssh <host> sudo reboot`, re-run.
- A process on a component's box port not started by this script → `blocked: foreign process`; never stopped. Free GPU memory below `gpu_mem_mib` → start refused with the GPU process list.

## Security
- Tokens (one per auth=token component): made on the Mac, kept in the service JSON (dir 700, file 600) and `~/.rojev/<name>/token` (600); sent only inside the SSH-streamed script as a here-document; compared by sha256; never printed, never on a command line (`curl -K -` reads the header from stdin).
- No Hugging Face token leaves the Mac: the gated tokenizer is downloaded there and only its files are copied; S2 weights are public.
- All services this script starts bind 127.0.0.1 on the box; only the SSH tunnels reach them. A hand-started `serve.py --h3-port` binds UDP on all interfaces (bearer token + pinned certificate).

## Verified (2026-10-21)
- `bash -n` + shellcheck 0.11 clean on all scripts and hooks; hook Mac subcommands for all three components; `hf_pins.py` pin parsing, errors, offline lookup; MJev base resolved from `config.json`.
- Mac-side logic run locally: component loading and clash detection, vendor dedup, local port picking, service JSON write/check (modes, tokens kept across rewrites, no token for onejev), restart scoping; the JSON loads through `workflow.service_addresses`.
- Box library simulated locally: token here-document, file modes, digest equality, launchers with and without token.
- `--dry-run` against `lambda` (A100, set up by hand): driver, packages, uv, vendors, envs, torch ×3 `ok`; code / assets / prepare-remote / tokens / JSON `would`; all three services reported `blocked: foreign process`; nothing written on the box. Pinned S2 weights confirmed in its HF cache.
- 2026-10-22, first real run on a fresh box (`lambda`: A100 SXM4 40 GB, Ubuntu 24.04 without a driver, Phoenix AZ): driver install, packages, envs, weights, tokens, the OneJev and MJev units came up healthy; the π0.5² unit died at import because the hook did not ship `embodiment.py` (imported by `serve.py` since the 2026-10-21 switch mapping) → added to `files()` and re-run (idempotent). Not yet exercised: memory refusal.
- `lambda` was terminated after a backup on 2026-10-21 (`results/gpu-backups/lambda-v1-test-20261021-1456/`); the WAN transport numbers in [`remote-models.md`](remote-models.md) come from it.
