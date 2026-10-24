# GPU setup: one command from the Mac to serving S1 + S2 models (2026-10-21, parallel since 2026-10-24)

Code: `scripts/gpu_setup.sh` (Mac orchestrator, bash 3.2), `scripts/gpu_box.sh` (box-side step library, streamed over SSH per call), `scripts/hf_pins.py` (pinned HF snapshots), `<kind>/<name>/gpu/hook.sh` (one per component). Decisions 2026-10-21 (components, one hook interface) and 2026-10-24 (independent steps at once). Transport and services: [`remote-models.md`](remote-models.md).

## Usage
- `scripts/gpu_setup.sh <ssh-host> [--executor robocasa-pi052|robocasa-xr1] [--services onejev,mjev,sam2]` (at least one component): fresh Ubuntu GPU box → services running, one SSH tunnel each, one service JSON for the box.
- `--dry-run` (alias `--check`): every check runs, nothing changes on the box or the Mac; prints `would …` with the files that differ, and `blocked: …` for a foreign process or too little GPU memory.
- `--tunnel`: only reopen the tunnels and check health (e.g. after the Mac slept). `--skip-serve`: set up, leave services and tunnels alone. `--restart`: restart every selected service.
- `--local-port KEY=N` (repeatable; default: the port already in the JSON, else the first free 8010–8099), `--driver-branch B` (default 580).
- Service JSON `~/.config/rojev/services/<host>.json` (dir 700, file 600): one entry per component key, `{url, token}` (`onejev`: url only). Other entries, and other keys inside an entry (e.g. `codec`), are kept; `url` is always rewritten to the tunnel's `http://127.0.0.1:<port>`, so an `h3://` service needs its own JSON file.
- Use from the Mac: `cd executors/robocasa-pi052 && uv run python realtime.py --policy remote [--s1 xr1] --service <json>`. `realtime.py` reads `pi052` / `xr1` (`RemotePolicy(**entry)`), `onejev` (`--chooser onejev`, `--verify onejev`; `onejev.KeptClient(url)`) and `mjev` (`--verify mjev`); `gpu_suite.py` and `calibrate_jev.py` take the same file.
- Needs: an SSH alias with key login (BatchMode; ssh-agent for passphrases), passwordless sudo on the box; for `robocasa-pi052` an HF login on the Mac with access to `google/paligemma-3b-pt-224`.

## Components
| Component | Key | Box port | Auth | GPU need (MiB) | Idle exit | Weights / env |
|---|---|---|---|---|---|---|
| `executors/robocasa-pi052` (S1) | `pi052` | 8000 | token | 10240 | 600 min | `lerobot/pi052_robocasa` + FAST tokenizer (box); gated PaliGemma tokenizer from the Mac; vendors LeRobot steerable, RoboCasa, robosuite |
| `executors/robocasa-xr1` (S1) | `xr1` | 8004 | token | 11776 | 600 min | `config/xr1_weights.txt` (Xiaomi-Robotics-1-RoboCasa365, ~10.2 GB); flash-attn 2.8.3 from its sdist, readiness imports `flash_attn_2_cuda`; no vendor |
| `services/onejev` (S2) | `onejev` | 8002 | none | 14336 | none | `config/onejev_weights.txt` (OneJev1.6-4B), `qev serve --multimodal --dtype bfloat16 --host 127.0.0.1` |
| `services/mjev` (S2) | `mjev` | 8001 | token | 9728 | 600 min | `config/mjev_weights.txt` + the base model named in its `config.json`; bf16 trunk; out of the v1 loop |
| `services/sam2` (measurement) | `sam2` | 8003 | token | 3072 | 600 min | `config/sam2_weights.txt` (SAM 2.1 Hiera Tiny); not in the loop |
- With an executor, the core's OneJev vendor (`config/onejev.commit`) is cloned too; vendors are deduplicated across components.
- Measured on one A100 40 GB with π0.5² + MJev + OneJev resident: 32.6 GB (π0.5² ~9.6, MJev ~9.0, OneJev ~13.2 GB; 2026-10-21). XR-1 alone: 10.2 GB (E23).
- HTTP only: no hook starts HTTP/3. By hand: `make_certificate`, `serve.py --h3-port N --cert C --key K`, a separate service JSON with `h3://host:N` and `cert_pem`.
- `PI052_SERVE_FLAGS` (e.g. `--planner-memory`) is read by the π0.5² hook's `serve`; `gpu_setup.sh` never sets it and rewrites the launcher on every start, so an A/B run sets it by hand.

## Steps (each prints `[step] ok | would … | done`)
1. `ssh` (BatchMode, accepts a new host key) → `gpu` (NVIDIA display / 3D controller on PCI).
2. `packages` (apt: build-essential git rsync curl), alone: apt holds the dpkg lock.
3. Three branches at once, synchronised by marker files:
   - per component `local-assets:<n>` on the Mac (hook `check_local` / `prepare_local`; a missing HF login fails here, early);
   - driver: `driver` (skipped if `nvidia-smi` works; else `nvidia-driver-<B>-server` + headers, utils by the built module's version, `modprobe`) → `gpu-mode` (MIG off; refused while GPU processes run);
   - code: `uv` (≥ the Mac's version) → `code:core` (rojev/ + root pyproject / uv.lock / README; only with an executor) → `code:shared` (config/, scripts/) → `code:<n>` (hook files + `gpu/`) → `vendor:<dir>` (clones at `config/*.commit`) → every component's chain at once: `env:<n>` (`uv sync --frozen`, uv-managed Python) → waits for its local assets → `assets:<n>` → waits for the driver (readiness may load CUDA kernels) → `prepare-remote:<n>` (weights) → `torch:<n>` (CUDA visible).
4. `service-json` → `token:<key>` (auth = token) → `service:<n>` started one at a time, GPU memory projected across the starts → health waits (`GET /health`, up to 1200 s) at once → `tunnel:<key>` at once.
- Trees mirror with `--delete`, compared by checksum; each job's log prints after its group finishes; any failure stops the setup.
- Restart scope: a service restarts only when its own steps changed (code, env, assets, weights, token) or a shared one did (driver, gpu-mode, packages, uv, config / scripts, vendors); `code:core` restarts only the executor.

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
| Box | `serve` | exec the service in the foreground; reads `$ROJEV_TOKEN` when auth = token |
- Full contract: `executors/robocasa-pi052/gpu/hook.sh` header. A new component = one hook; `gpu_setup.sh` refuses clashing names, keys or ports.

## Box layout
- `~/rojev/` code + `vendor/`; `~/.rojev/<name>/{token, serve.sh, serve.log}` (`serve.sh` exports `ROJEV_TOKEN` from the token file); transient unit `rojev-<name>` (`systemd-run`; `systemctl status rojev-<name>`; output appended to `serve.log`).

## Pitfalls (cause → fix)
- `nvidia-driver-570-server` is transitional to the 580 kernel module; `nvidia-utils-570-server` then mismatches → utils installed by the built module's version (`modinfo -F version nvidia`).
- MIG enabled with no instances hides the GPU from CUDA → `gpu-mode` turns MIG off (only with no GPU processes).
- System Python 3.12 has no `Python.h`; evdev (robosuite → pynput → evdev) builds from source → `UV_PYTHON_PREFERENCE=only-managed` + `build-essential`.
- `pkill -f 'python serve.py'` inside `ssh host '…'` kills its own shell → services run as systemd units; stop / status by unit name.
- qev binds 0.0.0.0 by default and has no auth → OneJev is always served with `--host 127.0.0.1`; the tunnel is the only access path.
- MJev loads its vision tower from a base model named in the checkpoint's `config.json` → the base is resolved from there, never hard-coded.
- flash-attn's prebuilt wheel names a local version its metadata lacks, so `uv sync --frozen --check` never passed → built from the sdist with `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE` (fetches the matching wheel); a kernel-less build fails readiness.
- macOS openrsync: `-c -L` checksums the symlink, not its target (every HF-cache file reads as changed, exit 23) → assets compare by size + mtime; `-R --delete` deletes inside implied parent dirs → trees with `--delete` synced one by one.
- Fresh boxes hold the dpkg lock (unattended-upgrades) → apt waits (`DPkg::Lock::Timeout=600`).
- `modprobe nvidia` fails (nouveau, Secure Boot) or `nvidia-smi` version mismatch → `ssh <host> sudo reboot`, re-run.
- A process on a component's box port not started by this script → `blocked: foreign process`; never stopped. Free GPU memory below `gpu_mem_mib` → start refused with the GPU process list.

## Security
- Tokens (one per auth = token component): made on the Mac, kept in the service JSON (600) and `~/.rojev/<name>/token` (600); sent only inside the SSH-streamed script as a here-document; compared by sha256; never printed, never on a command line (`curl -K -` reads the header from stdin).
- No Hugging Face token leaves the Mac: the gated tokenizer is downloaded there and only its files are copied; every other weight is public.
- Every service this script starts binds 127.0.0.1 on the box; only the SSH tunnels reach them. A hand-started `serve.py --h3-port` binds UDP on all interfaces (bearer token + pinned certificate).

## Verified
- 2026-10-21: `bash -n` + shellcheck 0.11 clean on all scripts and hooks; Mac-side logic run locally (component loading and clash detection, vendor dedup, port picking, service JSON modes and tokens, restart scoping); `--dry-run` against an A100 box set up by hand.
- 2026-10-22, first real run on a fresh box (A100 SXM4 40 GB, Ubuntu 24.04 without a driver): driver, packages, envs, weights, tokens, OneJev and MJev healthy; π0.5² died at import (hook did not ship `embodiment.py`) → added to `files`, re-run idempotent.
- 2026-10-24: parallel steps checked with `--dry-run`; XR-1 env with flash-attn from its sdist on the A100 box.
- Not exercised: the GPU memory refusal.

## After setup: the GPU test suite
- `cd executors/robocasa-pi052 && uv run python gpu_suite.py --service <json> [--only …] [--quick]` (π0.5²-based v1 suite).
- Steps, sequential on purpose (one GPU; π0.5²'s service serves one request at a time): `baseline` (π0.5² native per task), `k_samples` (K = 1 / 3 / 5 latency, spread, GPU memory), `jev` (OneJev latency; live wait / stop / ask runs with the deadline from the measured latency; calibration cases recorded), `mjev` (oracle arm + MJev step watcher; P(done) vs truth), `t8a_gate` (commitment ablation + gate run with video). Opt-in: `done_check` (`--done-frames DIR` from `done_check_ab.py build`), `commit_ab`.
- Output `results/gpu-suite/<stamp>/{<step>/…, suite.json, suite.md}`; a failed step is recorded and the suite goes on.
