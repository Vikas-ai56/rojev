# Remote models: services, transport, wire format (2026-10-21, updated 2026-10-24)

Code: `rojev/transport/` (`open_transport`, `HttpTransport`, `H3Transport`, `Endpoint`, `HttpServer`, `H3Server`, `make_certificate`), `rojev/system1/wire.py` (wire v3, codecs), `rojev/system1/remote.py` (`RemotePolicy`), `rojev/system1/service.py` (`PolicyService`, `serve_policy`), `executors/robocasa-pi052/serve.py` (π0.5²), `executors/robocasa-xr1/serve.py` (XR-1), `rojev/system2/models/onejev.py` (`KeptClient`), `rojev/system2/models/mjev.py` (`RemoteMJev`), `rojev/system2/models/sam2.py` (`RemoteSam2`), `services/mjev/serve_mjev.py`, `services/sam2/serve_sam2.py`. Tests: `tests/test_transport.py` (19: loopback, stand-in service, TCP proxy that loses replies / drops connections), `tests/test_onejev_client.py` (1: one shared connection, a dropped one reopened), `executors/robocasa-pi052/tests/test_serve.py` (5: stand-in policy, no GPU). Box setup and ports: `scripts/gpu_setup.sh`, `scripts/gpu_box.sh`.

## Services
| Service | Role | Endpoints | Auth | Server | Client |
|---|---|---|---|---|---|
| XR-1 (`executors/robocasa-xr1/serve.py`) | S1 (current RoboCasa S1) | same as π0.5² | bearer token (`Endpoint`) | `serve_policy`, HTTP/1.1 only (no `--h3-port` flag) | `RemotePolicy` inside `HistoryPolicy` |
| OneJev (`qev serve`) | S2 decision (Jev chooser, `--verify onejev`) | qev API | none | uvicorn, 127.0.0.1 (`--host 127.0.0.1` mandatory) | `onejev.KeptClient` (one kept connection) |
| SAM 2.1 (`serve_sam2.py`) | masks for measurement; not in the loop | `POST /segment`, `/track/start`, `/track/step`, `/track/end`, `GET /health` | bearer token (own check) | own threaded HTTP/1.1, 127.0.0.1 | `RemoteSam2` (`rojev.transport`) |
- Every service reaches the Mac through an SSH tunnel made by `scripts/gpu_setup.sh`; only `serve_policy` services can add HTTP/3, and no hook does.
- `--idle-min` (code default 30; the GPU hooks pass 600) exits after that long without requests; qev has no idle exit. `MAX_BODY` 32 MB on π0.5², XR-1 and MJev.
- XR-1: the RoboCasa365 checkpoint behind our wire, never its upstream pickle socket; history-encoded observation (`<cam>@k`, k = 0 oldest, 4 frames 2 ticks apart, + stacked `agent_pos`) → its prompt and processor → 16 × 12 raw actions, mapped into the action space on the robot side (`embodiment.to_action_space`); stateless, so a retried request answers the same. Measured on the A100: 10.2 GB, ~40 ms preprocess + 275–290 ms inference per query (E23).

## Transport (`rojev/transport/`; depends on nothing else in rojev)
- `open_transport(url, token, cert_pem=None, timeout_s=120)`: `h3://host:port` → `H3Transport` (needs `cert_pem`, else `ValueError`); anything else → `HttpTransport`. Both: `request(method, path, body) → bytes`, `health()`, `close()`.
- Replay-safe retry: every `request` sends a fresh `X-Request-Id` (128 random bits); its one retry after a lost connection reuses it. A lost connection can come after the service ran the request (reply lost), so the server de-duplicates by id (`Endpoint`).
- `HttpTransport`: one persistent `http.client` connection (keep-alive), `https` supported. A dropped connection (`RemoteDisconnected`, `BrokenPipeError`, `ConnectionResetError`, `CannotSendRequest`) is closed and the request sent once more, same id; non-200 → `ConnectionError`, not retried. `health()` opens its own short connection.
- `H3Transport` (aioquic, CUBIC): one QUIC connection held on its own asyncio thread; `request` blocks only the caller. Server certificate pinned as the only trust anchor (`cadata = cert_pem`, name `SERVER_NAME` = `rojev-gpu`, `CERT_REQUIRED`); bearer token on every request; session tickets kept for resumption.
  - `ConnectionLost` (connect failure, peer close, idle timeout; also for requests in flight on a connection closed locally) → reconnect and resend once, same id. `health()` = `request("GET", "/health")`, same path.
  - `ServiceError` (non-200 status) → raised, never retried.
- 0-RTT early data not used (a network replay beyond the replay cache's reach would run `/act` twice).

## Server layer (`endpoint.py`, `HttpServer`, `H3Server`): the S1 services (π0.5², XR-1)
- A service writes only its routing handler `(method, path, headers, body) → (status, body, type)` (headers lower-cased) and wraps it: `Endpoint(handler, token, max_body, replay_capacity=256)`.
- `admit(headers, length)` runs on the headers, before any body byte is kept: wrong token → 401 (`hmac.compare_digest` on bytes); declared length > `max_body` → 413; malformed length → 400.
- `handle(...)` after the body: `ReplayCache` = at most one execution per request id; a copy arriving while the first runs waits for it; later copies get the stored response (`replays` counted); id reused for a different method / path / body (SHA-256) → 409; handler exception → 500 (logged, also stored). No id → runs every time.
- Bound: last 256 completed responses kept (a retry follows its failed attempt at once); running entries never evicted.
- `HttpServer(endpoint, host, port)`: threaded HTTP/1.1 keep-alive. Refused request: body ≤ limit is discarded unbuffered (clean status, connection kept); larger → reply + `Connection: close`, body never read. `Transfer-Encoding` → 411.
- `H3Server(endpoint, host, port, cert, key)`: asyncio QUIC server on its own thread; refusal on headers, or once a stream's data passes `max_body` → reply 413 + `STOP_SENDING` (H3_NO_ERROR, RFC 9114 §4.1), nothing more buffered; handler in a worker thread so inference never stalls the QUIC loop (ACKs keep flowing).
- `make_certificate(cert, key, days=365)`: self-signed EC P-256 for `rojev-gpu`, key file mode 600; returns the PEM the client pins.

## Wire format (`wire.py`; numpy only, `allow_pickle=False`)
- `VERSION` = 3 in request and response (v3 adds the optional continuation); another version (v1 = no `version` key) → `ValueError` (service: 400).
- Request = compressed `.npz`: per camera `img/<cam>` (uint8 H×W×3) or `webp/<cam>` (WebP bytes); `state` float32; `task`, `subtask` unicode scalars; `samples` K ≥ 1; optional `prefix` float32 n×d + `delay_steps` (the `Continuation`). `decode_request` → `ActRequest(observation, subtask, samples, continuation)`.
- Response = compressed `.npz`: `actions` float32 K×n×d; `timings_ms`, `info` = JSON lists, one entry per sample. `decode_response` → `list[ActionChunk]`.
- Codecs (`CODECS`): `raw` lossless (~150 KB for three 256² cameras); `webp90` WebP quality 90, method 4 (~26 KB, ~3 ms encode per camera). `DEFAULT_CODEC` = `webp90`: the one place the default lives (`encode_request`, `RemotePolicy`).
- `RemotePolicy.act_samples(obs, subtask, k)`: one request; per chunk `network` = round trip − the server's stage times (also contains server decode / encode and lock wait), `round_trip`, `request_kb`; K ≠ returned count → `ValueError`. `act` = `act_samples(…, 1)[0]`.

## Server side (`PolicyService`, π0.5² `serve.py`)
- `rojev.system1.PolicyService(policy, about).handle` = routing only (`/health` without the model lock; `/act`, `/reset` serialized by `lock`; bad wire → 400; a continuation reaches only a policy that `accepts_continuation`). Each executor's `main` wraps it in one `Endpoint(service.handle, $ROJEV_TOKEN, MAX_BODY)` and calls `serve_policy`: `HttpServer` (127.0.0.1), plus `H3Server` (0.0.0.0) with `--h3-port N --cert C --key K` (π0.5² only; all three or refused).
- `Pi052Policy.act_samples(obs, subtask, k)`: raw observation → `preprocess_observation` → checkpoint preprocessor (once) → subtask: given, or π0.5²'s own planner run once (state kept until the task text changes or `/reset`; returned in each chunk's `info`) → PI052 adapter `select_action` over `Draws(policy, k)` → postprocessor per action → `to_action_space`, per chunk. `act` = K = 1.
- Real-time chunking (`--rtc H`): a request's `Continuation` (actions the robot runs from the observation's time; `delay_steps` = ⌈L/dt⌉) → checkpoint normalizer (MEAN_STD) → LeRobot RTC guidance (`prev_chunk_left_over`, `inference_delay`, `execution_horizon` = H, EXP prefix schedule) through `Draws`; under `no_grad` (RTC uses autograd), else `inference_mode`; `info["rtc"]` says whether it was used. Off by default.
- `Draws`: the adapter builds the prompt for one observation, then `predict_action_chunk` repeats every conditioned tensor K times along the batch → `sample_actions` draws K independent noises in one forward batch. Prefix (images + prompt) computed per row: LeRobot's `sample_actions` builds the prefix KV cache and the denoising loop in one method, so sharing one prefix would mean re-implementing it in the executor (rejected).
- Timings (shared by the K chunks): `preprocess`, `plan` (planner; ~0 with a given subtask), `infer` (K draws), `postprocess`.
- `embodiment.to_action_space` (π0.5²: in `serve.py`; XR-1: on the robot side): every PandaOmron dimension with a threshold (gripper, base mode) → its high value at or above the threshold, else low, as RoboCasa's gym wrapper does; continuous dimensions untouched. Before it, the first live run reported 486 of 600 ticks as violations, all on the two switches (outputs up to 1.018); continuous overshoots are still reported, as they should be.

## Measurements
Setup (2026-10-21): A100 40 GB box in Ashburn VA (`lambda`, since terminated), Mac in India, ping 216 ms; π0.5² + MJev + OneJev resident (32.6 GB); real π0.5², real RoboCasa frames.
| Path | Network (ms) | Round trip (ms) | Notes |
|---|---|---|---|
| HTTP/1.1, new connection per request, raw | 680 | 1,012 | SSH tunnel; inference 308 ms |
| HTTP/1.1 keep-alive, raw | 255 (p90 414) | 572 | |
| Keep-alive + WebP-90 | p50 239 / p90 282 | p50 559 / p90 597 | 60 interleaved requests per path |
| Keep-alive + raw | 283 / 482 | — | same 60-request run |
| HTTP/3 + WebP-90 | 302 / 482 | 621 / 799 | same run |
| HTTP/3 + raw | 324 / 902 | — | 15 / 60 above 800 ms |
| Box loopback | transport overhead 21 (HTTP/3) vs 45 (HTTP/1.1) | — | real π0.5² |
- Codec check (8 frames): actions from WebP-90 frames differ from lossless by 0.078 mean (actions 0–5) vs 0.070 between two lossless calls (π0.5²'s own sampling noise).
- One request ≥ one ping; distance dominates. At L ≈ 0.57 s the commitment floor is ~24 of 50 actions ([`s1-supply-and-commitment.md`](../utils/s1-supply-and-commitment.md)).

## Default and why
- Decided: HTTP/1.1 keep-alive + WebP-90. Best p50 and tail in the WAN run; WebP within sampling noise; 

## Not done
- Streamed frames and server-pushed chunks (removes one one-way trip); a native QUIC stack (quiche / msquic); a closer GPU region.
- HTTP/3 in the GPU setup: no certificate, UDP rule or `cert_pem` entry is made; started by hand.
- OneJev server keep-alive: `qev serve` calls `uvicorn.run` without `timeout_keep_alive` (`vendor/OneJev/qev/cli.py:135`), so uvicorn's 5 s default drops a connection idle ~5 s (E30) and `KeptClient` reconnects between decisions seconds apart (in the loop ~930 ms, not 604); qev exposes no flag for it (decision 2026-10-24: fix on the server side, no client ping).
- Shared prefix across K draws (needs a LeRobot entry point that takes a prefix cache).
