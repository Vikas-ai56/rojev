# Remote models: services, transport, wire format (2026-10-21 / 22)

Code: `rojev/transport/` (`open_transport`, `HttpTransport`, `H3Transport`, `Endpoint`, `HttpServer`, `H3Server`, `make_certificate`), `rojev/system1/remote.py` (`RemotePolicy`), `rojev/system1/wire.py` (codecs), `executors/robocasa-pi052/serve.py` (π0.5² service), `services/mjev/serve_mjev.py`, `rojev/system2/models/mjev.py` (`RemoteMJev`), `rojev/system2/models/onejev.py`. Tests: `tests/test_transport.py` (16, loopback, stand-in service, TCP proxy that loses replies / drops connections), `executors/robocasa-pi052/tests/test_serve.py` (3, stand-in policy, no GPU). Box setup: [`gpu-setup.md`](gpu-setup.md).

## Services
| Service | Role | Endpoints | Auth | Box bind | Client |
|---|---|---|---|---|---|
| π0.5² (`serve.py`) | S1 | `POST /act` (wire v2, K samples), `POST /reset`, `GET /health` | `Bearer $ROJEV_TOKEN` (`Endpoint`, before the body) | HTTP/1.1 127.0.0.1:8000; with `--h3-port`: HTTP/3 UDP 0.0.0.0 | `RemotePolicy` over `rojev.transport` |
| MJev (`serve_mjev.py`) | S2 evidence | `POST /ask` (JSON, base64 PNG images + questions), `GET /health` | bearer token | HTTP/1.1 keep-alive, threaded, 127.0.0.1:8001 | `RemoteMJev`, `urllib` (new connection per request) |
| OneJev (qev) | S2 decision | qev API | none | 127.0.0.1:8002 (`--host 127.0.0.1` mandatory) | qev `Client` |
- All three reach the Mac through SSH tunnels made by `scripts/gpu_setup.sh`; only π0.5² has an HTTP/3 path.
- `serve.py`: `--idle-min` (default 30; the GPU hook passes 600) exits after that long without requests; `MAX_BODY` 32 MB → 413 before buffering.

## Transport (`rojev/transport/`; depends on nothing else in rojev)
- `open_transport(url, token, cert_pem=None, timeout_s=120)`: `h3://host:port` → `H3Transport` (needs `cert_pem`, else `ValueError`); anything else → `HttpTransport`. Both: `request(method, path, body) → bytes`, `health()`, `close()`.
- Replay-safe retry: every `request` sends a fresh `X-Request-Id` (128 random bits); its one retry after a lost connection reuses it. A lost connection can come after the service ran the request (reply lost), so the server de-duplicates by id (`Endpoint`).
- `HttpTransport`: one persistent `http.client` connection (keep-alive), `https` supported. A dropped connection (`RemoteDisconnected`, `BrokenPipeError`, `ConnectionResetError`, `CannotSendRequest`) is closed and the request sent once more, same id; non-200 → `ConnectionError`, not retried. `health()` opens its own short connection.
- `H3Transport` (aioquic, CUBIC): one QUIC connection held on its own asyncio thread; `request` blocks only the caller. Server certificate pinned as the only trust anchor (`cadata = cert_pem`, name `SERVER_NAME` = `rojev-gpu`, `CERT_REQUIRED`); bearer token on every request; session tickets kept for resumption.
  - `ConnectionLost` (connect failure, peer close, idle timeout; also for requests in flight on a connection closed locally) → reconnect and resend once, same id. `health()` = `request("GET", "/health")`, same path.
  - `ServiceError` (non-200 status) → raised, never retried.
- 0-RTT early data not used (a network replay beyond the replay cache's reach would run `/act` twice).

## Server layer (`endpoint.py`, `HttpServer`, `H3Server`): every service gets it
- A service writes only its routing handler `(method, path, headers, body) → (status, body, type)` (headers lower-cased) and wraps it: `Endpoint(handler, token, max_body, replay_capacity=256)`.
- `admit(headers, length)` runs on the headers, before any body byte is kept: wrong token → 401 (`hmac.compare_digest` on bytes); declared length > `max_body` → 413; malformed length → 400.
- `handle(...)` after the body: `ReplayCache` = at most one execution per request id; a copy arriving while the first runs waits for it; later copies get the stored response (`replays` counted); id reused for a different method / path / body (SHA-256) → 409; handler exception → 500 (logged, also stored). No id → runs every time.
- Bound: last 256 completed responses kept (a retry follows its failed attempt at once); running entries never evicted.
- `HttpServer(endpoint, host, port)`: threaded HTTP/1.1 keep-alive. Refused request: body ≤ limit is discarded unbuffered (clean status, connection kept); larger → reply + `Connection: close`, body never read. `Transfer-Encoding` → 411.
- `H3Server(endpoint, host, port, cert, key)`: asyncio QUIC server on its own thread; refusal on headers, or once a stream's data passes `max_body` → reply 413 + `STOP_SENDING` (H3_NO_ERROR, RFC 9114 §4.1), nothing more buffered; handler in a worker thread so inference never stalls the QUIC loop (ACKs keep flowing).
- `make_certificate(cert, key, days=365)`: self-signed EC P-256 for `rojev-gpu`, key file mode 600; returns the PEM the client pins.

## Wire format (`wire.py`; numpy only, `allow_pickle=False`)
- `VERSION` = 2 in request and response; another version (v1 = no `version` key) → `ValueError` (service: 400).
- Request = compressed `.npz`: per camera `img/<cam>` (uint8 H×W×3) or `webp/<cam>` (WebP bytes); `state` float32; `task`, `subtask` unicode scalars; `samples` K ≥ 1. `decode_request` → `ActRequest(observation, subtask, samples)`.
- Response = compressed `.npz`: `actions` float32 K×n×d; `timings_ms`, `info` = JSON lists, one entry per sample. `decode_response` → `list[ActionChunk]`.
- Codecs (`CODECS`): `raw` lossless (~150 KB for three 256² cameras); `webp90` WebP quality 90, method 4 (~26 KB, ~3 ms encode per camera). `DEFAULT_CODEC` = `webp90`: the one place the default lives (`encode_request`, `RemotePolicy`).
- `RemotePolicy.act_samples(obs, subtask, k)`: one request; per chunk `network` = round trip − the server's stage times (also contains server decode / encode and lock wait), `round_trip`, `request_kb`; K ≠ returned count → `ValueError`. `act` = `act_samples(…, 1)[0]`.

## Server side (`serve.py`)
- `Service(policy, about).handle` = routing only (`/health` without the model lock; `/act`, `/reset` serialized by `lock`; bad wire → 400); `main` wraps it in one `Endpoint(service.handle, $ROJEV_TOKEN, MAX_BODY)` served by `HttpServer` (127.0.0.1) and, with `--h3-port N --cert C --key K` (all three or refused), `H3Server` (0.0.0.0).
- `Pi052Policy.act_samples(obs, subtask, k)`: raw observation → `preprocess_observation` → checkpoint preprocessor (once) → subtask: given, or π0.5²'s own planner run once (state kept until the task text changes or `/reset`; returned in each chunk's `info`) → PI052 adapter `select_action` over `Draws(policy, k)` → postprocessor per action → `to_action_space`, per chunk. `act` = K = 1.
- `Draws`: the adapter builds the prompt for one observation, then `predict_action_chunk` repeats every conditioned tensor K times along the batch → `sample_actions` draws K independent noises in one forward batch. Prefix (images + prompt) computed per row: LeRobot's `sample_actions` builds the prefix KV cache and the denoising loop in one method, so sharing one prefix would mean re-implementing it in the executor (rejected).
- Timings (shared by the K chunks): `preprocess`, `plan` (planner; ~0 with a given subtask), `infer` (K draws), `postprocess`.
- `to_action_space`: every PandaOmron dimension with a threshold (gripper, base mode) → its high value at or above the threshold, else low, as RoboCasa's gym wrapper does; continuous dimensions untouched. Before it, the first live run reported 486 of 600 ticks as violations, all on the two switches (outputs up to 1.018); continuous overshoots are still reported, as they should be.

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
- One request ≥ one ping; distance dominates. At L ≈ 0.57 s the commitment floor is ~24 of 50 actions ([`s1-supply-and-commitment.md`](s1-supply-and-commitment.md)).

## Default and why
- Decided (decisions 2026-10-21): HTTP/1.1 keep-alive + WebP-90. Best p50 and tail in the WAN run; WebP within sampling noise; ~6× fewer bytes → fewer packets, fewer loss episodes. HTTP/3 kept: aioquic was inconsistent across runs and not better here.
- In code: `gpu_setup.sh` writes `http://127.0.0.1:<port>` (keep-alive); `RemotePolicy` defaults to `wire.DEFAULT_CODEC` (WebP-90); a `"codec"` key in the service JSON entry overrides it (`service_addresses` → `RemotePolicy(**entry)`).

## Not done
- Streamed frames and server-pushed chunks (removes one one-way trip); a native QUIC stack (quiche / msquic); a closer GPU region.
- HTTP/3 in the GPU setup: no certificate, UDP rule or `cert_pem` entry is made; started by hand.
- MJev and OneJev on `rojev.transport`: `RemoteMJev` opens a new connection per request.
- Shared prefix across K draws (needs a LeRobot entry point that takes a prefix cache).
- MJev on `Endpoint` (replay cache, admission before the body): adopted by the main session.
- Not validated: codec check at scale (8 frames; recheck in T8a); batched K on the GPU (below).

## To measure (batched K; GPU box terminated 2026-10-22, no numbers yet)
1. Setup: same box class (A100 40 GB), `serve.py` resident, real π0.5², a fixed set of recorded RoboCasa observations (≥ 10 frames, three cameras), WebP-90; run on the box over loopback (inference only) and from the Mac (WAN).
2. Latency: per K ∈ {1, 3, 5}, interleaved, 30 requests per K, both modes (given subtask, planner); record server `preprocess` / `plan` / `infer` / `postprocess` and `round_trip`; report p50 / p90 per K and `torch.cuda.max_memory_allocated`.
3. Diversity: per frame, K = 5 → mean pairwise distance between the 5 chunks (actions 0–5, as the codec check); compare with the distance between independent K = 1 calls on the same frame (0.070 baseline). Batched draws must match independent ones (independent noise per row).
4. Consequence: L(K) → floor ⌈2L/dt⌉ + 1 vs the 50-action chunk; `S1Stats.floor_over_chunk` must stay 0 for the K chosen in T8a.

## Security
- Tokens: made on the Mac, sent to the box only inside the SSH-streamed script, read by the service from `ROJEV_TOKEN` (launcher reads the token file); never on a command line or printed.
- HTTP/3: pinned self-signed certificate (no public CA; no other server can stand in) + bearer token; binds all interfaces, so it needs the cloud firewall's UDP port.
- HTTP/1.1 services and qev bind loopback only; the SSH tunnel is the only path (qev has no auth).
- Request size and token: checked by `Endpoint.admit` on the headers, before any body is buffered, on both transports (HTTP/3 also stops a stream once its data passes the limit), so an unauthenticated client on the public HTTP/3 port cannot make the service buffer a body; `serve_mjev.py` checks the size before reading (moves to `Endpoint` with the main session).
