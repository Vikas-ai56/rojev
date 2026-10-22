"""Transport on loopback: HTTP/3 (certificate pinning, bearer token, persistent connection, reconnect), HTTP/1.1 keep-alive
through a TCP proxy that can lose replies or drop connections, the shared server `Endpoint` (admission before the body,
at-most-once execution per request id), the versioned wire format and `RemotePolicy` over both transports with a
stand-in policy service."""
import asyncio
import io
import socket
import threading
import time

import numpy as np
import pytest

from rojev.protocol import Observation
from rojev.system1 import ActionChunk, RemotePolicy, wire
from rojev.transport import Endpoint, H3Server, HttpServer, HttpTransport, make_certificate, open_transport
from rojev.transport.endpoint import REQUEST_ID, ReplayCache
from rojev.transport.h3 import ServiceError

TOKEN = "t0ken"
MAX_BODY = 64 * 1024
H3_PORT = 47433


class StandIn:
    """Stand-in policy service: `/act` answers `samples` chunks whose actions carry len(subtask) + sample index.
    `calls` records (path, request id) per execution; `on_call` runs once inside the next execution (to break the
    client's connection after the service has run the request)."""

    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []
        self.on_call = None

    def __call__(self, method, path, headers, body):
        self.calls.append((path, headers.get(REQUEST_ID)))
        hook, self.on_call = self.on_call, None
        if hook is not None:
            hook()
        if path == "/health":
            return 200, b'{"ok": true}', "application/json"
        if path == "/act":
            request = wire.decode_request(body)
            return 200, wire.encode_response([
                ActionChunk(np.full((50, 12), len(request.subtask) + i, np.float32), {"infer": 1.0},
                            {"subtask": request.subtask, "sample": i}) for i in range(request.samples)]), "x/npz"
        return 404, b"", "text/plain"

    def executed(self, path):
        return [rid for p, rid in self.calls if p == path]


class Proxy:
    """Loopback TCP proxy in front of an HTTP server. `drop()` closes every client connection (a restarted tunnel);
    `lose_next_reply = True` closes the client's connection instead of forwarding the next response (the service ran
    the request, its reply was lost)."""

    def __init__(self, upstream_port: int):
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port, self.upstream_port = self.listener.getsockname()[1], upstream_port
        self.clients: list[socket.socket] = []
        self.lose_next_reply = False
        threading.Thread(target=self._accept, daemon=True).start()

    def drop(self) -> None:
        for c in self.clients:
            _shut(c)

    def close(self) -> None:
        self.listener.close()
        self.drop()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            upstream = socket.create_connection(("127.0.0.1", self.upstream_port))
            self.clients.append(client)
            threading.Thread(target=self._pump, args=(client, upstream, False), daemon=True).start()
            threading.Thread(target=self._pump, args=(upstream, client, True), daemon=True).start()

    def _pump(self, src: socket.socket, dst: socket.socket, reply: bool) -> None:
        try:
            while data := src.recv(65536):
                if reply and self.lose_next_reply:
                    self.lose_next_reply = False
                    break
                dst.sendall(data)
        except OSError:
            pass
        _shut(src)
        _shut(dst)


def _shut(s: socket.socket) -> None:
    try:
        s.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    s.close()


@pytest.fixture(scope="module")
def h3_server(tmp_path_factory):
    d = tmp_path_factory.mktemp("h3")
    pem = make_certificate(d / "cert.pem", d / "key.pem")
    service = StandIn()
    endpoint = Endpoint(service, TOKEN, MAX_BODY)
    server = H3Server(endpoint, "127.0.0.1", H3_PORT, d / "cert.pem", d / "key.pem")
    server.start()
    yield pem, service, endpoint
    server.stop()


@pytest.fixture
def h3(h3_server):
    _, service, endpoint = h3_server
    service.calls.clear()
    service.on_call = None
    endpoint.replay.replays = 0
    return h3_server


@pytest.fixture
def http():
    service = StandIn()
    endpoint = Endpoint(service, TOKEN, MAX_BODY)
    server = HttpServer(endpoint, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proxy = Proxy(server.server_port)
    yield proxy, service, endpoint
    proxy.close()
    server.shutdown()


def obs():
    return Observation({"cam": np.zeros((8, 8, 3), np.uint8)}, np.zeros(16, np.float32), "task")


def lose_h3_connection(transport):
    """Close the client's QUIC connection and wait until it is gone, so the request in flight fails with
    `ConnectionLost` after the service has received it."""
    holder = transport._holder
    transport._drop()
    asyncio.run_coroutine_threadsafe(asyncio.wait({holder}), transport._loop).result(10)


def test_h3_round_trip_keeps_one_connection_and_info(h3):
    pem, service, _ = h3
    policy = RemotePolicy(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=pem)
    for subtask in ("pick", "place it"):
        chunk = policy.act(obs(), subtask)
        assert chunk.actions.shape == (50, 12) and chunk.actions[0, 0] == len(subtask)
        assert chunk.info == {"subtask": subtask, "sample": 0} and "network" in chunk.timings_ms
    first = policy.transport._protocol
    policy.act(obs(), "again")
    assert policy.transport._protocol is first
    ids = service.executed("/act")
    assert len(ids) == 3 and len(set(ids)) == 3 and all(ids)
    policy.transport.close()


def test_h3_rejects_a_wrong_token_and_an_unpinned_certificate(h3, tmp_path):
    pem, service, _ = h3
    bad = open_transport(f"h3://127.0.0.1:{H3_PORT}", "wrong", cert_pem=pem)
    with pytest.raises(ConnectionError, match="401"):
        bad.request("GET", "/health")
    with pytest.raises(ConnectionError, match="401"):
        bad.request("POST", "/act", b"x" * (4 * MAX_BODY))
    bad.close()
    other = make_certificate(tmp_path / "c.pem", tmp_path / "k.pem")
    impostor = open_transport(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=other, timeout_s=5)
    with pytest.raises(Exception):
        impostor.request("GET", "/health")
    impostor.close()
    with pytest.raises(ValueError, match="cert_pem"):
        open_transport(f"h3://127.0.0.1:{H3_PORT}", TOKEN)
    assert service.calls == []


def test_h3_reconnects_after_the_connection_drops_between_requests(h3):
    t = open_transport(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=h3[0])
    assert "ok" in t.health()
    t._drop()
    assert "ok" in t.health()
    t.close()


@pytest.mark.parametrize("path", ["/act", "/health"])
def test_h3_retry_after_the_service_ran_the_request_does_not_run_it_twice(h3, path):
    """The connection is lost inside the request, after the service received it: `request` (and `health`) reconnects
    and resends with the same request id, and the endpoint answers the copy from its replay cache."""
    pem, service, endpoint = h3
    policy = RemotePolicy(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=pem)
    assert "ok" in policy.health()
    first = policy.transport._protocol
    service.calls.clear()
    service.on_call = lambda: lose_h3_connection(policy.transport)
    if path == "/act":
        assert policy.act(obs(), "pick").actions[0, 0] == 4
    else:
        assert "ok" in policy.health()
    assert len(service.executed(path)) == 1 and endpoint.replay.replays == 1
    assert policy.transport._protocol is not first
    policy.transport.close()


def test_h3_stream_over_the_limit_is_refused_without_running_the_handler(h3):
    pem, service, _ = h3
    t = open_transport(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=pem)
    with pytest.raises(ServiceError, match="413"):
        t.request("POST", "/act", b"x" * (3 * MAX_BODY))
    assert "ok" in t.health() and service.executed("/act") == []
    t.close()


def test_k_samples_are_one_request_over_both_transports(h3, http):
    for policy, service in ((RemotePolicy(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=h3[0]), h3[1]),
                            (RemotePolicy(f"http://127.0.0.1:{http[0].port}", TOKEN), http[1])):
        chunks = policy.act_samples(obs(), "pick", 3)
        assert [c.actions[0, 0] for c in chunks] == [4, 5, 6] and [c.info["sample"] for c in chunks] == [0, 1, 2]
        assert all("round_trip" in c.timings_ms and c.timings_ms["infer"] == 1.0 for c in chunks)
        assert len(service.executed("/act")) == 1
        policy.transport.close()


def test_http_keep_alive_reuses_one_connection(http):
    proxy, service, _ = http
    policy = RemotePolicy(f"http://127.0.0.1:{proxy.port}", TOKEN)
    for subtask in ("a", "bb", "ccc"):
        assert policy.act(obs(), subtask).info == {"subtask": subtask, "sample": 0}
    assert isinstance(policy.transport, HttpTransport) and len(proxy.clients) == 1
    assert len(set(service.executed("/act"))) == 3


def test_http_retry_after_a_lost_reply_does_not_run_the_request_twice(http):
    proxy, service, endpoint = http
    policy = RemotePolicy(f"http://127.0.0.1:{proxy.port}", TOKEN)
    proxy.lose_next_reply = True
    assert policy.act(obs(), "pick").actions[0, 0] == 4
    assert len(service.executed("/act")) == 1 and endpoint.replay.replays == 1 and len(proxy.clients) == 2


def test_http_stale_connection_is_replaced_and_the_request_sent_once_more(http):
    proxy, service, endpoint = http
    policy = RemotePolicy(f"http://127.0.0.1:{proxy.port}", TOKEN)
    policy.act(obs(), "a")
    proxy.drop()
    time.sleep(0.05)
    assert policy.act(obs(), "bb").actions[0, 0] == 2
    ids = service.executed("/act")
    assert len(ids) == 2 and len(set(ids)) == 2 and endpoint.replay.replays == 0 and len(proxy.clients) == 2


def read_response(s: socket.socket) -> bytes:
    """One whole HTTP/1.1 response (head and its Content-Length body)."""
    data = b""
    while b"\r\n\r\n" not in data:
        data += s.recv(65536)
    head, rest = data.split(b"\r\n\r\n", 1)
    length = int(next(line.split(b":")[1] for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")))
    while len(rest) < length:
        rest += s.recv(65536)
    return head + b"\r\n\r\n" + rest


def raw_exchange(port: int, head: str, body: bytes = b"") -> tuple[bytes, socket.socket]:
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(head.encode() + b"\r\n" + body)
    return read_response(s), s


def test_http_admission_happens_before_the_body_is_read(http):
    proxy, service, _ = http
    port = proxy.port
    reply, s = raw_exchange(port, f"POST /act HTTP/1.1\r\nAuthorization: Bearer {TOKEN}\r\n"
                                  f"Content-Length: {MAX_BODY + 1}\r\n")
    assert reply.startswith(b"HTTP/1.1 413") and b"Connection: close" in reply
    s.close()
    reply, s = raw_exchange(port, f"POST /act HTTP/1.1\r\nAuthorization: Bearer wrong\r\nContent-Length: {10**9}\r\n")
    assert reply.startswith(b"HTTP/1.1 401") and b"Connection: close" in reply
    s.close()
    reply, s = raw_exchange(port, "POST /act HTTP/1.1\r\nAuthorization: Bearer wrong\r\nContent-Length: 5\r\n", b"hello")
    assert reply.startswith(b"HTTP/1.1 401") and b"Connection: close" not in reply
    s.sendall(f"GET /health HTTP/1.1\r\nAuthorization: Bearer {TOKEN}\r\n\r\n".encode())
    assert read_response(s).startswith(b"HTTP/1.1 200")
    s.close()
    reply, s = raw_exchange(port, f"POST /act HTTP/1.1\r\nAuthorization: Bearer {TOKEN}\r\nContent-Length: -1\r\n")
    assert reply.startswith(b"HTTP/1.1 400")
    s.close()
    assert service.executed("/act") == []


def test_replay_cache_runs_once_per_id_even_when_copies_overlap():
    started, release, runs = threading.Event(), threading.Event(), []

    def slow():
        runs.append(1)
        started.set()
        release.wait(5)
        return 200, b"done", "text/plain"

    cache = ReplayCache(capacity=2)
    results = []
    first = threading.Thread(target=lambda: results.append(cache.run("a", b"f", slow)))
    first.start()
    started.wait(5)
    copy = threading.Thread(target=lambda: results.append(cache.run("a", b"f", slow)))
    copy.start()
    time.sleep(0.05)
    assert results == []
    release.set()
    first.join(5)
    copy.join(5)
    assert runs == [1] and results == [(200, b"done", "text/plain")] * 2 and cache.replays == 1
    assert cache.run("a", b"other body", slow)[0] == 409


def test_replay_cache_is_bounded_and_endpoint_errors_are_500_and_kept():
    cache = ReplayCache(capacity=2)
    for rid in ("a", "b", "c"):
        cache.run(rid, b"f", lambda: (200, b"", "t"))
    assert list(cache._entries) == ["b", "c"]
    calls = []

    def boom(method, path, headers, body):
        calls.append(path)
        raise RuntimeError("out of memory")

    endpoint = Endpoint(boom, TOKEN, MAX_BODY)
    headers = {"authorization": f"Bearer {TOKEN}", REQUEST_ID: "x"}
    assert endpoint.admit(headers, 10) is None and endpoint.admit({**headers, "authorization": "Bearer t0ke"}, 1)[0] == 401
    for _ in range(2):
        status, body, _ = endpoint.handle("POST", "/act", headers, b"")
        assert status == 500 and b"RuntimeError: out of memory" in body
    assert calls == ["/act"]


def test_wire_is_versioned_and_carries_k_samples():
    request = wire.decode_request(wire.encode_request(obs(), "s", 4))
    assert request.samples == 4 and request.subtask == "s" and request.observation.task == "task"
    chunks = [ActionChunk(np.full((5, 3), i, np.float32), {"infer": float(i)}, {"i": i}) for i in range(4)]
    back = wire.decode_response(wire.encode_response(chunks))
    assert [c.info for c in back] == [{"i": i} for i in range(4)] and back[3].timings_ms == {"infer": 3.0}
    assert all(np.array_equal(a.actions, b.actions) for a, b in zip(chunks, back))
    with pytest.raises(ValueError):
        wire.encode_request(obs(), "s", 0)
    buf = io.BytesIO()
    np.savez_compressed(buf, version=np.array(1), state=np.zeros(2), task=np.array("t"), subtask=np.array(""),
                        samples=np.array(1))
    with pytest.raises(ValueError, match="wire version 1"):
        wire.decode_request(buf.getvalue())


def test_webp_codec_round_trips_within_a_few_levels_and_is_the_default():
    smooth = (np.add.outer(np.arange(64), np.arange(64)) % 255).astype(np.uint8)
    image = np.stack([smooth, smooth.T, (smooth // 2)], axis=-1)
    o = Observation({"a": image, "b": image}, np.arange(16, dtype=np.float32), "task")
    raw, webp = wire.encode_request(o, "s", codec="raw"), wire.encode_request(o, "s")
    back = wire.decode_request(webp).observation
    assert back.images["a"].shape == image.shape and np.allclose(back.state, o.state)
    assert np.abs(back.images["a"].astype(int) - image.astype(int)).mean() < 4 and len(webp) < len(raw)
    assert np.array_equal(wire.decode_request(raw).observation.images["a"], image)
    with pytest.raises(ValueError):
        wire.encode_request(o, "s", codec="jpeg")


def test_remote_policy_defaults_to_the_wire_default_codec(h3):
    default = RemotePolicy(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=h3[0])
    lossless = RemotePolicy(f"h3://127.0.0.1:{H3_PORT}", TOKEN, cert_pem=h3[0], codec="raw")
    assert default.codec == wire.DEFAULT_CODEC == "webp90" and lossless.act(obs(), "pick").actions.shape == (50, 12)
    default.transport.close()
    lossless.transport.close()


@pytest.mark.parametrize("transport", ["http", "h3"])
def test_a_request_past_its_deadline_is_abandoned_and_the_next_one_answers(request, transport):
    if transport == "http":
        proxy, service, _ = request.getfixturevalue("http")
        policy = RemotePolicy(f"http://127.0.0.1:{proxy.port}", TOKEN)
    else:
        pem, service, _ = request.getfixturevalue("h3")
        policy = RemotePolicy(f"h3://127.0.0.1:{H3_PORT}", TOKEN, pem)
    service.on_call = lambda: time.sleep(0.6)
    w0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        policy.act(obs(), "slow", timeout_s=0.2)
    assert time.perf_counter() - w0 < 0.5
    time.sleep(0.5)
    assert policy.act(obs(), "next", timeout_s=2.0).info["subtask"] == "next"
