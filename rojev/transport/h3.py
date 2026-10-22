"""HTTP/3 (QUIC over UDP) request/response transport between a robot-side client and a GPU-side model service.

Why QUIC for a remote model: one connection carries many independent streams (no TCP head-of-line blocking, so a lost
packet delays one request, not every request behind it); connection set-up and resumption are cheaper than TCP + TLS;
CUBIC congestion control keeps the window open between requests; a connection survives the client changing address
(robot Wi-Fi roaming). It cannot remove the network's round trip: one request still costs at least one ping.

Security: TLS 1.3 is part of QUIC. The server presents a self-signed certificate made by `make_certificate`; the client
trusts exactly that certificate (passed as `cert_pem`, the trust anchor) for the name `SERVER_NAME`, so no other
server can stand in for it. Every request also carries the service's bearer token.

  H3Server(endpoint, host, port, cert, key) runs an asyncio QUIC server on its own thread for an `Endpoint`: a request
                                            is admitted on its headers (token, declared length) before any body is
                                            kept, and a stream whose data passes `max_body` is answered 413 and stopped
                                            (STOP_SENDING, H3_NO_ERROR) without buffering more; the endpoint's handler
                                            runs in a worker thread so model inference never stalls the QUIC loop
  H3Transport(url, token, cert_pem)         one persistent connection on its own thread; `request` (and `health`)
                                            blocks the caller only; a lost connection (`ConnectionLost`: connect
                                            failure, peer close, idle timeout) is re-established and the request resent
                                            once with the same request id (replay-safe: the endpoint never runs it
                                            twice); an error status from the service (`ServiceError`) is never retried
"""
import asyncio
import datetime
import functools
import ssl
import threading
from pathlib import Path
from urllib.parse import urlsplit

from aioquic.asyncio import QuicConnectionProtocol, connect, serve
from aioquic.h3.connection import H3_ALPN, ErrorCode, H3Connection
from aioquic.h3.events import DataReceived, HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ConnectionTerminated
from aioquic.tls import SessionTicket

from .endpoint import BAD_LENGTH, REQUEST_ID, TOO_LARGE, Endpoint, Response, new_request_id, parse_length

SERVER_NAME = "rojev-gpu"


class ConnectionLost(ConnectionError):
    """The QUIC connection is gone; a new one may succeed."""


class ServiceError(ConnectionError):
    """The service answered with an error status; retrying the same request will not help."""

CONGESTION = "cubic"


def make_certificate(cert_path: Path, key_path: Path, days: int = 365) -> str:
    """A self-signed EC P-256 certificate for `SERVER_NAME` (usable as its own trust anchor); returns its PEM."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, SERVER_NAME)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(SERVER_NAME)]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    pem = cert.public_bytes(serialization.Encoding.PEM)
    cert_path.write_bytes(pem)
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    key_path.chmod(0o600)
    return pem.decode()


class _ServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, endpoint: Endpoint, **kwargs):
        super().__init__(*args, **kwargs)
        self._h3 = H3Connection(self._quic)
        self._endpoint = endpoint
        self._open: dict[int, tuple[str, str, dict[str, str], bytearray]] = {}

    def quic_event_received(self, event) -> None:
        for e in self._h3.handle_event(event):
            if isinstance(e, HeadersReceived):
                self._admit(e.stream_id, dict(e.headers), e.stream_ended)
            elif isinstance(e, DataReceived) and e.stream_id in self._open:
                body = self._open[e.stream_id][3]
                body.extend(e.data)
                if len(body) > self._endpoint.max_body:
                    del self._open[e.stream_id]
                    self._refuse(e.stream_id, TOO_LARGE, e.stream_ended)
            if isinstance(e, (HeadersReceived, DataReceived)) and e.stream_ended and e.stream_id in self._open:
                method, path, headers, body = self._open.pop(e.stream_id)
                asyncio.ensure_future(self._respond(e.stream_id, method, path, headers, bytes(body)))

    def _admit(self, stream_id: int, raw: dict[bytes, bytes], ended: bool) -> None:
        headers = {k.decode().lower(): v.decode() for k, v in raw.items() if not k.startswith(b":")}
        try:
            refusal = self._endpoint.admit(headers, parse_length(headers.get("content-length")))
        except ValueError:
            refusal = BAD_LENGTH
        if refusal is not None:
            self._refuse(stream_id, refusal, ended)
            return
        self._open[stream_id] = (raw[b":method"].decode(), raw[b":path"].decode(), headers, bytearray())

    def _refuse(self, stream_id: int, response: Response, ended: bool) -> None:
        """Answer at once and, if the client is still sending, ask it to stop (RFC 9114 §4.1: H3_NO_ERROR)."""
        self._send(stream_id, response)
        if not ended:
            self._quic.stop_stream(stream_id, ErrorCode.H3_NO_ERROR)
        self.transmit()

    async def _respond(self, stream_id: int, method: str, path: str, headers: dict[str, str], body: bytes) -> None:
        response = await asyncio.get_running_loop().run_in_executor(
            None, self._endpoint.handle, method, path, headers, body)
        self._send(stream_id, response)
        self.transmit()

    def _send(self, stream_id: int, response: Response) -> None:
        status, out, ctype = response
        self._h3.send_headers(stream_id, [(b":status", str(status).encode()), (b"content-type", ctype.encode()),
                                          (b"content-length", str(len(out)).encode())])
        self._h3.send_data(stream_id, out, end_stream=True)


class H3Server:
    def __init__(self, endpoint: Endpoint, host: str, port: int, cert: Path, key: Path):
        self.endpoint, self.host, self.port, self.cert, self.key = endpoint, host, port, cert, key
        self._tickets: dict[bytes, SessionTicket] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="h3-server", daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        config = QuicConfiguration(alpn_protocols=H3_ALPN, is_client=False, congestion_control_algorithm=CONGESTION,
                                   max_datagram_frame_size=65536)
        config.load_cert_chain(self.cert, self.key)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(serve(
            self.host, self.port, configuration=config,
            create_protocol=functools.partial(_ServerProtocol, endpoint=self.endpoint),
            session_ticket_fetcher=self._tickets.pop, session_ticket_handler=self._store_ticket))
        self._ready.set()
        self._loop.run_forever()

    def _store_ticket(self, ticket: SessionTicket) -> None:
        self._tickets[ticket.ticket] = ticket


class _ClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._h3 = H3Connection(self._quic)
        self._pending: dict[int, tuple[asyncio.Future, list, bytearray]] = {}
        self.terminated = False

    async def request(self, method: str, authority: str, path: str, headers: list, body: bytes) -> tuple[int, bytes]:
        stream_id = self._quic.get_next_available_stream_id()
        self._h3.send_headers(stream_id, [(b":method", method.encode()), (b":scheme", b"https"),
                                          (b":authority", authority.encode()), (b":path", path.encode()), *headers],
                              end_stream=not body)
        if body:
            self._h3.send_data(stream_id, body, end_stream=True)
        future = asyncio.get_running_loop().create_future()
        self._pending[stream_id] = (future, [0], bytearray())
        self.transmit()
        return await future

    def quic_event_received(self, event) -> None:
        if isinstance(event, ConnectionTerminated):
            self.terminated = True
            for future, _, _ in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionLost(f"QUIC connection closed: {event.reason_phrase or 'peer'}"))
            self._pending.clear()
        for e in self._h3.handle_event(event):
            entry = self._pending.get(e.stream_id) if isinstance(e, (HeadersReceived, DataReceived)) else None
            if entry is None:
                continue
            if isinstance(e, HeadersReceived):
                entry[1][0] = int(dict(e.headers)[b":status"])
            else:
                entry[2].extend(e.data)
            if e.stream_ended:
                self._pending.pop(e.stream_id)
                if not entry[0].done():
                    entry[0].set_result((entry[1][0], bytes(entry[2])))


class H3Transport:
    """`url` = h3://host:port[/base]; `cert_pem` = the server's certificate (its only trusted identity)."""

    def __init__(self, url: str, token: str, cert_pem: str, timeout_s: float = 120.0):
        parts = urlsplit(url.rstrip("/"))
        self.host, self.port, self.base = parts.hostname, parts.port, parts.path
        self.token, self.cert_pem, self.timeout_s = token, cert_pem, timeout_s
        self._ticket: SessionTicket | None = None
        self._protocol: _ClientProtocol | None = None
        self._closed: asyncio.Event | None = None
        self._holder: asyncio.Task | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="h3-client", daemon=True)
        self._thread.start()

    def request(self, method: str, path: str, body: bytes = b"") -> bytes:
        request_id = new_request_id()
        try:
            return self._call(method, path, body, request_id)
        except ConnectionLost:
            self._drop()
            return self._call(method, path, body, request_id)

    def health(self) -> str:
        return self.request("GET", "/health").decode()

    def close(self) -> None:
        asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(self.timeout_s)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _shutdown(self) -> None:
        await self._release()
        if self._holder is not None:
            await self._holder

    def _call(self, method: str, path: str, body: bytes, request_id: str) -> bytes:
        future = asyncio.run_coroutine_threadsafe(self._send(method, path, body, request_id), self._loop)
        status, data = future.result(self.timeout_s)
        if status != 200:
            raise ServiceError(f"{method} {path}: HTTP/3 status {status}")
        return data

    async def _send(self, method: str, path: str, body: bytes, request_id: str) -> tuple[int, bytes]:
        if self._protocol is None or self._protocol.terminated:
            await self._release()
            await self._connect()
        headers = [(b"authorization", f"Bearer {self.token}".encode()), (REQUEST_ID.encode(), request_id.encode()),
                   (b"content-type", b"application/octet-stream"), (b"content-length", str(len(body)).encode())]
        return await self._protocol.request(method, f"{self.host}:{self.port}", self.base + path, headers, body)

    async def _connect(self) -> None:
        config = QuicConfiguration(alpn_protocols=H3_ALPN, is_client=True, congestion_control_algorithm=CONGESTION,
                                   server_name=SERVER_NAME, cadata=self.cert_pem.encode(),
                                   verify_mode=ssl.CERT_REQUIRED, max_datagram_frame_size=65536)
        if self._ticket is not None:
            config.session_ticket = self._ticket
        ready = asyncio.get_running_loop().create_future()
        self._closed = asyncio.Event()
        self._holder = asyncio.ensure_future(self._hold(config, ready))
        await ready

    async def _hold(self, config: QuicConfiguration, ready: asyncio.Future) -> None:
        held: _ClientProtocol | None = None
        try:
            async with connect(self.host, self.port, configuration=config, create_protocol=_ClientProtocol,
                               session_ticket_handler=self._keep_ticket) as protocol:
                held = self._protocol = protocol
                ready.set_result(None)
                stop, peer = asyncio.ensure_future(self._closed.wait()), asyncio.ensure_future(protocol.wait_closed())
                await asyncio.wait({stop, peer}, return_when=asyncio.FIRST_COMPLETED)
                stop.cancel()
                peer.cancel()
        except Exception as e:
            if not ready.done():
                ready.set_exception(ConnectionLost(f"QUIC connect to {self.host}:{self.port} failed: {e}"))
        finally:
            if held is not None and self._protocol is held:
                self._protocol = None

    def _keep_ticket(self, ticket: SessionTicket) -> None:
        self._ticket = ticket

    def _drop(self) -> None:
        """Close the connection and forget it before returning, so the next request opens a new one."""
        asyncio.run_coroutine_threadsafe(self._release(), self._loop).result(self.timeout_s)

    async def _release(self) -> None:
        self._protocol = None
        if self._closed is not None:
            self._closed.set()
