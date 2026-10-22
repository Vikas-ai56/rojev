"""Transport: request/response bytes between the robot side and a remote model service (System 1 or 2 on a GPU).

Client: `open_transport(url, token, cert_pem)` picks the implementation from the URL scheme:
  h3://host:port        HTTP/3 over QUIC (UDP), certificate-pinned TLS 1.3 (`h3.H3Transport`)
  http(s)://host:port   HTTP/1.1 over one persistent TCP connection (`http.HttpTransport`), e.g. through an SSH tunnel
Both expose `request(method, path, body) -> bytes`, `health() -> str` and `close()`; each request carries a fresh
request id that its one retry (after a lost connection) reuses.

Server: a service writes its routing handler and wraps it in `Endpoint(handler, token, max_body)` (bearer token and
body limit checked before the body is buffered; at most one execution per request id, so a retry after a lost reply
never runs the handler twice); `HttpServer` (HTTP/1.1) and `H3Server` (HTTP/3) serve the same endpoint.
Depends on nothing else in rojev.
"""
from .endpoint import Endpoint
from .h3 import SERVER_NAME, H3Server, H3Transport, make_certificate
from .http import HttpServer, HttpTransport


def open_transport(url: str, token: str, cert_pem: str | None = None, timeout_s: float = 120.0):
    if url.startswith("h3://"):
        if cert_pem is None:
            raise ValueError("an h3:// service needs the server certificate (cert_pem) to pin")
        return H3Transport(url, token, cert_pem, timeout_s)
    return HttpTransport(url, token, timeout_s)


__all__ = ["SERVER_NAME", "Endpoint", "H3Server", "H3Transport", "HttpServer", "HttpTransport", "make_certificate",
           "open_transport"]
