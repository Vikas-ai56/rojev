"""`onejev.KeptClient`: qev requests over one kept connection, reopened when the server drops it."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rojev.system2.models.onejev import KeptClient


class Echo(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    connections: set = set()
    drop = False

    def do_POST(self):
        Echo.connections.add(self.client_address)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps({"echo": body}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        if Echo.drop:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_requests_share_one_connection_and_a_dropped_one_is_reopened():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = KeptClient(f"http://127.0.0.1:{server.server_address[1]}")
        assert [client._post("/x", {"n": n})[0]["echo"]["n"] for n in range(3)] == [0, 1, 2]
        assert len(Echo.connections) == 1
        Echo.drop = True
        client._post("/x", {"n": 3})
        Echo.drop = False
        assert client._post("/x", {"n": 4})[0]["echo"]["n"] == 4 and len(Echo.connections) == 2
        client.close()
    finally:
        server.shutdown()
