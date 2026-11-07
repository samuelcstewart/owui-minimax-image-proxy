"""Minimal stand-in for the MiniMax API, used by `poe integration-test`.

Serves just enough of the upstream contract for the proxy's image
sanity checks — NO real MiniMax account, key, or network needed:

- `GET /v1/models`        → 200 `{"data": [{"id": "image-01"}]}`
  (proxied by `GET /health`)
- `POST /v1/image_generation` → 200 with a synthetic base64 PNG per
  requested `n`, echoing the request's model/aspect_ratio. If the
  body carries `"_fail_with": {"status_code": N}` (test-only hook),
  responds with that HTTP status and a `base_resp` error payload.

Run standalone:  python3 tests/stub_upstream.py   (listens on :9000)
In compose it runs as the `stub` service on port 9000; the proxy is
pointed at it via MINIMAX_API_BASE_URL=http://stub:9000/v1.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# 1x1 red PNG, generated once (deterministic, no deps).
def _tiny_png() -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, RGB
    raw = zlib.compress(b"\x00\xff\x00\x00")  # filter byte + red pixel
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


TINY_PNG_B64 = base64.b64encode(_tiny_png()).decode()


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "image-01", "object": "model"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"base_resp": {"status_code": 2013, "status_msg": "bad json"}})
            return
        if self.path.rstrip("/").endswith("/image_generation"):
            fail = body.get("_fail_with")
            if isinstance(fail, dict):
                self._send(
                    int(fail.get("status_code", 500)),
                    {
                        "base_resp": {
                            "status_code": int(fail.get("code", 1000)),
                            "status_msg": fail.get("msg", "injected failure"),
                        }
                    },
                )
                return
            n = int(body.get("n", 1))
            self._send(
                200,
                {
                    "id": "stub",
                    "model": body.get("model", "image-01"),
                    "data": {"image_base64": [TINY_PNG_B64] * n},
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                },
            )
        else:
            self._send(404, {"base_resp": {"status_code": 404, "status_msg": "not found"}})

    def log_message(self, format: str, *args: object) -> None:  # quiet
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 9000), Handler).serve_forever()
