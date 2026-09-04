#!/usr/bin/env python3
"""Interactive N2S evaluation workbench. Port 8257."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_api as api  # noqa: E402

HOST = "127.0.0.1"


def _json_response(handler: SimpleHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(api.STATIC_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                return super().do_GET()
            if path == "/api/meta":
                return _json_response(self, api.meta_payload())
            if path == "/api/job":
                return _json_response(self, api.job_status())
            if path.startswith("/api/"):
                return _json_response(self, {"error": "not found"}, 404)
            return super().do_GET()
        except Exception as exc:  # noqa: BLE001
            return _json_response(
                self,
                {"error": str(exc), "trace": traceback.format_exc()},
                500,
            )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = _read_json_body(self)
            if path == "/api/evaluate":
                return _json_response(self, api.evaluate_fast(body))
            if path == "/api/deep-dive":
                return _json_response(self, api.submit_deep_dive(body))
            return _json_response(self, {"error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            return _json_response(
                self,
                {"error": str(exc), "trace": traceback.format_exc()},
                500,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="N2S interactive eval workbench")
    parser.add_argument("--port", type=int, default=api.PORT)
    args = parser.parse_args()
    if not api.STATIC_DIR.is_dir():
        raise SystemExit(f"missing static dir: {api.STATIC_DIR}")
    httpd = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"N2S eval workbench  http://{HOST}:{args.port}/")
    print(f"  Fast path         POST /api/evaluate  (Track A Dual + mismatch)")
    print(f"  Deep dive         POST /api/deep-dive (round-trip forensics)")
    print(f"  Lab               {api.LAB_ROOT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
