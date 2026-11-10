"""Loopback readiness regression, kept outside the locally ignored tests tree.

Run: uv run --no-sync --extra web pytest -q scripts/validation/test_workbench_launcher_proxy.py
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest


def _http_status(address: str) -> int:
    with urlopen(address, timeout=0.5) as response:
        return response.status


def test_readiness_probe_bypasses_proxy_without_changing_other_http_requests(monkeypatch) -> None:
    from photomatagent.web.launcher import _probe_ready

    ui_requests: list[str] = []
    proxy_requests: list[str] = []

    class LocalUI(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            ui_requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<main>workbench</main>")

        def log_message(self, *_args) -> None:
            pass

    class FailingProxy(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            proxy_requests.append(self.path)
            self.send_response(503)
            self.end_headers()

        def log_message(self, *_args) -> None:
            pass

    with (
        ThreadingHTTPServer(("127.0.0.1", 0), LocalUI) as ui,
        ThreadingHTTPServer(("127.0.0.1", 0), FailingProxy) as proxy,
    ):
        threads = [Thread(target=server.serve_forever, daemon=True) for server in (ui, proxy)]
        for thread in threads:
            thread.start()
        try:
            proxy_address = f"http://127.0.0.1:{proxy.server_port}"
            address = f"http://127.0.0.1:{ui.server_port}/"
            monkeypatch.setenv("http_proxy", proxy_address)
            monkeypatch.setenv("HTTP_PROXY", proxy_address)
            monkeypatch.setenv("no_proxy", "")
            monkeypatch.setenv("NO_PROXY", "")
            # urllib caches its default opener; isolate this test from earlier requests.
            monkeypatch.setattr("urllib.request._opener", None)

            with pytest.raises(HTTPError, match="503"):
                _http_status(address)
            _probe_ready(address)
            with pytest.raises(HTTPError, match="503"):
                _http_status(address)

            assert ui_requests == ["/"]
            assert proxy_requests == [address, address]
            assert os.environ["http_proxy"] == proxy_address
            assert os.environ["HTTP_PROXY"] == proxy_address
        finally:
            ui.shutdown()
            proxy.shutdown()
            for thread in threads:
                thread.join()
