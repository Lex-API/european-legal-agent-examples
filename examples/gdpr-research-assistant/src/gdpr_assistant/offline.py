"""A local stand-in for the LexAPI REST API, for tests and ``--dry-run``.

It listens on 127.0.0.1 only and replays the files in ``offline/fixtures`` at
the repository root (see ``routes.json`` there). The fixtures are synthetic:
they follow the response shapes the SDKs type, but were not recorded from the
live API.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

FIXTURES_DIR = Path(__file__).resolve().parents[4] / "offline" / "fixtures"
LOCAL_PREFIX = "http://127.0.0.1:"

DRY_RUN_BANNER = (
    "> DRY RUN - LexAPI was replaced by a local stub and the model by scripted replies.\n"
    "> The legal texts below are synthetic placeholders, not the wording of any act or judgment.\n"
)


def read_fixture(name: str) -> str:
    if not FIXTURES_DIR.is_dir():
        # The fixtures live in the repository, not in the installed package.
        raise RuntimeError(
            f"Offline fixtures not found at {FIXTURES_DIR}. --dry-run and the tests need a checkout of the "
            'repository with an editable install: pip install -e ".[anthropic]"'
        )
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def assert_local_base_url(base_url: str | None) -> str:
    """The offline rule: refuse any LexAPI base URL that is not a 127.0.0.1 stub."""
    if not base_url or not base_url.startswith(LOCAL_PREFIX):
        raise RuntimeError(f"Refusing to run: LEXAPI_BASE_URL must start with {LOCAL_PREFIX} (got {base_url!r}).")
    return base_url


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    query: str
    headers: dict[str, str]
    body: str


#: An override returns ``(status, body)``, ``(status, body, headers)`` or ``None`` to fall through to ``routes.json``.
Override = Callable[[RecordedRequest], "tuple[int, Any] | tuple[int, Any, dict[str, str]] | None"]


@dataclass
class StubServer:
    base_url: str
    requests: list[RecordedRequest] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()


def start_stub_server(override: Override | None = None) -> StubServer:
    routes: list[dict[str, Any]] = json.loads(read_fixture("routes.json"))
    stub = StubServer(base_url="")

    def reply(request: RecordedRequest) -> tuple[int, Any, dict[str, str]]:
        if override and (custom := override(request)) is not None:
            return (custom[0], custom[1], custom[2] if len(custom) > 2 else {})
        haystack = unquote(request.query) if request.method == "GET" else request.body
        for route in routes:
            if route["method"] != request.method or route["path"] != request.path:
                continue
            if route.get("includes") and route["includes"] not in haystack:
                continue
            return route.get("status", 200), read_fixture(route["fixture"]), {}
        message = f"The stub has no fixture for {request.method} {request.path}"
        return 404, {"success": False, "error": {"code": "NOT_FOUND", "message": message}}, {}

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            url = urlsplit(self.path)
            length = int(self.headers.get("content-length") or 0)
            request = RecordedRequest(
                method=self.command,
                path=unquote(url.path),
                query=url.query,
                headers={k.lower(): v for k, v in self.headers.items()},
                body=self.rfile.read(length).decode("utf-8") if length else "",
            )
            stub.requests.append(request)
            status, body, extra_headers = reply(request)
            payload = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
            self.send_response(status)
            headers = {
                "content-type": "application/json",
                "content-length": str(len(payload)),
                # The real API sends these on every /api/v1 response; the SDK reads them.
                "x-lexapi-version": "stub",
                "ratelimit-limit": "15",
                "ratelimit-remaining": "14",
                "ratelimit-reset": "60",
                **extra_headers,
            }
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _handle  # noqa: N815 - http.server's naming
        do_POST = _handle  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    stub.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    stub._server = server
    return stub
