"""Offline test harness.

Nothing may leave this machine: every test runs against the local stub, the
harness refuses any LexAPI base URL that is not ``http://127.0.0.1:...``, and
name lookups and socket connections for anything but 127.0.0.1 are blocked -
so an SDK (LexAPI or a model provider) left on its default URL fails loudly
instead of reaching a real service.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest
from lexapi import LexAPI

from gdpr_assistant.offline import StubServer, assert_local_base_url, start_stub_server

_real_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _loopback_only(self: socket.socket, address: Any) -> None:
    host = address[0] if isinstance(address, tuple) else address
    if host != "127.0.0.1":
        raise ConnectionRefusedError(f"Offline tests blocked a connection to {host!r}")
    _real_connect(self, address)


def _resolve_loopback_only(host: Any, *args: Any, **kwargs: Any) -> Any:
    if host != "127.0.0.1":
        raise socket.gaierror(f"Offline tests blocked a connection to {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _loopback_only)
    monkeypatch.setattr(socket, "getaddrinfo", _resolve_loopback_only)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[StubServer]:
    server = start_stub_server()
    monkeypatch.setenv("LEXAPI_BASE_URL", server.base_url)
    monkeypatch.setenv("LEXAPI_API_KEY", "lex_offline_test_key")
    yield server
    server.close()


@pytest.fixture
def client(stub: StubServer) -> Iterator[LexAPI]:
    base_url = assert_local_base_url(os.environ.get("LEXAPI_BASE_URL"))
    with LexAPI(os.environ["LEXAPI_API_KEY"], base_url=base_url, max_retries=0) as api:
        yield api
