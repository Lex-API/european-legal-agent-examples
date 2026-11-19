"""Offline test harness.

Nothing may leave this machine. For the whole session, socket connections and
name lookups for anything but 127.0.0.1 fail and httpx's real transports fail;
no model key is set. LexAPI is answered in-process by an ``httpx.MockTransport``
on an explicit ``http://127.0.0.1:`` base URL, and the model is a scripted
LlamaIndex LLM. Every fixture body is synthetic. "Today" is fixed.
"""

from __future__ import annotations

import datetime
import socket
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from law_on_a_date import cli, offline

BASE_URL = "http://127.0.0.1:9/api/v1"
TEST_KEY = "lex_offline_test_key_do_not_print_0123456789"
OFFLINE_ERROR = "Offline tests blocked a real network request"
TODAY = datetime.date(2026, 6, 30)

REAL_LOAD_ENV_FILE = cli.load_env_file  # the conftest replaces both for every test
REAL_TODAY = cli.today

_real_connect = socket.socket.connect
_real_getaddrinfo = socket.getaddrinfo


def _loopback_only(self: socket.socket, address: Any) -> None:
    host = address[0] if isinstance(address, tuple) else address
    if host != "127.0.0.1":
        raise ConnectionRefusedError(f"{OFFLINE_ERROR} to {host!r}")
    _real_connect(self, address)


def _resolve_loopback_only(host: Any, *args: Any, **kwargs: Any) -> Any:
    if host != "127.0.0.1":
        raise socket.gaierror(f"{OFFLINE_ERROR} to {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(OFFLINE_ERROR)


@pytest.fixture(scope="session", autouse=True)
def _no_network() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", _loopback_only)
        patch.setattr(socket, "getaddrinfo", _resolve_loopback_only)
        patch.setattr(httpx.HTTPTransport, "handle_request", _refuse)
        patch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _refuse)
        yield


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", TEST_KEY)
    monkeypatch.setenv("LEXAPI_BASE_URL", BASE_URL)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # A developer's .env must not leak into the tests.
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    monkeypatch.setattr(cli, "today", lambda: TODAY)


Stub = offline.LexAPIMock


def scripted(*replies: Any) -> offline.ScriptedLLM:
    return offline.ScriptedLLM(replies=list(replies))


@pytest.fixture
def stub() -> Stub:
    return Stub()
