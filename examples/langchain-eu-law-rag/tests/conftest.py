"""Offline test harness.

Nothing may leave this machine. For the whole session, socket connections and
name lookups for anything but 127.0.0.1 fail, httpx's real transports fail,
and LangSmith clients cannot be created; tracing is off and no model key is
set. LexAPI is answered in-process by an ``httpx.MockTransport`` on an
explicit ``http://127.0.0.1:`` base URL, and the model is a fake chat model.
Every fixture body is synthetic.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import langsmith
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from dora_rag import rag

# Before anything can read them.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
for _name in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT"):
    os.environ.pop(_name, None)

BASE_URL = "http://127.0.0.1:9/api/v1"
TEST_KEY = "lex_offline_test_key_do_not_print_0123456789"
OFFLINE_ERROR = "Offline tests blocked a real network request"

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
        patch.setattr(langsmith.Client, "__init__", _refuse)
        yield


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", TEST_KEY)
    monkeypatch.setenv("LEXAPI_BASE_URL", BASE_URL)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # A developer's .env must not leak into the tests.
    monkeypatch.setattr(rag, "load_env_file", lambda path: None)


def search_reply(as_of: str | None) -> dict[str, Any]:
    """The synthetic DORA search answer, echoing the date asked."""
    if as_of is None:
        loaded: dict[str, Any] = rag.read_fixture("legislation-semantic.dora.json")
        return loaded
    body: dict[str, Any] = rag.read_fixture("legislation-semantic.dora-asof.json")
    body["asOf"] = as_of
    for hit in body["results"]:
        hit["asOf"] = as_of
    return body


Reply = Callable[[dict[str, Any]], tuple[int, Any]]


def default_reply(body: dict[str, Any]) -> tuple[int, Any]:
    return 200, search_reply(body.get("asOf"))


@dataclass
class Stub:
    """The LexAPI legislation search, in-process; records every request."""

    reply: Reply = default_reply
    requests: list[httpx.Request] = field(default_factory=list)

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) for r in self.requests]

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            assert str(request.url).startswith("http://127.0.0.1:"), request.url
            self.requests.append(request)
            if request.method != "POST" or request.url.path != "/api/v1/legislation/semantic":
                return httpx.Response(404, json={"success": False, "error": {"code": "NOT_FOUND", "message": "none"}})
            status, payload = self.reply(json.loads(request.content))
            return httpx.Response(status, json=payload)

        return httpx.MockTransport(handle)


class RecordingChatModel(GenericFakeChatModel):
    """A fake chat model that keeps the messages it was sent."""

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def fake_model(*replies: AIMessage | str) -> RecordingChatModel:
    return RecordingChatModel(messages=iter(replies))


@pytest.fixture
def stub() -> Stub:
    return Stub()
