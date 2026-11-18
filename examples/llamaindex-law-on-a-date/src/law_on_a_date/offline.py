"""The --dry-run stand-ins: LexAPI answered by an in-process mock transport, the model by scripted replies.

The fixtures live in ``offline/fixtures`` at the repository root. They are synthetic: placeholder
texts in the response shapes the SDK reads, not recorded from the live API. The mock lists the act
as published for dates before 2020 and a placeholder consolidated text from 2020 on; that rule is
part of the fixture, not LexAPI's data.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_chat_callback

FIXTURES_DIR = Path(__file__).resolve().parents[4] / "offline" / "fixtures"
DRY_RUN_BASE_URL = "http://127.0.0.1:9/api/v1"  # never connected to: requests go to the mock transport
DRY_RUN_BANNER = (
    "DRY RUN - LexAPI was replaced by a local mock and the model by a scripted reply.\n"
    "The texts and versions are synthetic placeholders, not the wording or the versions of any act.\n"
)
CONSOLIDATED_FROM = "2020-01-01"  # the fixture's rule (see the module docstring)

Reply = Callable[[dict[str, Any]], tuple[int, Any]]


class FixturesNotFoundError(RuntimeError):
    pass


def read_fixture(name: str) -> Any:
    if not FIXTURES_DIR.is_dir():
        raise FixturesNotFoundError(
            f"Offline fixtures not found at {FIXTURES_DIR}. --dry-run needs a checkout of the repository "
            "with an editable install: pip install -e ."
        )
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _provision_id(value: str) -> str:
    digits = re.fullmatch(r"(?:article\s*)?(\d+)", value.strip(), re.IGNORECASE)
    return f"art_{digits.group(1)}" if digits else value


def as_of_reply(body: dict[str, Any]) -> dict[str, Any]:
    """The synthetic answer of POST /legislation/as-of, for the date and provision asked."""
    as_of = str(body.get("asOf"))
    name = "as-published" if as_of < CONSOLIDATED_FROM else "consolidated"
    reply: dict[str, Any] = read_fixture(f"legislation-as-of.gdpr.{name}.json")
    reply["asOf"] = as_of
    if body.get("provision"):
        reply["provision"]["id"] = _provision_id(str(body["provision"]))
    else:
        del reply["provision"]
        reply["credits"]["operation_weight"] = 1  # no text fetched
    return reply


def search_reply(body: dict[str, Any]) -> dict[str, Any]:
    """The synthetic answer of POST /legislation/semantic, echoing the date asked."""
    reply: dict[str, Any] = read_fixture("legislation-semantic.gdpr-on-a-date.json")
    as_of = body.get("asOf")
    reply["asOf"] = as_of
    version = read_fixture("legislation-as-of.gdpr.consolidated.json")
    for hit in reply["results"]:
        hit["asOf"] = as_of
        if str(as_of) >= CONSOLIDATED_FROM and hit["law_id"] == "32016R0679" and hit["asOfResolved"]:
            celex = version["asOfVersion"]["celex"]
            hit.update(
                law_id=celex,
                consolidated=True,
                baseCelex="32016R0679",
                consolidatedAsOf=version["consolidatedAsOf"],
                consolidatedNote=version["consolidatedNote"],
                asOfVersion=copy.deepcopy(version["asOfVersion"]),
                provenance={"sourceUrl": version["provenance"]["sourceUrl"]},
            )
    return reply


def _default(path: str) -> Reply | None:
    if path.endswith("/legislation/as-of"):
        return lambda body: (200, as_of_reply(body))
    if path.endswith("/legislation/semantic"):
        return lambda body: (200, search_reply(body))
    return None


@dataclass
class LexAPIMock:
    """LexAPI's two endpoints used here, in-process. Records every request and keeps a credit pool.

    ``replies`` replaces an endpoint's answer (key: the path's last part, ``as-of`` or ``semantic``).
    """

    replies: dict[str, Reply] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)
    pool: int = 500

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) for r in self.requests]

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            if not str(request.url).startswith("http://127.0.0.1:"):
                raise RuntimeError(f"The mock refuses {request.url}")
            self.requests.append(request)
            path = request.url.path
            reply = self.replies.get(path.rsplit("/", 1)[-1]) or _default(path)
            if request.method != "POST" or reply is None:
                return httpx.Response(404, json={"success": False, "error": {"code": "NOT_FOUND", "message": "none"}})
            status, payload = reply(json.loads(request.content or b"{}"))
            credits = payload.get("credits") if isinstance(payload, dict) else None
            if status == 200 and isinstance(credits, dict) and isinstance(credits.get("operation_weight"), int):
                self.pool -= credits["operation_weight"]
                credits["remaining"] = self.pool
            return httpx.Response(status, json=payload)

        return httpx.MockTransport(handle)


class ScriptedLLM(CustomLLM):
    """A LlamaIndex LLM that answers chats with scripted replies, in order, and keeps the messages it was sent."""

    replies: list[Any] = Field(default_factory=list)
    seen: list[list[ChatMessage]] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "ScriptedLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=True, model_name="scripted")

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError("ScriptedLLM answers chats only")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("ScriptedLLM answers chats only")

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        self.seen.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, ChatResponse):
            return reply
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=str(reply)), raw={})


def dry_run_model(with_provision: bool) -> ScriptedLLM:
    replies = read_fixture("llm/law-on-a-date.json")
    return ScriptedLLM(replies=[replies["answer" if with_provision else "answer_without_provision"]])
