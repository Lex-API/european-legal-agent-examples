"""The --dry-run stand-ins: LexAPI answered by an in-process mock transport, the model by a scripted reply.

The fixtures live in ``offline/fixtures`` at the repository root. They are synthetic: placeholder
texts in the response shapes the SDK reads, not recorded from the live API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

FIXTURES_DIR = Path(__file__).resolve().parents[4] / "offline" / "fixtures"
DRY_RUN_BASE_URL = "http://127.0.0.1:9/api/v1"  # never connected to: requests go to the mock transport
DRY_RUN_BANNER = (
    "DRY RUN - LexAPI was replaced by a local mock and the model by a scripted reply.\n"
    "The passages given to the model are synthetic placeholders, not the wording of any act.\n"
)


class FixturesNotFoundError(RuntimeError):
    pass


def read_fixture(name: str) -> Any:
    if not FIXTURES_DIR.is_dir():
        raise FixturesNotFoundError(
            f"Offline fixtures not found at {FIXTURES_DIR}. --dry-run needs a checkout of the repository "
            "with an editable install: pip install -e ."
        )
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def search_reply(as_of: str | None) -> dict[str, Any]:
    """The synthetic DORA search answer, echoing the date asked."""
    if as_of is None:
        plain: dict[str, Any] = read_fixture("legislation-semantic.dora.json")
        return plain
    body: dict[str, Any] = read_fixture("legislation-semantic.dora-asof.json")
    body["asOf"] = as_of
    for hit in body["results"]:
        hit["asOf"] = as_of
    return body


def dry_run_transport() -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith("/legislation/semantic"):
            return httpx.Response(404, json={"success": False, "error": {"code": "NOT_FOUND", "message": "no fixture"}})
        return httpx.Response(200, json=search_reply(json.loads(request.content or b"{}").get("asOf")))

    return httpx.MockTransport(handle)


def dry_run_model(as_of: str | None) -> GenericFakeChatModel:
    replies = read_fixture("llm/dora.json")
    return GenericFakeChatModel(messages=iter([AIMessage(content=replies["answer_as_of" if as_of else "answer"])]))
