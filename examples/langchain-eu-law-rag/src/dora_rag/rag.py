"""DORA questions answered from cited passages, with LangChain and LexAPI.

dora-rag "What does DORA say about ICT risk management frameworks?"
dora-rag --as-of 2025-01-17 "..."     (the text for a date, where LexAPI can establish it)
dora-rag --dry-run "..."              (no keys, synthetic fixtures)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, TextIO

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_lexapi import LexAPILegislationRetriever, RetrievalReport
from lexapi import CreditsExhaustedError, LexAPI, LexAPIError, PaymentPastDueError

#: Regulation (EU) 2022/2554, the Digital Operational Resilience Act.
DORA_CELEX = "32022R2554"
#: Put in front of the question, so the search looks for DORA articles.
DORA_CONTEXT = "Digital Operational Resilience Act (DORA, Regulation (EU) 2022/2554)"
#: Hits asked for. Only DORA hits are kept, after the search (see the README).
K = 10
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
USER_AGENT_SUFFIX = "european-legal-agent-examples"
LOCAL_PREFIX = "http://127.0.0.1:"
DRY_RUN_BASE_URL = "http://127.0.0.1:9/api/v1"  # never connected to: the dry run uses a mock transport
FIXTURES_DIR = Path(__file__).resolve().parents[4] / "offline" / "fixtures"

SYSTEM_PROMPT = resources.files("dora_rag").joinpath("prompt.txt").read_text(encoding="utf-8")
HUMAN_PROMPT = "Date asked: {date}\n\nPassages:\n\n{passages}\n\nQuestion: {question}"
PROMPT = ChatPromptTemplate.from_messages([SystemMessage(content=SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])

DRY_RUN_BANNER = (
    "DRY RUN - LexAPI was replaced by a local mock and the model by a scripted reply.\n"
    "The passages below are synthetic placeholders, not the wording of any act.\n"
)
CLOSING = (
    "Research support, not legal advice. The answer is model-written and may misread the passages; "
    "check it against the sources."
)

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class AnswerError(RuntimeError):
    """The model's reply could not be used as an answer."""


@dataclass(frozen=True)
class Result:
    question: str
    as_of: str | None
    documents: list[Document]
    report: RetrievalReport
    answer: str | None  # None when no model was called


def clean(text: str, *, keep_newlines: bool = False) -> str:
    """Remove control and format characters (terminal escapes, zero-width and bidi marks)."""
    kept = []
    for char in text:
        if char == "\n" and keep_newlines:
            kept.append(char)
        elif unicodedata.category(char) in ("Cc", "Cf", "Zl", "Zp"):
            kept.append(" ")
        else:
            kept.append(char)
    return "".join(kept).strip()


def as_of_date(value: str) -> str:
    """argparse type for --as-of: a real calendar date written YYYY-MM-DD."""
    try:
        if _DATE.match(value):
            return datetime.date.fromisoformat(value).isoformat()
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected a date written YYYY-MM-DD, got {value!r}")


def format_passages(documents: Sequence[Document]) -> str:
    """Number the passages [1], [2], ... Each page_content is the text (or a notice) and its Source line."""
    return "\n\n".join(f"[{n}]\n{doc.page_content}" for n, doc in enumerate(documents, start=1))


def read_answer(message: Any, passage_count: int) -> str:
    """The text of the model's reply. Raises AnswerError rather than return an unusable answer."""
    if not isinstance(message, AIMessage):
        raise AnswerError(f"The model returned {type(message).__name__}, not a message.")
    stop_reason = message.response_metadata.get("stop_reason")
    if stop_reason == "refusal":
        raise AnswerError("The model declined this request.")
    if stop_reason == "max_tokens":
        raise AnswerError("The model's reply was cut off (max_tokens).")
    answer = clean(message.text, keep_newlines=True)
    if not answer:
        raise AnswerError("The model returned no text.")
    cited = {int(n) for group in _CITATION.findall(answer) for n in group.split(",")}
    unknown = sorted(n for n in cited if not 1 <= n <= passage_count)
    if unknown:
        listed = ", ".join(f"[{n}]" for n in unknown)
        raise AnswerError(f"The answer cites {listed}; the passages are numbered 1 to {passage_count}.")
    return answer


def answer_question(
    question: str, *, retriever: LexAPILegislationRetriever, model: BaseChatModel, as_of: str | None = None
) -> Result:
    """Retrieve DORA passages, then have the model answer from them alone."""
    documents, report = retriever.invoke_with_report(f"{DORA_CONTEXT}: {question}", as_of=as_of)
    if not any(not doc.metadata["text_withheld"] for doc in documents):
        # Nothing with text to answer from: no model call.
        return Result(question, as_of, documents, report, answer=None)
    message = (PROMPT | model).invoke(
        {"date": as_of or "none", "passages": format_passages(documents), "question": question}
    )
    return Result(question, as_of, documents, report, answer=read_answer(message, len(documents)))


def credits_text(value: int | None) -> str:
    return "not reported" if value is None else str(value)


def render(result: Result, model_label: str) -> str:
    lines = [f"Question: {clean(result.question)}"]
    if result.as_of:
        lines.append(f"Text asked for: {result.as_of}")
    lines.append(f"Model: {model_label}")
    lines.append("")
    if result.answer is not None:
        lines += ["Answer (written by the model from the passages below):", result.answer, ""]
    elif result.documents:
        lines += [f"None of the passages has text for {result.as_of}, so no model was called.", ""]
    else:
        lines += ["No DORA passages were found for this question, so no model was called.", ""]
    if result.documents:
        lines.append("Sources:")
        for n, doc in enumerate(result.documents, start=1):
            note = " - notice only, no text was given to the model" if doc.metadata["text_withheld"] else ""
            lines.append(f"[{n}] {clean(str(doc.metadata['citation']))}{note}")
        lines.append("")
    report = result.report
    if result.as_of:
        lines.append(
            f"Withheld for {result.as_of}: {report.returned_withheld} passage(s) arrived as a notice without text "
            "(LexAPI did not establish their text for that date)."
        )
    if report.filtered_out:
        lines.append(f"Left out: {report.filtered_out} hit(s) from acts other than DORA.")
    lines.append(
        f"LexAPI credits used for this call: {credits_text(report.units_charged)}. "
        f"Credits remaining: {credits_text(report.credits_remaining)}."
    )
    lines += ["", CLOSING]
    return "\n".join(lines)


# ── setup ───────────────────────────────────────────────────────────────


def load_env_file(path: Path) -> None:
    """Load ``KEY=value`` lines into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def create_client(transport: httpx.BaseTransport | None = None) -> LexAPI:
    """The LexAPI client, from LEXAPI_API_KEY (never hard-code it) and the optional LEXAPI_BASE_URL.

    ``transport`` replaces the network (tests and --dry-run); the base URL must then be a 127.0.0.1 one.
    """
    api_key = os.environ.get("LEXAPI_API_KEY")
    if not api_key:
        raise SystemExit("LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard or use --dry-run.")
    base_url = os.environ.get("LEXAPI_BASE_URL")
    if transport is not None and not (base_url or "").startswith(LOCAL_PREFIX):
        raise RuntimeError(
            f"Refusing to run: with a replaced transport, LEXAPI_BASE_URL must start with {LOCAL_PREFIX}."
        )
    options: dict[str, Any] = {"user_agent_suffix": USER_AGENT_SUFFIX, "transport": transport}
    if base_url:
        options["base_url"] = base_url
    return LexAPI(api_key, **options)


def create_chat_model() -> tuple[BaseChatModel, str]:
    """Anthropic through langchain-anthropic: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default claude-sonnet-5-5)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to .env or use --dry-run.")
    name = os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    return ChatAnthropic(model=name, max_tokens=2000), f"anthropic:{name}"


def read_fixture(name: str) -> Any:
    if not FIXTURES_DIR.is_dir():
        raise RuntimeError(
            f"Offline fixtures not found at {FIXTURES_DIR}. --dry-run needs a checkout of the repository "
            "with an editable install: pip install -e ."
        )
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def dry_run_transport() -> httpx.MockTransport:
    """Answers the legislation search from the synthetic DORA fixtures, echoing the date asked."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if request.method != "POST" or not request.url.path.endswith("/legislation/semantic"):
            return httpx.Response(404, json={"success": False, "error": {"code": "NOT_FOUND", "message": "no fixture"}})
        as_of = body.get("asOf")
        if as_of is None:
            return httpx.Response(200, json=read_fixture("legislation-semantic.dora.json"))
        fixture = read_fixture("legislation-semantic.dora-asof.json")
        fixture["asOf"] = as_of
        for hit in fixture["results"]:
            hit["asOf"] = as_of
        return httpx.Response(200, json=fixture)

    return httpx.MockTransport(handle)


def dry_run_model(as_of: str | None) -> GenericFakeChatModel:
    replies = read_fixture("llm/dora.json")
    return GenericFakeChatModel(messages=iter([AIMessage(content=replies["answer_as_of" if as_of else "answer"])]))


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    chat_model: BaseChatModel | None = None,
    out: TextIO | None = None,
) -> int:
    """Command line. ``transport`` and ``chat_model`` are for the offline tests."""
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="dora-rag", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--as-of", type=as_of_date, metavar="YYYY-MM-DD", help="ask for the text for this date")
    parser.add_argument("--dry-run", action="store_true", help="synthetic fixtures and a scripted reply, no keys")
    args = parser.parse_args(argv)

    load_env_file(Path(__file__).resolve().parents[2] / ".env")

    model_label = "scripted (no model was called)"
    if args.dry_run:
        os.environ["LEXAPI_BASE_URL"] = DRY_RUN_BASE_URL
        os.environ["LEXAPI_API_KEY"] = "lex_dry_run_no_key"
        transport = dry_run_transport()
        chat_model = dry_run_model(args.as_of)
    if chat_model is None:
        chat_model, model_label = create_chat_model()

    try:
        with (
            create_client(transport) as client,
            LexAPILegislationRetriever(client=client, k=K, celex=DORA_CELEX) as retriever,
        ):
            result = answer_question(args.question, retriever=retriever, model=chat_model, as_of=args.as_of)
    except (CreditsExhaustedError, PaymentPastDueError) as error:
        # 402 / 403: the API's own message, never the key.
        print(f"LexAPI refused the request (HTTP {error.status}): {clean(error.message)}", file=sys.stderr)
        if isinstance(error, PaymentPastDueError) and error.billing_url:
            print(f"Billing: {clean(error.billing_url)}", file=sys.stderr)
        return 1
    except LexAPIError as error:
        print(f"LexAPI error {error.code} (HTTP {error.status}): {clean(error.message)}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(DRY_RUN_BANNER, file=out)
    print(render(result, model_label), file=out)
    return 0
