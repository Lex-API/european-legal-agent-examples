"""DORA questions answered from cited passages, with LangChain and LexAPI - command line.

dora-rag "What does DORA say about ICT risk management frameworks?"
dora-rag --as-of 2025-01-17 "..."     (the text for a date, where LexAPI can establish it)
dora-rag --dry-run "..."              (no keys, synthetic fixtures)
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_lexapi import LexAPILegislationRetriever, RetrievalReport
from lexapi import LexAPI, LexAPIError, PaymentPastDueError

from dora_rag import offline
from dora_rag.rag import (
    DORA_CELEX,
    AnswerError,
    K,
    ModelCallError,
    NoTextError,
    answer,
    format_passages,
    has_text,
    retrieve,
)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
USER_AGENT_SUFFIX = "european-legal-agent-examples"
LOCAL_PREFIX = "http://127.0.0.1:"
NOTICE_MEANING = (
    "A notice gives only LexAPI's reason for having no text for that date; "
    "check the source for whether the provision applied then."
)
CLOSING = (
    "Research support, not legal advice. The answer is model-written and may misread the passages; "
    "check it against the sources."
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ConfigError(RuntimeError):
    """A missing key or setting (exit code 2)."""


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


def question_text(value: str) -> str:
    """argparse type for the question: not empty or blank."""
    if not value.strip():
        raise argparse.ArgumentTypeError("the question is empty")
    return value.strip()


def credits_line(report: RetrievalReport) -> str:
    def shown(value: int | None) -> str:
        return "not reported" if value is None else str(value)

    return (
        f"LexAPI credits used for this call: {shown(report.units_charged)}. "
        f"Credits remaining: {shown(report.credits_remaining)}."
    )


def source_line(n: int, doc: Document) -> str:
    line = f"[{n}] {clean(str(doc.metadata['citation']))}"
    if doc.metadata.get("text_withheld"):
        reason = doc.metadata.get("as_of_reason")
        why = f" (reason: {clean(str(reason))})" if reason else ""
        return f"{line} - notice only, no text was given to the model{why}"
    if not has_text(doc):
        return f"{line} - no text returned"
    return line


def render(
    *,
    question: str,
    as_of: str | None,
    documents: Sequence[Document],
    report: RetrievalReport,
    answer_text: str | None,
    model_label: str,
    show_passages: bool,
) -> str:
    lines = [f"Question: {clean(question)}"]
    if as_of:
        lines.append(f"Text asked for: {as_of}")
    lines += [f"Model: {clean(model_label)}", ""]
    if show_passages and documents:
        lines += ["Passages given to the model:", "", clean(format_passages(documents), keep_newlines=True), ""]
    if answer_text is not None:
        where = "above" if show_passages else "given to it (not printed; see --show-passages)"
        lines += [f"Answer (written by the model from the passages {where}):", clean(answer_text, keep_newlines=True)]
    elif documents:
        lines.append("None of the passages has text, so no model was called.")
    else:
        lines.append("No DORA passages were found for this question, so no model was called.")
    lines.append("")
    if documents:
        lines += ["Sources:", *(source_line(n, doc) for n, doc in enumerate(documents, start=1)), ""]
    if as_of:
        lines.append(
            f"Withheld for {as_of}: {report.returned_withheld} passage(s) have no text for that date "
            "and arrived as a notice."
        )
        if report.returned_withheld:
            lines.append(NOTICE_MEANING)
    if report.filtered_out:
        lines.append(f"Left out: {report.filtered_out} hit(s) from acts other than DORA.")
    lines += [credits_line(report), "", CLOSING]
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
        raise ConfigError("LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard or use --dry-run.")
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
        raise ConfigError("ANTHROPIC_API_KEY is not set. Add it to .env or use --dry-run.")
    name = os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    return ChatAnthropic(model=name, max_tokens=2000), f"anthropic:{name}"


def one_line(text: str) -> str:
    """A provider's error text on one line, without control characters or any key from the environment."""
    for name in ("ANTHROPIC_API_KEY", "LEXAPI_API_KEY"):
        if secret := os.environ.get(name):
            text = text.replace(secret, "[key]")
    text = " ".join(clean(text).split())
    return text if len(text) <= 300 else text[:300] + "..."


def model_failure(error: AnswerError | ModelCallError) -> str:
    """One line for a failed model call or an unusable reply."""
    if isinstance(error, AnswerError):
        return one_line(f"The model's reply was not used: {error}")
    cause = error.__cause__
    if isinstance(cause, anthropic.APIError):
        return one_line(f"The model provider returned an error ({type(cause).__name__}): {cause}")
    return one_line(f"The model call failed ({type(cause).__name__}): {cause}")


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    chat_model: BaseChatModel | None = None,
    out: TextIO | None = None,
) -> int:
    """Exit codes: 0 printed; 1 LexAPI refused, or the model call failed; 2 usage or configuration error."""
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="dora-rag", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("question", type=question_text)
    parser.add_argument("--as-of", type=as_of_date, metavar="YYYY-MM-DD", help="ask for the text for this date")
    parser.add_argument("--dry-run", action="store_true", help="synthetic fixtures and a scripted reply, no keys")
    parser.add_argument("--show-passages", action="store_true", help="also print the passages given to the model")
    args = parser.parse_args(argv)

    load_env_file(Path(__file__).resolve().parents[2] / ".env")

    model_label = "scripted (no model was called)"
    try:
        if args.dry_run:
            # A dry run is never traced, whatever your LangChain tracing settings are.
            os.environ["LANGSMITH_TRACING"] = os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LEXAPI_BASE_URL"] = offline.DRY_RUN_BASE_URL
            os.environ["LEXAPI_API_KEY"] = "lex_dry_run_no_key"
            transport = offline.dry_run_transport()
            chat_model = offline.dry_run_model(args.as_of)
        if chat_model is None:
            chat_model, model_label = create_chat_model()
        client = create_client(transport)
    except (ConfigError, offline.FixturesNotFoundError) as error:
        print(error, file=sys.stderr)
        return 2

    answer_text = None
    with client:  # the retriever never closes a client passed in; this block closes it
        retriever = LexAPILegislationRetriever(client=client, k=K, celex=DORA_CELEX)
        try:
            documents, report = retrieve(retriever, args.question, args.as_of)
        except LexAPIError as error:
            status = f" (HTTP {error.status})" if error.status is not None else ""
            print(one_line(f"LexAPI error{status}: {error.code}: {error.message}"), file=sys.stderr)
            if isinstance(error, PaymentPastDueError) and error.billing_url:
                print(f"Billing: {one_line(error.billing_url)}", file=sys.stderr)
            return 1
        try:
            answer_text = answer(args.question, documents, chat_model, args.as_of)
        except NoTextError:
            pass  # rendered below: no passage has text, no model was called
        except (AnswerError, ModelCallError) as error:
            print(credits_line(report), file=sys.stderr)  # the search was charged
            print(model_failure(error), file=sys.stderr)
            return 1

    if args.dry_run:
        print(offline.DRY_RUN_BANNER, file=out)
    text = render(
        question=args.question,
        as_of=args.as_of,
        documents=documents,
        report=report,
        answer_text=answer_text,
        model_label=model_label,
        show_passages=args.show_passages,
    )
    print(text, file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
