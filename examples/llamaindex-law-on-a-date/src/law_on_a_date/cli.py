"""EU law for a date, answered from cited passages, with LlamaIndex and LexAPI - command line.

law-on-a-date --as-of 2019-06-01 --provision art_17 "What did Article 17 of the GDPR say about erasure?"
law-on-a-date --compare 2018-05-25 2024-01-01      (which version is listed for each date)
law-on-a-date --dry-run ...                        (no keys, synthetic fixtures)
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
from lexapi import CreditsExhaustedError, LexAPI, LexAPIError, PaymentPastDueError
from lexapi.passages import Passage, format_citation
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.llms.anthropic import Anthropic
from llama_index.retrievers.lexapi import LexAPILegislationRetriever
from llama_index.tools.lexapi import LexAPIToolError, LexAPIToolSpec

from law_on_a_date import offline
from law_on_a_date.pipeline import (
    GDPR_CELEX,
    AnswerError,
    CallReport,
    K,
    ModelCallError,
    NoTextError,
    ToolReportError,
    VersionLookup,
    answer,
    combine,
    format_passages,
    has_text,
    look_up_version,
    other_versions,
    retrieve,
)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
USER_AGENT_SUFFIX = "european-legal-agent-examples"
LOCAL_PREFIX = "http://127.0.0.1:"
DASHBOARD_URL = "https://lex-api.com/dashboard?ref=github-examples"
FIRST_DATE = datetime.date(1952, 1, 1)  # the earliest date LexAPI accepts
ACT = "Act: CELEX"
DATE_ASKED = "Date asked"
CREDITS_USED = "LexAPI credits used for this run"
NOTICE_MEANING = (
    "A notice gives only LexAPI's reason for having no text for that date; "
    "check the source for whether the provision applied then."
)
CLOSING = (
    "Research support, not legal advice. The answer is model-written and may misread the passages; "
    "check it against the sources."
)
CLOSING_COMPARE = (
    "Not legal advice. LexAPI provides source texts and citations; it does not assess whether anyone complies "
    "with the law."
)
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_EXPORT = re.compile(r"^export\s+")
PLACEHOLDERS = frozenset({"lex_your_key_here"})  # the key values this repository's .env.example files shipped
_DASHBOARD_CODES = frozenset({"UNAUTHORIZED", "INVALID_API_KEY", "CREDITS_EXHAUSTED"})
KEY_HELP = {
    "LEXAPI_API_KEY": f"Create a key at {DASHBOARD_URL} or use --dry-run.",
    "ANTHROPIC_API_KEY": "Add it to .env or use --dry-run.",
}
_ACT_CELEX = re.compile(r"^3\d{4}[RLD]\d{4}$")


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


def today() -> datetime.date:
    """Today in UTC: LexAPI answers for dates up to today."""
    return datetime.datetime.now(datetime.UTC).date()


def date_arg(value: str) -> str:
    """argparse type for a date: a real calendar day written YYYY-MM-DD, from 1952-01-01 to today (UTC)."""
    if not _DATE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"expected a date written YYYY-MM-DD, got {value!r}")
    try:
        day = datetime.date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a calendar date; expected YYYY-MM-DD") from None
    if day > today():
        raise argparse.ArgumentTypeError(f"{value} is in the future; LexAPI answers for dates up to today (UTC)")
    if day < FIRST_DATE:
        raise argparse.ArgumentTypeError(f"{value} is before {FIRST_DATE}, the earliest date LexAPI accepts")
    return day.isoformat()


def celex_arg(value: str) -> str:
    """argparse type for --celex: the CELEX number of an EU regulation, directive or decision."""
    celex = value.strip().upper()
    if not _ACT_CELEX.match(celex):
        raise argparse.ArgumentTypeError(
            f"expected the CELEX number of an EU regulation, directive or decision, such as 32016R0679, got {value!r}"
        )
    return celex


def not_blank(value: str) -> str:
    """argparse type for a text argument: not empty or blank."""
    if not value.strip():
        raise argparse.ArgumentTypeError("the value is empty")
    return value.strip()


def shown(value: int | None) -> str:
    return "not reported" if value is None else str(value)


def credits_line(calls: Sequence[tuple[str, CallReport]]) -> str:
    """The credits of each call of this run, their total, and the credits left after the last one."""
    charged = [report.units_charged for _label, report in calls]
    total = "not reported" if any(value is None for value in charged) else str(sum(v or 0 for v in charged))
    parts = ", ".join(f"{label}: {shown(report.units_charged)}" for label, report in calls)
    remaining = shown(calls[-1][1].credits_remaining)
    return f"{CREDITS_USED}: {total} ({parts}). Credits remaining: {remaining}."


def version_lines(passage: Passage, as_of: str) -> list[str]:
    """The version LexAPI lists for the date, in the API's terms, and its Source line."""
    version, kind = passage.version_celex, passage.version_kind
    if passage.as_of_resolved and version:
        if passage.consolidated or kind == "consolidated":
            cite = passage.cite_celex
            act = f"the act to cite is CELEX {cite}" if cite else "LexAPI did not name the act it consolidates"
            line = f"Version listed for {as_of}: {version}, a consolidated text (no legal force); {act}."
        elif kind == "as_published":
            line = f"Version listed for {as_of}: {version}, the act as published in the Official Journal."
        else:
            line = f"Version listed for {as_of}: {version}."
    elif version:
        what = f"the text of {passage.article_ref}" if passage.article_ref else "the text"
        line = (
            f"Version listed for {as_of}: {version}; {what} for that date is not established "
            f"(reason: {passage.as_of_reason or 'not given'})."
        )
    else:
        line = f"Version for {as_of}: not established by LexAPI (reason: {passage.as_of_reason or 'not given'})."
    lines = [clean(line)]
    if passage.as_of_resolved and passage.consolidated_note:
        lines.append(f"  LexAPI's note: {clean(passage.consolidated_note)}")
    lines.append(f"  Source: {clean(format_citation(passage))}")
    return lines


def answer_heading(*, show_passages: bool, scripted: bool) -> str:
    if scripted:
        where = "from the passages above" if show_passages else "passages not printed, see --show-passages"
        return f"Answer (scripted for this dry run; {where}):"
    if show_passages:
        return "Answer (written by the model from the passages above):"
    return "Answer (written by the model from the passages given to it; see --show-passages):"


def source_line(n: int, node: NodeWithScore) -> str:
    metadata = node.node.metadata
    line = f"[{n}] {clean(str(metadata['citation']))}"
    if metadata.get("text_withheld"):
        reason = metadata.get("as_of_reason")
        why = f" (reason: {clean(str(reason))})" if reason else ""
        return f"{line} - notice only, no text was given to the model{why}"
    if not has_text(node):
        return f"{line} - no text returned"
    return line


def render(
    *,
    question: str,
    celex: str,
    as_of: str,
    provision: str | None,
    lookup: VersionLookup,
    nodes: Sequence[NodeWithScore],
    calls: Sequence[tuple[str, CallReport]],
    filtered_out: int,
    answer_text: str | None,
    model_label: str,
    show_passages: bool,
    scripted: bool = False,
) -> str:
    lines = [f"Question: {clean(question)}", f"{ACT} {celex}", f"{DATE_ASKED}: {as_of}"]
    if provision:
        lines.append(f"Provision: {clean(provision)}")
    lines += [f"Model: {clean(model_label)}", "", *version_lines(lookup.passage, as_of)]
    for version in other_versions(lookup.node(), nodes):
        lines.append(
            f"The search listed a different version for this provision ({clean(version)}); both passages are shown."
        )
    lines.append("")
    if show_passages and nodes:
        lines += ["Passages given to the model:", "", clean(format_passages(nodes), keep_newlines=True), ""]
    if answer_text is not None:
        lines += [
            answer_heading(show_passages=show_passages, scripted=scripted),
            clean(answer_text, keep_newlines=True),
        ]
    elif nodes:
        lines.append("None of the passages has text for that date, so no model was called.")
    else:
        lines.append(f"No passages of CELEX {celex} were found for this question, so no model was called.")
    lines.append("")
    if nodes:
        lines += ["Sources:", *(source_line(n, node) for n, node in enumerate(nodes, start=1)), ""]
    withheld = sum(bool(node.node.metadata.get("text_withheld")) for node in nodes)
    lines.append(f"Withheld for {as_of}: {withheld} passage(s) have no text for that date and arrived as a notice.")
    if withheld:
        lines.append(NOTICE_MEANING)
    if filtered_out:
        # The adapter keeps a hit when its CELEX number or the act it consolidates is the one asked for;
        # a hit from another act, and a consolidated text that does not name its act, are left out.
        lines.append(f"Left out: {filtered_out} hit(s) not attributed to CELEX {celex}.")
    lines += [credits_line(calls), "", CLOSING]
    return "\n".join(lines)


def render_compare(celex: str, lookups: Sequence[VersionLookup], calls: Sequence[tuple[str, CallReport]]) -> str:
    first, second = lookups
    lines = [f"{ACT} {celex}", f"Dates compared: {first.as_of} and {second.as_of}", ""]
    for lookup in lookups:
        lines += version_lines(lookup.passage, lookup.as_of)
    lines.append("")
    a, b = clean(first.passage.version_celex or ""), clean(second.passage.version_celex or "")
    unknown = [lookup.as_of for lookup in lookups if not (lookup.resolved and lookup.passage.version_celex)]
    if unknown:
        lines.append(f"Not compared: LexAPI did not establish the version for {' and '.join(unknown)}.")
    elif a == b:
        lines.append(
            f"Same version: {a} is listed for both dates. "
            "This says which version LexAPI lists, not that nothing changed between the dates."
        )
    else:
        lines.append(
            f"Different versions: {a} is listed for {first.as_of} and {b} for {second.as_of}. "
            "This compares the versions LexAPI lists for the dates, not their texts."
        )
    lines += [credits_line(calls), "", CLOSING_COMPARE]
    return "\n".join(lines)


# ── setup ───────────────────────────────────────────────────────────────


def load_env_file(path: Path) -> None:
    """Load ``KEY=value`` lines (also ``export KEY=value``) into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = _EXPORT.sub("", line.strip(), count=1).partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env_key(name: str) -> str | None:
    """The key in ``name``, trimmed; None when unset, blank, a placeholder this repository shipped, or ``<...>``."""
    value = os.environ.get(name, "").strip()
    if not value or value.lower() in PLACEHOLDERS or (value.startswith("<") and value.endswith(">")):
        return None
    return value


def require_keys(names: Sequence[str]) -> None:
    """Raise ConfigError naming every key in ``names`` that is not set, one line per key."""
    missing = [name for name in names if env_key(name) is None]
    if missing:
        raise ConfigError("\n".join(f"{name} is not set. {KEY_HELP[name]}" for name in missing))


def create_client(transport: httpx.BaseTransport | None = None) -> LexAPI:
    """The LexAPI client, from LEXAPI_API_KEY (never hard-code it) and the optional LEXAPI_BASE_URL.

    ``transport`` replaces the network (tests and --dry-run); the base URL must then be a 127.0.0.1 one.
    """
    require_keys(["LEXAPI_API_KEY"])
    api_key = env_key("LEXAPI_API_KEY") or ""  # set: checked just above
    base_url = os.environ.get("LEXAPI_BASE_URL")
    if transport is not None and not (base_url or "").startswith(LOCAL_PREFIX):
        raise RuntimeError(
            f"Refusing to run: with a replaced transport, LEXAPI_BASE_URL must start with {LOCAL_PREFIX}."
        )
    options: dict[str, Any] = {"user_agent_suffix": USER_AGENT_SUFFIX, "transport": transport}
    if base_url:
        options["base_url"] = base_url
    return LexAPI(api_key, **options)


def create_llm() -> tuple[LLM, str]:
    """Anthropic through llama-index-llms-anthropic: ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default claude-sonnet-5-5)."""
    require_keys(["ANTHROPIC_API_KEY"])
    name = os.environ.get("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
    return Anthropic(model=name, max_tokens=2000, api_key=env_key("ANTHROPIC_API_KEY")), f"anthropic:{name}"


def one_line(text: str, *, keep: str = "") -> str:
    """Error text on one line, without control characters or any key from the environment.

    Text over 300 characters is cut; ``keep`` (a link) follows it in full.
    """

    def tidy(part: str) -> str:
        for name in ("ANTHROPIC_API_KEY", "LEXAPI_API_KEY"):
            if secret := os.environ.get(name, "").strip():
                part = part.replace(secret, "[key]")
        return " ".join(clean(part).split())

    message, link = tidy(text), tidy(keep)
    if len(message) > 300:
        message = message[:300] + "..."
    return f"{message} {link}" if link else message


def error_fields(error: LexAPIError) -> dict[str, Any]:
    """What the adapters put after the API's message, from the error's own fields."""
    return {
        "resets_at": error.resets_at if isinstance(error, CreditsExhaustedError) else None,
        "billing_url": error.billing_url if isinstance(error, PaymentPastDueError) else None,
        "retry_after": error.retry_after,
    }


def refusal_line(
    text: str,
    code: str,
    *,
    resets_at: str | None = None,
    billing_url: str | None = None,
    retry_after: float | None = None,
) -> str:
    """The adapter's error line on one line; after a key or credits error without a billing link, the dashboard.

    The adapters end the line with what the error's fields say (credits reset time, billing link, retry
    delay), in that order. A long message is cut; that end and the dashboard link never are.
    """
    tail = [
        f"Credits reset at {resets_at}." if resets_at else "",
        f"Billing: {billing_url}" if billing_url else "",
        f"Retry after {retry_after:g} s." if retry_after is not None else "",
    ]
    end = "".join(f" {part}" for part in tail if part)
    message = text[: len(text) - len(end)] if end and text.endswith(end) else text
    kept = end if message is not text else ""
    if code in _DASHBOARD_CODES and not billing_url:
        kept += f" Dashboard: {DASHBOARD_URL}"
    return one_line(message, keep=kept)


def refusal(error: LexAPIToolError | LexAPIError) -> str:
    """A LexAPI refusal or unreadable answer, in the tools' one-line form for both adapters."""
    if isinstance(error, LexAPIToolError):  # the tools adapter keeps the code and the retry delay
        return refusal_line(str(error), error.code, retry_after=error.retry_after)
    tool_error = LexAPIToolError.from_sdk(error)
    return refusal_line(str(tool_error), tool_error.code, **error_fields(error))


def model_failure(error: AnswerError | ModelCallError) -> str:
    """One line for a failed model call or an unusable reply."""
    if isinstance(error, AnswerError):
        return one_line(f"The model's reply was not used: {error}")
    cause = error.__cause__
    if isinstance(cause, anthropic.APIError):
        return one_line(f"The model provider returned an error ({type(cause).__name__}): {cause}")
    return one_line(f"The model call failed ({type(cause).__name__}): {cause}")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="law-on-a-date", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("question", nargs="?", type=not_blank, help="the question (not with --compare)")
    when = parser.add_mutually_exclusive_group(required=True)
    when.add_argument("--as-of", type=date_arg, metavar="YYYY-MM-DD", help="the date to ask for")
    when.add_argument(
        "--compare", type=date_arg, nargs=2, metavar="YYYY-MM-DD", help="list the version for each of two dates"
    )
    parser.add_argument("--celex", type=celex_arg, default=GDPR_CELEX, help=f"the act (default {GDPR_CELEX})")
    parser.add_argument("--provision", type=not_blank, help="an article (art_17), annex (anx_III) or recitals")
    parser.add_argument("--dry-run", action="store_true", help="synthetic fixtures and a scripted reply, no keys")
    parser.add_argument("--show-passages", action="store_true", help="also print the passages given to the model")
    args = parser.parse_args(argv)
    if args.as_of and args.question is None:
        parser.error("a question is needed with --as-of")
    if args.compare:
        if args.question is not None or args.provision or args.show_passages:
            parser.error("--compare lists versions only: it takes no question, --provision or --show-passages")
        if args.compare[0] == args.compare[1]:
            parser.error("--compare needs two different dates")
        args.compare = sorted(args.compare)  # earlier date first
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    llm: LLM | None = None,
    out: TextIO | None = None,
) -> int:
    """Exit codes: 0 printed; 1 LexAPI refused, or the model call failed; 2 usage or configuration error."""
    out = out or sys.stdout
    args = parse_args(argv)
    load_env_file(Path(__file__).resolve().parents[2] / ".env")

    model_label = "scripted (no model was called)"
    try:
        if args.dry_run:
            if args.celex != GDPR_CELEX:
                raise ConfigError(f"The dry run has fixtures for CELEX {GDPR_CELEX} only.")
            os.environ["LEXAPI_BASE_URL"] = offline.DRY_RUN_BASE_URL
            os.environ["LEXAPI_API_KEY"] = "lex_dry_run_no_key"
            transport = offline.LexAPIMock().transport()
            llm = offline.dry_run_model(bool(args.provision))
        model_needed = llm is None and not args.compare
        require_keys(["LEXAPI_API_KEY", *(["ANTHROPIC_API_KEY"] if model_needed else [])])
        if model_needed:
            llm, model_label = create_llm()
        client = create_client(transport)
    except (ConfigError, offline.FixturesNotFoundError) as error:
        print(error, file=sys.stderr)
        return 2

    calls: list[tuple[str, CallReport]] = []
    with client:  # the adapters never close a client passed in; this block closes it
        spec = LexAPIToolSpec(client=client)
        try:
            if args.compare:
                lookups = []
                for day in args.compare:
                    lookups.append(look_up_version(spec, args.celex, day))
                    calls.append((f"lookup for {day}", lookups[-1].report))
            else:
                lookup = look_up_version(spec, args.celex, args.as_of, args.provision)
                calls.append(("version lookup", lookup.report))
                retriever = LexAPILegislationRetriever(
                    client=client, similarity_top_k=K, as_of=args.as_of, celex=args.celex
                )
                found, report = retrieve(retriever, args.question)
                calls.append(("search", report))
        except (LexAPIToolError, LexAPIError, ToolReportError) as error:
            if calls:
                print(credits_line(calls), file=sys.stderr)  # the calls before it were charged
            print(one_line(str(error)) if isinstance(error, ToolReportError) else refusal(error), file=sys.stderr)
            return 1

        if args.compare:
            if args.dry_run:
                print(offline.DRY_RUN_BANNER, file=out)
            print(render_compare(args.celex, lookups, calls), file=out)
            return 0

        nodes = combine(lookup.node(), found)
        answer_text = None
        assert llm is not None  # set above for every run but --compare
        try:
            answer_text = answer(args.question, nodes, llm, args.as_of)
        except NoTextError:
            pass  # rendered below: no passage has text, no model was called
        except (AnswerError, ModelCallError) as error:
            print(credits_line(calls), file=sys.stderr)  # the LexAPI calls were charged
            print(model_failure(error), file=sys.stderr)
            return 1

    if args.dry_run:
        print(offline.DRY_RUN_BANNER, file=out)
    text = render(
        question=args.question,
        celex=args.celex,
        as_of=args.as_of,
        provision=args.provision,
        lookup=lookup,
        nodes=nodes,
        calls=calls,
        filtered_out=report.filtered_out,
        answer_text=answer_text,
        model_label=model_label,
        show_passages=args.show_passages,
        scripted=args.dry_run,
    )
    print(text, file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
