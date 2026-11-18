"""Grounding: the rule that keeps model-written text tied to retrieved sources.

The model never writes the final answer directly. It returns a draft:

- ``quotes``     - excerpts it wants shown, each naming the source it came from;
- ``statements`` - its own sentences, each naming the sources that support it.

``validate_draft`` then enforces, in code:

1. every source id in the draft was actually retrieved in this run        -> otherwise REJECT;
2. every quote is at least three words long and a verbatim excerpt of the
   source it names                                                        -> otherwise REJECT;
3. every quotation inside a statement (two words or more, or eight
   characters or more, in any style of quotation marks) is verbatim in a
   source that statement cites; quotation marks must be balanced          -> otherwise REJECT;
4. quotes and statements are single lines of plain text: no line breaks,
   no leading Markdown block marker (heading, blockquote, list, table,
   code fence)                                                            -> otherwise REJECT;
5. a statement that cites no source is removed and reported.

"Verbatim" is checked after collapsing whitespace, unifying quotation-mark
styles and dropping soft hyphens, on both sides.

"Reject" raises ``GroundingError``; the agent retries once, then falls back to
showing the sources only.

This checks that citations are real and excerpts are verbatim. It cannot check
that an excerpt is quoted in context, or that a model-written statement is a
correct reading of the source - that is why the rendered output keeps quoted
text and model text visibly apart, and escapes Markdown in everything the
model wrote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote as url_quote


@dataclass(frozen=True)
class Source:
    """One retrieved piece of text the model may quote or cite."""

    id: str  # run-local: S1, S2, ... - the only handle the model is given
    kind: Literal["legislation", "case-law"]
    celex: str
    title: str
    url: str
    text: str  # exactly as the API returned it
    article_ref: str | None = None
    paragraph: int | None = None
    ecli: str | None = None
    notes: tuple[str, ...] = ()  # caveats the output must carry


@dataclass(frozen=True)
class Quote:
    source: str
    text: str


@dataclass(frozen=True)
class Statement:
    text: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class GroundedAnswer:
    status: Literal["answered", "no_support"]
    quotes: tuple[Quote, ...] = ()
    statements: tuple[Statement, ...] = ()
    removed: tuple[str, ...] = ()  # statements dropped because they cited no source


class GroundingError(Exception):
    def __init__(self, violations: list[str]) -> None:
        super().__init__("Draft rejected by the grounding check:\n- " + "\n- ".join(violations))
        self.violations = violations


@dataclass
class _Parsed:
    status: str
    quotes: list[Quote] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)


_SINGLE_QUOTES = dict.fromkeys(map(ord, "‘’‚‛‹›"), "'")
_DOUBLE_QUOTES = dict.fromkeys(map(ord, "“”„‟«»"), '"')
_TYPOGRAPHY = {**_SINGLE_QUOTES, **_DOUBLE_QUOTES, 0x00AD: None}
_LINE_BREAK = re.compile("[\n\r  ]")
_STATEMENT_MARKER = re.compile(r"^(#|>|\||```|~~~|[-*+]\s|\d+[.)]\s)")
# A quote may open with a paragraph number ("1. The controller ..."): laws are written that way.
_QUOTE_MARKER = re.compile(r"^(#|>|\||```|~~~|[-*+]\s)")
_MARKDOWN_CONTROL = re.compile(r"[\\`*_#>|\[\]<~]")
MIN_QUOTE_WORDS = 3


def normalise(text: str) -> str:
    """Collapse whitespace, unify quotation-mark styles and drop soft hyphens.

    Line wrapping and typography then do not break a verbatim match.
    """
    return re.sub(r"\s+", " ", text.translate(_TYPOGRAPHY)).strip()


def _shape_problem(text: str, markers: re.Pattern[str]) -> str | None:
    """Why a piece of model text may not be rendered as one plain line, or ``None`` if it may."""
    if _LINE_BREAK.search(text):
        return "contains a line break"
    if markers.match(text.lstrip()):
        return "starts with a Markdown block marker"
    return None


def embedded_quotations(statement: str) -> list[str] | None:
    """The quotations inside a statement, in any style of quotation marks.

    The text is normalised first, so only ``"`` and ``'`` remain. A ``'``
    between two letters, or closing nothing, is an apostrophe ("provider's",
    "subjects'"). Returns ``None`` when a quotation mark is left open.
    """
    text = normalise(statement)
    spans: list[str] = []
    opened: tuple[str, int] | None = None
    for i, ch in enumerate(text):
        if ch not in "\"'":
            continue
        before = text[i - 1] if i > 0 else " "
        after = text[i + 1] if i + 1 < len(text) else " "
        if opened:
            if ch != opened[0]:
                continue
            if ch == "'" and before.isalnum() and after.isalnum():
                continue  # apostrophe inside a word
            spans.append(text[opened[1] + 1 : i].strip())
            opened = None
        elif ch == '"' or (not before.isalnum() and after != " "):
            opened = (ch, i)  # a single quote opens only at the start of a word
    return None if opened else spans


def _must_be_verbatim(quotation: str) -> bool:
    """Quotations long enough to carry meaning: two words or more, or eight characters or more."""
    return len(quotation) >= 8 or len(quotation.split()) >= 2


def escape_markdown(text: str) -> str:
    """Escape Markdown control characters in text the model wrote.

    It then renders as literal characters: it cannot open a heading, a
    blockquote, a link or a citation marker of its own.
    """
    return _MARKDOWN_CONTROL.sub(lambda m: "\\" + m.group(0), normalise(text))


def _parse(draft: Any) -> _Parsed:
    """Narrow the model's reply; the schema is not trusted blindly."""
    wrong_shape = GroundingError(["the model's reply does not have the expected shape"])
    if not isinstance(draft, dict) or draft.get("status") not in ("answered", "no_support"):
        raise wrong_shape
    quotes, statements = draft.get("quotes"), draft.get("statements")
    if not isinstance(quotes, list) or not isinstance(statements, list):
        raise wrong_shape
    parsed = _Parsed(status=draft["status"])
    for q in quotes:
        if not isinstance(q, dict) or not isinstance(q.get("source"), str) or not isinstance(q.get("text"), str):
            raise wrong_shape
        parsed.quotes.append(Quote(q["source"], q["text"]))
    for s in statements:
        ids = s.get("sources") if isinstance(s, dict) else None
        if not isinstance(s, dict) or not isinstance(s.get("text"), str) or not isinstance(ids, list):
            raise wrong_shape
        if not all(isinstance(i, str) for i in ids):
            raise wrong_shape
        parsed.statements.append(Statement(s["text"], tuple(ids)))
    return parsed


def validate_draft(draft: Any, sources: list[Source]) -> GroundedAnswer:
    parsed = _parse(draft)
    if parsed.status == "no_support":
        return GroundedAnswer(status="no_support")

    # Normalise each source once; drafts can carry many quotes.
    by_id = {s.id: normalise(s.text) for s in sources}
    violations: list[str] = []

    quotes: list[Quote] = []
    for q in parsed.quotes:
        source_text = by_id.get(q.source)
        wanted = normalise(q.text)
        if source_text is None:
            violations.append(f'quote cites "{q.source}", which was not retrieved')
        elif shape := _shape_problem(q.text.strip(), _QUOTE_MARKER):
            violations.append(f"quote from {q.source} {shape}; a quote is one line of plain text")
        elif len(wanted.split()) < MIN_QUOTE_WORDS:
            violations.append(f"quote from {q.source} is shorter than {MIN_QUOTE_WORDS} words")
        elif wanted not in source_text:
            violations.append(f"quote from {q.source} is not a verbatim excerpt of that source")
        else:
            quotes.append(Quote(q.source, wanted))

    kept: list[Statement] = []
    removed: list[str] = []
    for st in parsed.statements:
        ids = tuple(dict.fromkeys(st.sources))  # de-duplicated, order kept
        unknown = [i for i in ids if i not in by_id]
        if unknown:
            cited = ", ".join(f'"{i}"' for i in unknown)
            violations.append(f"statement cites {cited}, which was not retrieved")
            continue
        if shape := _shape_problem(st.text.strip(), _STATEMENT_MARKER):
            violations.append(f"a statement {shape}; each statement is one line of plain text")
            continue
        text = normalise(st.text)
        if not ids:
            removed.append(text)
            continue
        quotations = embedded_quotations(text)
        if quotations is None:
            violations.append("a statement has an unbalanced quotation mark")
            continue
        cited_texts = [by_id[i] for i in ids]
        invented = [q for q in quotations if _must_be_verbatim(q) and not any(q in t for t in cited_texts)]
        if invented:
            violations.append(f'statement contains a quotation that is not in the sources it cites: "{invented[0]}"')
            continue
        kept.append(Statement(text, ids))

    if violations:
        raise GroundingError(violations)
    status: Literal["answered", "no_support"] = "answered" if kept or quotes else "no_support"
    return GroundedAnswer(status, tuple(quotes), tuple(kept), tuple(removed))


#: JSON Schema of the draft, used as the model's tool input schema.
DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["answered", "no_support"],
            "description": "no_support when the sources do not contain text that answers the question.",
        },
        "quotes": {
            "type": "array",
            "description": "Verbatim excerpts copied character for character from one source: "
            "one line each, at least three words.",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source id, e.g. S1."},
                    "text": {"type": "string", "description": "The excerpt, copied exactly."},
                },
                "required": ["source", "text"],
                "additionalProperties": False,
            },
        },
        "statements": {
            "type": "array",
            "description": "Your own sentences, one plain-text line each (no Markdown, no line breaks). "
            "Each lists the source ids that support it.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "sources"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "quotes", "statements"],
    "additionalProperties": False,
}


# -- Rendering ---------------------------------------------------------------


def eur_lex_url(celex: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{url_quote(celex, safe='')}"


def article_label(article_ref: str) -> str:
    """``art_46`` -> ``Article 46``; anything else is shown as returned."""
    match = re.fullmatch(r"art_(\w+)", article_ref)
    return f"Article {match.group(1)}" if match else article_ref


def citation(source: Source) -> str:
    """The citation line for one source: title, pinpoint, ECLI, CELEX, link."""
    parts = [source.title]
    if source.article_ref:
        parts.append(f"{article_label(source.article_ref)} (article id {source.article_ref})")
    if source.paragraph is not None:
        parts.append(f"paragraph {source.paragraph}")
    if source.ecli:
        parts.append(source.ecli)
    parts.append(f"CELEX {source.celex}")
    parts.append(source.url)
    return " - ".join(parts)


def render_answer(answer: GroundedAnswer, sources: list[Source]) -> str:
    """Quoted source text and model-written text go in separate sections; every line carries its source ids."""
    by_id = {s.id: s for s in sources}
    used: set[str] = set()
    lines = ["## Quoted from the sources", ""]
    if not answer.quotes:
        lines += ["_The model selected no excerpts._", ""]
    for q in answer.quotes:
        used.add(q.source)
        lines += [f'> "{escape_markdown(q.text)}" [{q.source}]', ""]

    lines += ["## Summary (written by the model - not legal text)", ""]
    if not answer.statements:
        lines += ["_None._", ""]
    for st in answer.statements:
        used.update(st.sources)
        lines.append(f"- {escape_markdown(st.text)} " + "".join(f"[{i}]" for i in st.sources))
    if answer.statements:
        lines.append("")

    if answer.removed:
        lines += ["## Removed by the grounding check", ""]
        lines += [f"- (cited no source) {escape_markdown(text)}" for text in answer.removed]
        lines.append("")

    lines += ["## Sources", ""]
    for source_id in sorted(used, key=lambda i: int(i[1:])):
        source = by_id[source_id]
        lines.append(f"- [{source_id}] {citation(source)}")
        lines += [f"  - Note: {note}" for note in source.notes]
    return "\n".join(lines) + "\n"


def sources_for_prompt(sources: list[Source]) -> str:
    """The sources as shown to the model: id, citation line, caveats, then the text."""
    blocks = []
    for s in sources:
        notes = [f"Note: {n}" for n in s.notes]
        blocks.append("\n".join([f'<source id="{s.id}">', citation(s), *notes, "", s.text, "</source>"]))
    return "\n\n".join(blocks)
