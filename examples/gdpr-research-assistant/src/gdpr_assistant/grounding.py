"""Grounding: the rule that keeps model-written text tied to retrieved sources.

The model never writes the final answer directly. It returns a draft:

- ``quotes``     - excerpts it wants shown, each naming the source it came from;
- ``statements`` - its own sentences, each naming the sources that support it.

``validate_draft`` then enforces, in code:

1. every source id in the draft was actually retrieved in this run   -> otherwise REJECT;
2. every quote is a verbatim excerpt of the source it names           -> otherwise REJECT;
3. a quotation embedded in a statement is verbatim in a cited source  -> otherwise REJECT;
4. a statement that cites no source is removed and reported.

"Reject" raises ``GroundingError``; the agent retries once, then falls back to
showing the sources only.

This checks that citations are real and quotes are verbatim. It cannot check
that a model-written statement is a correct reading of the source - that is why
the rendered output keeps quoted text and model text visibly apart.
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


def normalise(text: str) -> str:
    """Collapse whitespace and unify quote characters so line wrapping does not break a verbatim match."""
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


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


def validate_draft(draft: Any, sources: list[Source], *, min_embedded_quote: int = 20) -> GroundedAnswer:
    parsed = _parse(draft)
    if parsed.status == "no_support":
        return GroundedAnswer(status="no_support")

    by_id = {s.id: s for s in sources}
    violations: list[str] = []

    for q in parsed.quotes:
        source = by_id.get(q.source)
        if source is None:
            violations.append(f'quote cites "{q.source}", which was not retrieved')
        elif not normalise(q.text):
            violations.append(f"quote from {q.source} is empty")
        elif normalise(q.text) not in normalise(source.text):
            violations.append(f"quote from {q.source} is not a verbatim excerpt of that source")

    kept: list[Statement] = []
    removed: list[str] = []
    for st in parsed.statements:
        unknown = [i for i in st.sources if i not in by_id]
        if unknown:
            cited = ", ".join(f'"{i}"' for i in unknown)
            violations.append(f"statement cites {cited}, which was not retrieved")
            continue
        if not st.sources:
            removed.append(st.text)
            continue
        cited_texts = [normalise(by_id[i].text) for i in st.sources]
        embedded = [m.strip() for m in re.findall(r'"([^"]+)"', normalise(st.text))]
        invented = [e for e in embedded if len(e) >= min_embedded_quote and not any(e in t for t in cited_texts)]
        if invented:
            violations.append(f'statement contains a quotation that is not in the sources it cites: "{invented[0]}"')
            continue
        kept.append(st)

    if violations:
        raise GroundingError(violations)
    status: Literal["answered", "no_support"] = "answered" if kept or parsed.quotes else "no_support"
    return GroundedAnswer(status, tuple(parsed.quotes), tuple(kept), tuple(removed))


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
            "description": "Verbatim excerpts copied character for character from one source.",
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
            "description": "Your own sentences. Each one lists the source ids that support it.",
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
        lines += [f'> "{normalise(q.text)}" [{q.source}]', ""]

    lines += ["## Summary (written by the model - not legal text)", ""]
    if not answer.statements:
        lines += ["_None._", ""]
    for st in answer.statements:
        used.update(st.sources)
        lines.append(f"- {st.text} " + "".join(f"[{i}]" for i in st.sources))
    if answer.statements:
        lines.append("")

    if answer.removed:
        lines += ["## Removed by the grounding check", ""]
        lines += [f"- (cited no source) {text}" for text in answer.removed]
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
