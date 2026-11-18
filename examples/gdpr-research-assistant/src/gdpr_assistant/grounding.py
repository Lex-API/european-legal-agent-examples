"""Grounding: the rule that keeps model-written text tied to retrieved sources.

The model never writes the final answer directly. It returns a draft:

- ``quotes``     - excerpts it wants shown, each naming the source it came from;
- ``statements`` - its own sentences, each naming the sources that support it.

``validate_draft`` REJECTS the draft (raises ``GroundingError``) when:

1. a quote or statement cites a source id that was not retrieved in this run;
2. a quote is not a verbatim excerpt of the source it names, or is shorter
   than three words;
3. a quotation inside a statement (two words or more, or eight characters or
   more) is verbatim neither in a source that statement cites nor in the
   user's own question, or a quotation mark is left open;
4. a quote or statement breaks the rules for model text in ``text.py`` (line
   breaks, control characters, leading block markers, length; for statements
   also HTML character references and ``[S1]``-style markers);
5. a statement contains a URL that is not the URL of a source it cites;
6. there are more than 12 quotes or more than 12 statements.

It REMOVES, and counts without repeating them:

7. statements that cite no source;
8. statements that match the conclusion filter (``conclusions.py``).

"Verbatim" means: after normalisation (see ``text.normalise``), the words
appear in the source in the same order, starting and ending on a word
boundary. Punctuation and ellipses at the two ends of a quotation are ignored,
and the case of its first character may differ. A quote is printed with the
source's own characters, not the model's.

The agent retries a rejected draft once, then shows the sources only.

What this does not check: that an excerpt is quoted in context, or that a
model-written statement is a correct reading of the source it cites. That is
why the output keeps quoted text and model text in separate sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote as url_quote

from gdpr_assistant.conclusions import states_conclusion
from gdpr_assistant.text import (
    find_verbatim,
    model_text_problem,
    normalise,
    plain_text,
    quotations_in,
    trim_quotation,
    urls_in,
    word_count,
)

MAX_QUOTES = 12
MAX_STATEMENTS = 12
MIN_QUOTE_WORDS = 3


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
    #: The statement quotes words from the user's own question (not from a source).
    quotes_user_input: bool = False


@dataclass(frozen=True)
class Removed:
    """How many statements were removed, by reason. Their text is not kept."""

    uncited: int = 0
    conclusion: int = 0


@dataclass(frozen=True)
class GroundedAnswer:
    status: Literal["answered", "no_support"]
    quotes: tuple[Quote, ...] = ()  # text is the excerpt as it stands in the source
    statements: tuple[Statement, ...] = ()
    removed: Removed = Removed()


class GroundingError(Exception):
    def __init__(self, violations: list[str]) -> None:
        super().__init__("Draft rejected by the grounding check:\n- " + "\n- ".join(violations))
        self.violations = violations


@dataclass
class _Parsed:
    status: str
    quotes: list[Quote] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)


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


def _must_be_verbatim(quotation: str) -> bool:
    """Quotations long enough to carry meaning: two words or more, or eight characters or more."""
    return len(quotation) >= 8 or word_count(quotation) >= 2


def validate_draft(
    draft: Any,
    sources: list[Source],
    *,
    user_text: str = "",
    conclusion_patterns: tuple[str, ...] | None = None,
) -> GroundedAnswer:
    """Check a draft against the retrieved sources.

    ``user_text`` is the user's own input: a statement may quote it, and such a
    quotation is not presented as law. Statements matching one of
    ``conclusion_patterns`` are removed.
    """
    parsed = _parse(draft)
    if parsed.status == "no_support":
        return GroundedAnswer(status="no_support")

    # Normalise each source once; drafts can carry many quotes.
    texts = {s.id: normalise(s.text) for s in sources}
    urls = {s.id: s.url for s in sources}
    user = normalise(user_text)
    violations: list[str] = []
    if len(parsed.quotes) > MAX_QUOTES:
        violations.append(f"more than {MAX_QUOTES} quotes")
    if len(parsed.statements) > MAX_STATEMENTS:
        violations.append(f"more than {MAX_STATEMENTS} statements")

    quotes: list[Quote] = []
    for q in parsed.quotes[:MAX_QUOTES]:
        source_text = texts.get(q.source)
        if source_text is None:
            violations.append(f'quote cites "{q.source}", which was not retrieved')
            continue
        if problem := model_text_problem(q.text, "quote"):
            violations.append(f"quote from {q.source} {problem}")
            continue
        # A model often wraps the excerpt in quotation marks or ends it with "..."; neither is part of the excerpt.
        wanted = trim_quotation(normalise(q.text))
        if len(wanted) >= 2 and wanted.startswith('"') and wanted.endswith('"'):
            wanted = trim_quotation(wanted[1:-1])
        if word_count(wanted) < MIN_QUOTE_WORDS:
            violations.append(f"quote from {q.source} is shorter than {MIN_QUOTE_WORDS} words")
            continue
        excerpt = find_verbatim(source_text, wanted)
        if excerpt is None:
            violations.append(f"quote from {q.source} is not a verbatim excerpt of that source (whole words, in order)")
            continue
        quotes.append(Quote(q.source, excerpt))

    kept: list[Statement] = []
    uncited = conclusion = 0
    for st in parsed.statements[:MAX_STATEMENTS]:
        ids = tuple(dict.fromkeys(st.sources))  # de-duplicated, order kept
        unknown = [i for i in ids if i not in texts]
        if unknown:
            cited = ", ".join(f'"{i}"' for i in unknown)
            violations.append(f"statement cites {cited}, which was not retrieved")
            continue
        if not ids:
            uncited += 1
            continue
        if problem := model_text_problem(st.text, "statement"):
            violations.append(f"a statement {problem}")
            continue
        text = normalise(st.text)
        if conclusion_patterns and states_conclusion(text, conclusion_patterns):
            conclusion += 1
            continue
        if any(url not in [urls[i] for i in ids] for url in urls_in(text)):
            violations.append("a statement contains a URL that is not the URL of a source it cites")
            continue
        quotations = quotations_in(text)
        if quotations is None:
            violations.append("a statement has an unbalanced quotation mark")
            continue
        cited_texts = [texts[i] for i in ids]
        quotes_user_input = False
        invented: str | None = None
        for quotation in map(trim_quotation, quotations):
            if not _must_be_verbatim(quotation):
                continue
            if any(find_verbatim(t, quotation) is not None for t in cited_texts):
                continue
            if find_verbatim(user, quotation) is not None:
                quotes_user_input = True
            elif invented is None:
                invented = quotation
        if invented is not None:
            violations.append(
                "a statement contains a quotation that is not in the sources it cites "
                f'(whole words, in order): "{invented}"'
            )
            continue
        kept.append(Statement(text, ids, quotes_user_input))

    if violations:
        raise GroundingError(violations)
    status: Literal["answered", "no_support"] = "answered" if kept or quotes else "no_support"
    return GroundedAnswer(status, tuple(quotes), tuple(kept), Removed(uncited, conclusion))


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
            "description": "At most 12 verbatim excerpts, each copied character for character from one source: "
            "one line, whole words, 3 words to 1,200 characters.",
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
            "description": "At most 12 of your own sentences, one plain-text line of at most 600 characters each "
            "(no Markdown, no line breaks, no URLs, no [S1] markers in the text). "
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


def render_removed(removed: Removed) -> list[str]:
    """What the check removed: counts and reasons only - the removed text is never repeated."""
    lines = []
    if removed.uncited:
        lines.append(f"- {removed.uncited} statement(s) removed: cited no source.")
    if removed.conclusion:
        lines.append(f"- {removed.conclusion} statement(s) removed: matched the conclusion filter.")
    return lines


def render_answer(answer: GroundedAnswer, sources: list[Source]) -> str:
    """Quoted source text and model-written text go in separate sections; every line carries its source ids."""
    by_id = {s.id: s for s in sources}
    used: set[str] = set()
    lines = ["## Quoted from the sources", ""]
    if not answer.quotes:
        lines += ["_The model selected no excerpts._", ""]
    for q in answer.quotes:
        used.add(q.source)
        lines += [f'> "{plain_text(q.text)}" [{q.source}]', ""]

    lines += ["## Summary (written by the model - not legal text)", ""]
    if not answer.statements:
        lines += ["_None._", ""]
    for st in answer.statements:
        used.update(st.sources)
        note = " (the quoted words are from your question, not from a source)" if st.quotes_user_input else ""
        lines.append(f"- {plain_text(st.text)} " + "".join(f"[{i}]" for i in st.sources) + note)
    if answer.statements:
        lines.append("")

    removed = render_removed(answer.removed)
    if removed:
        lines += ["## Removed by the grounding check", "", *removed, ""]

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
