"""Retrieval: GDPR articles and CJEU judgment paragraphs, as ``Source`` objects.

Per question (see the README for the credit arithmetic):

1. ``semantic_legislation_search`` with ``include_text=True`` -> full text of the
   matched articles; only GDPR articles are kept.
2. ``semantic_search`` -> the most similar CJEU cases. A hit's ``summary`` and
   ``snippet`` are machine-written digests, NOT the Court's words, so they are
   never used as quotable sources.
3. ``get_document`` for the top cases -> the judgment's own numbered paragraphs.
   A few paragraphs are selected per case and become the quotable sources.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from lexapi import APIResponse, LexAPI
from lexapi.models import Document, SemanticCaseLawResult, SemanticLegislationResult

from gdpr_assistant.grounding import Source, eur_lex_url

#: Regulation (EU) 2016/679 - the General Data Protection Regulation.
GDPR_CELEX = "32016R0679"

MAX_ARTICLES = 6
MAX_CASES = 2
PARAGRAPHS_PER_CASE = 4
#: Longest article text handed to the model; longer ones are cut and labelled.
MAX_ARTICLE_CHARS = 8000

_CONSOLIDATED = re.compile(r"^0(\d{4}[A-Z]\d{4})-\d{8}$")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it may must of on or shall that the "
    "their this to under what when where which who with".split()
)


@dataclass
class Retrieved:
    sources: list[Source] = field(default_factory=list)
    #: Things the reader should know (cases skipped, hits from other acts, ...).
    notices: list[str] = field(default_factory=list)
    credits_reported: int = 0
    #: Calls whose response carried no credits figure (never counted as 0).
    credits_unreported: int = 0
    #: Calls counted so far, with or without a credits figure.
    calls: int = 0

    def add_credits(self, charged: int | None) -> None:
        self.calls += 1
        if charged is None:
            self.credits_unreported += 1
        else:
            self.credits_reported += charged

    def describe_credits(self) -> str:
        if self.credits_unreported and not self.credits_reported:
            return "not reported"
        if self.credits_unreported:
            return f"{self.credits_reported} ({self.credits_unreported} call(s) did not report credits)"
        return str(self.credits_reported)


def reported_credits(response: APIResponse) -> int | None:
    """Credits a response reports for its own call; ``None`` when it reports none (not 0)."""
    return response.credits.operation_weight if response.credits else None


def base_act(hit: SemanticLegislationResult) -> str | None:
    """The act a hit belongs to. A consolidated hit names it in ``base_celex``."""
    if hit.base_celex:
        return hit.base_celex
    match = _CONSOLIDATED.match(hit.law_id or "")
    return f"3{match.group(1)}" if match else hit.law_id


def version_notes(hit: SemanticLegislationResult, as_of: str | None) -> list[str]:
    """What may be said about which version of the law this text is.

    "In force on <date>" is claimed only when the API says ``asOfResolved: true``.
    """
    notes: list[str] = []
    if as_of:
        if hit.as_of_resolved is True:
            version = hit.as_of_version
            label = "the act as published in the Official Journal"
            if version and version.kind == "consolidated":
                label = f"the consolidated version of {version.date or 'unknown date'} (CELEX {version.celex})"
            notes.append(f"Text in force on {as_of}, according to LexAPI: {label}.")
        else:
            reason = hit.as_of_reason or "no reason given"
            notes.append(
                f"The version in force on {as_of} is not established (LexAPI: {reason}). "
                "The text shown is the latest version held, not necessarily the text on that date."
            )
    consolidated = hit.consolidated or bool(hit.as_of_version and hit.as_of_version.kind == "consolidated")
    if consolidated:
        base = base_act(hit) or GDPR_CELEX
        note = hit.consolidated_note or "Consolidated text; not the authentic text."
        notes.append(f"{note} Authentic act: CELEX {base} as published in the Official Journal ({eur_lex_url(base)}).")
    return notes


def legislation_sources(
    hits: list[SemanticLegislationResult], as_of: str | None, first_id: int = 1
) -> tuple[list[Source], int]:
    """GDPR hits with text and an article id, one per article, best score first. Returns (sources, other-act hits)."""
    best: dict[str, SemanticLegislationResult] = {}
    other_acts = 0
    for hit in hits:
        if base_act(hit) != GDPR_CELEX:
            other_acts += 1
            continue
        if not hit.article_ref or not hit.text:
            continue
        seen = best.get(hit.article_ref)
        if seen is None or (hit.score or 0) > (seen.score or 0):
            best[hit.article_ref] = hit

    ranked = sorted(best.values(), key=lambda h: h.score or 0, reverse=True)[:MAX_ARTICLES]
    sources = []
    for offset, hit in enumerate(ranked):
        celex = hit.law_id or GDPR_CELEX
        text = hit.text or ""
        notes = version_notes(hit, as_of)
        if len(text) > MAX_ARTICLE_CHARS:
            text = text[:MAX_ARTICLE_CHARS]
            notes.append(f"Only the first {MAX_ARTICLE_CHARS} characters of this article were given to the model.")
        sources.append(
            Source(
                id=f"S{first_id + offset}",
                kind="legislation",
                celex=celex,
                article_ref=hit.article_ref,
                title=hit.law_title or f"CELEX {celex}",
                url=eur_lex_url(celex),
                text=text,
                notes=tuple(notes),
            )
        )
    return sources, other_acts


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS}


def select_paragraphs(document: Document, question: str, hint: int | None) -> list[tuple[int, str]]:
    """Pick the paragraphs of a judgment to hand to the model.

    When the search hit says which paragraph matched, take it and its
    neighbours; otherwise take the paragraphs sharing most words with the
    question. Unnumbered paragraphs are skipped - they cannot be cited.
    """
    numbered = [
        (p.number, p.text) for p in (document.content.paragraphs if document.content else []) if p.number and p.text
    ]
    if hint is not None and any(number == hint for number, _ in numbered):
        return [(n, t) for n, t in numbered if abs(n - hint) <= 1]
    wanted = _terms(question)
    scored = sorted(numbered, key=lambda p: len(wanted & _terms(p[1])), reverse=True)[:PARAGRAPHS_PER_CASE]
    return sorted(p for p in scored if wanted & _terms(p[1]))


def case_sources(
    client: LexAPI,
    cases: list[SemanticCaseLawResult],
    question: str,
    first_id: int,
    on_credits: Callable[[int | None], None] | None = None,
) -> tuple[list[Source], list[str], list[int | None]]:
    """Paragraphs of the judgments found. ``on_credits`` is told each read's credits as soon as it returns."""
    sources: list[Source] = []
    notices: list[str] = []
    credits: list[int | None] = []
    for case in cases[:MAX_CASES]:
        if not case.celex:
            continue
        response = client.get_document(case.celex, include=["metadata", "paragraphs"])
        credits.append(reported_credits(response))
        if on_credits:
            on_credits(credits[-1])
        document = response.document
        name = case.ecli or document.ecli or case.celex
        availability = document.text_availability
        if document.content is None or (availability and availability.available is False):
            reason = (availability.reason if availability else None) or "no reason given"
            notices.append(f"{name}: found by the search, but its text is not available ({reason}).")
            continue
        chosen = select_paragraphs(document, question, case.paragraph_number)
        if not chosen:
            notices.append(f"{name}: found by the search, but no numbered paragraph matched the question.")
            continue
        url = document.urls.get("html") if isinstance(document.urls.get("html"), str) else None
        for number, text in chosen:
            sources.append(
                Source(
                    id=f"S{first_id + len(sources)}",
                    kind="case-law",
                    celex=case.celex,
                    ecli=case.ecli or document.ecli,
                    paragraph=number,
                    title=document.title or case.case_name or name,
                    url=url or eur_lex_url(case.celex),
                    text=text,
                )
            )
    return sources, notices, credits


def retrieve(client: LexAPI, question: str, as_of: str | None = None, result: Retrieved | None = None) -> Retrieved:
    """Search, then read the judgments found. Pass ``result`` to see the credits counted when a later call fails."""
    result = Retrieved() if result is None else result

    legislation = client.semantic_legislation_search(question, limit=10, include_text=True, as_of=as_of)
    result.add_credits(reported_credits(legislation))
    articles, other_acts = legislation_sources(legislation.results, as_of)
    result.sources += articles
    if other_acts:
        result.notices.append(f"{other_acts} legislation hit(s) from acts other than the GDPR were not used.")
    if as_of and legislation.as_of_summary and legislation.as_of_summary.unresolved:
        result.notices.append(
            f"For {legislation.as_of_summary.unresolved} hit(s) LexAPI could not establish the version in force on "
            f"{as_of}; they are labelled in the sources."
        )

    case_law = client.semantic_search(question, limit=5)
    result.add_credits(reported_credits(case_law))
    if case_law.hint:
        result.notices.append(f"Case-law search: {case_law.hint}")
    paragraphs, notices, _credits = case_sources(
        client, case_law.results, question, len(result.sources) + 1, result.add_credits
    )
    result.sources += paragraphs
    result.notices += notices
    return result
