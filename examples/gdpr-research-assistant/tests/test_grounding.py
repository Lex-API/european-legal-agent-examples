from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gdpr_assistant.conclusions import READER_PATTERNS, states_conclusion
from gdpr_assistant.grounding import (
    GroundedAnswer,
    GroundingError,
    Removed,
    Source,
    render_answer,
    render_removed,
    validate_draft,
)
from gdpr_assistant.text import model_text_problem, normalise, plain_text, printable_url, quotations_in

# The cases in shared/grounding-vectors.json are run by this suite AND by the TypeScript suite.
VECTORS_PATH = Path(__file__).resolve().parents[3] / "shared" / "grounding-vectors.json"


def expand(value: Any) -> Any:
    """``{"repeat": [unit, n]}`` stands for the unit repeated n times."""
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        if "repeat" in value:
            unit, count = value["repeat"]
            return unit * count
        return {k: expand(v) for k, v in value.items()}
    return value


VECTORS: dict[str, Any] = expand(json.loads(VECTORS_PATH.read_text(encoding="utf-8")))
SOURCES = [Source(s["id"], "legislation", "39999R0001", "Test source", s["url"], s["text"]) for s in VECTORS["sources"]]


def check(draft: Any, patterns: str | None = None) -> GroundedAnswer | list[str]:
    try:
        return validate_draft(draft, SOURCES, conclusion_patterns=READER_PATTERNS if patterns else None)
    except GroundingError as error:
        return error.violations


@pytest.mark.parametrize("case", VECTORS["drafts"], ids=lambda c: f"{c['group']}: {c['name']}")
def test_shared_vectors_drafts(case: dict[str, Any]) -> None:
    result = check(case["draft"], case.get("patterns"))
    if not case["accept"]:
        assert isinstance(result, list), "expected a rejection"
        assert any(case["violation"] in v for v in result), result
        return
    assert isinstance(result, GroundedAnswer), result
    if "quotes" in case:
        assert [q.text for q in result.quotes] == case["quotes"]
    if "statements" in case:
        assert len(result.statements) == case["statements"]
    if "sources" in case:
        assert list(result.statements[0].sources) == case["sources"]
    if "removed" in case:
        assert result.removed == Removed(case["removed"]["uncited"], case["removed"]["conclusion"])


def test_shared_vectors_text_functions() -> None:
    for text, expected in VECTORS["normalise"]:
        assert normalise(text) == expected, repr(text)
    for text, expected in VECTORS["quotations"]:
        assert quotations_in(text) == expected, text
    for text, expected in VECTORS["plainText"]:
        assert plain_text(text) == expected, text
    for text, kind, expected in VECTORS["modelTextProblems"]:
        problem = model_text_problem(text, kind)
        if expected is None:
            assert problem is None, repr(text)
        else:
            assert problem is not None and expected in problem, repr(text)


def test_shared_vectors_conclusion_filter_reader_patterns() -> None:
    for case in VECTORS["conclusions"]:
        assert states_conclusion(normalise(case["text"]), READER_PATTERNS) is case["reader"], case["text"]


def draft(**overrides: Any) -> dict[str, Any]:
    return {"status": "answered", "quotes": [], "statements": [], **overrides}


def test_a_quote_is_returned_with_the_sources_own_characters() -> None:
    typeset = [Source("S1", "legislation", "X", "T", "u", "The keeper’s duty “at once” ends   here.")]
    answer = validate_draft(draft(quotes=[{"source": "S1", "text": 'the keeper\'s duty "at once" ends...'}]), typeset)
    assert [q.text for q in answer.quotes] == ['The keeper\'s duty "at once" ends']


def test_removed_statements_are_counted_never_echoed() -> None:
    statements = [
        {"text": "You must appoint a representative.", "sources": ["S1"]},
        {"text": 'An uncited claim with an invented "quotation of the law".', "sources": []},
        {"text": "The article covers consent.", "sources": ["S1"]},
    ]
    answer = validate_draft(draft(statements=statements), SOURCES, conclusion_patterns=READER_PATTERNS)
    assert [s.text for s in answer.statements] == ["The article covers consent."]
    assert answer.removed == Removed(uncited=1, conclusion=1)
    assert render_removed(answer.removed) == [
        "- 1 statement(s) removed: cited no source.",
        "- 1 statement(s) removed: matched the conclusion filter.",
    ]
    out = render_answer(answer, SOURCES)
    assert "appoint a representative" not in out
    assert "quotation of the law" not in out


def test_no_support_and_emptied_drafts_end_as_no_support() -> None:
    declared = draft(status="no_support", quotes=[{"source": "S9", "text": "x"}])
    assert validate_draft(declared, SOURCES).status == "no_support"
    assert validate_draft(draft(statements=[{"text": "Uncited.", "sources": []}]), SOURCES).status == "no_support"


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"status": "maybe", "quotes": [], "statements": []},
        {"status": "answered", "quotes": "none", "statements": []},
        {"status": "answered", "quotes": [], "statements": [{"text": "x", "sources": [1]}]},
        {"status": "answered", "quotes": [{"source": "S1"}], "statements": []},
    ],
)
def test_rejects_replies_with_the_wrong_shape(bad: Any) -> None:
    with pytest.raises(GroundingError):
        validate_draft(bad, SOURCES)


def test_render_separates_quotes_from_model_text_and_cites_celex_article_ecli_link() -> None:
    cited = [
        Source(
            "S1",
            "legislation",
            "32016R0679",
            "Fixture act",
            "https://example.org/a",
            "A placeholder process shall be kept running.",
            article_ref="art_46",
            notes=("Consolidated text; not the authentic text.",),
        ),
        Source(
            "S2",
            "case-law",
            "62099CJ9901",
            "Fixture judgment",
            "https://example.org/b",
            "Placeholder ground three.",
            paragraph=3,
            ecli="ECLI:EU:C:2099:901",
        ),
    ]
    statements = [
        {"text": "1. A note with **bold**   spacing & an <i>tag</i>.", "sources": ["S2"]},
    ]
    answer = validate_draft(
        draft(quotes=[{"source": "S1", "text": "process shall be kept"}], statements=statements),
        cited,
    )
    out = render_answer(answer, cited)
    assert out.index("## Quoted from the sources") < out.index("## Summary (written by the model")
    assert '> "process shall be kept" [S1]\n' in out
    assert "- 1\\. A note with \\*\\*bold\\*\\* spacing \\& an \\<i\\>tag\\</i\\>. [S2]\n" in out
    assert "[S1] Fixture act - Article 46 (article id art\\_46) - CELEX 32016R0679 - https://example.org/a" in out
    assert "[S2] Fixture judgment - paragraph 3 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901" in out
    assert "Note: Consolidated text; not the authentic text." in out


def test_strings_from_the_api_are_printed_as_literal_text() -> None:
    hostile = [
        Source(
            "S1",
            "case-law",
            "62099CJ9901\n## Fake",
            'A title\n\n## Quoted from the sources\n\n> "Transfers are always lawful" [S1]',
            "javascript:alert(1)",
            "Placeholder ground three of the judgment.",
            article_ref="art_9 [S7]",
            ecli="ECLI:EU:C:2099:901 **x**",
            notes=("A note with `code`\n# heading and \u001b[31mcolour",),
        )
    ]
    answer = validate_draft(draft(quotes=[{"source": "S1", "text": "Placeholder ground three"}]), hostile)
    out = render_answer(answer, hostile)
    headings = [line for line in out.split("\n") if line.lstrip().startswith("#")]
    assert headings == [
        "## Quoted from the sources",
        "## Summary (written by the model - not legal text)",
        "## Sources",
    ]
    assert len([line for line in out.split("\n") if line.lstrip().startswith(">")]) == 1  # the one real quote
    assert (
        '- [S1] A title \\#\\# Quoted from the sources \\> "Transfers are always lawful" \\[S1\\] - art\\_9 \\[S7\\] '
        "(article id art\\_9 \\[S7\\]) - ECLI:EU:C:2099:901 \\*\\*x\\*\\* - CELEX 62099CJ9901 \\#\\# Fake\n"
    ) in out
    assert "javascript:" not in out
    assert "  - Note: A note with 'code' \\# heading and \\[31mcolour\n" in out
    assert printable_url("https://example.org/a_b?x=1&y=2") == "https://example.org/a\\_b?x=1\\&y=2"
    assert printable_url("not a url") == ""


def test_conclusion_vectors_meet_the_review_targets() -> None:
    def of(kind: str) -> list[dict[str, Any]]:
        return [c for c in VECTORS["conclusions"] if c["kind"] == kind]

    assert len(of("hedged attack")) >= 14
    assert len(of("legitimate")) >= 13 and not any(c["reader"] for c in of("legitimate"))
