from __future__ import annotations

from typing import Any

import pytest

from gdpr_assistant.grounding import GroundingError, Source, render_answer, validate_draft

SOURCES = [
    Source(
        id="S1",
        kind="legislation",
        celex="32016R0679",
        article_ref="art_46",
        title="Fixture act",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679",
        text="A controller may transfer personal data\n  only if appropriate safeguards are provided.",
        notes=("Consolidated text; not the authentic text.",),
    ),
    Source(
        id="S2",
        kind="case-law",
        celex="62018CJ0311",
        ecli="ECLI:EU:C:2020:559",
        paragraph=3,
        title="Fixture judgment",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62018CJ0311",
        text="The safeguards must ensure an essentially equivalent level of protection.",
    ),
]


def draft(**overrides: Any) -> dict[str, Any]:
    return {"status": "answered", "quotes": [], "statements": [], **overrides}


def violations(d: dict[str, Any]) -> list[str]:
    with pytest.raises(GroundingError) as caught:
        validate_draft(d, SOURCES)
    return caught.value.violations


def test_accepts_verbatim_quotes_and_cited_statements() -> None:
    answer = validate_draft(
        draft(
            quotes=[{"source": "S1", "text": "only if appropriate safeguards are provided"}],
            statements=[{"text": "A summary.", "sources": ["S1", "S2"]}],
        ),
        SOURCES,
    )
    assert answer.status == "answered"
    assert len(answer.statements) == 1
    assert answer.removed == ()


def test_quote_may_differ_in_whitespace_only() -> None:
    validate_draft(draft(quotes=[{"source": "S1", "text": "personal data   only\nif appropriate"}]), SOURCES)
    found = violations(draft(quotes=[{"source": "S1", "text": "only when appropriate safeguards are provided"}]))
    assert "not a verbatim excerpt" in found[0]


def test_rejects_ids_that_were_not_retrieved() -> None:
    assert '"S9", which was not retrieved' in violations(draft(quotes=[{"source": "S9", "text": "x"}]))[0]
    assert '"S7", which was not retrieved' in violations(draft(statements=[{"text": "c", "sources": ["S1", "S7"]}]))[0]


def test_rejects_a_quote_attributed_to_the_wrong_source_and_an_empty_quote() -> None:
    assert len(violations(draft(quotes=[{"source": "S2", "text": "appropriate safeguards are provided"}]))) == 1
    assert "is empty" in violations(draft(quotes=[{"source": "S1", "text": "  "}]))[0]


def test_removes_and_reports_uncited_statements() -> None:
    answer = validate_draft(
        draft(statements=[{"text": "Supported.", "sources": ["S1"]}, {"text": "General knowledge.", "sources": []}]),
        SOURCES,
    )
    assert [s.text for s in answer.statements] == ["Supported."]
    assert answer.removed == ("General knowledge.",)


def test_rejects_a_quotation_embedded_in_a_statement_that_its_sources_do_not_contain() -> None:
    invented = 'The Court held that "transfers are always prohibited without consent".'
    found = violations(draft(statements=[{"text": invented, "sources": ["S2"]}]))
    assert "quotation that is not in the sources" in found[0]
    real = 'It refers to an "essentially equivalent level of protection".'
    validate_draft(draft(statements=[{"text": real, "sources": ["S2"]}]), SOURCES)


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
    answer = validate_draft(
        draft(
            quotes=[{"source": "S1", "text": "appropriate safeguards"}],
            statements=[{"text": "A note.", "sources": ["S2"]}],
        ),
        SOURCES,
    )
    out = render_answer(answer, SOURCES)
    assert out.index("## Quoted from the sources") < out.index("## Summary (written by the model")
    assert '> "appropriate safeguards" [S1]' in out
    assert "- A note. [S2]" in out
    assert "[S1] Fixture act - Article 46 (article id art_46) - CELEX 32016R0679 - https://eur-lex.europa.eu/" in out
    assert "[S2] Fixture judgment - paragraph 3 - ECLI:EU:C:2020:559 - CELEX 62018CJ0311" in out
    assert "Note: Consolidated text; not the authentic text." in out
