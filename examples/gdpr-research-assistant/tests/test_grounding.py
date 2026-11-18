from __future__ import annotations

from typing import Any

import pytest

from gdpr_assistant.grounding import (
    GroundingError,
    Source,
    embedded_quotations,
    escape_markdown,
    normalise,
    render_answer,
    validate_draft,
)

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
        celex="62099CJ9901",
        ecli="ECLI:EU:C:2099:901",
        paragraph=3,
        title="Fixture judgment",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62099CJ9901",
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
    validate_draft(draft(quotes=[{"source": "S1", "text": "personal data   only\tif appropriate"}]), SOURCES)
    found = violations(draft(quotes=[{"source": "S1", "text": "only when appropriate safeguards are provided"}]))
    assert "not a verbatim excerpt" in found[0]


def test_rejects_ids_that_were_not_retrieved() -> None:
    assert '"S9", which was not retrieved' in violations(draft(quotes=[{"source": "S9", "text": "x"}]))[0]
    assert '"S7", which was not retrieved' in violations(draft(statements=[{"text": "c", "sources": ["S1", "S7"]}]))[0]


def test_rejects_a_quote_attributed_to_the_wrong_source_and_an_empty_quote() -> None:
    assert len(violations(draft(quotes=[{"source": "S2", "text": "appropriate safeguards are provided"}]))) == 1
    for text in ("  ", "a", "appropriate safeguards"):
        assert "shorter than 3 words" in violations(draft(quotes=[{"source": "S1", "text": text}]))[0]


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
            quotes=[{"source": "S1", "text": "appropriate safeguards are provided"}],
            statements=[{"text": "A note.", "sources": ["S2"]}],
        ),
        SOURCES,
    )
    out = render_answer(answer, SOURCES)
    assert out.index("## Quoted from the sources") < out.index("## Summary (written by the model")
    assert '> "appropriate safeguards are provided" [S1]' in out
    assert "- A note. [S2]" in out
    assert "[S1] Fixture act - Article 46 (article id art_46) - CELEX 32016R0679 - https://eur-lex.europa.eu/" in out
    assert "[S2] Fixture judgment - paragraph 3 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901" in out
    assert "Note: Consolidated text; not the authentic text." in out


def test_soft_hyphens_and_typographic_quotes_do_not_break_a_verbatim_match() -> None:
    typeset = [Source("S1", "legislation", "X", "T", "u", "the trans\u00adfer shall not \u201ctake place\u201d here")]
    validate_draft(draft(quotes=[{"source": "S1", "text": 'the transfer shall not "take place"'}]), typeset)
    plain = [Source("S1", "legislation", "X", "T", "u", "the transfer shall not take place")]
    validate_draft(draft(quotes=[{"source": "S1", "text": "the trans\u00adfer shall not"}]), plain)
    with pytest.raises(GroundingError):  # dropping the negation is still not verbatim
        validate_draft(draft(quotes=[{"source": "S1", "text": "the transfer shall take place"}]), typeset)


@pytest.mark.parametrize(
    "text",
    [
        "ok.\n\n## Quoted from the sources\n\n> Transfers are always lawful [S1]",
        "# Heading",
        "> quoted",
        "- item",
        "* item",
        "+ item",
        "1. item",
        "2) item",
        "| a | b |",
        "```code",
        "~~~code",
        "  ## indented",
        "a\u2028b",
        "a\rb",
    ],
)
def test_rejects_statements_that_are_not_one_line_of_plain_text(text: str) -> None:
    found = violations(draft(statements=[{"text": text, "sources": ["S1"]}]))
    assert "line break" in found[0] or "Markdown block marker" in found[0]


def test_shape_rule_details() -> None:
    quote = violations(draft(quotes=[{"source": "S1", "text": "transfer personal data\nonly if"}]))
    assert "quote from S1 contains a line break" in quote[0]
    # An uncited statement is rejected for its shape too - it would otherwise be printed in the "removed" list.
    violations(draft(statements=[{"text": "x\n## y", "sources": []}]))
    # Hyphenated words and a quote that opens with a paragraph number are fine.
    validate_draft(draft(statements=[{"text": "-risk is not a list marker.", "sources": ["S1"]}]), SOURCES)
    numbered = [Source("S1", "legislation", "X", "T", "u", "1. A numbered paragraph of law.")]
    validate_draft(draft(quotes=[{"source": "S1", "text": "1. A numbered paragraph"}]), numbered)


def test_model_text_is_escaped_when_rendered() -> None:
    assert escape_markdown("see [S9] and *this* `c` <b> # > | ~ _x_ \\") == (
        "see \\[S9\\] and \\*this\\* \\`c\\` \\<b\\> \\# \\> \\| \\~ \\_x\\_ \\\\"
    )
    styled = {"text": "A note with [S2] and **bold**   spacing.", "sources": ["S1"]}
    out = render_answer(validate_draft(draft(statements=[styled]), SOURCES), SOURCES)
    assert "- A note with \\[S2\\] and \\*\\*bold\\*\\* spacing. [S1]\n" in out
    assert "[S2] Fixture judgment" not in out  # the escaped marker did not pull S2 into the sources


def test_quotations_in_every_mark_style_and_apostrophes() -> None:
    assert embedded_quotations('says "never allowed" here') == ["never allowed"]
    assert embedded_quotations("says 'never allowed' here") == ["never allowed"]
    mixed = "says \u00abnever allowed\u00bb and \u201enever\u201c and \u2039x y\u203a and \u201aa b\u2018"
    assert embedded_quotations(mixed) == ["never allowed", "never", "x y", "a b"]
    assert embedded_quotations("the provider's duty and data subjects' rights, and it's fine") == []
    assert embedded_quotations("it says 'the provider's duty' here") == ["the provider's duty"]
    assert embedded_quotations('says "never allowed here') is None
    assert embedded_quotations("says 'never allowed here") is None
    assert normalise("\u201ea\u201c \u00abb\u00bb \u2018c\u2019") == '"a" "b" \'c\''


@pytest.mark.parametrize(
    "text",
    [
        'Article 45 says transfers are "never allowed".',
        "Article 45 says 'transfers always require prior authorisation'.",
        "Article 45 says \u00abtransfers always require prior authorisation\u00bb.",
        "Article 45 says \u201etransfers always require prior authorisation\u201c.",
        'Article 45 says "transfers always require prior authorisation.',
        'A single long word: "unauthorised".',
        'It says "an essentially equivalent level of protection".',  # verbatim, but in S2, which is not cited
    ],
)
def test_rejects_invented_quotations_in_any_style(text: str) -> None:
    found = violations(draft(statements=[{"text": text, "sources": ["S1"]}]))
    assert "quotation that is not in the sources it cites" in found[0] or "unbalanced quotation mark" in found[0]


def test_short_single_word_quotations_and_possessives_pass() -> None:
    text = "The status was \"fresh\" and the provider's duty and subjects' rights are mentioned."
    validate_draft(draft(statements=[{"text": text, "sources": ["S1"]}]), SOURCES)


def test_duplicate_ids_are_collapsed_and_ids_must_match_exactly() -> None:
    answer = validate_draft(draft(statements=[{"text": "A note.", "sources": ["S1", "S1", "S2", "S1"]}]), SOURCES)
    assert answer.statements[0].sources == ("S1", "S2")
    for bad in ("s1", " S1", "S1 "):
        violations(draft(statements=[{"text": "A note.", "sources": [bad]}]))
