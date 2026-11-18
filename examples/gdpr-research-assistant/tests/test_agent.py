from __future__ import annotations

import json
from typing import Any

import pytest
from lexapi import LexAPI

from gdpr_assistant import cli
from gdpr_assistant.agent import answer_question
from gdpr_assistant.llm import ScriptedLlm
from gdpr_assistant.offline import StubServer, read_fixture
from gdpr_assistant.retrieve import base_act, select_paragraphs

QUESTION = "When may personal data be transferred to a third country?"


def script(name: str = "gdpr") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(read_fixture(f"llm/{name}.json"))
    return loaded


def test_answers_with_quotes_from_the_gdpr_and_a_judgment(stub: StubServer, client: LexAPI) -> None:
    llm = ScriptedLlm(script())
    result = answer_question(QUESTION, client=client, llm=llm)

    assert result.outcome == "answered"
    # The requests the SDK sent, with the wire names the API documents.
    assert [(r.method, r.path) for r in stub.requests] == [
        ("POST", "/legislation/semantic"),
        ("POST", "/search/semantic"),
        ("POST", "/documentContent"),
        ("POST", "/documentContent"),
    ]
    assert json.loads(stub.requests[0].body) == {"query": QUESTION, "limit": 10, "include": ["text"]}
    assert json.loads(stub.requests[1].body) == {"query": QUESTION, "limit": 5}
    assert json.loads(stub.requests[2].body) == {"celexNumber": "62018CJ0311", "include": ["metadata", "paragraphs"]}
    assert stub.requests[0].headers["x-api-key"] == "lex_offline_test_key"
    assert result.credits_reported == 20  # 5 + 5 + 5 + 5 as the fixtures report (before any refund)

    # Sources: GDPR articles only (the other act is dropped), then the judgment's own paragraphs.
    assert [(s.id, s.kind, s.article_ref, s.paragraph) for s in result.sources] == [
        ("S1", "legislation", "art_46", None),
        ("S2", "legislation", "art_45", None),
        ("S3", "case-law", None, 2),
        ("S4", "case-law", None, 3),
        ("S5", "case-law", None, 4),
    ]
    md = result.markdown
    assert "Article 45 (article id art_45) - CELEX 32016R0679 - https://eur-lex.europa.eu/legal-content/EN/TXT/" in md
    assert "paragraph 3 - ECLI:EU:C:2020:559 - CELEX 62018CJ0311 - https://eur-lex.europa.eu/" in md
    # The consolidated hit says it has no legal force and names the authentic act.
    assert "not the authentic text" in md
    assert "Authentic act: CELEX 32016R0679 as published in the Official Journal" in md
    # No as_of was asked, so nothing is said about the text in force on a date.
    assert "in force on" not in md
    assert md.index("## Quoted from the sources") < md.index("## Summary (written by the model")
    assert "(cited no source) Most companies rely on standard contractual clauses." in md
    assert "1 legislation hit(s) from acts other than the GDPR were not used." in md
    assert "ECLI:EU:T:2099:1: found by the search, but its text is not available in English (not_in_language)." in md
    assert "not legal advice" in md


def test_machine_written_case_summaries_never_reach_the_model(client: LexAPI) -> None:
    llm = ScriptedLlm(script())
    answer_question(QUESTION, client=client, llm=llm)
    prompt = llm.calls[0].user
    assert '<source id="S4">' in prompt
    assert "machine-written digest" not in prompt  # the search hits' summary/snippet
    assert "32018R1725" not in prompt  # the non-GDPR hit


def test_as_of_claims_in_force_only_when_the_api_resolved_the_version(stub: StubServer, client: LexAPI) -> None:
    result = answer_question(QUESTION, client=client, llm=ScriptedLlm(script("gdpr-asof")), as_of="2019-06-01")
    assert json.loads(stub.requests[0].body)["asOf"] == "2019-06-01"
    resolved, unresolved = result.sources[0], result.sources[1]
    assert resolved.notes == (
        "Text in force on 2019-06-01, according to LexAPI: the act as published in the Official Journal.",
    )
    assert unresolved.notes[0].startswith(
        "The version in force on 2019-06-01 is not established (LexAPI: fetch_timeout)."
    )
    assert not any("Text in force on" in note for note in unresolved.notes)
    assert "For 1 hit(s) LexAPI could not establish the version in force on 2019-06-01" in result.markdown
    assert "**Question (as of 2019-06-01):**" in result.markdown


def test_no_supporting_text_found_skips_the_model(client: LexAPI) -> None:
    llm = ScriptedLlm(script())
    result = answer_question("no-such-topic placeholder question", client=client, llm=llm)
    assert result.outcome == "no_supporting_text"
    assert "## No supporting text found" in result.markdown
    assert llm.calls == []


def test_a_model_reply_of_no_support_gives_no_answer(client: LexAPI) -> None:
    llm = ScriptedLlm({"submit_answer": {"status": "no_support", "quotes": [], "statements": []}})
    result = answer_question(QUESTION, client=client, llm=llm)
    assert result.outcome == "no_supporting_text"
    assert "nothing in them that answers the question" in result.markdown


def test_a_draft_citing_an_unretrieved_source_is_rejected_then_retried(client: LexAPI) -> None:
    bad = {"status": "answered", "quotes": [], "statements": [{"text": "See Article 49.", "sources": ["S12"]}]}
    llm = ScriptedLlm({"submit_answer": [bad, script()["submit_answer"]]})
    result = answer_question(QUESTION, client=client, llm=llm)
    assert result.outcome == "answered"
    assert len(llm.calls) == 2
    assert 'previous draft was rejected:\n- statement cites "S12", which was not retrieved' in llm.calls[1].user


def test_after_two_rejected_drafts_only_the_sources_are_shown(client: LexAPI) -> None:
    quote = {"source": "S4", "text": "transfers are always lawful"}
    invented = {"status": "answered", "quotes": [quote], "statements": []}
    result = answer_question(QUESTION, client=client, llm=ScriptedLlm({"submit_answer": invented}))
    assert result.outcome == "sources_only"
    assert "transfers are always lawful" not in result.markdown
    assert "failed the grounding check twice" in result.markdown
    assert "[S4] " in result.markdown


def test_base_act_of_consolidated_and_plain_hits(client: LexAPI) -> None:
    hits = client.semantic_legislation_search(QUESTION, include_text=True).results
    assert [base_act(h) for h in hits] == ["32016R0679", "32016R0679", "32018R1725"]


def test_paragraph_selection_without_a_position_hint(client: LexAPI) -> None:
    document = client.get_document("62018CJ0311").document
    chosen = select_paragraphs(document, "level of protection for a transfer to a third country", None)
    assert [number for number, _ in chosen][:2] == [2, 3]
    assert select_paragraphs(document, "zzz qqq", None) == []


def test_cli_dry_run_needs_no_keys(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("LEXAPI_API_KEY", "LEXAPI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    assert cli.main(["--dry-run", QUESTION]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("> DRY RUN")
    assert "## Quoted from the sources" in out.out
    assert "credits reported for this run: 20" in out.err
