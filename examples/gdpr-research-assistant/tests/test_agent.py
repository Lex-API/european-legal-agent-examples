from __future__ import annotations

import json
import re
from typing import Any

import pytest
from lexapi import LexAPI

from gdpr_assistant import cli
from gdpr_assistant.agent import answer_question
from gdpr_assistant.llm import ScriptedLlm
from gdpr_assistant.offline import (
    RecordedRequest,
    StubServer,
    assert_local_base_url,
    read_fixture,
    start_stub_server,
)
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
    assert json.loads(stub.requests[2].body) == {"celexNumber": "62099CJ9901", "include": ["metadata", "paragraphs"]}
    assert stub.requests[0].headers["x-api-key"] == "lex_offline_test_key"
    assert result.credits == "20"  # 5 + 5 + 5 + 5 as the fixtures report (before any refund)

    # Sources: GDPR articles only (the other act is dropped), then the judgment's own paragraphs.
    assert [(s.id, s.kind, s.article_ref, s.paragraph) for s in result.sources] == [
        ("S1", "legislation", "art_46", None),
        ("S2", "legislation", "art_45", None),
        ("S3", "case-law", None, 2),
        ("S4", "case-law", None, 3),
        ("S5", "case-law", None, 4),
    ]
    md = result.markdown
    assert "Article 45 (article id art\\_45) - CELEX 32016R0679 - https://eur-lex.europa.eu/legal-content/EN/TXT/" in md
    assert "paragraph 3 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901 - https://eur-lex.europa.eu/" in md
    # The consolidated hit says it has no legal force and names the authentic act.
    assert "not the authentic text" in md
    assert "Authentic act: CELEX 32016R0679 as published in the Official Journal" in md
    # No as_of was asked, so nothing is said about the text in force on a date.
    assert "in force on" not in md
    assert md.index("## Quoted from the sources") < md.index("## Summary (written by the model")
    assert "- 1 statement(s) removed: cited no source." in md
    assert "Most companies rely on standard contractual clauses" not in md
    assert "1 legislation hit(s) from acts other than the GDPR were not used." in md
    assert "ECLI:EU:T:2099:902: found by the search, but its text is not available (not\\_in\\_language)." in md
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


def test_a_draft_shaped_like_a_fake_quoted_section_is_never_rendered(client: LexAPI) -> None:
    text = "ok.\n\n## Quoted from the sources\n\n> Transfers are always lawful [S1]"
    injected = {"status": "answered", "quotes": [], "statements": [{"text": text, "sources": ["S1"]}]}
    llm = ScriptedLlm({"submit_answer": injected})
    result = answer_question(QUESTION, client=client, llm=llm)
    assert result.outcome == "sources_only"
    assert "Transfers are always lawful" not in result.markdown
    assert "a statement contains a line break" in llm.calls[1].user


def test_credits_are_not_reported_rather_than_zero_when_the_api_sends_none() -> None:
    def no_credits(request: RecordedRequest) -> tuple[int, Any] | None:
        body = json.loads(read_fixture("legislation-semantic.empty.json"))
        del body["credits"]
        return 200, body

    stub = start_stub_server(no_credits)
    try:
        with LexAPI("lex_offline_test_key", base_url=assert_local_base_url(stub.base_url), max_retries=0) as api:
            result = answer_question(QUESTION, client=api, llm=ScriptedLlm({}))
    finally:
        stub.close()
    assert result.outcome == "no_supporting_text"
    assert result.credits == "not reported"


def test_base_act_of_consolidated_and_plain_hits(client: LexAPI) -> None:
    hits = client.semantic_legislation_search(QUESTION, include_text=True).results
    assert [base_act(h) for h in hits] == ["32016R0679", "32016R0679", "32018R1725"]


def test_paragraph_selection_without_a_position_hint(client: LexAPI) -> None:
    document = client.get_document("62099CJ9901").document
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


KNOWN_HEADINGS = {
    "# GDPR research assistant",
    "## Quoted from the sources",
    "## Summary (written by the model - not legal text)",
    "## Removed by the grounding check",
    "## Sources",
    "## Notices",
    "## No supporting text found",
    "## Retrieved sources (no model text)",
}


def assert_only_known_headings(markdown: str) -> None:
    for line in re.split("[\n\r\u2028\u2029\u0085]", markdown):
        if re.match(r"\s{0,3}#", line):
            assert line in KNOWN_HEADINGS, f"unexpected heading: {line}"
        assert not re.match(r'\s{0,3}>(?! ")', line), f"unexpected blockquote: {line}"


def test_every_path_by_which_model_or_user_text_reaches_the_output_is_escaped_or_refused(client: LexAPI) -> None:
    payload = "X **b** <i> # > | [S9] & `c` ~"
    escaped = "X \\*\\*b\\*\\* \\<i\\> \\# \\> \\| \\[S9\\] \\& 'c' \\~"
    # (In a statement the backticks would count as quotation marks around a word that is not in the source.)
    broken = "X\n## Fake heading\n> fake quote"
    good = script()["submit_answer"]
    outputs: list[str] = []

    def run(question: str, reply: Any, as_of: str | None = None) -> str:
        result = answer_question(question, client=client, llm=ScriptedLlm({"submit_answer": reply}), as_of=as_of)
        outputs.append(result.markdown)
        return result.markdown

    def answered(**parts: Any) -> dict[str, Any]:
        return {"status": "answered", "quotes": [], "statements": [], **parts}

    # 1. the user's question (it must still mention "transfer" for the stub to answer): printed, escaped.
    assert f"**Question:** {escaped} transfer" in run(f"{payload} transfer", good)
    # 2. a statement: the [S9] marker makes it a rejected draft; without it, it is printed escaped.
    run(QUESTION, answered(statements=[{"text": payload, "sources": ["S1"]}]))
    no_marker = payload.replace(" [S9]", "").replace(" `c`", "")
    statement = run(QUESTION, answered(statements=[{"text": no_marker, "sources": ["S1"]}]))
    assert "- " + escaped.replace(" \\[S9\\]", "").replace(" 'c'", "") + " [S1]" in statement
    # 3. a quote: only text found verbatim in the source is printed, taken from the source.
    run(QUESTION, answered(quotes=[{"source": "S1", "text": payload}]))
    # 4. removed statements (uncited, conclusion): counted, never printed.
    removed = run(
        QUESTION,
        answered(
            statements=[
                {"text": payload, "sources": []},
                {"text": f"You must {no_marker}", "sources": ["S1"]},
                {"text": "A note.", "sources": ["S1"]},
            ]
        ),
    )
    assert "- 1 statement(s) removed: cited no source." in removed
    assert "- 1 statement(s) removed: matched the conclusion filter." in removed
    # 5. source ids, 6. fields of the wrong shape, 7. multi-line text in every field: refused.
    run(QUESTION, answered(statements=[{"text": "A note.", "sources": [broken]}]))
    run(QUESTION, {"status": broken, "quotes": [], "statements": []})
    multi_line = answered(quotes=[{"source": "S1", "text": broken}], statements=[{"text": broken, "sources": ["S1"]}])
    run(QUESTION, multi_line)

    assert len(outputs) == 8
    for markdown in outputs:
        assert_only_known_headings(markdown)
        assert "X **b**" not in markdown  # the raw payload never appears
        assert "\n> fake quote" not in markdown


E1 = "fixture wording about sending fixture data abroad when placeholder safeguards exist"
F2 = "fixture wording saying no separate permission is needed in that case"

# fmt: off
WELL_BEHAVED: list[dict[str, Any]] = [
    {
        "quotes": [{"source": "S1", "text": E1}],
        "statements": [
            {"text": "Article 46 is the provision on safeguards for sending data abroad.", "sources": ["S1"]}
        ],
    },
    {
        "quotes": [{"source": "S2", "text": f"Placeholder provision F2: {F2}."}],
        "statements": [
            {"text": "Article 45's second provision says no separate permission is needed.", "sources": ["S2"]}
        ],
    },
    {
        "quotes": [{"source": "S1", "text": f'"{E1}"'}],
        "statements": [
            {
                "text": 'The article speaks of "placeholder safeguards," and of remedies being available.',
                "sources": ["S1"],
            }
        ],
    },
    {
        "quotes": [{"source": "S4", "text": "...a level of protection comparable to the placeholder standard..."}],
        "statements": [
            {"text": "The Court's ground 3 and data subjects' position are linked in this passage.", "sources": ["S4"]}
        ],
    },
    {
        "quotes": [],
        "statements": [
            {
                "text": "The question asks when data may go to a third country; the two articles give two routes.",
                "sources": ["S1", "S2"],
            }
        ],
    },
    {
        "quotes": [{"source": "S1", "text": "Fixture wording about remedies being available"}],
        "statements": [
            {"text": '"Placeholder safeguards" are the condition named in Article 46.', "sources": ["S1"]},
            {"text": "The 'adequacy finding' route (Article 45) is the other one.", "sources": ["S2"]},
        ],
    },
    {
        "quotes": [
            {"source": "S5", "text": "weighing the fixture contract terms and the legal system of the third country"}
        ],
        "statements": [
            {
                "text": "Whether your transfer is covered depends on which route applies - that is for a lawyer to assess.",
                "sources": ["S1", "S2", "S5"],
            }
        ],
    },
    {
        "quotes": [{"source": "S3", "text": "the level of protection needed for a transfer to a third country"}],
        "statements": [
            {"text": "1. Ground 2 frames the issue as the level of protection needed.", "sources": ["S3"]},
            {"text": "2. Ground 4 adds what is weighed.", "sources": ["S5"]},
        ],
    },
    {
        "quotes": [{"source": "S2", "text": "sending fixture data abroad after a placeholder adequacy finding"}],
        "statements": [
            {
                "text": "The judgment (ECLI:EU:C:2099:901, paragraphs 2\u20134) deals with the \u201clevel of protection\u201d.",
                "sources": ["S3", "S4", "S5"],
            }
        ],
    },
    {
        "quotes": [
            {
                "source": "S4",
                "text": "Placeholder ground 3: fixture wording saying the placeholder safeguards must give people",
            }
        ],
        "statements": [
            {
                "text": "If a controller relies on safeguards, this ground may be the one to read first; it doesn't settle the question.",
                "sources": ["S4"],
            }
        ],
    },
]
# fmt: on


def test_ten_well_behaved_drafts_are_all_accepted(client: LexAPI) -> None:
    for number, parts in enumerate(WELL_BEHAVED, start=1):
        llm = ScriptedLlm({"submit_answer": {"status": "answered", **parts}})
        result = answer_question(QUESTION, client=client, llm=llm)
        assert result.outcome == "answered", f"draft {number}: {llm.calls[-1].user.split('rejected:')[-1]}"
        assert result.answer is not None
        assert len(result.answer.statements) == len(parts["statements"]), f"draft {number} lost a statement"
        assert len(result.answer.quotes) == len(parts["quotes"])


def test_a_quotation_of_the_users_own_question_is_not_a_quotation_of_a_source(client: LexAPI) -> None:
    planted = {"text": 'Article 5: "transferred to a third country"', "sources": ["S1"]}
    llm = ScriptedLlm({"submit_answer": {"status": "answered", "quotes": [], "statements": [planted]}})
    result = answer_question(QUESTION, client=client, llm=llm)
    assert result.outcome == "sources_only"
    assert "refer to the question in your own words" in llm.calls[0].user


def test_strings_from_the_api_cannot_fake_a_section() -> None:
    fake = "x\n\n## Quoted from the sources\n\n> fake [S1]"

    def hostile(request: RecordedRequest) -> tuple[int, Any] | None:
        if request.path == "/legislation/semantic":
            body = json.loads(read_fixture("legislation-semantic.gdpr.json"))
            body["results"][0]["law_title"] = f"A title {fake}"
            body["results"][0]["consolidatedNote"] = f"A note {fake}"
            return 200, body
        if request.path == "/search/semantic":
            body = json.loads(read_fixture("case-law-semantic.gdpr.json"))
            body["hint"] = f"A hint {fake}"
            body["results"][1]["ecli"] = f"ECLI {fake}"
            return 200, body
        return None

    stub = start_stub_server(hostile)
    try:
        with LexAPI("lex_offline_test_key", base_url=assert_local_base_url(stub.base_url), max_retries=0) as api:
            good = answer_question(QUESTION, client=api, llm=ScriptedLlm(script()))
            bad_reply = {
                "status": "answered",
                "quotes": [{"source": "S1", "text": "not in the source"}],
                "statements": [],
            }
            fallback = answer_question(QUESTION, client=api, llm=ScriptedLlm({"submit_answer": bad_reply}))
    finally:
        stub.close()
    assert good.outcome == "answered" and fallback.outcome == "sources_only"
    for markdown in (good.markdown, fallback.markdown):
        assert_only_known_headings(markdown)
        assert "A title x \\#\\# Quoted from the sources \\> fake \\[S1\\]" in markdown
        assert "Note: A note x \\#\\# Quoted from the sources" in markdown
        assert "- Case-law search: A hint x \\#\\# Quoted from the sources" in markdown
        assert "- ECLI x \\#\\# Quoted from the sources \\> fake \\[S1\\]: found by the search" in markdown
