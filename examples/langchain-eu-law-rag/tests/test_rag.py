from __future__ import annotations

import io
import re
from typing import Any

import httpx
import pytest
from conftest import BASE_URL, OFFLINE_ERROR, TEST_KEY, RecordingChatModel, Stub, fake_model, search_reply
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_lexapi import LexAPILegislationRetriever
from lexapi import LexAPI

from dora_rag import rag

QUESTION = "What does the act say about reviewing the framework?"
AS_OF = "2025-01-17"
ANSWER = "The first passage describes a placeholder review [1]. The second lists placeholder elements [2]."
ANSWER_AS_OF = "The first passage describes a placeholder review [1]. There is no text of the second for the date [2]."


def run(stub: Stub, model: RecordingChatModel, *args: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    code = rag.main([*args, QUESTION], transport=stub.transport(), chat_model=model, out=out)
    captured = capsys.readouterr()
    return code, out.getvalue() + captured.out, captured.err


def prompt_text(model: RecordingChatModel) -> tuple[str, str]:
    assert len(model.seen) == 1
    system, human = model.seen[0]
    return str(system.content), str(human.content)


def sources_map(output: str) -> list[str]:
    block = output.split("Sources:\n", 1)[1].split("\n\n", 1)[0]
    return block.splitlines()


def expected_citations(as_of: str | None) -> list[str]:
    """What the passages' Source lines must say, from the fixture data (not from the code under test)."""
    article = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554"
    if as_of is None:
        consolidated = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02022R2554-20990101"
        return [
            f"CELEX 32022R2554, art_5, {article}",
            f"CELEX 32022R2554, art_6, consolidated text 02022R2554-20990101 (no legal force), {consolidated}",
        ]
    return [
        f"CELEX 32022R2554, art_5, version for {as_of}: 32022R2554, {article}",
        f"CELEX 32022R2554, art_6, text for {as_of} not established, {article}",
    ]


# ── the request ─────────────────────────────────────────────────────────


def test_one_search_for_dora_articles_without_a_date(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model(ANSWER), capsys=capsys)

    assert code == 0
    assert [(r.method, r.url.path) for r in stub.requests] == [("POST", "/api/v1/legislation/semantic")]
    assert stub.bodies() == [{"query": f"{rag.DORA_CONTEXT}: {QUESTION}", "limit": 10, "include": ["text"]}]
    assert stub.requests[0].headers["x-api-key"] == TEST_KEY
    # No date was asked, so nothing is said about a date.
    assert "Withheld" not in out
    assert "in force" not in out
    assert "Text asked for" not in out


def test_as_of_is_sent_with_the_search(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model(ANSWER_AS_OF), "--as-of", AS_OF, capsys=capsys)

    assert code == 0
    assert stub.bodies() == [
        {"query": f"{rag.DORA_CONTEXT}: {QUESTION}", "limit": 10, "include": ["text"], "asOf": AS_OF}
    ]
    assert f"Text asked for: {AS_OF}" in out


@pytest.mark.parametrize("value", ["2025-02-30", "17/01/2025", "2025-1-17", "2025-01-17T00:00", "", "tomorrow"])
def test_an_invalid_as_of_is_refused_before_any_request(
    stub: Stub, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    model = fake_model(ANSWER)
    with pytest.raises(SystemExit) as exit_info:
        rag.main(["--as-of", value, QUESTION], transport=stub.transport(), chat_model=model)
    assert exit_info.value.code == 2
    assert "expected a date written YYYY-MM-DD" in capsys.readouterr().err
    assert stub.requests == []
    assert model.seen == []


# ── what the model is sent ──────────────────────────────────────────────


def test_the_prompt_holds_every_source_line_and_no_withheld_text(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    model = fake_model(ANSWER_AS_OF)
    run(stub, model, "--as-of", AS_OF, capsys=capsys)
    system, human = prompt_text(model)

    for citation in expected_citations(AS_OF):
        assert f"Source: {citation}" in human
    assert human.count("Source: ") == 2
    assert "[1]\n" in human and "[2]\n" in human and "[3]" not in human
    assert f"Date asked: {AS_OF}" in human
    assert f"Question: {QUESTION}" in human
    # The withheld passage arrives as a notice, never with its latest text or snippet.
    assert f"[No text: CELEX 32022R2554 art_6 is not established as the text for {AS_OF} (reason: fetch_timeout).]" in (
        human
    )
    assert "UNRESOLVED-LATEST" not in system + human
    # Hits from other acts never reach the model.
    assert "OTHER-ACT" not in human
    assert "32099L0009" not in human


def test_the_prompt_states_the_rules(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    model = fake_model(ANSWER)
    run(stub, model, capsys=capsys)
    system, human = prompt_text(model)

    assert system == rag.SYSTEM_PROMPT
    assert "Answer only from the passages" in system
    assert '"[No text: CELEX ... is not established as the text for <date> ...]"' in system
    assert "has no text for that date" in system
    assert "Consolidated texts have no legal force" in system
    assert "name the one given by the CELEX number at the start of the Source line" in system
    assert "do not give legal advice" in system
    assert "do not say whether anyone complies" in system
    for citation in expected_citations(None):
        assert f"Source: {citation}" in human
    assert "Date asked: none" in human


# ── what is printed ─────────────────────────────────────────────────────


def test_answer_then_a_citation_map_that_matches_the_passages(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model(ANSWER), capsys=capsys)

    assert code == 0
    assert out.index(ANSWER) < out.index("Sources:")
    assert sources_map(out) == [f"[{n}] {c}" for n, c in enumerate(expected_citations(None), start=1)]
    # Every number the answer cites is in the map.
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", ANSWER)}
    assert cited <= set(range(1, len(sources_map(out)) + 1))
    assert "Left out: 1 hit(s) from acts other than DORA." in out
    assert "LexAPI credits used for this call: 5. Credits remaining: 495." in out
    assert "not legal advice" in out


def test_withheld_count_and_credits_are_printed(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model(ANSWER_AS_OF), "--as-of", AS_OF, capsys=capsys)

    assert code == 0
    assert (
        f"Withheld for {AS_OF}: 1 passage(s) arrived as a notice without text "
        "(LexAPI did not establish their text for that date)."
    ) in out
    citations = expected_citations(AS_OF)
    assert sources_map(out) == [
        f"[1] {citations[0]}",
        f"[2] {citations[1]} - notice only, no text was given to the model",
    ]
    assert "LexAPI credits used for this call: 5. Credits remaining: 495." in out
    assert "UNRESOLVED-LATEST" not in out


def test_credits_missing_from_the_response_print_as_not_reported(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = search_reply(None)
        del payload["credits"]
        return 200, payload

    code, out, _ = run(Stub(reply=reply), fake_model(ANSWER), capsys=capsys)
    assert code == 0
    assert "LexAPI credits used for this call: not reported. Credits remaining: not reported." in out


def test_no_dora_passages_means_no_model_call(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = search_reply(None)
        payload["results"] = [hit for hit in payload["results"] if hit["law_id"] == "32099L0009"]
        return 200, payload

    model = fake_model(ANSWER)
    code, out, _ = run(Stub(reply=reply), model, capsys=capsys)
    assert code == 0
    assert model.seen == []
    assert "No DORA passages were found for this question, so no model was called." in out
    assert "Sources:" not in out
    assert "Left out: 1 hit(s) from acts other than DORA." in out


def test_only_notices_means_no_model_call(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = search_reply(body["asOf"])
        for hit in payload["results"]:
            hit["asOfResolved"] = False
            hit["asOfReason"] = "fetch_timeout"
        return 200, payload

    model = fake_model(ANSWER)
    code, out, _ = run(Stub(reply=reply), model, "--as-of", AS_OF, capsys=capsys)
    assert code == 0
    assert model.seen == []
    assert f"None of the passages has text for {AS_OF}, so no model was called." in out
    assert f"Withheld for {AS_OF}: 2 passage(s)" in out


def test_control_characters_from_the_model_are_removed(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model("Placeholder \x1b[31mred\x1b[0m and ‮flip [1]."), capsys=capsys)
    assert code == 0
    assert "\x1b" not in out and "‮" not in out


# ── unreadable answers ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (AIMessage(content=""), "The model returned no text."),
        (AIMessage(content="   \n "), "The model returned no text."),
        (
            AIMessage(content=[{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]),
            "The model returned no text.",
        ),
        (AIMessage(content="Placeholder [1][3]."), "The answer cites [3]; the passages are numbered 1 to 2."),
        (AIMessage(content="Placeholder [0, 2]."), "The answer cites [0]; the passages are numbered 1 to 2."),
        (AIMessage(content="Partial", response_metadata={"stop_reason": "max_tokens"}), "cut off"),
        (AIMessage(content="No", response_metadata={"stop_reason": "refusal"}), "declined"),
    ],
)
def test_an_unreadable_answer_raises(stub: Stub, reply: AIMessage, message: str) -> None:
    with pytest.raises(rag.AnswerError, match=re.escape(message)):
        rag.main([QUESTION], transport=stub.transport(), chat_model=fake_model(reply), out=io.StringIO())


def test_read_answer_refuses_a_non_message() -> None:
    with pytest.raises(rag.AnswerError, match="not a message"):
        rag.read_answer("plain string", 2)


# ── errors and the key ──────────────────────────────────────────────────

CREDITS_EXHAUSTED = {
    "success": False,
    "error": {
        "code": "CREDITS_EXHAUSTED",
        "message": "Credit pool exhausted",
        "details": {"operation": "semantic_legislation", "weight": 5, "available": 2},
    },
}
PAYMENT_PAST_DUE = {
    "success": False,
    "code": "PAYMENT_PAST_DUE",
    "error": "Payment past due",
    "message": "Your subscription payment failed, so API requests are refused.",
    "billingUrl": "https://lex-api.com/dashboard#billing",
}


@pytest.mark.parametrize(
    ("status", "body", "printed"),
    [
        (402, CREDITS_EXHAUSTED, "LexAPI refused the request (HTTP 402): Credit pool exhausted"),
        (
            403,
            PAYMENT_PAST_DUE,
            "LexAPI refused the request (HTTP 403): Payment past due: Your subscription payment failed, so API requests"
            " are refused.\nBilling: https://lex-api.com/dashboard#billing",
        ),
    ],
)
def test_402_and_403_print_the_api_message_and_exit_non_zero(
    status: int, body: dict[str, Any], printed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = Stub(reply=lambda _body: (status, body))
    model = fake_model(ANSWER)
    code, out, err = run(stub, model, capsys=capsys)

    assert code == 1
    assert printed in err
    assert "Answer" not in out
    assert model.seen == []
    assert len(stub.requests) == 1  # not retried


def test_the_key_is_never_printed(capsys: pytest.CaptureFixture[str]) -> None:
    outputs = []
    for stub, args in [
        (Stub(), ()),
        (Stub(), ("--as-of", AS_OF)),
        (Stub(reply=lambda _b: (402, CREDITS_EXHAUSTED)), ()),
        (Stub(reply=lambda _b: (403, PAYMENT_PAST_DUE)), ()),
        (Stub(reply=lambda _b: (500, {"success": False, "error": {"code": "INTERNAL", "message": "x"}})), ()),
    ]:
        outputs.append(run(stub, fake_model(ANSWER_AS_OF if args else ANSWER), *args, capsys=capsys))
        assert stub.requests[0].headers["x-api-key"] == TEST_KEY  # the key was used ...
    for _code, out, err in outputs:
        assert TEST_KEY not in out + err  # ... and never printed
        assert "lex_offline_test_key" not in out + err


def test_a_missing_lexapi_key_stops_with_a_message(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    with pytest.raises(SystemExit, match="LEXAPI_API_KEY is not set"):
        rag.main([QUESTION], transport=stub.transport(), chat_model=fake_model(ANSWER))
    assert stub.requests == []


# ── the model ───────────────────────────────────────────────────────────


def test_the_model_is_anthropic_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set"):
        rag.create_chat_model()

    model_key = "sk-ant-offline-test-not-a-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", model_key)
    model, label = rag.create_chat_model()
    assert label == "anthropic:claude-sonnet-5-5"
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5-5"
    assert model_key not in repr(model) and model_key not in str(model)

    monkeypatch.setenv("ANTHROPIC_MODEL", "some-other-model")
    model, label = rag.create_chat_model()
    assert label == "anthropic:some-other-model"
    assert isinstance(model, ChatAnthropic)
    assert model.model == "some-other-model"


# ── offline guards and the dry run ──────────────────────────────────────


def test_a_replaced_transport_needs_a_loopback_base_url(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_BASE_URL", "https://api.example.invalid/api/v1")
    with pytest.raises(RuntimeError, match="must start with http://127.0.0.1:"):
        rag.create_client(stub.transport())
    monkeypatch.delenv("LEXAPI_BASE_URL")
    with pytest.raises(RuntimeError, match="must start with http://127.0.0.1:"):
        rag.create_client(stub.transport())


def test_the_real_network_is_blocked() -> None:
    # The SDK's default base URL is the real API: with the real transport it must fail here.
    with LexAPI(TEST_KEY, max_retries=0) as client:
        with pytest.raises(Exception, match=OFFLINE_ERROR):
            client.semantic_legislation_search("x", limit=1)
    with pytest.raises(RuntimeError, match=OFFLINE_ERROR):
        httpx.get(BASE_URL)


def test_the_retriever_is_closed_after_the_run(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[LexAPILegislationRetriever] = []
    real = LexAPILegislationRetriever

    def recording(**kwargs: Any) -> LexAPILegislationRetriever:
        retriever = real(**kwargs)
        made.append(retriever)
        return retriever

    monkeypatch.setattr(rag, "LexAPILegislationRetriever", recording)
    rag.main([QUESTION], transport=stub.transport(), chat_model=fake_model(ANSWER), out=io.StringIO())
    assert len(made) == 1
    assert made[0].celex == rag.DORA_CELEX and made[0].k == rag.K
    assert made[0].unresolved == "withhold" and made[0].emit_withheld is True
    with pytest.raises(Exception, match="is closed"):
        made[0].invoke("x")


@pytest.mark.parametrize("args", [(), ("--as-of", AS_OF)])
def test_the_dry_run_needs_no_keys(
    args: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    out = io.StringIO()
    assert rag.main(["--dry-run", *args, QUESTION], out=out) == 0
    text = out.getvalue()
    assert text.startswith(rag.DRY_RUN_BANNER)
    assert "Model: scripted (no model was called)" in text
    assert sources_map(text) and "LexAPI credits used for this call: 5." in text
    assert "lex_dry_run_no_key" not in text + capsys.readouterr().err
