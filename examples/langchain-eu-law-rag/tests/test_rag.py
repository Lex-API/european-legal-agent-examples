from __future__ import annotations

import io
import os
import re
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from conftest import BASE_URL, OFFLINE_ERROR, TEST_KEY, RecordingChatModel, Stub, fake_model, search_reply
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_lexapi import LexAPILegislationRetriever, RetrievalReport
from lexapi import LexAPI

from dora_rag import cli, offline, rag

QUESTION = "What does the act say about reviewing the framework?"
AS_OF = "2025-01-17"
ANSWER = "The first passage describes a placeholder review [1]. The second lists placeholder elements [2]."
ANSWER_AS_OF = "The first passage describes a placeholder review [1]. There is no text of the second for the date [2]."


def run(stub: Stub, model: RecordingChatModel, *args: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    code = cli.main([*args, QUESTION], transport=stub.transport(), chat_model=model, out=out)
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


def without_text(body: dict[str, Any]) -> tuple[int, Any]:
    """Hits with no text and no date: the passages become "[No text returned ...]" notices."""
    payload = search_reply(None)
    for hit in payload["results"]:
        del hit["text"]
    return 200, payload


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
        cli.main(["--as-of", value, QUESTION], transport=stub.transport(), chat_model=model)
    assert exit_info.value.code == 2
    assert "expected a date written YYYY-MM-DD" in capsys.readouterr().err
    assert stub.requests == []
    assert model.seen == []


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_an_empty_question_is_refused_before_any_request(
    stub: Stub, question: str, capsys: pytest.CaptureFixture[str]
) -> None:
    model = fake_model(ANSWER)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([question], transport=stub.transport(), chat_model=model)
    assert exit_info.value.code == 2
    assert "the question is empty" in capsys.readouterr().err
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
    notice = f"[No text: CELEX 32022R2554 art_6 is not established as the text for {AS_OF} (reason: fetch_timeout).]"
    assert notice in human
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
    assert f'reply "{rag.NO_ANSWER}" and nothing else' in system
    assert '"[No text: CELEX ... is not established as the text for <date> ...]"' in system
    assert '"[No text returned for CELEX ...]"' in system
    assert "There is nothing to quote or summarise in it" in system
    assert "say that there is no text of that provision for the date" in system
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
    assert "Answer (written by the model from the passages given to it (not printed; see --show-passages)):" in out
    assert "Passages given to the model" not in out
    assert "Placeholder provision" not in out  # no passage text is printed by default
    assert sources_map(out) == [f"[{n}] {c}" for n, c in enumerate(expected_citations(None), start=1)]
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", ANSWER)}
    assert cited <= set(range(1, len(sources_map(out)) + 1))
    assert "Left out: 1 hit(s) from acts other than DORA." in out
    assert "LexAPI credits used for this call: 5. Credits remaining: 495." in out
    assert "not legal advice" in out


def test_show_passages_prints_what_the_model_was_given(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    model = fake_model(ANSWER_AS_OF)
    code, out, _ = run(stub, model, "--show-passages", "--as-of", AS_OF, capsys=capsys)
    _system, human = prompt_text(model)

    assert code == 0
    passages = human.split("Passages:\n\n", 1)[1].split("\n\nQuestion:", 1)[0]
    assert f"Passages given to the model:\n\n{passages}\n" in out
    assert "Answer (written by the model from the passages above):" in out
    assert "UNRESOLVED-LATEST" not in out


def test_withheld_count_reason_and_credits_are_printed(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model(ANSWER_AS_OF), "--as-of", AS_OF, capsys=capsys)

    assert code == 0
    assert (
        f"Withheld for {AS_OF}: 1 passage(s) arrived as a notice without text "
        "(LexAPI did not establish their text for that date).\n" + cli.NOTICE_MEANING
    ) in out
    citations = expected_citations(AS_OF)
    assert sources_map(out) == [
        f"[1] {citations[0]}",
        f"[2] {citations[1]} - notice only, no text was given to the model (reason: fetch_timeout)",
    ]
    assert "LexAPI credits used for this call: 5. Credits remaining: 495." in out
    assert "UNRESOLVED-LATEST" not in out


def test_strings_from_the_api_and_the_environment_are_sanitised(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = search_reply(body["asOf"])
        payload["results"][1]["asOfReason"] = "fetch\x1b[2J_timeout\u202e"
        return 200, payload

    code, out, _ = run(Stub(reply=reply), fake_model(ANSWER_AS_OF), "--as-of", AS_OF, capsys=capsys)
    assert code == 0
    assert "\x1b" not in out and "\u202e" not in out
    assert "(reason: fetch [2J_timeout)" in out

    label = cli.render(
        question="q",
        as_of=None,
        documents=[],
        report=RetrievalReport(requested=10, returned=0),
        answer_text=None,
        model_label="anthropic:evil\x1b]0;title\x07",
        show_passages=False,
    )
    assert "\x1b" not in label and "\x07" not in label


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


def test_only_notices_for_the_date_means_no_model_call(capsys: pytest.CaptureFixture[str]) -> None:
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
    assert "None of the passages has text, so no model was called." in out
    assert f"Withheld for {AS_OF}: 2 passage(s)" in out


def test_hits_without_text_and_without_a_date_mean_no_model_call(capsys: pytest.CaptureFixture[str]) -> None:
    model = fake_model(ANSWER)
    code, out, _ = run(Stub(reply=without_text), model, capsys=capsys)
    assert code == 0
    assert model.seen == []
    assert "None of the passages has text, so no model was called." in out
    assert sources_map(out) == [f"[{n}] {c} - no text returned" for n, c in enumerate(expected_citations(None), 1)]


def test_one_passage_with_text_is_enough_for_a_model_call(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        status, payload = without_text(body)
        payload["results"][0]["text"] = "[SYNTHETIC FIXTURE - not the text of the act] Placeholder provision."
        return status, payload

    model = fake_model(ANSWER)
    code, _out, _ = run(Stub(reply=reply), model, capsys=capsys)
    assert code == 0
    _system, human = prompt_text(model)
    assert "[No text returned for CELEX 02022R2554-20990101 art_6.]" in human


def test_control_characters_from_the_model_are_removed(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model("Placeholder \x1bcred\x07 and \u202eflip [1]."), capsys=capsys)
    assert code == 0
    assert "\x1b" not in out and "\u202e" not in out


# ── the citation check and unusable replies ─────────────────────────────


@pytest.mark.parametrize(
    "reply",
    [
        "Placeholder [1][3].",
        "Placeholder [0].",
        "Placeholder [1-2].",
        "Placeholder [1\u20132].",
        "Placeholder [1; 2].",
        "Placeholder [1, 2].",
        "Placeholder [ 2 ].",
        "Placeholder [2a].",
        "Placeholder [1] \uff3b1\uff3d.",
        "Placeholder [1] \u30101\u3011.",
        "Placeholder [1] [\u0661].",
        "Placeholder [1] [\uff11].",
        "Placeholder [01].",
        "Placeholder [1\u200b].",
        "Placeholder without any citation.",
        "Placeholder [sic] without a numbered citation.",
    ],
)
def test_the_citation_check_is_strict(stub: Stub, reply: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(stub, fake_model(reply), capsys=capsys)
    assert code == 1
    assert "The model's reply was not used:" in err
    assert "Answer" not in out


@pytest.mark.parametrize(
    "reply",
    ["Placeholder [1].", "Placeholder [2][1].", "Placeholder [sic] [2].", rag.NO_ANSWER, f"  {rag.NO_ANSWER}\n"],
)
def test_the_citation_check_accepts_plain_numbers(stub: Stub, reply: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, fake_model(reply), capsys=capsys)
    assert code == 0
    assert reply.strip() in out


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (AIMessage(content=""), "The model returned no text."),
        (AIMessage(content="   \n "), "The model returned no text."),
        (
            AIMessage(content=[{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]),
            "The model returned no text.",
        ),
        (AIMessage(content="Placeholder [3]."), "The answer cites '[3]'; citations must be [1] to [2]."),
        (AIMessage(content="Partial [1]", response_metadata={"stop_reason": "max_tokens"}), "cut off"),
        (AIMessage(content="No [1]", response_metadata={"stop_reason": "refusal"}), "declined"),
    ],
)
def test_an_unusable_reply_exits_1_with_the_credits_and_a_message(
    stub: Stub, reply: AIMessage, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(stub, fake_model(reply), capsys=capsys)
    assert code == 1
    assert out == ""
    lines = err.strip().splitlines()
    assert lines[0] == "LexAPI credits used for this call: 5. Credits remaining: 495."
    assert len(lines) == 2 and lines[1].startswith("The model's reply was not used:") and message in lines[1]
    assert "Traceback" not in err


def test_read_answer_refuses_a_non_message() -> None:
    with pytest.raises(rag.AnswerError, match="not a message"):
        rag.read_answer("plain string", 2)


class FailingModel(RecordingChatModel):
    error: Any = None

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error


def provider_error(kind: str) -> Exception:
    request = httpx2.Request("POST", "http://127.0.0.1:9/v1/messages")  # the anthropic SDK's HTTP client
    if kind == "connection":
        return anthropic.APIConnectionError(request=request)
    status = {"auth": 401, "rate": 429, "overloaded": 529}[kind]
    response = httpx2.Response(status, request=request, json={"error": {"message": f"placeholder {kind}"}})
    return anthropic.APIStatusError(f"placeholder {kind} \x1b[31m{TEST_KEY}", response=response, body=None)


@pytest.mark.parametrize("kind", ["connection", "auth", "rate", "overloaded"])
def test_a_model_provider_error_exits_1_without_a_traceback(
    stub: Stub, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    model = FailingModel(messages=iter([]), error=provider_error(kind))
    code, out, err = run(stub, model, capsys=capsys)
    assert code == 1
    assert out == ""
    lines = err.strip().splitlines()
    assert lines[0] == "LexAPI credits used for this call: 5. Credits remaining: 495."
    assert len(lines) == 2 and lines[1].startswith("The model provider returned an error (")
    assert "Traceback" not in err and "\x1b" not in err and TEST_KEY not in err


def test_any_other_model_failure_exits_1_without_a_traceback(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    model = FailingModel(messages=iter([]), error=ValueError("placeholder failure"))
    code, out, err = run(stub, model, capsys=capsys)
    assert code == 1
    assert err.strip().splitlines()[1] == "The model call failed (ValueError): placeholder failure"


# ── LexAPI errors, configuration and the key ────────────────────────────

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
        (402, CREDITS_EXHAUSTED, "LexAPI refused the request (HTTP 402): CREDITS_EXHAUSTED: Credit pool exhausted\n"),
        (
            403,
            PAYMENT_PAST_DUE,
            "LexAPI refused the request (HTTP 403): PAYMENT_PAST_DUE: Payment past due: Your subscription payment"
            " failed, so API requests are refused.\nBilling: https://lex-api.com/dashboard#billing\n",
        ),
    ],
)
def test_402_and_403_print_the_api_message_and_exit_1(
    status: int, body: dict[str, Any], printed: str, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = Stub(reply=lambda _body: (status, body))
    model = fake_model(ANSWER)
    code, out, err = run(stub, model, capsys=capsys)

    assert code == 1
    assert err == printed
    assert out == ""
    assert model.seen == []
    assert len(stub.requests) == 1  # not retried


def test_an_error_code_from_the_api_is_sanitised(capsys: pytest.CaptureFixture[str]) -> None:
    body = {"success": False, "error": {"code": "ODD\x1b[2J", "message": "placeholder\u202e"}}
    code, _out, err = run(Stub(reply=lambda _b: (400, body)), fake_model(ANSWER), capsys=capsys)
    assert code == 1
    assert "\x1b" not in err and "\u202e" not in err


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


def test_a_missing_lexapi_key_exits_2(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    assert cli.main([QUESTION], transport=stub.transport(), chat_model=fake_model(ANSWER)) == 2
    assert "LEXAPI_API_KEY is not set" in capsys.readouterr().err
    assert stub.requests == []


def test_a_missing_model_key_exits_2(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([QUESTION], transport=stub.transport()) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    assert stub.requests == []


def test_the_model_is_anthropic_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    model_key = "sk-ant-offline-test-not-a-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", model_key)
    model, label = cli.create_chat_model()
    assert label == "anthropic:claude-sonnet-5-5"
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5-5"
    assert model_key not in repr(model) and model_key not in str(model)

    monkeypatch.setenv("ANTHROPIC_MODEL", "some-other-model")
    model, label = cli.create_chat_model()
    assert label == "anthropic:some-other-model"
    assert isinstance(model, ChatAnthropic)
    assert model.model == "some-other-model"


# ── offline guards and the dry run ──────────────────────────────────────


def test_a_replaced_transport_needs_a_loopback_base_url(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_BASE_URL", "https://api.example.invalid/api/v1")
    with pytest.raises(RuntimeError, match="must start with http://127.0.0.1:"):
        cli.create_client(stub.transport())
    monkeypatch.delenv("LEXAPI_BASE_URL")
    with pytest.raises(RuntimeError, match="must start with http://127.0.0.1:"):
        cli.create_client(stub.transport())


def test_the_real_network_is_blocked() -> None:
    # The SDK's default base URL is the real API: with the real transport it must fail here.
    with LexAPI(TEST_KEY, max_retries=0) as client:
        with pytest.raises(Exception, match=OFFLINE_ERROR):
            client.semantic_legislation_search("x", limit=1)
    with pytest.raises(RuntimeError, match=OFFLINE_ERROR):
        httpx.get(BASE_URL)


def test_the_client_is_closed_after_the_run(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[LexAPI] = []
    real = cli.create_client

    def recording(transport: httpx.BaseTransport | None = None) -> LexAPI:
        client = real(transport)
        made.append(client)
        return client

    monkeypatch.setattr(cli, "create_client", recording)
    cli.main([QUESTION], transport=stub.transport(), chat_model=fake_model(ANSWER), out=io.StringIO())
    assert len(made) == 1
    with pytest.raises(RuntimeError, match="closed"):
        made[0].semantic_legislation_search("x", limit=1)


def test_the_retriever_settings(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[LexAPILegislationRetriever] = []

    def recording(**kwargs: Any) -> LexAPILegislationRetriever:
        retriever = LexAPILegislationRetriever(**kwargs)
        made.append(retriever)
        return retriever

    monkeypatch.setattr(cli, "LexAPILegislationRetriever", recording)
    cli.main([QUESTION], transport=stub.transport(), chat_model=fake_model(ANSWER), out=io.StringIO())
    assert len(made) == 1
    assert made[0].celex == rag.DORA_CELEX and made[0].k == rag.K == 10
    assert made[0].unresolved == "withhold" and made[0].emit_withheld is True


@pytest.mark.parametrize("args", [(), ("--as-of", AS_OF)])
def test_the_dry_run_needs_no_keys_and_turns_tracing_off(
    args: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    out = io.StringIO()
    assert cli.main(["--dry-run", *args, QUESTION], out=out) == 0
    assert os.environ["LANGSMITH_TRACING"] == "false" and os.environ["LANGCHAIN_TRACING_V2"] == "false"
    text = out.getvalue()
    assert text.startswith(offline.DRY_RUN_BANNER)
    assert "Model: scripted (no model was called)" in text
    assert sources_map(text) and "LexAPI credits used for this call: 5." in text
    assert "lex_dry_run_no_key" not in text + capsys.readouterr().err


def test_missing_dry_run_fixtures_exit_2_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(offline, "FIXTURES_DIR", tmp_path / "missing")
    assert cli.main(["--dry-run", QUESTION], out=io.StringIO()) == 2
    err = capsys.readouterr().err
    assert "Offline fixtures not found" in err and "Traceback" not in err
