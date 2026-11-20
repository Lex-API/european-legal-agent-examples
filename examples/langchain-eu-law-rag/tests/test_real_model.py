"""The real model path: ``ChatAnthropic`` (langchain-anthropic and the anthropic client) builds and sends the
request to a local stand-in for the Anthropic API on 127.0.0.1, and its answer is parsed."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import Stub
from fake_providers import FakeProviders

from dora_rag import cli

QUESTION = "What does the act say about reviewing the framework?"
ANSWER = "The first passage describes a placeholder review [1]."


def run(stub: Stub, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    code = cli.main([QUESTION], transport=stub.transport(), out=out)
    return code, out.getvalue(), capsys.readouterr().err


def test_the_real_chat_model_answers_through_the_stand_in(
    stub: Stub, providers: FakeProviders, capsys: pytest.CaptureFixture[str]
) -> None:
    providers.text = ANSWER
    code, out, err = run(stub, capsys)
    assert code == 0, err
    assert ANSWER in out and "Model: anthropic:claude-sonnet-5-5" in out
    path, body, headers = providers.requests[0]
    assert path == "/v1/messages" and len(providers.requests) == 1
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 2000
    # The key the example read, in the header the client version uses (x-api-key, or a bearer authorization).
    sent = headers.get("x-api-key") or headers.get("authorization", "").removeprefix("Bearer ")
    assert sent == "sk-fake-offline-test-not-a-key"


@pytest.mark.parametrize(
    ("fault", "printed"),
    [
        (
            (401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}),
            "Model error: invalid x-api-key. Check ANTHROPIC_API_KEY.",
        ),
        ((400, "<html>bad\ngateway</html>"), "Model error: HTTP 400: <html>bad gateway</html>"),
    ],
)
def test_a_provider_error_is_its_message_or_its_status(
    fault: tuple[int, object], printed: str, stub: Stub, providers: FakeProviders, capsys: pytest.CaptureFixture[str]
) -> None:
    providers.fault = fault
    code, out, err = run(stub, capsys)
    assert code == 1 and out == ""
    assert err.splitlines()[-1] == printed
    assert "sk-fake-offline-test-not-a-key" not in err


ERRORS = json.loads(
    (Path(__file__).resolve().parents[3] / "offline" / "fixtures" / "llm" / "provider-errors.json").read_text()
)


@pytest.mark.parametrize("answer", ERRORS["cases"], ids=lambda answer: answer["name"])
def test_a_provider_error_reads_the_same_in_every_example(
    answer: dict[str, Any], stub: Stub, providers: FakeProviders, capsys: pytest.CaptureFixture[str]
) -> None:
    providers.fault = (answer["status"], answer["raw"], answer["contentType"])
    code, out, err = run(stub, capsys)
    assert code == 1 and out == ""
    assert err.splitlines()[-1] == answer["printed"]
