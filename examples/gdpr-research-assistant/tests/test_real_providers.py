"""The real model clients: ``anthropic.Anthropic`` and ``openai.OpenAI`` build and send the request to a local
stand-in for each API on 127.0.0.1, and the answer is parsed; provider errors read as the examples print them."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_providers import FakeProviders

from gdpr_assistant import cli
from gdpr_assistant.llm import LlmError, LlmRequest, create_llm_from_env, provider_message

REQUEST = LlmRequest(
    system="s",
    user="u",
    tool_name="submit_answer",
    tool_description="d",
    schema={"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]},
)
ENVS = {
    "anthropic": {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-fake-offline-test-not-a-key"},
    "openai": {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-fake-offline-test-not-a-key", "OPENAI_MODEL": "fake"},
}


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeProviders]:
    fake = FakeProviders(reply=lambda name: {"status": "no_support"}).start()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", fake.base_url)
    monkeypatch.setenv("OPENAI_BASE_URL", f"{fake.base_url}/v1")
    monkeypatch.setenv("HOME", str(Path(os.environ.get("TMPDIR", "/tmp"))))  # no machine credentials are read
    yield fake
    fake.close()


@pytest.mark.parametrize("provider", list(ENVS))
def test_the_real_client_sends_the_request_and_its_answer_is_parsed(provider: str, providers: FakeProviders) -> None:
    llm = create_llm_from_env(ENVS[provider])
    assert llm.generate(REQUEST) == {"status": "no_support"}
    path, body, headers = providers.requests[0]
    assert path == ("/v1/messages" if provider == "anthropic" else "/v1/chat/completions")
    sent = headers.get("x-api-key") or headers.get("authorization", "").removeprefix("Bearer ")
    assert sent == "sk-fake-offline-test-not-a-key"
    tools = body["tools"]
    assert (tools[0].get("name") or tools[0]["function"]["name"]) == "submit_answer"


@pytest.mark.parametrize("provider", list(ENVS))
@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ((401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}), "auth"),
        ((400, "<html>bad gateway</html>"), "HTTP 400: <html>bad gateway</html>"),
    ],
)
def test_a_provider_error_is_its_message_or_its_status(
    provider: str, fault: tuple[int, Any], message: str, providers: FakeProviders
) -> None:
    providers.fault = fault
    with pytest.raises(LlmError) as raised:
        create_llm_from_env(ENVS[provider]).generate(REQUEST)
    variable = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    expected = f"invalid x-api-key. Check {variable}." if message == "auth" else message
    assert str(raised.value) == expected
    assert cli.model_error_line(raised.value) == f"Model error: {expected}"


def test_provider_message_never_returns_a_repr() -> None:
    class WithBody(Exception):
        body = {"type": "error", "error": {"type": "x", "message": "the provider's words"}}

    class WithoutMessage(Exception):
        body = {"type": "error", "error": {"type": "x"}}
        status_code = 500

    assert provider_message(WithBody("Error code: 400 - {'type': 'error'}")) == "the provider's words"
    text = provider_message(WithoutMessage("Error code: 500 - {'type': 'error', 'error': {'type': 'x'}}"))
    assert text == "HTTP 500: no message" and "{" not in text


@pytest.mark.parametrize(("value", "loaded"), [("1", False), ("0", True), ("", True), ("true", True), (" 1", True)])
def test_the_dotenv_test_hook_is_honoured_only_for_the_exact_value_1(
    value: str, loaded: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text("EXAMPLES_DOTENV_PROBE=loaded\n", encoding="utf-8")
    monkeypatch.delenv("EXAMPLES_DOTENV_PROBE", raising=False)
    monkeypatch.setenv(cli.TEST_IGNORE_DOTENV, value)
    cli.load_env_file(env)
    assert (os.environ.get("EXAMPLES_DOTENV_PROBE") == "loaded") is loaded
    monkeypatch.delenv("EXAMPLES_DOTENV_PROBE", raising=False)
