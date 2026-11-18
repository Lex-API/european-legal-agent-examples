from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from lexapi import LexAPI, LexAPIConnectionError, LexAPIError

from gdpr_assistant.llm import AnthropicLlm, LlmError, LlmRequest, OpenAiLlm, ScriptedLlm, create_llm_from_env
from gdpr_assistant.offline import assert_local_base_url

REQUEST = LlmRequest("system", "user", "submit", "d", {"type": "object", "properties": {}, "required": []})


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://lex-api.com/api/v1",
        "http://localhost:3000",
        "http://127.0.0.1.example.com:80",
        "https://127.0.0.1:1",
    ],
)
def test_harness_refuses_any_base_url_that_is_not_a_local_stub(url: str | None) -> None:
    with pytest.raises(RuntimeError, match="Refusing to run"):
        assert_local_base_url(url)


def test_a_client_left_on_the_default_base_url_is_blocked() -> None:
    with (
        LexAPI("lex_offline_test_key", max_retries=0) as client,
        pytest.raises(LexAPIConnectionError, match="Offline tests blocked a connection to 'lex-api.com'"),
    ):
        client.get_info()


def test_stub_answers_404_for_unknown_routes(client: LexAPI) -> None:
    with pytest.raises(LexAPIError, match="no fixture"):
        client.resolve("ECLI:EU:C:2099:901")


class FakeAnthropic:
    """Stands in for the provider client: records requests, returns canned responses. No network."""

    def __init__(self, responses: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self._responses = responses

    def _create(self, **params: Any) -> Any:
        self.calls.append({**params, "messages": list(params["messages"])})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def message(stop_reason: str, *blocks: Any) -> Any:
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


TOOL_USE = SimpleNamespace(type="tool_use", name="submit", input={"a": 1})
TEXT = SimpleNamespace(type="text", text="Here is my answer.")


def test_anthropic_adapter_sends_a_strict_tool_without_forcing_it() -> None:
    fake = FakeAnthropic([message("tool_use", TEXT, TOOL_USE)])
    llm = AnthropicLlm(client=fake)
    assert llm.generate(REQUEST) == {"a": 1}
    assert llm.label == "anthropic:claude-sonnet-5-5"
    call = fake.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert "tool_choice" not in call
    assert call["tools"] == [{"name": "submit", "description": "d", "input_schema": REQUEST.schema, "strict": True}]


def test_anthropic_adapter_reprompts_once_then_gives_up() -> None:
    recovered = FakeAnthropic([message("end_turn", TEXT), message("tool_use", TOOL_USE)])
    assert AnthropicLlm(client=recovered).generate(REQUEST) == {"a": 1}
    assert len(recovered.calls) == 2
    assert len(recovered.calls[1]["messages"]) == 3

    stuck = FakeAnthropic([message("end_turn", TEXT)])
    with pytest.raises(LlmError, match="did not call the submit tool"):
        AnthropicLlm(client=stuck).generate(REQUEST)
    assert len(stuck.calls) == 2


def test_anthropic_adapter_surfaces_refusals_and_truncation() -> None:
    with pytest.raises(LlmError, match="declined"):
        AnthropicLlm(client=FakeAnthropic([message("refusal")])).generate(REQUEST)
    with pytest.raises(LlmError, match="cut off"):
        AnthropicLlm(client=FakeAnthropic([message("max_tokens")])).generate(REQUEST)


def fake_openai(reply: Any, calls: list[dict[str, Any]]) -> Any:
    def create(**params: Any) -> Any:
        calls.append(params)
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def tool_call(arguments: str) -> Any:
    function = SimpleNamespace(name="submit", arguments=arguments)
    return SimpleNamespace(refusal=None, tool_calls=[SimpleNamespace(type="function", function=function)])


def test_openai_adapter_forces_the_function_and_parses_its_arguments() -> None:
    calls: list[dict[str, Any]] = []
    assert OpenAiLlm("test-model", client=fake_openai(tool_call('{"a": 3}'), calls)).generate(REQUEST) == {"a": 3}
    assert calls[0]["tool_choice"] == {"type": "function", "function": {"name": "submit"}}
    assert calls[0]["model"] == "test-model"
    with pytest.raises(LlmError, match="declined"):
        OpenAiLlm("m", client=fake_openai(SimpleNamespace(refusal="no", tool_calls=None), calls)).generate(REQUEST)
    with pytest.raises(LlmError, match="did not call"):
        OpenAiLlm("m", client=fake_openai(SimpleNamespace(refusal=None, tool_calls=None), calls)).generate(REQUEST)
    with pytest.raises(LlmError, match="not valid JSON"):
        OpenAiLlm("m", client=fake_openai(tool_call("{"), calls)).generate(REQUEST)


def test_scripted_llm_and_provider_selection() -> None:
    llm = ScriptedLlm({"submit": [{"n": 1}, {"n": 2}]})
    assert [llm.generate(REQUEST) for _ in range(3)] == [{"n": 1}, {"n": 2}, {"n": 2}]
    with pytest.raises(LlmError):
        ScriptedLlm({}).generate(REQUEST)
    with pytest.raises(LlmError, match="OPENAI_MODEL"):
        create_llm_from_env({"LLM_PROVIDER": "openai"})
    with pytest.raises(LlmError, match="Unknown LLM_PROVIDER"):
        create_llm_from_env({"LLM_PROVIDER": "other"})
