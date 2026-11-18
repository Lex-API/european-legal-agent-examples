"""The model call, behind one small interface.

Every model call asks for ONE structured object: the model is given a tool
whose input schema is the object we want, and the adapter returns that tool
input. Nothing else the model writes is used.

- ``ScriptedLlm``  - deterministic, no network. Used by the tests and ``--dry-run``.
- ``AnthropicLlm`` - Anthropic Messages API with tool use.
- ``OpenAiLlm``    - OpenAI Chat Completions with function calling.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"


@dataclass(frozen=True)
class LlmRequest:
    system: str
    user: str
    tool_name: str
    tool_description: str
    schema: dict[str, Any]  # strict JSON Schema: every property required, no extras


class Llm(Protocol):
    label: str

    def generate(self, request: LlmRequest) -> Any:
        """Return the tool input object produced by the model (unvalidated - callers parse it)."""
        ...


class LlmError(Exception):
    pass


class ScriptedLlm:
    """A fake model. Replies are looked up by tool name; a list is consumed one per call (the last repeats)."""

    label = "scripted (no model was called)"

    def __init__(self, replies: Mapping[str, Any]) -> None:
        self._replies = {tool: list(r) if isinstance(r, list) else [r] for tool, r in replies.items()}
        self.calls: list[LlmRequest] = []

    def generate(self, request: LlmRequest) -> Any:
        self.calls.append(request)
        queue = self._replies.get(request.tool_name)
        if not queue:
            raise LlmError(f'ScriptedLlm has no reply for tool "{request.tool_name}"')
        return copy.deepcopy(queue.pop(0) if len(queue) > 1 else queue[0])


class AnthropicLlm:
    """Anthropic Messages API with tool use.

    The tool is declared ``strict``, so its input is guaranteed to match the
    schema. ``tool_choice`` is left at its default (``auto``): current models
    reject a forced tool choice, so the prompt asks for the tool call and a
    reply without one is re-prompted once.
    """

    def __init__(self, model: str | None = None, client: Any = None) -> None:
        if client is None:
            import anthropic  # optional dependency: pip install ".[anthropic]"

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._client = client
        self._model = model or DEFAULT_ANTHROPIC_MODEL
        self.label = f"anthropic:{self._model}"

    def generate(self, request: LlmRequest) -> Any:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"{request.user}\n\nReply by calling the {request.tool_name} tool."}
        ]
        for _attempt in range(2):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=16000,
                system=request.system,
                tools=[
                    {
                        "name": request.tool_name,
                        "description": request.tool_description,
                        "input_schema": request.schema,
                        "strict": True,
                    }
                ],
                messages=messages,
            )
            if response.stop_reason == "refusal":
                raise LlmError("The model declined this request (stop_reason: refusal).")
            if response.stop_reason == "max_tokens":
                raise LlmError("The model's reply was cut off (stop_reason: max_tokens).")
            for block in response.content:
                if block.type == "tool_use" and block.name == request.tool_name:
                    return block.input
            # No tool call: keep the reply in the conversation and ask again.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": f"Call the {request.tool_name} tool with your answer."})
        raise LlmError(f"The model did not call the {request.tool_name} tool.")


class OpenAiLlm:
    """OpenAI Chat Completions with one strict function the model is required to call."""

    def __init__(self, model: str, client: Any = None) -> None:
        if client is None:
            import openai  # optional dependency: pip install ".[openai]"

            client = openai.OpenAI()  # reads OPENAI_API_KEY
        self._client = client
        self._model = model
        self.label = f"openai:{model}"

    def generate(self, request: LlmRequest) -> Any:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": request.tool_name,
                        "description": request.tool_description,
                        "parameters": request.schema,
                        "strict": True,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": request.tool_name}},
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise LlmError(f"The model declined this request: {message.refusal}")
        for call in message.tool_calls or []:
            if call.type == "function" and call.function.name == request.tool_name:
                try:
                    return json.loads(call.function.arguments)
                except json.JSONDecodeError as error:
                    raise LlmError("The model returned function arguments that are not valid JSON.") from error
        raise LlmError(f"The model did not call the {request.tool_name} function.")


def create_llm_from_env(env: Mapping[str, str] | None = None) -> Llm:
    """Pick the real adapter from the environment.

    ``LLM_PROVIDER=anthropic`` (default): ``ANTHROPIC_API_KEY``, ``ANTHROPIC_MODEL`` (default claude-sonnet-5-5).
    ``LLM_PROVIDER=openai``: ``OPENAI_API_KEY``, ``OPENAI_MODEL`` (required).
    """
    env = os.environ if env is None else env
    provider = env.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicLlm(env.get("ANTHROPIC_MODEL"))
    if provider == "openai":
        if not env.get("OPENAI_MODEL"):
            raise LlmError("Set OPENAI_MODEL to the model id to use (there is no default).")
        return OpenAiLlm(env["OPENAI_MODEL"])
    raise LlmError(f'Unknown LLM_PROVIDER "{provider}" (use "anthropic" or "openai").')
