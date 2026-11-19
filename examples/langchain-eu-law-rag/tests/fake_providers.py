"""Local stand-ins for the Anthropic and OpenAI HTTP APIs, on 127.0.0.1 only.

The real provider client classes send their requests here, so a test covers how the
client builds the request and how its answer is parsed, without reaching a provider.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: ``(status, body)``: a body that is not a dict is sent as text/html.
Fault = tuple[int, Any]


@dataclass
class FakeProviders:
    """Anthropic ``POST /v1/messages`` and OpenAI ``POST /v1/chat/completions``.

    ``reply(tool_name)`` gives the tool input to return when the request offers a tool;
    without a tool, ``text`` is the answer. ``fault`` answers every request with an error.
    """

    reply: Callable[[str], Any] = lambda name: {}
    text: str = ""
    fault: Fault | None = None
    requests: list[tuple[str, dict[str, Any], dict[str, str]]] = field(default_factory=list)
    base_url: str = ""

    def start(self) -> FakeProviders:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server's naming
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
                fake.requests.append((self.path, body, {k.lower(): v for k, v in self.headers.items()}))
                if fake.fault is not None:
                    status, payload = fake.fault
                    self._send(status, payload)
                elif self.path.endswith("/messages"):
                    self._send(200, fake._anthropic(body))
                elif self.path.endswith("/chat/completions"):
                    self._send(200, fake._openai(body))
                else:
                    self._send(404, {"error": {"message": f"no stand-in for {self.path}"}})

            def _send(self, status: int, payload: Any) -> None:
                raw = json.dumps(payload).encode() if isinstance(payload, dict) else str(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json" if isinstance(payload, dict) else "text/html")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _anthropic(self, body: dict[str, Any]) -> dict[str, Any]:
        tools = body.get("tools") or []
        if tools:
            name = tools[0]["name"]
            content: list[dict[str, Any]] = [
                {"type": "tool_use", "id": "toolu_fake", "name": name, "input": self.reply(name)}
            ]
            stop = "tool_use"
        else:
            content, stop = [{"type": "text", "text": self.text}], "end_turn"
        return {
            "id": "msg_fake",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "fake"),
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def _openai(self, body: dict[str, Any]) -> dict[str, Any]:
        tools = body.get("tools") or []
        message: dict[str, Any] = {"role": "assistant", "content": self.text if not tools else None}
        if tools:
            name = tools[0]["function"]["name"]
            arguments = json.dumps(self.reply(name))
            message["tool_calls"] = [
                {"id": "call_fake", "type": "function", "function": {"name": name, "arguments": arguments}}
            ]
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake"),
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tools else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
