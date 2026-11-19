"""What the command line prints when a run cannot finish: LexAPI errors in the adapters' format, missing keys,
model errors and interruptions - in process, and as a real process against the local stub."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from lexapi import LexAPIError

from gdpr_assistant import cli
from gdpr_assistant.llm import LlmError, LlmRequest, ScriptedLlm
from gdpr_assistant.offline import RecordedRequest, StubServer, read_fixture, start_stub_server

QUESTION = "When may personal data be transferred to a third country?"
KEY = "lex_offline_test_key_do_not_print_0123456789"
DASHBOARD = "Dashboard: https://lex-api.com/dashboard?ref=github-examples"
EXAMPLE = Path(__file__).resolve().parents[1]

# The four answers that stop a run for a while (synthetic values) and the text the adapters print for each.
BODIES: dict[str, tuple[int, dict[str, Any], dict[str, str]]] = {
    "credits": (
        402,
        {
            "success": False,
            "error": {
                "code": "CREDITS_EXHAUSTED",
                "message": "Credit pool exhausted",
                "details": {"operation": "semantic_legislation", "weight": 5, "available": 2},
            },
            "resetsAt": "2099-07-01T00:00:00.000Z",
        },
        {},
    ),
    "quota": (
        429,
        {
            "error": "Daily quota exhausted",
            "code": "QUOTA_EXHAUSTED",
            "usage": {"resetsAt": "2099-07-02T00:00:00.000Z"},
        },
        {},
    ),
    "past due": (
        403,
        {
            "code": "PAYMENT_PAST_DUE",
            "error": "Payment past due",
            "message": "Your subscription payment failed, so API requests are refused.",
            "accessEndedAt": "2099-06-25T09:14:03.000Z",
            "billingUrl": "https://lex-api.com/dashboard#billing",
        },
        {},
    ),
    "rate limit": (
        429,
        {"error": "Rate limit exceeded", "code": "RATE_LIMITED", "retryAfter": 7},
        {"retry-after": "7", "ratelimit-remaining": "0", "ratelimit-reset": "7"},
    ),
}
TEXTS = {
    "credits": (
        "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted. Credits reset at 2099-07-01T00:00:00.000Z."
    ),
    "quota": (
        "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily quota exhausted. This is an exhausted quota, not a rate limit."
        " Quota resets at 2099-07-02T00:00:00.000Z."
    ),
    "past due": (
        "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Your subscription payment failed, so API requests"
        " are refused. Access ended at 2099-06-25T09:14:03.000Z. Billing: https://lex-api.com/dashboard#billing"
    ),
    "rate limit": "LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded. Retry after 7 s.",
}
LINES = {
    "credits": [TEXTS["credits"], DASHBOARD],
    "quota": [TEXTS["quota"]],
    "past due": [TEXTS["past due"]],
    "rate limit": [TEXTS["rate limit"]],
}


def faulty_stub(reply: tuple[int, Any, dict[str, str]] | None, nth: int = 1) -> StubServer:
    """A stub whose ``nth`` request (1-based) and every later one get ``reply``."""
    holder: list[StubServer] = []

    def override(request: RecordedRequest) -> tuple[int, Any, dict[str, str]] | None:
        return reply if reply is not None and len(holder[0].requests) >= nth else None

    holder.append(start_stub_server(override))
    return holder[0]


@pytest.fixture
def real_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-offline-test-not-a-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    monkeypatch.setattr(cli, "create_llm_from_env", lambda: ScriptedLlm(json.loads(read_fixture("llm/gdpr.json"))))
    monkeypatch.setattr("lexapi.client.time.sleep", lambda seconds: None)  # the SDK retries a rate limit


def run_main(
    stub: StubServer, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    monkeypatch.setenv("LEXAPI_BASE_URL", stub.base_url)
    try:
        code = cli.main([QUESTION])
    finally:
        stub.close()
    out, err = capsys.readouterr()
    return code, out, err


# ── in process ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", list(BODIES))
@pytest.mark.usefixtures("real_env")
def test_the_four_answers_print_the_adapters_text_and_exit_1(
    case: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub = faulty_stub(BODIES[case])
    code, out, err = run_main(stub, monkeypatch, capsys)
    assert code == 1
    assert out == ""
    assert err.splitlines() == LINES[case]
    assert "upgrade" not in err.lower()  # a failed payment is not a plan to upgrade
    assert len(stub.requests) == (4 if case == "rate limit" else 1)  # only a rate limit is retried
    assert KEY not in out + err


@pytest.mark.usefixtures("real_env")
def test_a_refusal_after_a_charged_call_prints_its_credits_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_main(faulty_stub(BODIES["credits"], nth=2), monkeypatch, capsys)
    assert code == 1 and out == ""
    assert err.splitlines() == ["LexAPI credits reported for this run: 5.", *LINES["credits"]]


@pytest.mark.parametrize(
    ("status", "body", "printed"),
    [
        (401, {"error": "Invalid API key"}, ["LexAPI error UNAUTHORIZED (HTTP 401): Invalid API key.", DASHBOARD]),
        (
            500,
            {"success": False, "error": {"code": "INTERNAL", "message": "x"}},
            ["LexAPI error INTERNAL (HTTP 500): x."],
        ),
        (
            504,
            {"success": False, "error": {"code": "TIMEOUT", "message": "Upstream fetch timed out"}},
            ["LexAPI error TIMEOUT (HTTP 504): Upstream fetch timed out."],
        ),
        (200, {"success": True}, ["LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read."]),
    ],
)
@pytest.mark.usefixtures("real_env")
def test_other_failures_are_one_line_too(
    status: int, body: Any, printed: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_main(faulty_stub((status, body, {})), monkeypatch, capsys)
    assert code == 1 and out == ""
    assert err.splitlines() == printed


@pytest.mark.usefixtures("real_env")
def test_a_model_error_is_one_line_after_the_credits_of_the_searches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Failing:
        label = "failing"

        def generate(self, request: LlmRequest) -> Any:
            raise LlmError(f"invalid x-api-key {KEY}\nmore")

    monkeypatch.setattr(cli, "create_llm_from_env", Failing)
    code, out, err = run_main(faulty_stub(None), monkeypatch, capsys)
    assert code == 1 and out == ""
    assert err.splitlines() == [
        "LexAPI credits reported for this run: 20.",
        "Model error: invalid x-api-key lex_*** more",
    ]


@pytest.mark.parametrize("missing", ["LEXAPI_API_KEY", "ANTHROPIC_API_KEY", "both"])
@pytest.mark.usefixtures("real_env")
def test_missing_keys_exit_2_before_any_request(
    missing: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("LEXAPI_API_KEY", "ANTHROPIC_API_KEY") if missing == "both" else (missing,):
        monkeypatch.setenv(name, "  ")
    stub = faulty_stub(None)
    code, out, err = run_main(stub, monkeypatch, capsys)
    assert code == 2 and out == ""
    expected = {
        "LEXAPI_API_KEY": [cli.LEXAPI_KEY_MISSING],
        "ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run."],
    }
    assert err.splitlines() == (
        [*expected["LEXAPI_API_KEY"], *expected["ANTHROPIC_API_KEY"]] if missing == "both" else expected[missing]
    )
    assert stub.requests == []


@pytest.mark.usefixtures("real_env")
def test_ctrl_c_exits_130_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "answer_question", interrupt)
    code, out, err = run_main(faulty_stub(None), monkeypatch, capsys)
    assert code == 130
    assert "Traceback" not in err and out == ""


def test_the_format_caps_and_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", KEY)
    error = LexAPIError(f"line\x1b[2J\u202e {KEY}\n{'x' * 2000}", code="ODD", status=400)
    text = cli.lexapi_error_text(error)
    assert "\x1b" not in text and "\u202e" not in text and "\n" not in text and KEY not in text
    assert text.startswith("LexAPI error ODD (HTTP 400): line [2J lex_*** x")
    assert text.endswith("x….")
    connection = LexAPIError("no route", code="CONNECTION_ERROR", status=None)
    assert cli.lexapi_error_text(connection) == "LexAPI error CONNECTION_ERROR: no route."


# ── as a process ────────────────────────────────────────────────────────

# Run inside the child before the command line: nothing but 127.0.0.1 can be reached.
GUARD = """
import socket, sys
_connect, _resolve = socket.socket.connect, socket.getaddrinfo
def connect(self, address):
    if (address[0] if isinstance(address, tuple) else address) != "127.0.0.1":
        raise ConnectionRefusedError("offline test")
    return _connect(self, address)
def resolve(host, *args, **kwargs):
    if host != "127.0.0.1":
        raise socket.gaierror("offline test")
    return _resolve(host, *args, **kwargs)
socket.socket.connect, socket.getaddrinfo = connect, resolve
from gdpr_assistant.cli import main
raise SystemExit(main(sys.argv[1:]))
"""


class FakeModel:
    """A local stand-in for the Anthropic Messages API: it calls the requested tool with the scripted input."""

    def __init__(self) -> None:
        replies = json.loads(read_fixture("llm/gdpr.json"))
        self.requests = 0
        model = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server's naming
                model.requests += 1
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
                name = body["tools"][0]["name"]
                payload = json.dumps(
                    {
                        "id": "msg_fake",
                        "type": "message",
                        "role": "assistant",
                        "model": body.get("model", "fake"),
                        "content": [{"type": "tool_use", "id": "toolu_fake", "name": name, "input": replies[name]}],
                        "stop_reason": "tool_use",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_model() -> Iterator[FakeModel]:
    model = FakeModel()
    yield model
    model.close()


def run_process(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    # A clean environment, with an empty home: no model-provider configuration of this machine is read.
    with tempfile.TemporaryDirectory() as home:
        child_env = {"PATH": os.environ.get("PATH", ""), "HOME": home, **env}
        return subprocess.run(
            [sys.executable, "-c", GUARD, QUESTION], env=child_env, capture_output=True, text=True, timeout=120
        )


def process_env(stub: StubServer, model: FakeModel) -> dict[str, str]:
    return {
        "LEXAPI_API_KEY": KEY,
        "LEXAPI_BASE_URL": stub.base_url,
        "ANTHROPIC_API_KEY": "sk-fake-offline-test-not-a-key",
        "ANTHROPIC_BASE_URL": model.base_url,
    }


def test_the_process_answers_against_the_stand_ins(fake_model: FakeModel) -> None:
    stub = faulty_stub(None)
    try:
        done = run_process(process_env(stub, fake_model))
    finally:
        stub.close()
    assert done.returncode == 0, done.stderr
    assert "## Quoted from the sources" in done.stdout
    assert done.stderr.strip() == "LexAPI credits reported for this run: 20."
    assert fake_model.requests >= 1


@pytest.mark.parametrize("case", ["credits", "past due", "quota"])
def test_a_refusal_as_a_process_stderr_only_exit_1_no_traceback(case: str, fake_model: FakeModel) -> None:
    stub = faulty_stub(BODIES[case])
    try:
        done = run_process(process_env(stub, fake_model))
    finally:
        stub.close()
    assert done.returncode == 1
    assert done.stdout == ""
    assert done.stderr.splitlines() == LINES[case]
    assert "Traceback" not in done.stderr
    assert len(stub.requests) == 1 and fake_model.requests == 0


def test_a_rate_limit_as_a_process(fake_model: FakeModel) -> None:
    status, body, _headers = BODIES["rate limit"]
    stub = faulty_stub((status, {**body, "retryAfter": 0}, {"retry-after": "0", "ratelimit-reset": "0"}))
    try:
        done = run_process(process_env(stub, fake_model))
    finally:
        stub.close()
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.splitlines() == ["LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded."]
    assert len(stub.requests) == 4


@pytest.mark.skipif((EXAMPLE / ".env").exists(), reason="this example has its own .env")
def test_missing_keys_as_a_process_exit_2_with_no_request(fake_model: FakeModel) -> None:
    stub = faulty_stub(None)
    try:
        done = run_process({"LEXAPI_BASE_URL": stub.base_url, "ANTHROPIC_BASE_URL": fake_model.base_url})
    finally:
        stub.close()
    assert done.returncode == 2 and done.stdout == ""
    assert done.stderr.splitlines() == [
        cli.LEXAPI_KEY_MISSING,
        "ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run.",
    ]
    assert stub.requests == [] and fake_model.requests == 0
