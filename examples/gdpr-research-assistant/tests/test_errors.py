"""A refused LexAPI call: the run stops with one line, the SDK's own text, and exit code 1."""

from __future__ import annotations

from typing import Any

import pytest

from gdpr_assistant import cli
from gdpr_assistant.llm import ScriptedLlm
from gdpr_assistant.offline import RecordedRequest, start_stub_server

QUESTION = "When may personal data be transferred to a third country?"
KEY = "lex_offline_test_key_do_not_print_0123456789"

# The four answers that stop a run for a while, with every field the API sends for them (synthetic).
REFUSALS: dict[str, tuple[int, dict[str, Any], str]] = {
    "credits": (
        402,
        {
            "success": False,
            "error": {
                "code": "CREDITS_EXHAUSTED",
                "message": "Credit pool exhausted",
                "details": {"operation": "semantic_legislation", "weight": 5, "available": 2},
            },
            "resetsAt": "2026-07-01T00:00:00Z",
        },
        "LexAPI error: [CREDITS_EXHAUSTED] Credit pool exhausted (HTTP 402)",
    ),
    "quota": (
        429,
        {
            "success": False,
            "code": "QUOTA_EXHAUSTED",
            "error": "Daily quota exceeded",
            "message": "The daily call quota of this plan is used up.",
            "usage": {"resetsAt": "2026-07-01T00:00:00Z"},
        },
        "LexAPI error: [QUOTA_EXHAUSTED] Daily quota exceeded: The daily call quota of this plan is used up."
        " (HTTP 429)",
    ),
    "past due": (
        403,
        {
            "success": False,
            "code": "PAYMENT_PAST_DUE",
            "error": "Payment past due",
            "message": "Your subscription payment failed, so API requests are refused.",
            "accessEndedAt": "2026-06-20T00:00:00Z",
            "billingUrl": "https://lex-api.com/dashboard#billing",
        },
        "LexAPI error: [PAYMENT_PAST_DUE] Payment past due: Your subscription payment failed, so API requests are"
        " refused. (HTTP 403)",
    ),
    "rate limit": (
        429,
        {"success": False, "error": {"code": "RATE_LIMITED", "message": "Too many requests"}, "retryAfter": 7},
        "LexAPI error: [RATE_LIMITED] Too many requests (HTTP 429)",
    ),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_refusal_prints_the_sdks_text_and_exits_1(
    case: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    status, body, printed = REFUSALS[case]

    def refuse(request: RecordedRequest) -> tuple[int, Any]:
        return status, body

    stub = start_stub_server(refuse)
    monkeypatch.setenv("LEXAPI_BASE_URL", stub.base_url)
    monkeypatch.setenv("LEXAPI_API_KEY", KEY)
    monkeypatch.setattr(cli, "load_env_file", lambda path: None)
    monkeypatch.setattr(cli, "create_llm_from_env", lambda: ScriptedLlm({}))
    monkeypatch.setattr("lexapi.client.time.sleep", lambda seconds: None)  # the SDK retries a rate limit
    try:
        code = cli.main([QUESTION])
    finally:
        stub.close()
    out, err = capsys.readouterr()

    assert code == 1
    assert out == ""
    assert err == printed + "\n"
    # Nothing is added: no wait, no link, no plan advice.
    assert "Retry" not in err and "https:" not in err and "upgrade" not in err.lower()
    assert len(stub.requests) == (4 if case == "rate limit" else 1)  # only a rate limit is retried
    assert stub.requests[0].headers["x-api-key"] == KEY and KEY not in out + err


def test_the_error_line_removes_control_characters_and_masks_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from lexapi import LexAPIError

    monkeypatch.setenv("LEXAPI_API_KEY", KEY)
    error = LexAPIError(f"placeholder\x1b[2J‮ {KEY}\nmore", code="ODD", status=400)
    assert cli.lexapi_error_line(error) == "LexAPI error: [ODD] placeholder [2J [key] more (HTTP 400)"
