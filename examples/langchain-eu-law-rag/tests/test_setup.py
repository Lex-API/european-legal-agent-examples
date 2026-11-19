"""The .env file, the keys, "today" in UTC and the one-line LexAPI errors."""

from __future__ import annotations

import datetime
import io
import os
import shutil
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import REAL_LOAD_ENV_FILE, REAL_TODAY, Stub, fake_model

from dora_rag import cli

EXAMPLE = Path(__file__).resolve().parents[1]
DASHBOARD = "Dashboard: https://lex-api.com/dashboard?ref=github-examples"
LEXAPI_MISSING = (
    "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples or use --dry-run."
)
MODEL_MISSING = "ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run."
QUESTION = "What does DORA say about ICT risk management frameworks?"
REAL_RUNS = [
    ([QUESTION], [LEXAPI_MISSING, MODEL_MISSING]),
    (["--as-of", "2025-01-17", QUESTION], [LEXAPI_MISSING, MODEL_MISSING]),
]
NAMES = ("LEXAPI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "PLAIN")


def unset(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        monkeypatch.setenv(name, "placeholder")  # restored to its state before the test
        monkeypatch.delenv(name)


def test_export_lines_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unset(monkeypatch, *NAMES)
    env = tmp_path / ".env"
    env.write_text(
        "export LEXAPI_API_KEY=lex_AbC123\nexport\tANTHROPIC_MODEL='some-model'\n# export PLAIN=no\nPLAIN=yes\n",
        encoding="utf-8",
    )
    REAL_LOAD_ENV_FILE(env)
    assert os.environ["LEXAPI_API_KEY"] == "lex_AbC123"
    assert os.environ["ANTHROPIC_MODEL"] == "some-model"
    assert os.environ["PLAIN"] == "yes"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_the_env_example_has_no_key_and_says_where_to_create_one() -> None:
    lines = (EXAMPLE / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "LEXAPI_API_KEY=" in lines and "ANTHROPIC_API_KEY=" in lines
    assert any(line.startswith("#") and "https://lex-api.com/dashboard?ref=github-examples" in line for line in lines)


@pytest.mark.parametrize(("args", "printed"), REAL_RUNS)
def test_an_unfilled_copy_of_the_env_example_stops_every_real_run(
    args: list[str], printed: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    unset(monkeypatch, "LEXAPI_API_KEY", "ANTHROPIC_API_KEY")
    shutil.copy(EXAMPLE / ".env.example", tmp_path / ".env")  # cp .env.example .env
    monkeypatch.setattr(cli, "load_env_file", lambda path: REAL_LOAD_ENV_FILE(tmp_path / ".env"))
    stub = Stub()
    assert cli.main(args, transport=stub.transport(), out=io.StringIO()) == 2
    assert stub.requests == []
    assert capsys.readouterr().err.splitlines() == printed


SHIPPED_PLACEHOLDERS = ["lex_your_key_here"]  # every key value a .env.example of this repository has shipped
REAL_LOOKING_KEYS = [
    "FAKE-KEY-FOR-TESTS-" + "contains-yOuR-by-chance",  # not a key; "your" inside a value is not a placeholder
    "lex_replace" + "a1b2c3d4e5f6",
    "lex_placeholder" + "9f8e7d6c",
    "lex_ab...cd" + "0123456789",
    "lex_live_" + "0123456789abcdef" * 4,
]


@pytest.mark.parametrize(
    "value",
    ["", "   ", *SHIPPED_PLACEHOLDERS, "LEX_YOUR_KEY_HERE", " lex_your_key_here ", "<key>", "<your LexAPI key>"],
)
def test_a_blank_or_placeholder_key_is_not_set(value: str, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", value)
    stub = Stub()
    assert cli.main([QUESTION], transport=stub.transport(), chat_model=fake_model("x [1]")) == 2
    assert stub.requests == []
    assert capsys.readouterr().err == LEXAPI_MISSING + "\n"
    assert cli.env_key("LEXAPI_API_KEY") is None


@pytest.mark.parametrize("value", REAL_LOOKING_KEYS)
def test_real_looking_keys_are_accepted(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"  {value}  ")
    assert cli.env_key("LEXAPI_API_KEY") == value and cli.env_key("ANTHROPIC_API_KEY") == value
    cli.require_keys(["LEXAPI_API_KEY", "ANTHROPIC_API_KEY"])


def test_a_real_looking_key_is_used_without_surrounding_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", "  lex_AbC123  ")
    assert cli.env_key("LEXAPI_API_KEY") == "lex_AbC123"
    stub = Stub()
    assert cli.main([QUESTION], transport=stub.transport(), chat_model=fake_model("x [1]"), out=io.StringIO()) == 0
    assert stub.requests[0].headers["x-api-key"] == "lex_AbC123"


def test_every_missing_key_is_named_in_one_message(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    unset(monkeypatch, "LEXAPI_API_KEY", "ANTHROPIC_API_KEY")
    assert cli.main(["--as-of", "2019-06-01", QUESTION]) == 2
    assert capsys.readouterr().err == f"{LEXAPI_MISSING}\n{MODEL_MISSING}\n"


# ── "today" is the UTC date ─────────────────────────────────────────────


def frozen_clock(instant: datetime.datetime) -> types.SimpleNamespace:
    """A stand-in for the datetime module whose clock reads ``instant``, in UTC or in the local zone."""

    class FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz: datetime.tzinfo | None = None) -> Any:
            return instant.astimezone(tz) if tz else instant.astimezone().replace(tzinfo=None)

    class FrozenDate(datetime.date):
        @classmethod
        def today(cls) -> Any:
            return instant.astimezone().date()

    return types.SimpleNamespace(datetime=FrozenDateTime, date=FrozenDate, UTC=datetime.UTC, timezone=datetime.timezone)


@pytest.fixture
def local_zone() -> Iterator[None]:
    before = os.environ.get("TZ")
    yield
    if before is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = before
    time.tzset()


@pytest.mark.parametrize(
    ("zone", "instant"),
    [
        ("Pacific/Pago_Pago", "2026-06-30T05:00:00+00:00"),  # local date 2026-06-29
        ("Pacific/Kiritimati", "2026-06-30T20:00:00+00:00"),  # local date 2026-07-01
    ],
)
def test_today_is_the_utc_date_whatever_the_local_zone(
    zone: str, instant: str, local_zone: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.environ["TZ"] = zone
    time.tzset()
    clock = frozen_clock(datetime.datetime.fromisoformat(instant))
    assert clock.date.today() != datetime.date(2026, 6, 30)  # the local date differs from the UTC one
    monkeypatch.setattr(cli, "datetime", clock)
    monkeypatch.setattr(cli, "today", REAL_TODAY)
    assert cli.today() == datetime.date(2026, 6, 30)
    assert cli.date_arg("2026-06-30") == "2026-06-30"
    with pytest.raises(Exception, match="2026-07-01 is in the future"):
        cli.date_arg("2026-07-01")


# ── LexAPI errors: the adapter's line, then the dashboard link ──────────


@pytest.mark.parametrize("code", ["UNAUTHORIZED", "INVALID_API_KEY", "CREDITS_EXHAUSTED"])
def test_a_key_or_credits_error_is_followed_by_the_dashboard_link_on_its_own_line(code: str) -> None:
    text = f"LexAPI error {code} (HTTP 402): Placeholder."
    assert cli.refusal_lines(text, code, 402) == f"{text}\n{DASHBOARD}"


BILLING_URL = "https://lex-api.com/dashboard#billing"


@pytest.mark.parametrize("code", ["RATE_LIMITED", "QUOTA_EXHAUSTED", "PAYMENT_PAST_DUE", "TIER_FORBIDDEN", "INTERNAL"])
def test_other_errors_get_no_dashboard_link(code: str) -> None:
    text = f"LexAPI error {code} (HTTP 403): Placeholder. Billing: {BILLING_URL}"
    assert cli.refusal_lines(text, code, 403) == text


def test_the_adapters_text_is_printed_whole() -> None:
    end = f" Credits reset at 2026-07-01T00:00:00Z. Billing: {BILLING_URL} Retry after 7 s."
    text = "LexAPI error CREDITS_EXHAUSTED (HTTP 402): " + "x" * 1000 + "." + end
    assert cli.refusal_lines(text, "CREDITS_EXHAUSTED", 402) == f"{text}\n{DASHBOARD}"


def test_a_key_from_the_environment_is_masked_in_the_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "sk-ant-api03-" + "Zq" * 20
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    assert (
        cli.refusal_lines(f"LexAPI error X (HTTP 400): {key} more.", "X", 400)
        == "LexAPI error X (HTTP 400): [key] more."
    )


def test_a_key_across_the_cut_stays_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "sk-ant-api03-" + "Zq" * 20
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    line = cli.one_line("The model call failed (X): " + "x" * 270 + key + " more.")
    assert line.endswith("x[ke...")  # masked first, then cut
    assert not any(key[i : i + 4] in line for i in range(len(key) - 3))


def test_control_characters_in_the_error_text_are_removed() -> None:
    url = "https://lex-api.com/\x1b[2Jbilling‮"
    line = cli.refusal_lines(f"LexAPI error P (HTTP 403): Due.\nBilling: {url}", "P", 403)
    assert "\x1b" not in line and "‮" not in line and "\n" not in line
    assert line.endswith("Due. Billing: https://lex-api.com/ [2Jbilling")


def test_a_connection_error_is_one_line_without_an_http_status(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("placeholder: no route", request=request)

    monkeypatch.setattr("lexapi.client.time.sleep", lambda seconds: None)  # the SDK retries a connection error
    code = cli.main([QUESTION], transport=httpx.MockTransport(refuse), chat_model=fake_model("x [1]"))
    err = capsys.readouterr().err
    assert code == 1
    assert len(err.splitlines()) == 1 and err.startswith("LexAPI error CONNECTION_ERROR: ") and "(HTTP" not in err


def test_the_adapters_error_text_is_available() -> None:
    try:
        from langchain_lexapi.tools import tool_error
    except ImportError:
        pytest.fail(
            "langchain_lexapi.tools.tool_error is not available in this adapter version; "
            "update cli.adapter_error_text to the adapter's error text"
        )
    assert callable(tool_error)


def test_a_byte_order_mark_does_not_hide_the_first_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unset(monkeypatch, "LEXAPI_API_KEY")
    env = tmp_path / ".env"
    env.write_bytes("LEXAPI_API_KEY=lex_AbC123\n".encode("utf-8-sig"))
    REAL_LOAD_ENV_FILE(env)
    assert os.environ["LEXAPI_API_KEY"] == "lex_AbC123"


@pytest.mark.parametrize("value", ["\uff12\uff10\uff12\uff15-01-17", "2025-01-\u0661\u0667"])
def test_non_ascii_digits_are_a_malformed_date(value: str) -> None:
    with pytest.raises(Exception, match="expected a date written YYYY-MM-DD"):
        cli.date_arg(value)


def test_the_model_client_gets_the_trimmed_key(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> object:
        made.append(kwargs)
        return object()

    monkeypatch.setattr(cli, "ChatAnthropic", record)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "  sk-ant-api03-placeholder-test  \t")
    cli.create_chat_model()
    assert made and made[0]["api_key"] == "sk-ant-api03-placeholder-test"


def test_without_the_adapters_error_text_the_run_prints_one_line_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_lexapi.tools", types.ModuleType("langchain_lexapi.tools"))
    body = {"success": False, "error": {"code": "CREDITS_EXHAUSTED", "message": "Credit pool exhausted"}}
    stub = Stub(reply=lambda _b: (402, body))
    assert cli.main([QUESTION], transport=stub.transport(), chat_model=fake_model("x [1]")) == 1
    assert capsys.readouterr().err == "LexAPI error: CREDITS_EXHAUSTED\n"


def test_ctrl_c_exits_130_without_a_traceback(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_main", interrupt)
    assert cli.main([QUESTION]) == 130
    captured = capsys.readouterr()
    assert captured.out == "" and "Traceback" not in captured.err


@pytest.mark.parametrize("code", ["KEY_REVOKED", "UNAUTHORIZED", "SOMETHING_NEW"])
def test_the_dashboard_link_follows_any_401_whatever_its_code(code: str) -> None:
    text = f"LexAPI error {code} (HTTP 401): Placeholder."
    assert cli.refusal_lines(text, code, 401) == f"{text}\n{DASHBOARD}"


def test_a_model_error_is_the_providers_message_capped_and_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-offline-test-not-a-key")

    class WithBody(Exception):
        body = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad\nrequest lex_abcdef0123"}}

    assert (
        cli.model_error_line(cli.provider_message(WithBody("Error code: 400 - {...}")))
        == "Model error: bad request lex_***"
    )
    long = cli.model_error_line("x" * 2000)
    assert long == "Model error: " + "x" * 1000 + "…"
    assert cli.model_error_line("key sk-fake-offline-test-not-a-key") == "Model error: key [key]"
