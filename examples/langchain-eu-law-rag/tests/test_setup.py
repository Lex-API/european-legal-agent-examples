"""The .env file, the keys, "today" in UTC and the one-line LexAPI errors."""

from __future__ import annotations

import datetime
import io
import os
import shutil
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
MODEL_MISSING = "ANTHROPIC_API_KEY is not set. Add it to .env or use --dry-run."
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


@pytest.mark.parametrize("value", ["", "   ", "lex_your_key_here", "<your LexAPI key>", "lex_...", " YOUR_KEY "])
def test_a_blank_or_placeholder_key_is_not_set(value: str, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setenv("LEXAPI_API_KEY", value)
    stub = Stub()
    assert cli.main([QUESTION], transport=stub.transport(), chat_model=fake_model("x [1]")) == 2
    assert stub.requests == []
    assert capsys.readouterr().err == LEXAPI_MISSING + "\n"
    assert cli.env_key("LEXAPI_API_KEY") is None


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


# ── LexAPI errors on one line ───────────────────────────────────────────


@pytest.mark.parametrize("code", ["UNAUTHORIZED", "INVALID_API_KEY", "CREDITS_EXHAUSTED"])
def test_a_key_or_credits_error_ends_with_the_dashboard_link(code: str) -> None:
    line = cli.refusal_line(f"LexAPI error {code} (HTTP 402): Placeholder.", code)
    assert line == f"LexAPI error {code} (HTTP 402): Placeholder. {DASHBOARD}"


def test_other_errors_and_billing_errors_get_no_dashboard_link() -> None:
    assert cli.refusal_line("LexAPI error RATE_LIMITED (HTTP 429): Slow down.", "RATE_LIMITED") == (
        "LexAPI error RATE_LIMITED (HTTP 429): Slow down."
    )
    billing = "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Placeholder. Billing: https://lex-api.com/dashboard#billing"
    assert cli.refusal_line(billing, "CREDITS_EXHAUSTED") == billing


@pytest.mark.parametrize(
    ("tail", "code", "kept"),
    [
        (
            " Billing: https://lex-api.com/dashboard#billing",
            "PAYMENT_PAST_DUE",
            "Billing: https://lex-api.com/dashboard#billing",
        ),
        ("", "CREDITS_EXHAUSTED", DASHBOARD),
    ],
)
def test_a_long_message_is_cut_but_never_its_link(tail: str, code: str, kept: str) -> None:
    line = cli.refusal_line(f"LexAPI error {code} (HTTP 402): " + "x" * 1000 + "." + tail, code)
    assert line.endswith("... " + kept)
    assert len(line) == 303 + 1 + len(kept)


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
