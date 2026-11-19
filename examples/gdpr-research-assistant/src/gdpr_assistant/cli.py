"""GDPR research assistant - command line.

gdpr-assistant "When may personal data be transferred to a third country?"
gdpr-assistant --as-of 2019-06-01 "..."     (version in force on a date, where LexAPI can establish it)
gdpr-assistant --dry-run "..."              (no keys, fixtures)

Exit codes: 0 printed; 1 LexAPI or the model failed; 2 usage or configuration error; 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from lexapi import (
    CreditsExhaustedError,
    InvalidResponseError,
    LexAPI,
    LexAPIError,
    NationalAllowanceExhaustedError,
    PaymentPastDueError,
    RateLimitedError,
)

from gdpr_assistant.agent import answer_question
from gdpr_assistant.llm import Llm, LlmError, ScriptedLlm, create_llm_from_env
from gdpr_assistant.offline import DRY_RUN_BANNER, assert_local_base_url, read_fixture, start_stub_server
from gdpr_assistant.retrieve import Retrieved

DASHBOARD_URL = "https://lex-api.com/dashboard?ref=github-examples"
LEXAPI_KEY_MISSING = f"LEXAPI_API_KEY is not set. Create a key at {DASHBOARD_URL} or use --dry-run."
PLACEHOLDERS = frozenset({"lex_your_key_here"})  # the key values this repository's .env.example files shipped
CREDITS_LINE = "LexAPI credits reported for this run: {}."
_MESSAGE_CAP = 1000
_VALUE_CAP = 200
_KEY_PATTERN = re.compile(r"lex_[A-Za-z0-9_\-]+")
_NOT_REACHED = frozenset({521, 522, 523})  # Cloudflare could not reach LexAPI: the SDK retries these
_DASHBOARD_CODES = frozenset({"UNAUTHORIZED", "INVALID_API_KEY", "CREDITS_EXHAUSTED"})
_UNREADABLE = "The answer could not be read."


class ConfigError(RuntimeError):
    """A missing key or setting (exit code 2), found before any request."""


def load_env_file(path: Path) -> None:
    """Load ``KEY=value`` lines into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def env_key(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """The key in ``name``, trimmed; None when unset, blank, a placeholder this repository shipped, or ``<...>``."""
    value = (os.environ if env is None else env).get(name, "").strip()
    if not value or value.lower() in PLACEHOLDERS or (value.startswith("<") and value.endswith(">")):
        return None
    return value


def missing_settings(env: Mapping[str, str], *, model: bool) -> list[str]:
    """One line per setting a real run needs that is missing; empty when nothing is."""
    lines = [] if env_key("LEXAPI_API_KEY", env) else [LEXAPI_KEY_MISSING]
    if not model:
        return lines
    provider = (env.get("LLM_PROVIDER", "").strip() or "anthropic").lower()
    if provider == "anthropic":
        if not env_key("ANTHROPIC_API_KEY", env):
            lines.append("ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run.")
    elif provider == "openai":
        if not env_key("OPENAI_API_KEY", env):
            lines.append("OPENAI_API_KEY is not set. Add an OpenAI API key to .env or use --dry-run.")
        if not env.get("OPENAI_MODEL", "").strip():
            lines.append("OPENAI_MODEL is not set. Name the OpenAI model to use (there is no default).")
    else:
        lines.append(f'LLM_PROVIDER is "{clean(provider, _VALUE_CAP) or ""}". Use "anthropic" or "openai".')
    return lines


def create_client() -> LexAPI:
    """The LexAPI client, from LEXAPI_API_KEY (never hard-code it) and the optional LEXAPI_BASE_URL."""
    api_key = env_key("LEXAPI_API_KEY")
    if not api_key:
        raise ConfigError(LEXAPI_KEY_MISSING)
    suffix = "european-legal-agent-examples"
    base_url = os.environ.get("LEXAPI_BASE_URL")
    if base_url:
        return LexAPI(api_key, base_url=base_url, user_agent_suffix=suffix)
    return LexAPI(api_key, user_agent_suffix=suffix)


# ── the error format (the same as the LexAPI adapters for LangChain and LlamaIndex) ──


def clean(value: object, cap: int) -> str | None:
    """A server string fit for one line: control, format and line-break characters become spaces, whitespace
    collapses, and the result is cut at ``cap`` characters (``…`` marks the cut). None when not a string or blank."""
    if not isinstance(value, str):
        return None
    text = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in value)
    text = " ".join(text.split())
    if not text:
        return None
    return text[:cap].rstrip() + "…" if len(text) > cap else text


def mask(text: str) -> str:
    """Anything shaped like a LexAPI key, then every key from the environment, masked."""
    text = _KEY_PATTERN.sub("lex_***", text)
    for name in ("LEXAPI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name, "").strip()
        if len(secret) >= 8:
            text = text.replace(secret, "[key]")
    return text


def _sentence(message: str) -> str:
    message = message.rstrip(": ")
    return message if not message or message[-1] in ".!?" else message + "."


def _wait_text(seconds: object) -> str | None:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    value = float(seconds)
    if not math.isfinite(value) or value <= 0:
        return None
    return str(int(value)) if value.is_integer() else f"{value:.6f}".rstrip("0").rstrip(".")


def _quota(error: LexAPIError) -> bool:
    return error.code == "QUOTA_EXHAUSTED" or (isinstance(error, RateLimitedError) and error.quota_exhausted)


def _sdk_retries(error: LexAPIError) -> bool:
    """Whether the SDK retries an error of this kind: a rate-limit 429, or a 521/522/523."""
    if error.status not in (429, *_NOT_REACHED) or error.code == "PAYMENT_PAST_DUE" or _quota(error):
        return False
    return not isinstance(error, (PaymentPastDueError, CreditsExhaustedError, NationalAllowanceExhaustedError))


def lexapi_error_text(error: LexAPIError) -> str:
    """``LexAPI error CODE (HTTP n): <message>.`` and, each only when the error carries it and in this order, the
    credits reset time, the quota notice and its reset time, when access ended, the billing link and the wait."""
    status = error.status
    http = f" (HTTP {status})" if status is not None else ""
    if isinstance(error, InvalidResponseError) or (status is not None and 200 <= status < 300):
        return f"LexAPI error INVALID_RESPONSE{http}: {_UNREADABLE}"
    message = clean(error.message, _MESSAGE_CAP)
    said = _sentence(message) if message else ""
    parts = [f"LexAPI error {clean(error.code, _VALUE_CAP) or 'UNKNOWN'}{http}" + (f": {said}" if said else ".")]
    if isinstance(error, CreditsExhaustedError) and (resets_at := clean(error.resets_at, _VALUE_CAP)):
        parts.append(f"Credits reset at {resets_at}.")
    if _quota(error):
        parts.append("This is an exhausted quota, not a rate limit.")
        quota_reset = clean(error.resets_at, _VALUE_CAP) if isinstance(error, RateLimitedError) else None
        if quota_reset:
            parts.append(f"Quota resets at {quota_reset}.")
    if isinstance(error, PaymentPastDueError):
        if ended := clean(error.access_ended_at, _VALUE_CAP):
            parts.append(f"Access ended at {ended}.")
        full_url = clean(error.billing_url, len(error.billing_url or ""))
        full_message = clean(error.message, len(error.message or "")) or ""
        if full_url and full_url not in full_message:
            parts.append(f"Billing: {clean(error.billing_url, _VALUE_CAP)}")
    if _sdk_retries(error) and (wait := _wait_text(error.retry_after)):
        parts.append(f"Retry after {wait} s.")
    return mask(" ".join(parts))


def lexapi_error_lines(error: LexAPIError) -> list[str]:
    """The error's text, then the dashboard link on a line of its own after a key or credits error."""
    lines = [lexapi_error_text(error)]
    if error.status == 401 or error.code in _DASHBOARD_CODES:
        lines.append(f"Dashboard: {DASHBOARD_URL}")
    return lines


def model_error_line(error: LlmError) -> str:
    return "Model error: " + mask(clean(str(error), _MESSAGE_CAP) or "no message")


# ── the run ─────────────────────────────────────────────────────────────


def _run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog="gdpr-assistant", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--as-of", metavar="YYYY-MM-DD", help="ask for the text in force on this date")
    parser.add_argument("--dry-run", action="store_true", help="run against bundled fixtures, with no keys")
    args = parser.parse_args(argv)

    load_env_file(Path(__file__).resolve().parents[2] / ".env")
    if not args.dry_run:
        # Every key this run needs, before any request.
        missing = missing_settings(os.environ, model=True)
        if missing:
            print("\n".join(missing), file=sys.stderr)
            return 2

    stub = None
    llm: Llm
    progress = Retrieved()
    try:
        if args.dry_run:
            stub = start_stub_server()
            os.environ["LEXAPI_BASE_URL"] = assert_local_base_url(stub.base_url)
            os.environ["LEXAPI_API_KEY"] = "lex_dry_run_no_key"
            llm = ScriptedLlm(json.loads(read_fixture("llm/gdpr-asof.json" if args.as_of else "llm/gdpr.json")))
        else:
            llm = create_llm_from_env()
        with create_client() as client:
            result = answer_question(args.question, client=client, llm=llm, as_of=args.as_of, progress=progress)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    except (LexAPIError, LlmError) as error:
        if progress.calls:  # the calls before it were charged
            print(CREDITS_LINE.format(progress.describe_credits()), file=sys.stderr)
        lines = lexapi_error_lines(error) if isinstance(error, LexAPIError) else [model_error_line(error)]
        print("\n".join(lines), file=sys.stderr)
        return 1
    finally:
        if stub:
            stub.close()

    if args.dry_run:
        print(DRY_RUN_BANNER)
    print(result.markdown)
    print(CREDITS_LINE.format(result.credits), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 printed; 1 LexAPI or the model failed; 2 usage or configuration error; 130 interrupted."""
    try:
        return _run(argv)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
