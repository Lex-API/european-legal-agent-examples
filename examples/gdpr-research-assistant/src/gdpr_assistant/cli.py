"""GDPR research assistant - command line.

gdpr-assistant "When may personal data be transferred to a third country?"
gdpr-assistant --as-of 2019-06-01 "..."     (version in force on a date, where LexAPI can establish it)
gdpr-assistant --dry-run "..."              (no keys, fixtures)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

from lexapi import LexAPI, LexAPIError

from gdpr_assistant.agent import answer_question
from gdpr_assistant.llm import Llm, ScriptedLlm, create_llm_from_env
from gdpr_assistant.offline import DRY_RUN_BANNER, assert_local_base_url, read_fixture, start_stub_server


def load_env_file(path: Path) -> None:
    """Load ``KEY=value`` lines into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def create_client() -> LexAPI:
    """The LexAPI client, from LEXAPI_API_KEY (never hard-code it) and the optional LEXAPI_BASE_URL."""
    api_key = os.environ.get("LEXAPI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples"
            " or use --dry-run."
        )
    suffix = "european-legal-agent-examples"
    base_url = os.environ.get("LEXAPI_BASE_URL")
    if base_url:
        return LexAPI(api_key, base_url=base_url, user_agent_suffix=suffix)
    return LexAPI(api_key, user_agent_suffix=suffix)


def lexapi_error_line(error: LexAPIError) -> str:
    """One line for a refused or failed LexAPI call: the SDK's own text, ``[CODE] message (HTTP status)``.

    Control characters become spaces and the key from the environment is masked; nothing is added.
    """
    text = "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in str(error))
    key = os.environ.get("LEXAPI_API_KEY", "").strip()
    if key:
        text = text.replace(key, "[key]")
    return "LexAPI error: " + " ".join(text.split())


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 printed; 1 LexAPI refused a request or the call failed; 2 usage error."""
    parser = argparse.ArgumentParser(prog="gdpr-assistant", description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--as-of", metavar="YYYY-MM-DD", help="ask for the text in force on this date")
    parser.add_argument("--dry-run", action="store_true", help="run against bundled fixtures, with no keys")
    args = parser.parse_args(argv)

    load_env_file(Path(__file__).resolve().parents[2] / ".env")

    stub = None
    llm: Llm
    if args.dry_run:
        stub = start_stub_server()
        os.environ["LEXAPI_BASE_URL"] = assert_local_base_url(stub.base_url)
        os.environ["LEXAPI_API_KEY"] = "lex_dry_run_no_key"
        llm = ScriptedLlm(json.loads(read_fixture("llm/gdpr-asof.json" if args.as_of else "llm/gdpr.json")))
    else:
        llm = create_llm_from_env()

    try:
        with create_client() as client:
            result = answer_question(args.question, client=client, llm=llm, as_of=args.as_of)
    except LexAPIError as error:
        print(lexapi_error_line(error), file=sys.stderr)
        return 1
    finally:
        if stub:
            stub.close()

    if args.dry_run:
        print(DRY_RUN_BANNER)
    print(result.markdown)
    print(f"LexAPI credits reported for this run: {result.credits}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
