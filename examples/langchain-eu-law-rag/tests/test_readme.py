"""The README: the snippet in "The core" runs, against the mock LexAPI and a fake model; the sample output is real."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import langchain_anthropic
import langchain_lexapi
import pytest
from conftest import TEST_KEY, Stub, fake_model
from lexapi import LexAPI

from dora_rag import cli

README = Path(__file__).resolve().parents[1] / "README.md"


def core_snippet() -> str:
    section = README.read_text(encoding="utf-8").split("## The core\n", 1)[1]
    return section.split("```python\n", 1)[1].split("```", 1)[0]


def test_the_core_snippet_is_short() -> None:
    assert len(core_snippet().strip().splitlines()) <= 25


def test_the_core_snippet_runs(stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    client = LexAPI(TEST_KEY, base_url="http://127.0.0.1:9/api/v1", transport=stub.transport(), max_retries=0)
    real_retriever = langchain_lexapi.LexAPILegislationRetriever
    model = fake_model("The first passage describes a placeholder review [1].")
    made: list[dict[str, Any]] = []

    def retriever(**kwargs: Any) -> Any:
        return real_retriever(client=client, **kwargs)  # the snippet's settings, on the mock

    def chat_model(**kwargs: Any) -> Any:
        made.append(kwargs)
        return model

    monkeypatch.setattr(langchain_lexapi, "LexAPILegislationRetriever", retriever)
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", chat_model)
    with client:
        exec(compile(core_snippet(), str(README), "exec"), {})

    assert made == [{"model": "claude-sonnet-5-5"}]
    body = json.loads(stub.requests[0].content)
    assert body["asOf"] == "2025-01-17" and body["limit"] == 10
    assert body["query"].startswith("Digital Operational Resilience Act (DORA): ")
    human = str(model.seen[0][1].content)
    assert "Source: CELEX 32022R2554, art_5, version for 2025-01-17: 32022R2554" in human
    assert "[No text: CELEX 32022R2554 art_6 is not established as the text for 2025-01-17" in human
    assert "UNRESOLVED-LATEST" not in human and "OTHER-ACT" not in human
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "The first passage describes a placeholder review [1].",
        "[1] CELEX 32022R2554, art_5, version for 2025-01-17: 32022R2554, "
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554",
        "[2] CELEX 32022R2554, art_6, text for 2025-01-17 not established, "
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554",
        "Credits used: 5 remaining: 495",
    ]


@pytest.mark.parametrize(
    "args",
    [
        [
            "--dry-run",
            "--show-passages",
            "--as-of",
            "2025-01-17",
            "What does DORA say about ICT risk management frameworks?",
        ],
    ],
)
def test_the_sample_outputs_are_what_the_dry_run_prints(args: list[str]) -> None:
    command = "$ dora-rag " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n"
    readme = README.read_text(encoding="utf-8")
    start = readme.index(command) + len(command)
    out = io.StringIO()
    assert cli.main(args, out=out) == 0
    assert readme[start : readme.index("\n```", start)] == out.getvalue().rstrip("\n")
