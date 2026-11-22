"""The snippet in the README's "The core" section runs, against the mock LexAPI and a scripted LLM."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import llama_index.llms.anthropic
import llama_index.retrievers.lexapi
import llama_index.tools.lexapi
import pytest
from conftest import TEST_KEY, Stub, scripted
from lexapi import LexAPI

from law_on_a_date import cli, offline

README = Path(__file__).resolve().parents[1] / "README.md"
QUESTION = "What did Article 17 of the GDPR say about erasure?"
ARTICLE = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"


def core_snippet() -> str:
    section = README.read_text(encoding="utf-8").split("## The core\n", 1)[1]
    return section.split("```python\n", 1)[1].split("```", 1)[0]


def test_the_core_snippet_is_short() -> None:
    assert len(core_snippet().strip().splitlines()) <= 25


def patched(monkeypatch: pytest.MonkeyPatch, stub: Stub, llm: Any) -> tuple[LexAPI, list[dict[str, Any]]]:
    client = LexAPI(TEST_KEY, base_url="http://127.0.0.1:9/api/v1", transport=stub.transport(), max_retries=0)
    real_tools = llama_index.tools.lexapi.LexAPIToolSpec
    real_retriever = llama_index.retrievers.lexapi.LexAPILegislationRetriever
    made: list[dict[str, Any]] = []

    def llm_factory(**kwargs: Any) -> Any:
        made.append(kwargs)
        return llm

    # The snippet's settings, on the mock client.
    monkeypatch.setattr(llama_index.tools.lexapi, "LexAPIToolSpec", lambda **kw: real_tools(client=client, **kw))
    monkeypatch.setattr(
        llama_index.retrievers.lexapi,
        "LexAPILegislationRetriever",
        lambda **kw: real_retriever(client=client, **kw),
    )
    monkeypatch.setattr(llama_index.llms.anthropic, "Anthropic", llm_factory)
    return client, made


def test_the_core_snippet_runs(stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted("The first passage describes a placeholder objection [1].")
    client, made = patched(monkeypatch, stub, llm)
    with client:
        exec(compile(core_snippet(), str(README), "exec"), {})

    assert made == [{"model": "claude-sonnet-5-5"}]
    assert [json.loads(r.content) for r in stub.requests] == [
        {"document": "32016R0679", "asOf": "2019-06-01"},
        {
            "query": "What did Article 17 of the GDPR say about erasure?",
            "limit": 10,
            "include": ["text"],
            "asOf": "2019-06-01",
        },
    ]
    system, user = llm.seen[0]
    assert "cite each claim as [n]" in str(system.content)
    human = str(user.content)
    assert f"Source: CELEX 32016R0679, art_21, version for 2019-06-01: 32016R0679, {ARTICLE}" in human
    assert "text_status: withheld\n" in human
    assert "[No text: CELEX 32016R0679 art_18 is not established as the text for 2019-06-01" in human
    assert "UNRESOLVED-LATEST" not in human and "OTHER-ACT" not in human
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "[Version 32016R0679 (as_published) is the version listed for 2019-06-01; no provision was requested.]",
        f"Source: CELEX 32016R0679, version for 2019-06-01: 32016R0679, {ARTICLE}",
        "The first passage describes a placeholder objection [1].",
        f"[1] CELEX 32016R0679, art_21, version for 2019-06-01: 32016R0679, {ARTICLE}",
        f"[2] CELEX 32016R0679, art_18, text for 2019-06-01 not established, {ARTICLE}",
        "Search credits used: 5 remaining: 494",
    ]


def test_the_core_snippet_calls_no_model_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def all_withheld(body: dict[str, Any]) -> tuple[int, Any]:
        reply = offline.search_reply(body)
        for hit in reply["results"]:
            hit.update(asOfResolved=False, asOfReason="fetch_timeout")
        return 200, reply

    llm = scripted("unused [1]")
    client, _made = patched(monkeypatch, Stub(replies={"semantic": all_withheld}), llm)
    with client, pytest.raises(SystemExit, match="no model is called"):
        exec(compile(core_snippet(), str(README), "exec"), {})
    assert llm.seen == []


NOTICE_RUN = ["--dry-run", "--show-passages", "--as-of", "2019-06-01", "--provision", "art_17", QUESTION]


def test_the_notice_block_is_what_the_dry_run_gives_the_model() -> None:
    readme = README.read_text(encoding="utf-8")
    section = readme.split("### What the model sees for a notice\n", 1)[1].split("\n## ", 1)[0]
    command = "law-on-a-date " + " ".join(f'"{a}"' if " " in a else a for a in NOTICE_RUN)
    assert f"`{command}`" in " ".join(section.split())
    block = section.split("```text\n", 1)[1].split("\n```", 1)[0]
    out = io.StringIO()
    assert cli.main(NOTICE_RUN, out=out) == 0
    passages = out.getvalue().split("Passages given to the model:\n\n", 1)[1].split("\n\nAnswer (", 1)[0]
    assert block == "[3]\n" + passages.split("\n\n[3]\n", 1)[1]


@pytest.mark.parametrize(
    "args",
    [
        [
            "--dry-run",
            "--as-of",
            "2019-06-01",
            "--provision",
            "art_17",
            "What did Article 17 of the GDPR say about erasure?",
        ],
        ["--dry-run", "--compare", "2018-05-25", "2024-01-01"],
    ],
)
def test_the_sample_outputs_are_what_the_dry_run_prints(args: list[str]) -> None:
    command = "$ law-on-a-date " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n"
    readme = README.read_text(encoding="utf-8")
    start = readme.index(command) + len(command)
    out = io.StringIO()
    assert cli.main(args, out=out) == 0
    assert readme[start : readme.index("\n```", start)] == out.getvalue().rstrip("\n")
