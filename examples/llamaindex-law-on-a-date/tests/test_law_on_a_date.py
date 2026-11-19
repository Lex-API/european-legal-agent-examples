from __future__ import annotations

import io
import json
import sys
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from conftest import BASE_URL, OFFLINE_ERROR, TEST_KEY, Stub, scripted
from lexapi import LexAPI
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.anthropic import Anthropic
from llama_index.retrievers.lexapi import LexAPICallReport, LexAPILegislationRetriever
from llama_index.tools.lexapi import LexAPIToolSpec

from law_on_a_date import cli, offline, pipeline

QUESTION = "What did Article 17 of the GDPR say about erasure?"
EARLY, LATE = "2019-06-01", "2024-01-01"  # the mock lists the act as published before 2020, a consolidated text after
ARTICLE = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679"
CONSOLIDATED = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02016R0679-20990101"
ANSWER = "The provision describes a placeholder request [1]. Another passage a placeholder objection [2]."
ANSWER_3 = ANSWER + " There is no text of the third for the date [3]."
DASHBOARD = "\nDashboard: https://lex-api.com/dashboard?ref=github-examples"  # a line of its own


def run(
    stub: Stub, llm: offline.ScriptedLLM | None, *args: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    out = io.StringIO()
    code = cli.main(list(args), transport=stub.transport(), llm=llm, out=out)
    captured = capsys.readouterr()
    return code, out.getvalue() + captured.out, captured.err


def ask(stub: Stub, llm: offline.ScriptedLLM, *extra: str, as_of: str = EARLY, capsys: Any) -> tuple[int, str, str]:
    return run(stub, llm, "--as-of", as_of, "--provision", "art_17", *extra, QUESTION, capsys=capsys)


def prompt_text(llm: offline.ScriptedLLM) -> tuple[str, str]:
    assert len(llm.seen) == 1
    system, human = llm.seen[0]
    assert system.role == MessageRole.SYSTEM and human.role == MessageRole.USER
    return str(system.content), str(human.content)


def sources_map(output: str) -> list[str]:
    return output.split("Sources:\n", 1)[1].split("\n\n", 1)[0].splitlines()


def expected_citations(as_of: str) -> list[str]:
    """What the passages' Source lines must say, from the fixture data (not from the code under test)."""
    if as_of < offline.CONSOLIDATED_FROM:
        version = f"version for {as_of}: 32016R0679, {ARTICLE}"
        return [
            f"CELEX 32016R0679, art_17, {version}",
            f"CELEX 32016R0679, art_21, {version}",
            f"CELEX 32016R0679, art_18, text for {as_of} not established, {ARTICLE}",
        ]
    version = f"consolidated text 02016R0679-20990101 (no legal force), version for {as_of}: 02016R0679-20990101"
    return [
        f"CELEX 32016R0679, art_17, {version}, {CONSOLIDATED}",
        f"CELEX 32016R0679, art_21, {version}, {CONSOLIDATED}",
        f"CELEX 32016R0679, art_18, text for {as_of} not established, {ARTICLE}",
    ]


def unresolved_lookup(body: dict[str, Any]) -> tuple[int, Any]:
    reply = offline.read_fixture("legislation-as-of.gdpr.unresolved.json")
    reply["asOf"] = body["asOf"]
    return 200, reply


def all_withheld(body: dict[str, Any]) -> tuple[int, Any]:
    reply = offline.search_reply(body)
    for hit in reply["results"]:
        hit.update(asOfResolved=False, asOfReason="fetch_timeout")
        hit.pop("asOfVersion", None)
    return 200, reply


# ── the requests ────────────────────────────────────────────────────────


def test_a_version_lookup_then_one_search_for_the_date(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = ask(stub, scripted(ANSWER_3), capsys=capsys)

    assert code == 0
    assert [(r.method, r.url.path) for r in stub.requests] == [
        ("POST", "/api/v1/legislation/as-of"),
        ("POST", "/api/v1/legislation/semantic"),
    ]
    assert stub.bodies() == [
        {"document": "32016R0679", "asOf": EARLY, "provision": "art_17"},
        {"query": QUESTION, "limit": 10, "include": ["text"], "asOf": EARLY},
    ]
    assert all(r.headers["x-api-key"] == TEST_KEY for r in stub.requests)
    assert f"Date asked: {EARLY}" in out and "Provision: art_17" in out
    assert "in force" not in out


def test_without_a_provision_the_lookup_names_the_version_only(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted("The first passage describes a placeholder objection [1]. No text of the second [2].")
    code, out, _ = run(stub, llm, "--as-of", EARLY, QUESTION, capsys=capsys)

    assert code == 0
    assert stub.bodies()[0] == {"document": "32016R0679", "asOf": EARLY}
    assert f"Version listed for {EARLY}: 32016R0679, the act as published in the Official Journal." in out
    assert sources_map(out) == [
        f"[1] {expected_citations(EARLY)[1]}",
        f"[2] {expected_citations(EARLY)[2]} - notice only, no text was given to the model (reason: fetch_timeout)",
    ]
    _system, human = prompt_text(llm)
    assert "no provision was requested" not in human  # the version-only answer is not a passage
    assert "LexAPI credits used for this run: 6 (version lookup: 1, search: 5). Credits remaining: 494." in out


def test_another_act_can_be_named(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(
        stub, scripted("Placeholder [1]."), "--as-of", EARLY, "--celex", "32099l0009", QUESTION, capsys=capsys
    )
    assert stub.bodies()[0]["document"] == "32099L0009"
    assert code == 0 and "Act: CELEX 32099L0009" in out
    assert "Left out: 2 hit(s) not attributed to CELEX 32099L0009." in out


# ── what the model is sent ──────────────────────────────────────────────


@pytest.mark.parametrize("as_of", [EARLY, LATE])
def test_the_prompt_holds_every_source_line_and_no_withheld_text(
    stub: Stub, as_of: str, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = scripted(ANSWER_3)
    ask(stub, llm, as_of=as_of, capsys=capsys)
    system, human = prompt_text(llm)

    for citation in expected_citations(as_of):
        assert f"Source: {citation}" in human
    assert human.count("Source: ") == 3
    assert "[1]\n" in human and "[3]\n" in human and "[4]" not in human
    assert f"Date asked: {as_of}" in human and f"Question: {QUESTION}" in human
    # The withheld passage arrives as a notice, with its metadata lines, never with its latest text or snippet.
    notice = f"[No text: CELEX 32016R0679 art_18 is not established as the text for {as_of} (reason: fetch_timeout).]"
    withheld = human.split("[3]\n", 1)[1]
    assert withheld.startswith("text_status: withheld\n")
    assert "as_of_resolved: False\nas_of_reason: fetch_timeout\n" in withheld
    assert f"\n\n{notice}\nSource: " in withheld
    assert "UNRESOLVED-LATEST" not in system + human
    assert "OTHER-ACT" not in human and "32099L0009" not in human
    # Keys hidden from the model's view of a node (the citation is read in the Source line).
    assert "citation:" not in human and "text_withheld:" not in human and "\nid: " not in human


def test_the_prompt_states_the_rules(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(ANSWER_3)
    ask(stub, llm, as_of=LATE, capsys=capsys)
    system, human = prompt_text(llm)

    assert system == pipeline.SYSTEM_PROMPT
    assert 'lines of the form "key: value"' in system and "they are not text of the act" in system
    assert "Answer only from the passages" in system
    assert f'reply "{pipeline.NO_ANSWER}" and nothing else' in system
    assert "Do not use square brackets for anything else (dates, act numbers, notices, editorial notes)." in system
    assert '"[No text: CELEX ... is not established as the text for <date> ...]"' in system
    assert '"[No text returned for CELEX ...]"' in system and '"text_status: withheld"' in system
    assert "There is nothing to quote or summarise in it" in system
    assert "Consolidated texts have no legal force" in system and '"consolidated: True"' in system
    assert "name the one given by the CELEX number at the start of the Source line" in system
    assert "do not give legal advice" in system and "do not say whether anyone complies" in system
    assert "Do not say whether a provision applied on the date" in system
    # What the prompt describes is what the model sees.
    assert "consolidated: True\n" in human and "cite_celex: 32016R0679\n" in human


def test_the_provision_node_is_shaped_like_the_retrievers(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(ANSWER_3)
    ask(stub, llm, capsys=capsys)
    _system, human = prompt_text(llm)
    first = human.split("[1]\n", 1)[1].split("\n\n[2]\n", 1)[0]
    assert first.startswith("text_status: provided\n")
    assert "[SYNTHETIC FIXTURE - not the text of the act] Placeholder provision G1" in first


def passage(status: str, part: str = "art_17", text: str = "[SYNTHETIC FIXTURE] t", **extra: Any) -> NodeWithScore:
    body = text if status == "provided" else "[No text: placeholder notice]"
    metadata = {
        "text_status": status,
        "cite_celex": "32016R0679",
        "article_ref": part,
        "as_of": EARLY,
        "language": "en",
        **extra,
    }
    return NodeWithScore(node=TextNode(text=f"{body}\nSource: placeholder", metadata=metadata), score=None)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"part": "art_17"}, {"part": "art_17a"}),
        ({"part": "17"}, {"part": "art_17a"}),
        ({"part": "rct_17"}, {"part": "art_17"}),
        ({"part": "anx_I"}, {"part": "art_1"}),
        ({"part": "art_1"}, {"part": "art_10"}),
        ({"part": "art_17"}, {"part": "art_17", "language": "de"}),
        ({"part": "art_17"}, {"part": "art_17", "cite_celex": "32022R2554"}),
    ],
)
def test_different_provisions_are_never_the_same_passage(first: dict[str, Any], second: dict[str, Any]) -> None:
    provision, other = passage("withheld", **first), passage("provided", **second)
    assert pipeline._passage_key(provision) != pipeline._passage_key(other)
    combined = pipeline.combine(provision, [other])
    assert len(combined) == 2 and combined[0] is provision and combined[1] is other


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"part": "art_17"}, {"part": "17"}),
        ({"part": "art_17a"}, {"part": "Article 17A"}),
        ({"part": "art_17"}, {"part": "Art. 17"}),
        # A consolidated version and the act as published, with the same act to cite.
        (
            {"celex": "02016R0679-20990101", "consolidated": True},
            {"celex": "32016R0679", "consolidated": False},
        ),
    ],
)
def test_the_same_provision_is_the_same_passage(first: dict[str, Any], second: dict[str, Any]) -> None:
    provision, copy = passage("withheld", **first), passage("provided", **second)
    assert pipeline._passage_key(provision) == pipeline._passage_key(copy)
    assert [n is copy for n in pipeline.combine(provision, [copy])] == [True]


def test_two_texts_of_the_provision_from_different_versions_are_both_kept() -> None:
    lookup = passage("provided", version_celex="32016R0679")
    found = passage("provided", version_celex="02016R0679-20990101", celex="02016R0679-20990101")
    other = passage("provided", part="art_21")
    combined = pipeline.combine(lookup, [found, other])
    assert len(combined) == 3 and combined[0] is lookup and combined[1] is found and combined[2] is other
    # The same version, or one copy without text, is still one passage.
    same = passage("provided", version_celex="32016R0679")
    assert [n is lookup for n in pipeline.combine(lookup, [same])] == [True]
    notice = passage("withheld", version_celex="02016R0679-20990101")
    assert [n is lookup for n in pipeline.combine(lookup, [notice])] == [True]
    notice_first = passage("withheld", version_celex="02016R0679-20990101")
    assert [n is found for n in pipeline.combine(notice_first, [found])] == [True]
    assert pipeline.other_versions(lookup, combined) == ["02016R0679-20990101"]
    assert pipeline.other_versions(None, combined) == []


def test_the_different_version_line_names_only_this_provisions_versions() -> None:
    lookup = passage("provided", version_celex="32016R0679")
    other_provision = passage("provided", part="art_21", version_celex="02016R0679-20990101")
    combined = pipeline.combine(lookup, [other_provision])
    assert len(combined) == 2
    assert pipeline.other_versions(lookup, combined) == []


def test_version_ids_are_compared_ignoring_case() -> None:
    lookup = passage("provided", version_celex="02016R0679-20990101")
    copy = passage("provided", version_celex=" 02016r0679-20990101")
    combined = pipeline.combine(lookup, [copy])
    assert [n is lookup for n in combined] == [True]
    assert pipeline.other_versions(lookup, combined) == []


@pytest.mark.parametrize("missing", ["provision", "copy", "both"])
def test_a_text_without_a_version_id_counts_as_the_same_version(missing: str) -> None:
    # Taken as one passage: the provision's copy wins the tie.
    lookup = passage("provided", **({} if missing in ("provision", "both") else {"version_celex": "32016R0679"}))
    copy = passage("provided", **({} if missing in ("copy", "both") else {"version_celex": "02016R0679-20990101"}))
    combined = pipeline.combine(lookup, [copy])
    assert [n is lookup for n in combined] == [True]
    assert pipeline.other_versions(lookup, combined) == []


def test_a_search_text_from_another_version_is_kept_beside_the_provision(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        _status, payload = add_art17(as_of_resolved=True)(body)
        version = offline.read_fixture("legislation-as-of.gdpr.consolidated.json")
        payload["results"][0].update(
            law_id=version["asOfVersion"]["celex"],
            consolidated=True,
            baseCelex="32016R0679",
            asOfVersion=version["asOfVersion"],
        )
        return 200, payload

    llm = scripted(ANSWER_3 + " Placeholder [4].")
    code, out, _ = ask(Stub(replies={"semantic": reply}), llm, capsys=capsys)
    assert code == 0
    _system, human = prompt_text(llm)
    assert "Placeholder provision G1" in human and "RETRIEVED-ART-17-TEXT" in human
    assert human.count("Source: ") == 4
    assert sources_map(out)[0] == f"[1] {expected_citations(EARLY)[0]}"
    assert sources_map(out)[1].startswith("[2] CELEX 32016R0679, art_17, consolidated text 02016R0679-20990101")
    version_block = out.split("Version listed for", 1)[1].split("\n\n", 1)[0]
    assert version_block.splitlines()[-1] == (
        "The search listed a different version for this provision (02016R0679-20990101); both passages are shown."
    )
    assert out.count("The search listed a different version") == 1


def test_no_different_version_line_when_one_copy_is_kept(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = ask(stub, scripted(ANSWER_3), capsys=capsys)
    assert code == 0 and "The search listed a different version" not in out


def test_combine_keeps_the_better_of_two_copies_in_the_provisions_place() -> None:
    other = passage("provided", part="art_21")
    for provision, copy, winner in [
        (passage("withheld", part="17"), passage("provided"), "copy"),
        (passage("withheld", part="Article 17"), passage("not_established_for_date"), "copy"),
        (passage("provided"), passage("withheld"), "provision"),
        (passage("not_established_for_date", part="article 17"), passage("withheld"), "provision"),
        (passage("provided"), passage("provided"), "provision"),  # a tie keeps the provision
        (passage("provided", text=" \u200b "), passage("withheld"), "provision"),  # blank text ranks as a notice
        (passage("provided", text=" \u200b "), passage("not_established_for_date"), "copy"),
    ]:
        combined = pipeline.combine(provision, [other, copy])
        assert len(combined) == 2 and combined[1] is other
        assert combined[0] is (copy if winner == "copy" else provision)
    assert [n is other for n in pipeline.combine(None, [other])] == [True]
    art_18 = passage("provided", part="art_18")
    assert [n is m for n, m in zip(pipeline.combine(art_18, [other]), [art_18, other], strict=True)] == [True, True]


def add_art17(as_of_resolved: bool) -> Any:
    """The search fixture with one more hit: art_17 of the act, resolved or not for the date."""

    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.search_reply(body)
        template = payload["results"][0 if as_of_resolved else 1]
        hit = json.loads(json.dumps(template))
        hit.update(id="fx-gdpr-date-0017", article_ref="art_17")
        if as_of_resolved:
            hit["text"] = "[SYNTHETIC FIXTURE - not the text of the act] RETRIEVED-ART-17-TEXT"
        payload["results"].insert(0, hit)
        return 200, payload

    return reply


@pytest.mark.parametrize("spelling", ["17", "art_17", "Article 17"])
def test_a_retrieved_text_replaces_a_provision_notice(spelling: str, capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(ANSWER_3)
    stub = Stub(replies={"as-of": unresolved_lookup, "semantic": add_art17(as_of_resolved=True)})
    code, out, _ = run(stub, llm, "--as-of", EARLY, "--provision", spelling, QUESTION, capsys=capsys)

    assert code == 0
    assert stub.bodies()[0]["provision"] == spelling
    _system, human = prompt_text(llm)
    assert human.split("[1]\n", 1)[1].startswith("text_status: provided\n")
    assert "RETRIEVED-ART-17-TEXT" in human
    assert "art_17 is not established" not in human and human.count("Source: ") == 3
    assert sources_map(out) == [
        f"[1] CELEX 32016R0679, art_17, version for {EARLY}: 32016R0679, {ARTICLE}",
        f"[2] {expected_citations(EARLY)[1]}",
        f"[3] {expected_citations(EARLY)[2]} - notice only, no text was given to the model (reason: fetch_timeout)",
    ]
    assert f"Withheld for {EARLY}: 1 passage(s)" in out


@pytest.mark.parametrize("spelling", ["17", "art_17", "Article 17"])
def test_a_retrieved_notice_never_replaces_the_provisions_text(
    spelling: str, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = scripted(ANSWER_3)
    stub = Stub(replies={"semantic": add_art17(as_of_resolved=False)})
    code, out, _ = run(stub, llm, "--as-of", EARLY, "--provision", spelling, QUESTION, capsys=capsys)

    assert code == 0
    _system, human = prompt_text(llm)
    assert "Placeholder provision G1" in human.split("[1]\n", 1)[1].split("\n\n[2]\n", 1)[0]
    assert "art_17 is not established" not in human and human.count("Source: ") == 3
    assert sources_map(out)[0] == f"[1] {expected_citations(EARLY)[0]}"
    assert f"Withheld for {EARLY}: 1 passage(s)" in out


# ── what is printed ─────────────────────────────────────────────────────


def test_answer_sources_withheld_and_credits(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = ask(stub, scripted(ANSWER_3), capsys=capsys)

    assert code == 0
    assert out.index("Version listed for") < out.index(ANSWER_3) < out.index("Sources:")
    assert "Answer (written by the model from the passages given to it; see --show-passages):" in out
    assert "Placeholder provision" not in out  # no passage text is printed by default
    citations = expected_citations(EARLY)
    assert sources_map(out) == [
        f"[1] {citations[0]}",
        f"[2] {citations[1]}",
        f"[3] {citations[2]} - notice only, no text was given to the model (reason: fetch_timeout)",
    ]
    assert (
        f"Withheld for {EARLY}: 1 passage(s) have no text for that date and arrived as a notice.\n"
        + cli.NOTICE_MEANING
        + "\n"
        "Left out: 1 hit(s) not attributed to CELEX 32016R0679.\n"
        "LexAPI credits used for this run: 9 (version lookup: 4, search: 5). Credits remaining: 491."
    ) in out
    assert out.rstrip().endswith(cli.CLOSING)
    assert "UNRESOLVED-LATEST" not in out


def test_a_consolidated_version_is_labelled_with_the_act_to_cite(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = ask(stub, scripted(ANSWER_3), as_of=LATE, capsys=capsys)
    assert code == 0
    assert (
        f"Version listed for {LATE}: 02016R0679-20990101, a consolidated text (no legal force); "
        "the act to cite is CELEX 32016R0679.\n  LexAPI's note: [SYNTHETIC FIXTURE] The act as amended"
    ) in out
    assert f"  Source: {expected_citations(LATE)[0]}" in out


def test_a_version_not_established_is_said_so_and_the_provision_is_a_notice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = scripted("The second passage describes a placeholder objection [2]. No text of the provision [1].")
    code, out, _ = ask(Stub(replies={"as-of": unresolved_lookup}), llm, capsys=capsys)

    assert code == 0
    assert f"Version for {EARLY}: not established by LexAPI (reason: amendment_not_consolidated)." in out
    _system, human = prompt_text(llm)
    assert human.split("[1]\n", 1)[1].startswith("text_status: withheld\n")
    assert f"[No text: CELEX 32016R0679 art_17 is not established as the text for {EARLY}" in human
    assert sources_map(out)[0].endswith("(reason: amendment_not_consolidated)")
    assert f"Withheld for {EARLY}: 2 passage(s) have no text for that date and arrived as a notice." in out
    assert "version lookup: 1, search: 5" in out


def test_a_version_listed_with_a_provision_not_established(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        _status, payload = unresolved_lookup(body)
        payload["asOfReason"] = "article_not_in_version"
        payload["asOfVersion"] = {"celex": "32016R0679", "kind": "as_published", "date": None}
        return 200, payload

    code, out, _ = ask(Stub(replies={"as-of": reply}), scripted(ANSWER_3), capsys=capsys)
    assert code == 0
    assert (
        f"Version listed for {EARLY}: 32016R0679; the text of art_17 for that date is not established "
        "(reason: article_not_in_version)."
    ) in out


def test_show_passages_prints_what_the_model_was_given(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(ANSWER_3)
    code, out, _ = ask(stub, llm, "--show-passages", capsys=capsys)
    _system, human = prompt_text(llm)

    assert code == 0
    passages = human.split("Passages:\n\n", 1)[1].split("\n\nQuestion:", 1)[0]
    assert f"Passages given to the model:\n\n{passages}\n" in out
    assert "Answer (written by the model from the passages above):" in out
    assert "UNRESOLVED-LATEST" not in out


def test_credits_missing_from_a_response_print_as_not_reported(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.search_reply(body)
        del payload["credits"]
        return 200, payload

    code, out, _ = ask(Stub(replies={"semantic": reply}), scripted(ANSWER_3), capsys=capsys)
    assert code == 0
    assert "LexAPI credits used for this run: not reported (version lookup: 4, search: not reported)." in out
    assert "Credits remaining: not reported." in out


def test_strings_from_the_api_are_sanitised(capsys: pytest.CaptureFixture[str]) -> None:
    def search(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.search_reply(body)
        payload["results"][1]["asOfReason"] = "fetch\x1b[2J_timeout‮"
        return 200, payload

    def lookup(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.as_of_reply(body)
        payload["asOfVersion"]["celex"] = "32016R0679\x1b]0;title\x07"
        return 200, payload

    stub = Stub(replies={"semantic": search, "as-of": lookup})
    code, out, _ = ask(stub, scripted(ANSWER_3), capsys=capsys)
    assert code == 0
    assert "\x1b" not in out and "‮" not in out and "\x07" not in out
    assert "(reason: fetch [2J_timeout)" in out


def test_control_characters_from_the_model_are_removed(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = ask(stub, scripted("Placeholder \x1bcred\x07 and ‮flip [1]."), capsys=capsys)
    assert code == 0
    assert "\x1b" not in out and "‮" not in out


# ── no model call without text ──────────────────────────────────────────


def test_only_notices_for_the_date_means_no_model_call(capsys: pytest.CaptureFixture[str]) -> None:
    llm = scripted(ANSWER)
    stub = Stub(replies={"as-of": unresolved_lookup, "semantic": all_withheld})
    code, out, _ = ask(stub, llm, capsys=capsys)
    assert code == 0
    assert llm.seen == []
    assert "None of the passages has text for that date, so no model was called." in out
    assert f"Withheld for {EARLY}: 3 passage(s) have no text for that date and arrived as a notice." in out
    assert "Answer" not in out


def test_no_passages_of_the_act_means_no_model_call(capsys: pytest.CaptureFixture[str]) -> None:
    def search(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.search_reply(body)
        payload["results"] = [hit for hit in payload["results"] if hit["law_id"] == "32099L0009"]
        return 200, payload

    llm = scripted(ANSWER)
    code, out, _ = run(Stub(replies={"semantic": search}), llm, "--as-of", EARLY, QUESTION, capsys=capsys)
    assert code == 0 and llm.seen == []
    assert "No passages of CELEX 32016R0679 were found for this question, so no model was called." in out
    assert "Sources:" not in out


def test_a_provided_status_with_a_blank_text_is_not_text(capsys: pytest.CaptureFixture[str]) -> None:
    def lookup(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.as_of_reply(body)
        payload["provision"]["text"] = "  \n "
        return 200, payload

    llm = scripted(ANSWER)
    code, out, _ = ask(Stub(replies={"as-of": lookup, "semantic": all_withheld}), llm, capsys=capsys)
    assert code == 0 and llm.seen == []
    assert sources_map(out)[0].endswith(" - no text returned")


def test_the_answer_function_never_calls_the_model_without_text() -> None:
    notice = TextNode(text="[No text: placeholder]\nSource: CELEX 32016R0679", metadata={"text_status": "withheld"})
    blank = TextNode(text=" \nSource: CELEX 32016R0679", metadata={"text_status": "provided"})
    llm = scripted(ANSWER)
    with pytest.raises(pipeline.NoTextError, match="no model was called"):
        pipeline.answer(QUESTION, [NodeWithScore(node=notice), NodeWithScore(node=blank)], llm, EARLY)
    assert llm.seen == []


# ── the citation check and unusable replies ─────────────────────────────


@pytest.mark.parametrize(
    "reply",
    [
        "Placeholder [1][4].",
        "Placeholder [0].",
        "Placeholder [1-2].",
        "Placeholder [1–2].",
        "Placeholder [1; 2].",
        "Placeholder [1, 2].",
        "Placeholder [ 2 ].",
        "Placeholder [01].",
        "Placeholder [1] [2019-06-01].",
        "Placeholder [1] [2016/679].",
        "Placeholder [1] [passage 2].",
        "Placeholder [1] [Source 2].",
        "Placeholder [1] ［1］.",
        "Placeholder [1] 【1】.",
        "Placeholder [1] [١].",
        "Placeholder [1] [１].",
        "Placeholder [1] [½].",
        "Placeholder without any citation.",
        "Placeholder [sic] without a numbered citation.",
        f"{pipeline.NO_ANSWER} But placeholder.",
    ],
)
def test_the_citation_check_is_strict(stub: Stub, reply: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = ask(stub, scripted(reply), capsys=capsys)
    assert code == 1
    assert "The model's reply was not used:" in err
    assert out == ""


@pytest.mark.parametrize(
    "reply",
    [
        "Placeholder [1].",
        "Placeholder [3][1].",
        "Placeholder [sic] [2].",
        "Placeholder [2a] [1].",
        "As the notice says, [No text: CELEX 32016R0679 art_18 is not established as the text for 2019-06-01"
        " (reason: fetch_timeout).] [3]",
        pipeline.NO_ANSWER,
        f"  {pipeline.NO_ANSWER}\n",
        pipeline.NO_ANSWER.rstrip("."),
        f'"{pipeline.NO_ANSWER}"',
    ],
)
def test_the_citation_check_accepts_plain_numbers(stub: Stub, reply: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = ask(stub, scripted(reply), capsys=capsys)
    assert code == 0
    assert reply.strip() in out


def assistant(text: str, **raw: Any) -> ChatResponse:
    return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=raw)


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (assistant(""), "The model returned no text."),
        (assistant("   \n "), "The model returned no text."),
        (assistant("Placeholder [4]."), "The answer cites '[4]'; citations must be [1] to [3]."),
        (assistant("Partial [1]", stop_reason="max_tokens"), "cut off"),
        (assistant("No [1]", stop_reason="refusal"), "declined"),
    ],
)
def test_an_unusable_reply_exits_1_with_the_credits_and_a_message(
    stub: Stub, reply: ChatResponse, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = ask(stub, scripted(reply), capsys=capsys)
    assert code == 1
    assert out == ""
    lines = err.strip().splitlines()
    assert lines[0] == "LexAPI credits used for this run: 9 (version lookup: 4, search: 5). Credits remaining: 491."
    assert len(lines) == 2 and lines[1].startswith("The model's reply was not used:") and message in lines[1]
    assert "Traceback" not in err


def test_read_answer_refuses_a_non_response() -> None:
    with pytest.raises(pipeline.AnswerError, match="not a chat response"):
        pipeline.read_answer("plain string", 2)


def provider_error(kind: str) -> Exception:
    request = httpx2.Request("POST", "http://127.0.0.1:9/v1/messages")  # the anthropic SDK's HTTP client
    if kind == "connection":
        return anthropic.APIConnectionError(request=request)
    status = {"auth": 401, "rate": 429, "overloaded": 529}[kind]
    response = httpx2.Response(status, request=request, json={"error": {"message": f"placeholder {kind}"}})
    return anthropic.APIStatusError(f"placeholder {kind} \x1b[31m{TEST_KEY}", response=response, body=None)


@pytest.mark.parametrize("kind", ["connection", "auth", "rate", "overloaded"])
def test_a_model_provider_error_exits_1_without_a_traceback(
    stub: Stub, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = ask(stub, scripted(provider_error(kind)), capsys=capsys)
    assert code == 1
    assert out == ""
    lines = err.strip().splitlines()
    assert lines[0].startswith("LexAPI credits used for this run: 9 ")
    expected = "Connection error." if kind == "connection" else f"placeholder {kind}"
    assert lines[1:] == [f"Model error: {expected}"]  # the provider's message, not the error's repr
    assert "Traceback" not in err and "\x1b" not in err and TEST_KEY not in err


def test_any_other_model_failure_exits_1_without_a_traceback(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, _out, err = ask(stub, scripted(ValueError("placeholder failure")), capsys=capsys)
    assert code == 1
    assert err.strip().splitlines()[1] == "Model error: placeholder failure"


def test_a_bug_outside_the_model_call_is_not_reported_as_a_model_failure(
    stub: Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(nodes: Any) -> str:
        raise KeyError("placeholder bug")

    monkeypatch.setattr(pipeline, "format_passages", broken)
    with pytest.raises(KeyError, match="placeholder bug"):
        cli.main(["--as-of", EARLY, QUESTION], transport=stub.transport(), llm=scripted(ANSWER), out=io.StringIO())


# ── LexAPI errors ───────────────────────────────────────────────────────

ERRORS: dict[int, dict[str, Any]] = {
    401: {"success": False, "error": {"code": "INVALID_API_KEY", "message": "Invalid API key"}},
    402: {
        "success": False,
        "error": {
            "code": "CREDITS_EXHAUSTED",
            "message": "Credit pool exhausted",
            "details": {"operation": "legislation_as_of", "weight": 1, "available": 0},
        },
    },
    403: {
        "success": False,
        "code": "PAYMENT_PAST_DUE",
        "error": "Payment past due",
        "message": "Your subscription payment failed, so API requests are refused.",
        "billingUrl": "https://lex-api.com/dashboard#billing",
    },
    429: {"success": False, "error": {"code": "RATE_LIMITED", "message": "Too many requests"}},
    500: {"success": False, "error": {"code": "INTERNAL", "message": "placeholder failure"}},
}


@pytest.mark.parametrize(
    ("status", "printed"),
    [
        (401, "LexAPI error INVALID_API_KEY (HTTP 401): Invalid API key." + DASHBOARD),
        (402, "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted." + DASHBOARD),
        (
            403,
            "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Your subscription payment failed, so API"
            " requests are refused. Billing: https://lex-api.com/dashboard#billing",
        ),
        (429, "LexAPI error RATE_LIMITED (HTTP 429): Too many requests."),
        (500, "LexAPI error INTERNAL (HTTP 500): placeholder failure."),
    ],
)
def test_a_refused_lookup_prints_one_line_and_exits_1(
    status: int, printed: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("lexapi.client.time.sleep", lambda seconds: None)  # the SDK retries a 429
    stub = Stub(replies={"as-of": lambda _body: (status, ERRORS[status])})
    llm = scripted(ANSWER)
    code, out, err = run(stub, llm, "--as-of", EARLY, QUESTION, capsys=capsys)

    assert code == 1
    assert err == printed + "\n"
    assert out == "" and llm.seen == []
    assert [r.url.path for r in stub.requests][-1] == "/api/v1/legislation/as-of"


def test_a_refused_search_prints_the_lookup_credits_first(capsys: pytest.CaptureFixture[str]) -> None:
    stub = Stub(replies={"semantic": lambda _body: (402, ERRORS[402])})
    code, out, err = ask(stub, scripted(ANSWER), capsys=capsys)
    assert code == 1 and out == ""
    assert err.splitlines() == [
        "LexAPI credits used for this run: 4 (version lookup: 4). Credits remaining: 496.",
        "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted.",
        "Dashboard: https://lex-api.com/dashboard?ref=github-examples",
    ]
    assert len(stub.requests) == 2  # not retried


# The four answers that stop a run for a while, with every field the API sends for them.
STOPPED: dict[str, tuple[int, dict[str, Any], str]] = {
    "credits": (
        402,
        {
            "success": False,
            "error": {
                "code": "CREDITS_EXHAUSTED",
                "message": "Credit pool exhausted",
                "details": {"operation": "legislation_as_of", "weight": 1, "available": 0},
            },
            "resetsAt": "2026-07-01T00:00:00Z",
        },
        "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted. Credits reset at 2026-07-01T00:00:00Z."
        + DASHBOARD,
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
        "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily quota exceeded: The daily call quota of this plan is used up."
        " This is an exhausted quota, not a rate limit. Quota resets at 2026-07-01T00:00:00Z.",
    ),
    "past due": (
        403,
        {**ERRORS[403], "accessEndedAt": "2026-06-20T00:00:00Z"},
        "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Your subscription payment failed, so API"
        " requests are refused. Access ended at 2026-06-20T00:00:00Z. Billing: https://lex-api.com/dashboard#billing",
    ),
    "rate limit": (
        429,
        {**ERRORS[429], "retryAfter": 7},
        "LexAPI error RATE_LIMITED (HTTP 429): Too many requests. Retry after 7 s.",
    ),
}


@pytest.mark.parametrize("case", list(STOPPED))
@pytest.mark.parametrize("call", ["as-of", "semantic"])
def test_credits_quota_payment_and_rate_limit_print_the_tools_text(
    case: str, call: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("lexapi.client.time.sleep", lambda seconds: None)  # the SDK retries a rate limit
    status, body, printed = STOPPED[case]
    stub = Stub(replies={call: lambda _body: (status, body)})
    code, out, err = ask(stub, scripted(ANSWER), capsys=capsys)

    assert code == 1 and out == ""
    lines = err.splitlines()
    if call == "semantic":  # the lookup before it was charged
        assert lines.pop(0) == "LexAPI credits used for this run: 4 (version lookup: 4). Credits remaining: 496."
    assert "\n".join(lines) == printed
    assert "upgrade" not in err.lower()
    refused = [r for r in stub.requests if r.url.path.endswith("/as-of" if call == "as-of" else "/semantic")]
    assert len(refused) == (4 if case == "rate limit" else 1)  # only a rate limit is retried


@pytest.mark.parametrize("missing", ["act", "provenance"])
def test_an_unreadable_lookup_answer_exits_1(missing: str, capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.as_of_reply(body)
        del payload[missing]
        return 200, payload

    code, out, err = ask(Stub(replies={"as-of": reply}), scripted(ANSWER), capsys=capsys)
    assert code == 1 and out == ""
    assert err == "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read.\n"


def test_an_error_text_from_the_api_is_sanitised_and_never_shows_the_key(capsys: pytest.CaptureFixture[str]) -> None:
    body = {"success": False, "error": {"code": "ODD\x1b[2J", "message": f"placeholder‮ {TEST_KEY}"}}
    code, _out, err = ask(Stub(replies={"semantic": lambda _b: (400, body)}), scripted(ANSWER), capsys=capsys)
    assert code == 1
    assert "\x1b" not in err and "‮" not in err and TEST_KEY not in err


def test_the_key_is_never_printed(capsys: pytest.CaptureFixture[str]) -> None:
    outputs = []
    for stub, args in [
        (Stub(), ("--as-of", EARLY, "--provision", "art_17", QUESTION)),
        (Stub(), ("--compare", EARLY, LATE)),
        (Stub(replies={"as-of": lambda _b: (402, ERRORS[402])}), ("--as-of", EARLY, QUESTION)),
        (Stub(replies={"semantic": lambda _b: (403, ERRORS[403])}), ("--as-of", EARLY, QUESTION)),
        (Stub(replies={"as-of": lambda _b: (500, ERRORS[500])}), ("--compare", EARLY, LATE)),
    ]:
        outputs.append(run(stub, scripted(ANSWER_3), *args, capsys=capsys))
        assert stub.requests[0].headers["x-api-key"] == TEST_KEY  # the key was used ...
    for _code, out, err in outputs:
        assert TEST_KEY not in out + err  # ... and never printed
        assert "lex_offline_test_key" not in out + err


# ── compare two dates ───────────────────────────────────────────────────


def test_compare_lists_a_different_version_for_each_date(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, None, "--compare", EARLY, LATE, capsys=capsys)

    assert code == 0
    assert stub.bodies() == [{"document": "32016R0679", "asOf": EARLY}, {"document": "32016R0679", "asOf": LATE}]
    assert f"Dates compared: {EARLY} and {LATE}" in out
    assert f"Version listed for {EARLY}: 32016R0679, the act as published in the Official Journal." in out
    assert f"Version listed for {LATE}: 02016R0679-20990101, a consolidated text (no legal force)" in out
    assert (
        f"Different versions: 32016R0679 is listed for {EARLY} and 02016R0679-20990101 for {LATE}. "
        "This compares the versions LexAPI lists for the dates, not their texts."
    ) in out
    assert (
        f"LexAPI credits used for this run: 2 (lookup for {EARLY}: 1, lookup for {LATE}: 1). Credits remaining: 498."
    ) in out
    assert out.rstrip().endswith(cli.CLOSING_COMPARE)
    assert "Placeholder provision" not in out and "in force" not in out


def test_compare_says_when_the_same_version_is_listed(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, None, "--compare", "2018-05-25", EARLY, capsys=capsys)
    assert code == 0
    assert (
        "Same version: 32016R0679 is listed for both dates. "
        "This says which version LexAPI lists, not that nothing changed between the dates."
    ) in out
    assert "Different" not in out


def test_compare_puts_the_earlier_date_first(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(stub, None, "--compare", LATE, EARLY, capsys=capsys)
    assert code == 0
    assert [body["asOf"] for body in stub.bodies()] == [EARLY, LATE]
    assert out.index(f"Version listed for {EARLY}") < out.index(f"Version listed for {LATE}")


def test_compare_does_not_compare_a_version_not_established(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        return unresolved_lookup(body) if body["asOf"] == LATE else (200, offline.as_of_reply(body))

    code, out, _ = run(Stub(replies={"as-of": reply}), None, "--compare", EARLY, LATE, capsys=capsys)
    assert code == 0
    assert f"Version for {LATE}: not established by LexAPI (reason: amendment_not_consolidated)." in out
    assert f"Not compared: LexAPI did not establish the version for {LATE}." in out
    assert "Same version" not in out and "Different" not in out


def test_compare_needs_no_model_key(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--compare", EARLY, LATE], transport=stub.transport(), out=io.StringIO()) == 0
    assert len(stub.requests) == 2


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--compare", EARLY, EARLY], "--compare needs two different dates"),
        (["--compare", EARLY, LATE, QUESTION], "--compare lists versions only"),
        (["--compare", EARLY, LATE, "--provision", "art_17"], "--compare lists versions only"),
        (["--compare", EARLY], "expected 2 arguments"),
        (["--compare", EARLY, LATE, "--as-of", EARLY], "not allowed with argument"),
    ],
)
def test_compare_usage_errors_exit_2_before_any_request(
    stub: Stub, args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args, transport=stub.transport())
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert stub.requests == []


# ── input checks before any request ─────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("2025-02-30", "2025-02-30 is not a calendar date; expected YYYY-MM-DD"),
        ("01/06/2019", "expected a date written YYYY-MM-DD"),
        ("2019-6-1", "expected a date written YYYY-MM-DD"),
        ("2019-06-01T00:00", "expected a date written YYYY-MM-DD"),
        ("", "expected a date written YYYY-MM-DD"),
        ("tomorrow", "expected a date written YYYY-MM-DD"),
        ("2026-07-01", "is in the future"),
        ("2099-06-01", "is in the future"),
        ("1951-12-31", "1951-12-31 is before 1952-01-01, the earliest date LexAPI accepts"),
    ],
)
@pytest.mark.parametrize("mode", ["--as-of", "--compare"])
def test_a_bad_date_is_refused_before_any_request(
    stub: Stub, value: str, message: str, mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = scripted(ANSWER)
    args = ["--as-of", value, QUESTION] if mode == "--as-of" else ["--compare", EARLY, value]
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args, transport=stub.transport(), llm=llm)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert stub.requests == [] and llm.seen == []


def test_today_and_the_first_date_are_accepted(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(stub, None, "--compare", "1952-01-01", "2026-06-30", capsys=capsys)[0] == 0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--as-of", EARLY, ""], "the value is empty"),
        (["--as-of", EARLY, "  \n\t"], "the value is empty"),
        (["--as-of", EARLY], "a question is needed with --as-of"),
        (["--as-of", EARLY, "--provision", " ", QUESTION], "the value is empty"),
        (["--as-of", EARLY, "--celex", "CELEX:32016R0679", QUESTION], "expected the CELEX number"),
        (["--as-of", EARLY, "--celex", "02016R0679-20160504", QUESTION], "expected the CELEX number"),
        (["--as-of", EARLY, "--celex", "62014CJ0362", QUESTION], "expected the CELEX number"),
        ([QUESTION], "one of the arguments --as-of --compare is required"),
    ],
)
def test_usage_errors_exit_2_before_any_request(
    stub: Stub, args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = scripted(ANSWER)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args, transport=stub.transport(), llm=llm)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    assert stub.requests == [] and llm.seen == []


# ── configuration, the model and the key ────────────────────────────────


def test_a_missing_lexapi_key_exits_2(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    assert cli.main(["--as-of", EARLY, QUESTION], transport=stub.transport(), llm=scripted(ANSWER)) == 2
    assert capsys.readouterr().err == (
        "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples"
        " or use --dry-run.\n"
    )
    assert stub.requests == []


def test_a_missing_model_key_exits_2(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--as-of", EARLY, QUESTION], transport=stub.transport()) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    assert stub.requests == []


def test_the_model_is_anthropic_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    model_key = "sk-ant-offline-test-not-a-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", model_key)
    llm, label = cli.create_llm()
    assert label == "anthropic:claude-sonnet-5-5"
    assert isinstance(llm, Anthropic)
    assert llm.model == "claude-sonnet-5-5" and llm.max_tokens == 2000
    assert model_key not in repr(llm) and model_key not in str(llm) and model_key not in llm.model_dump_json()

    monkeypatch.setenv("ANTHROPIC_MODEL", "some-other-model")
    llm, label = cli.create_llm()
    assert label == "anthropic:some-other-model"
    assert isinstance(llm, Anthropic) and llm.model == "some-other-model"


def test_no_embedding_model_or_openai_is_needed(stub: Stub, capsys: pytest.CaptureFixture[str]) -> None:
    ask(stub, scripted(ANSWER_3), capsys=capsys)
    assert "openai" not in sys.modules
    assert "llama_index.embeddings.openai" not in sys.modules


# ── offline guards and the dry run ──────────────────────────────────────


def test_a_replaced_transport_needs_a_loopback_base_url(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXAPI_BASE_URL", "https://api.example.invalid/api/v1")
    with pytest.raises(RuntimeError, match="must start with http://127.0.0.1:"):
        cli.create_client(stub.transport())
    monkeypatch.delenv("LEXAPI_BASE_URL")
    with pytest.raises(RuntimeError, match="must start with http://127.0.0.1:"):
        cli.create_client(stub.transport())


def test_the_real_network_is_blocked() -> None:
    # The SDK's default base URL is the real API: with the real transport it must fail here.
    with LexAPI(TEST_KEY, max_retries=0) as client:
        with pytest.raises(Exception, match=OFFLINE_ERROR):
            client.legislation_as_of("32016R0679", EARLY)
    with pytest.raises(RuntimeError, match=OFFLINE_ERROR):
        httpx.get(BASE_URL)


def test_the_client_is_closed_after_the_run(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[LexAPI] = []
    real = cli.create_client

    def recording(transport: httpx.BaseTransport | None = None) -> LexAPI:
        client = real(transport)
        made.append(client)
        return client

    monkeypatch.setattr(cli, "create_client", recording)
    cli.main(["--as-of", EARLY, QUESTION], transport=stub.transport(), llm=scripted(ANSWER), out=io.StringIO())
    assert len(made) == 1
    with pytest.raises(RuntimeError, match="closed"):
        made[0].legislation_as_of("32016R0679", EARLY)


def test_the_retriever_settings(stub: Stub, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[LexAPILegislationRetriever] = []

    def recording(**kwargs: Any) -> LexAPILegislationRetriever:
        retriever = LexAPILegislationRetriever(**kwargs)
        made.append(retriever)
        return retriever

    monkeypatch.setattr(cli, "LexAPILegislationRetriever", recording)
    cli.main(["--as-of", EARLY, QUESTION], transport=stub.transport(), llm=scripted(ANSWER), out=io.StringIO())
    assert len(made) == 1
    retriever = made[0]
    assert retriever.celex == (pipeline.GDPR_CELEX,) and retriever.similarity_top_k == pipeline.K == 10
    assert retriever.as_of == EARLY and retriever.unresolved == "withhold"
    assert retriever._emit_withheld is True  # the adapter's default: withheld passages stay, as notices


def test_the_credits_line_adds_up_the_calls() -> None:
    def report(charged: int | None, remaining: int | None) -> LexAPICallReport:
        fields: dict[str, Any] = dict.fromkeys(LexAPICallReport.__dataclass_fields__, 0)
        fields.update(requested=None, as_of=None, hint=None, units_charged=charged, credits_remaining=remaining)
        return LexAPICallReport(**fields)

    assert cli.credits_line([("a", report(4, 10)), ("b", report(5, 5))]) == (
        "LexAPI credits used for this run: 9 (a: 4, b: 5). Credits remaining: 5."
    )
    assert cli.credits_line([("a", report(None, None))]) == (
        "LexAPI credits used for this run: not reported (a: not reported). Credits remaining: not reported."
    )


@pytest.mark.parametrize(
    "args",
    [
        ("--as-of", EARLY, "--provision", "art_17", QUESTION),
        ("--as-of", LATE, QUESTION),
        ("--compare", "2018-05-25", LATE),
    ],
)
def test_the_dry_run_needs_no_keys(
    args: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    out = io.StringIO()
    assert cli.main(["--dry-run", *args], out=out) == 0
    text = out.getvalue()
    assert text.startswith(offline.DRY_RUN_BANNER)
    assert "LexAPI credits used for this run: " in text
    if args[0] == "--as-of":
        assert "Model: scripted (no model was called)" in text and sources_map(text)
    assert "lex_dry_run_no_key" not in text + capsys.readouterr().err


def test_missing_dry_run_fixtures_exit_2_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(offline, "FIXTURES_DIR", tmp_path / "missing")
    assert cli.main(["--dry-run", "--as-of", EARLY, QUESTION], out=io.StringIO()) == 2
    err = capsys.readouterr().err
    assert "Offline fixtures not found" in err and "Traceback" not in err


def test_the_fixtures_are_marked_synthetic() -> None:
    for name in (
        "legislation-as-of.gdpr.as-published.json",
        "legislation-as-of.gdpr.consolidated.json",
        "legislation-semantic.gdpr-on-a-date.json",
    ):
        body = json.dumps(offline.read_fixture(name))
        texts = body.count('"text": ')
        assert texts and body.count('"text": "[SYNTHETIC FIXTURE - not the text of the act]') == texts


def test_the_dry_run_refuses_another_act(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("LEXAPI_API_KEY")
    for args in (["--as-of", EARLY, QUESTION], ["--compare", EARLY, LATE]):
        out = io.StringIO()
        assert cli.main(["--dry-run", "--celex", "32022R2554", *args], out=out) == 2
        assert out.getvalue() == ""
        assert capsys.readouterr().err == "The dry run has fixtures for CELEX 32016R0679 only.\n"


# ── the tool's event ────────────────────────────────────────────────────


def handlers() -> tuple[list[Any], list[Any]]:
    from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]

    return list(pipeline._tools_dispatcher().event_handlers), list(get_dispatcher().event_handlers)


def test_the_result_is_read_from_the_tools_own_dispatcher() -> None:
    assert pipeline._tools_dispatcher().name == "llama_index.tools.lexapi.base"


def test_the_event_handler_is_removed_after_each_lookup(capsys: pytest.CaptureFixture[str]) -> None:
    before = handlers()
    assert run(Stub(), scripted(ANSWER_3), "--as-of", EARLY, "--provision", "art_17", QUESTION, capsys=capsys)[0] == 0
    assert handlers() == before
    assert run(Stub(replies={"as-of": lambda _b: (402, ERRORS[402])}), None, "--compare", EARLY, LATE, capsys=capsys)[0]
    assert handlers() == before


def test_a_lookup_without_its_event_exits_1(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(LexAPIToolSpec, "lexapi_legislation_as_of", lambda self, *a, **k: "[placeholder]")
    code, out, err = run(stub, scripted(ANSWER), "--compare", EARLY, LATE, capsys=capsys)
    assert code == 1 and out == ""
    assert err == "The tool did not report its result.\n"


def test_an_event_for_another_date_is_not_taken(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = LexAPIToolSpec.lexapi_legislation_as_of

    def other_date(self: LexAPIToolSpec, celex: str, as_of: str, **kwargs: Any) -> str:
        return real(self, celex, "2018-05-25", **kwargs)

    monkeypatch.setattr(LexAPIToolSpec, "lexapi_legislation_as_of", other_date)
    code, _out, err = run(stub, scripted(ANSWER), "--as-of", EARLY, QUESTION, capsys=capsys)
    assert code == 1
    assert err == "The tool did not report its result.\n"


def test_an_event_of_another_tool_is_not_taken(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = LexAPIToolSpec._emit

    def as_another_tool(self: LexAPIToolSpec, tool: str, report: Any, result: Any) -> None:
        real(self, "lexapi_search_legislation", report, result)

    monkeypatch.setattr(LexAPIToolSpec, "_emit", as_another_tool)
    code, _out, err = run(stub, None, "--compare", EARLY, LATE, capsys=capsys)
    assert code == 1
    assert err == "The tool did not report its result.\n"


# ── hits not attributed to the act ──────────────────────────────────────


def test_a_consolidated_hit_that_names_no_act_is_left_out_and_counted(capsys: pytest.CaptureFixture[str]) -> None:
    def reply(body: dict[str, Any]) -> tuple[int, Any]:
        payload = offline.search_reply(body)
        hit = json.loads(json.dumps(payload["results"][0]))
        hit.update(
            id="fx-gdpr-date-0099",
            law_id="02016R0679-20990101",
            consolidated=True,
            text="[SYNTHETIC FIXTURE - not the text of the act] UNNAMED-CONSOLIDATED-TEXT",
        )
        payload["results"].append(hit)
        return 200, payload

    llm = scripted(ANSWER_3)
    code, out, _ = ask(Stub(replies={"semantic": reply}), llm, capsys=capsys)
    assert code == 0
    _system, human = prompt_text(llm)
    assert "UNNAMED-CONSOLIDATED-TEXT" not in human
    assert "Left out: 2 hit(s) not attributed to CELEX 32016R0679." in out
    assert "other acts" not in out
