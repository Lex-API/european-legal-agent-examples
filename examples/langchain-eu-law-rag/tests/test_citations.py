"""The citation check, on crafted replies (three passages)."""

from __future__ import annotations

import pytest

from dora_rag import rag

PASSAGES = 3
NOTICE = "[No text: CELEX 32022R2554 art_6 is not established as the text for 2025-01-17 (reason: fetch_timeout).]"

ACCEPTED = [
    "Placeholder [1].",
    "[1]",
    "Placeholder [3].",
    "Placeholder [2][3].",
    "Placeholder [1] and [3].",
    "Placeholder [1], [2] and [3].",
    "Placeholder [1].[2]",
    "Placeholder ([1]).",
    "Placeholder [sic] [1].",
    "Placeholder [emphasis added] [2].",
    "Article 5(2)[a] says so [1].",
    f"Placeholder {NOTICE} [2]",
    "Placeholder [as amended in 2024] [1].",
    "Placeholder [Article 5] [1].",
    "Placeholder [i] [1].",
    "Placeholder [a-b] [1].",
    "Placeholder [x/y] [1].",
    "Placeholder [ ] [1].",
    "Placeholder [...] [1].",
    "Placeholder [-] [1].",
    "Placeholder [?] [1].",
    "A stray ] closer [1].",
    "Placeholder \u00abquoted\u00bb [2].",
    "Placeholder \u27e6note\u27e7 [2].",
    "First line [1].\nSecond line [2].",
    "Passage [1] says one thing; passage [2] another.",
    rag.NO_ANSWER,
    rag.NO_ANSWER.rstrip("."),
    f"  {rag.NO_ANSWER}\n",
    f'"{rag.NO_ANSWER}"',
    f"'{rag.NO_ANSWER}'",
    f"\u201c{rag.NO_ANSWER}\u201d",
    f"\u2018{rag.NO_ANSWER}\u2019",
]

REJECTED = [
    # out of range or not plain ASCII [n]
    "Placeholder [4].",
    "Placeholder [0].",
    "Placeholder [01].",
    "Placeholder [1][4].",
    "Placeholder [32022] [1].",
    # ranges, lists and other separators
    "Placeholder [1-2].",
    "Placeholder [1\u20132].",
    "Placeholder [1\u20142].",
    "Placeholder [1\u20133] [1].",
    "Placeholder [1; 2].",
    "Placeholder [1 ; 2].",
    "Placeholder [1, 2].",
    "Placeholder [1,2].",
    "Placeholder [1/2].",
    "Placeholder [1.].",
    "Placeholder [1.2].",
    # spaces and invisible characters
    "Placeholder [ 1 ].",
    "Placeholder [1 ].",
    "Placeholder [ 1].",
    "Placeholder [1\u200b].",
    "Placeholder [\u200b1].",
    # lead words
    "Placeholder [passage 1].",
    "Placeholder [Passage 2].",
    "Placeholder [source 1].",
    "Placeholder [ref. 1].",
    "Placeholder [ref: 2].",
    "Placeholder [1] [passage 2].",
    "Placeholder [1] [ref. 3].",
    # other bracket styles
    "Placeholder \uff3b1\uff3d.",
    "Placeholder [1] \uff3b2\uff3d.",
    "Placeholder \u30101\u3011.",
    "Placeholder [1] \u30102\u3011.",
    "Placeholder \ufe5d1\ufe5e.",
    "Placeholder \u27e61\u27e7.",
    "Placeholder \u30141\u3015.",
    "Placeholder \u30161\u3017.",
    "Placeholder \u00ab1\u00bb.",
    "Placeholder \u20451\u2046.",
    # numeric characters of other scripts
    "Placeholder [1] [\u0661].",
    "Placeholder [1] [\u06f1].",
    "Placeholder [1] [\u09e7].",
    "Placeholder [1] [\u0e51].",
    "Placeholder [1] [\uff11].",
    "Placeholder [1] [\u00b9].",
    "Placeholder [1] [\u2460].",
    "Placeholder [1] [\u2163].",
    "Placeholder [1] [\u2170].",
    "Placeholder [1] [\u0967].",
    "Placeholder [1] [\u3007].",
    "Placeholder [1] [\u96f6].",
    # dates and act numbers in brackets
    "Placeholder [2025-01-17] [1].",
    "Placeholder [2022/2554] [1].",
    # brackets not closed on their line
    "Placeholder [1",
    "Placeholder [1] and [2\n].",
    "Placeholder [\n1] [1].",
    # no citation
    "Placeholder without any citation.",
    "Placeholder [sic] without a numbered citation.",
    "Placeholder [2a].",
    f"Placeholder {NOTICE}",
    "",
    # the no-answer sentence with anything else
    f"{rag.NO_ANSWER} More text.",
    f"{rag.NO_ANSWER} [x]",
    "The passages do not answer this question!",
    "The passages do not answer",
    f"Note: {rag.NO_ANSWER}",
]


@pytest.mark.parametrize("reply", ACCEPTED)
def test_accepted(reply: str) -> None:
    rag.check_citations(reply, PASSAGES)


@pytest.mark.parametrize("reply", REJECTED)
def test_rejected(reply: str) -> None:
    with pytest.raises(rag.AnswerError):
        rag.check_citations(reply, PASSAGES)


def test_the_table_is_complete() -> None:
    assert len(ACCEPTED) + len(REJECTED) >= 78
    assert len(set(ACCEPTED) & set(REJECTED)) == 0
