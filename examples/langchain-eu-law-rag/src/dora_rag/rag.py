"""Retrieval-augmented answers about DORA: retrieve, prompt, answer, check the citations."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from importlib import resources
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_lexapi import LexAPILegislationRetriever, RetrievalReport

DORA_CELEX = "32022R2554"  # Regulation (EU) 2022/2554, the Digital Operational Resilience Act
DORA_CONTEXT = "Digital Operational Resilience Act (DORA, Regulation (EU) 2022/2554)"  # put in front of the question
K = 10  # hits asked for; only DORA hits are kept, after the search (see the README)
NO_ANSWER = "The passages do not answer this question."

SYSTEM_PROMPT = resources.files("dora_rag").joinpath("prompt.txt").read_text(encoding="utf-8")
HUMAN_PROMPT = "Date asked: {date}\n\nPassages:\n\n{passages}\n\nQuestion: {question}"
PROMPT = ChatPromptTemplate.from_messages([SystemMessage(content=SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])

# Any bracketed token: ASCII, full-width and other square-ish brackets.
_BRACKETED = re.compile(
    r"[\[\uff3b\u3010\u3014\u27e6\u3016][^\]\uff3d\u3011\u3015\u27e7\u3017\n]{0,60}[\]\uff3d\u3011\u3015\u27e7\u3017]"
)
_CITATION = re.compile(r"\[([1-9][0-9]*)\]")  # the only accepted form: ASCII brackets and ASCII digits


class AnswerError(RuntimeError):
    """The model's reply could not be used as an answer."""


def has_text(document: Document) -> bool:
    """True when the passage carries text from the API (not a notice)."""
    return document.metadata.get("text_status") == "provided"


def retrieve(
    retriever: LexAPILegislationRetriever, question: str, as_of: str | None = None
) -> tuple[list[Document], RetrievalReport]:
    """One search; the DORA context goes in front of the question. ``as_of`` is per call."""
    return retriever.invoke_with_report(f"{DORA_CONTEXT}: {question}", as_of=as_of)


def format_passages(documents: Sequence[Document]) -> str:
    """Number the passages [1], [2], ... Each page_content is the text (or a notice) and its Source line."""
    return "\n\n".join(f"[{n}]\n{doc.page_content}" for n, doc in enumerate(documents, start=1))


def check_citations(answer: str, passage_count: int) -> None:
    """Every bracketed token with a digit must be a plain [n], 1 <= n <= passage_count; at least one is required."""
    cited = 0
    for match in _BRACKETED.finditer(answer):
        token = match.group()
        if not any(unicodedata.category(char) == "Nd" for char in token):
            continue  # not a citation, e.g. [sic]
        plain = _CITATION.fullmatch(token)
        if plain is None or not 1 <= int(plain.group(1)) <= passage_count:
            raise AnswerError(f"The answer cites {token!r}; citations must be [1] to [{passage_count}].")
        cited += 1
    if cited == 0 and answer != NO_ANSWER:
        raise AnswerError("The answer cites no passage.")


def read_answer(message: Any, passage_count: int) -> str:
    """The text of the model's reply. Raises AnswerError rather than return an unusable answer.

    The stop reasons checked are Anthropic's; with another provider, add that provider's check.
    """
    if not isinstance(message, AIMessage):
        raise AnswerError(f"The model returned {type(message).__name__}, not a message.")
    stop_reason = message.response_metadata.get("stop_reason")
    if stop_reason == "refusal":
        raise AnswerError("The model declined this request.")
    if stop_reason == "max_tokens":
        raise AnswerError("The model's reply was cut off (max_tokens).")
    answer = message.text.strip()
    if not answer:
        raise AnswerError("The model returned no text.")
    check_citations(answer, passage_count)
    return answer


def answer(question: str, documents: Sequence[Document], model: BaseChatModel, as_of: str | None = None) -> str:
    """Have the model answer from the passages alone; call only when some passage has text."""
    message = (PROMPT | model).invoke(
        {"date": as_of or "none", "passages": format_passages(documents), "question": question}
    )
    return read_answer(message, len(documents))
