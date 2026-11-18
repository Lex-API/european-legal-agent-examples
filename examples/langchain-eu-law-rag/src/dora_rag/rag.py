"""Retrieval-augmented answers about DORA: retrieve, prompt, answer, check the citations."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Sequence
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

# Opening and closing brackets that a citation could be written with.
_BRACKETS = {
    "[": "]",
    "\uff3b": "\uff3d",
    "\u3010": "\u3011",
    "\ufe5d": "\ufe5e",
    "\u27e6": "\u27e7",
    "\u3014": "\u3015",
    "\u3016": "\u3017",
    "\u00ab": "\u00bb",
    "\u2045": "\u2046",
}
_SEPARATORS = frozenset(",;-\u2013\u2014/.")
_LEAD_WORD = re.compile(r"(?:passage|source|ref)\b[.:]?", re.IGNORECASE)
_CITATION = re.compile(r"\[([1-9][0-9]*)\]")  # the only accepted form: ASCII brackets and ASCII digits


class AnswerError(RuntimeError):
    """The model's reply could not be used as an answer."""


class NoTextError(RuntimeError):
    """No passage has text, so the model was not called."""


class ModelCallError(RuntimeError):
    """The model call itself failed; the original error is the ``__cause__``."""


def _blank(char: str) -> bool:
    return char.isspace() or unicodedata.category(char) == "Cf"


def has_text(document: Document) -> bool:
    """True when the passage carries text from the API (not a notice) that is not blank."""
    if document.metadata.get("text_status") != "provided":
        return False
    text = document.page_content.rsplit("\nSource: ", 1)[0]
    return not all(_blank(char) for char in text)


def retrieve(
    retriever: LexAPILegislationRetriever, question: str, as_of: str | None = None
) -> tuple[list[Document], RetrievalReport]:
    """One search; the DORA context goes in front of the question. ``as_of`` is per call."""
    return retriever.invoke_with_report(f"{DORA_CONTEXT}: {question}", as_of=as_of)


def format_passages(documents: Sequence[Document]) -> str:
    """Number the passages [1], [2], ... Each page_content is the text (or a notice) and its Source line."""
    return "\n\n".join(f"[{n}]\n{doc.page_content}" for n, doc in enumerate(documents, start=1))


def citation_like(content: str) -> bool:
    """Numeric characters of any script and separators only, optionally after "passage", "source" or "ref"."""
    body = "".join(" " if _blank(char) else char for char in content).strip()
    if lead := _LEAD_WORD.match(body):
        body = body[lead.end() :]
    allowed = all(char.isnumeric() or char in _SEPARATORS or char.isspace() for char in body)
    return allowed and any(char.isnumeric() for char in body)


def _bracketed(answer: str) -> Iterator[tuple[str, str, bool]]:
    """Each bracketed token: (token, content, closed on the same line)."""
    for start, opener in enumerate(answer):
        if opener not in _BRACKETS:
            continue
        end = answer.find(_BRACKETS[opener], start + 1)
        newline = answer.find("\n", start + 1)
        if end != -1 and (newline == -1 or end < newline):
            yield answer[start : end + 1], answer[start + 1 : end], True
        else:  # not closed on its line: everything up to the next closing bracket counts
            stop = len(answer) if end == -1 else end
            yield answer[start:stop], answer[start + 1 : stop], False


def is_no_answer(answer: str) -> bool:
    """The fixed no-answer sentence, alone (quotes and the final full stop optional)."""
    sentence = answer.strip().strip("\"'\u201c\u201d\u2018\u2019").strip()
    return sentence in (NO_ANSWER, NO_ANSWER.rstrip("."))


def check_citations(answer: str, passage_count: int) -> None:
    """Every citation-like bracketed token must be exactly [n], 1 <= n <= passage_count; one at least is required."""
    cited = 0
    for token, content, closed in _bracketed(answer):
        if closed and not citation_like(content):
            continue  # not a citation: [sic], [a], an echoed notice, ...
        if not closed and not any(char.isnumeric() for char in content):
            continue
        plain = _CITATION.fullmatch(token) if closed else None
        if plain is None or not 1 <= int(plain.group(1)) <= passage_count:
            raise AnswerError(f"The answer cites {token!r}; citations must be [1] to [{passage_count}].")
        cited += 1
    if cited == 0 and not is_no_answer(answer):
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
    """Have the model answer from the passages alone. Without a passage that has text, no model is called."""
    if not any(has_text(doc) for doc in documents):
        raise NoTextError("None of the passages has text, so no model was called.")
    prompt = PROMPT.invoke({"date": as_of or "none", "passages": format_passages(documents), "question": question})
    try:
        message = model.invoke(prompt)
    except Exception as error:  # only the model call: connection, auth, rate limit, overload, ...
        raise ModelCallError(f"{type(error).__name__}: {error}") from error
    return read_answer(message, len(documents))
