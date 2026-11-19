"""EU law for a date: look up the version for the date, retrieve passages for it, answer, check the citations."""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from typing import Any

from lexapi.passages import Passage
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]  # re-exported
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.llms import LLM
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
from llama_index.retrievers.lexapi import EXCLUDED_LLM_METADATA_KEYS, LexAPICallReport, LexAPILegislationRetriever
from llama_index.tools.lexapi import LexAPICallReport as ToolCallReport
from llama_index.tools.lexapi import LexAPIToolEvent, LexAPIToolSpec

GDPR_CELEX = "32016R0679"  # Regulation (EU) 2016/679, the default act
K = 10  # hits asked for; only hits of the act are kept, after the search (see the README)
NO_ANSWER = "The passages do not answer this question."
CallReport = LexAPICallReport | ToolCallReport  # each adapter has its own report class, with the same fields

SYSTEM_PROMPT = resources.files("law_on_a_date").joinpath("prompt.txt").read_text(encoding="utf-8")
HUMAN_PROMPT = "Date asked: {date}\n\nPassages:\n\n{passages}\n\nQuestion: {question}"
PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=HUMAN_PROMPT),
    ]
)

# The citation check (_BRACKETS to check_citations) is kept identical to examples/langchain-eu-law-rag;
# tests/test_citations.py checks that the two copies match.
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


class ToolReportError(RuntimeError):
    """The tool returned, but did not dispatch the event that carries its result."""


class ModelCallError(RuntimeError):
    """The model call itself failed; the original error is the ``__cause__``."""


def _blank(char: str) -> bool:
    return char.isspace() or unicodedata.category(char) == "Cf"


# ── step 1: the version listed for the date ─────────────────────────────


@dataclass(frozen=True)
class VersionLookup:
    """One ``lexapi_legislation_as_of`` call: what the tool returned and the passage it was built from."""

    as_of: str
    tool_output: str
    passage: Passage
    report: ToolCallReport

    @property
    def resolved(self) -> bool:
        return self.passage.as_of_resolved is True

    def node(self) -> NodeWithScore | None:
        """The provision as a node, shaped like the retriever's; ``None`` when no provision was asked for."""
        if not self.passage.article_ref:
            return None
        metadata = self.passage.metadata()
        node = TextNode(
            text=self.tool_output,  # the text or a notice, then the Source line
            id_=self.passage.id,
            metadata=metadata,
            excluded_embed_metadata_keys=sorted(metadata),
            excluded_llm_metadata_keys=list(EXCLUDED_LLM_METADATA_KEYS),
        )
        return NodeWithScore(node=node, score=None)


class _ToolEvents(BaseEventHandler):
    """Keeps the LexAPIToolEvent of each tool call (its report and passages)."""

    events: list[LexAPIToolEvent] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "LawOnADateToolEvents"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        if isinstance(event, LexAPIToolEvent):
            self.events.append(event)


def _tools_dispatcher() -> Any:
    """The dispatcher the tools adapter sends its events through (its module's), or the root one."""
    root = get_dispatcher()
    found = getattr(sys.modules.get(LexAPIToolSpec.__module__), "dispatcher", None)
    return found if isinstance(found, type(root)) else root


@contextmanager
def _tool_events() -> Iterator[_ToolEvents]:
    dispatcher = _tools_dispatcher()
    handler = _ToolEvents()
    dispatcher.add_event_handler(handler)
    try:
        yield handler
    finally:
        dispatcher.event_handlers = [h for h in dispatcher.event_handlers if h is not handler]


def look_up_version(spec: LexAPIToolSpec, celex: str, as_of: str, provision: str | None = None) -> VersionLookup:
    """The version LexAPI lists for the date and, with ``provision``, that provision's text for the date.

    The event is matched to this call by tool name and date. This runs single-threaded; with concurrent calls,
    match each event to its call (for example by the passage's CELEX number and provision as well).
    """
    with _tool_events() as seen:
        output = spec.lexapi_legislation_as_of(celex, as_of, provision=provision)
    matching = [
        event
        for event in seen.events
        if event.tool == "lexapi_legislation_as_of" and event.result.as_of == as_of and event.result.passages
    ]
    if len(matching) != 1:
        raise ToolReportError("The tool did not report its result.")
    event = matching[0]
    return VersionLookup(as_of=as_of, tool_output=output, passage=event.result.passages[0], report=event.report)


# ── step 2: passages for the question, for the date ─────────────────────


def retrieve(retriever: LexAPILegislationRetriever, question: str) -> tuple[list[NodeWithScore], LexAPICallReport]:
    """One search; the retriever carries the date (``as_of``) and the act (``celex``)."""
    nodes, report, _result = retriever.retrieve_with_result(question)
    return nodes, report


_ARTICLE = re.compile(r"(?:art(?:icle)?\.?[\s_]*)?(\d+[a-z]*)", re.IGNORECASE)


def _passage_key(node: NodeWithScore) -> tuple[str, str, str, str]:
    """What identifies a passage across the two calls: the act to cite, the part, the date and the language.

    The part is compared as the adapter ids it (``art_17``), whatever spelling of the provision was sent.
    """
    meta = node.node.metadata
    part = str(meta.get("article_ref") or "").strip()
    if article := _ARTICLE.fullmatch(part):
        part = f"art_{article.group(1).lower()}"
    act = str(meta.get("cite_celex") or meta.get("celex") or "")
    return act.upper(), part, str(meta.get("as_of") or ""), str(meta.get("language") or "")


def _rank(node: NodeWithScore) -> int:
    """The adapters' rule: a text over a labelled latest text over a notice."""
    if has_text(node):
        return 2
    return 1 if node.node.metadata.get("text_status") == "not_established_for_date" else 0


def _other_version(a: NodeWithScore, b: NodeWithScore) -> bool:
    """Both carry text, from different versions: two passages, not two copies of one."""
    if not (has_text(a) and has_text(b)):
        return False
    first, second = (str(n.node.metadata.get("version_celex") or "").strip().upper() for n in (a, b))
    return bool(first and second and first != second)


def combine(provision: NodeWithScore | None, nodes: Sequence[NodeWithScore]) -> list[NodeWithScore]:
    """The provision first, then the retrieved nodes.

    A retrieved node for the same passage is not repeated: whichever of the two ranks higher takes the
    provision's place, so a text is never replaced by a notice. Two texts of the same provision from
    different versions are both kept.
    """
    if provision is None:
        return list(nodes)
    key = _passage_key(provision)
    same = [n for n in nodes if _passage_key(n) == key and not _other_version(provision, n)]
    best = max([provision, *same], key=_rank)  # the provision wins a tie
    return [best, *(n for n in nodes if not any(n is s for s in same))]


# ── step 3: the answer ──────────────────────────────────────────────────


def has_text(node: NodeWithScore) -> bool:
    """True when the node carries text from the API (not a notice) before its Source line."""
    if node.node.metadata.get("text_status") != "provided":
        return False
    text = node.node.get_content(metadata_mode=MetadataMode.NONE).rsplit("\nSource: ", 1)[0]
    return not all(_blank(char) for char in text)


def format_passages(nodes: Sequence[NodeWithScore]) -> str:
    """Number the passages [1], [2], ... as the model sees each node: metadata lines, text or notice, Source line."""
    return "\n\n".join(
        f"[{n}]\n{node.node.get_content(metadata_mode=MetadataMode.LLM)}" for n, node in enumerate(nodes, start=1)
    )


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


def read_answer(response: Any, passage_count: int) -> str:
    """The text of the model's reply. Raises AnswerError rather than return an unusable answer.

    The stop reasons checked are Anthropic's; with another provider, add that provider's check.
    """
    if not isinstance(response, ChatResponse):
        raise AnswerError(f"The model returned {type(response).__name__}, not a chat response.")
    raw = response.raw if isinstance(response.raw, dict) else {}
    stop_reason = raw.get("stop_reason")
    if stop_reason == "refusal":
        raise AnswerError("The model declined this request.")
    if stop_reason == "max_tokens":
        raise AnswerError("The model's reply was cut off (max_tokens).")
    answer = (response.message.content or "").strip()
    if not answer:
        raise AnswerError("The model returned no text.")
    check_citations(answer, passage_count)
    return answer


def answer(question: str, nodes: Sequence[NodeWithScore], llm: LLM, as_of: str) -> str:
    """Have the model answer from the passages alone. Without a passage that has text, the model is not called."""
    if not any(has_text(node) for node in nodes):
        raise NoTextError("None of the passages has text for the date, so no model was called.")
    messages = PROMPT.format_messages(date=as_of, passages=format_passages(nodes), question=question)
    try:
        response = llm.chat(messages)
    except Exception as error:  # only the model call: connection, auth, rate limit, overload, ...
        raise ModelCallError(f"{type(error).__name__}: {error}") from error
    return read_answer(response, len(nodes))
