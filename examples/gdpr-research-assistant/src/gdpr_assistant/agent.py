"""GDPR research assistant - the agent loop.

1. retrieve  GDPR articles and CJEU judgment paragraphs from LexAPI
2. draft     the model picks excerpts and writes a summary, citing source ids
3. check     the grounding validator (one retry, then sources only)
4. render    quoted text and model text in separate sections, with citations

The loop is a fixed pipeline: the number of API calls per question - and so
its credit cost - is bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lexapi import LexAPI

from gdpr_assistant.grounding import (
    DRAFT_SCHEMA,
    GroundedAnswer,
    GroundingError,
    Source,
    citation,
    render_answer,
    sources_for_prompt,
    validate_draft,
)
from gdpr_assistant.llm import Llm, LlmRequest
from gdpr_assistant.retrieve import retrieve

SYSTEM = """You answer research questions about the General Data Protection Regulation \
(Regulation (EU) 2016/679) and the case law of the Court of Justice of the European Union.

You only know the sources given to you: articles of the regulation and numbered paragraphs of \
judgments. Do not rely on anything else you may remember. Quote the passages that answer the \
question, copied exactly, and summarise what they say in your own words, citing the source ids for \
every sentence. If a source carries a note about which version of the text it is, respect it: do \
not describe a text as the law in force on a date unless its note says so.

If the sources do not answer the question, set status to no_support. This is research support, \
not legal advice: do not tell the reader what they must do or whether they comply."""

NOT_ADVICE = (
    "_Research support, not legal advice. The summary is model-written and may misread the sources; "
    "check it against the quoted text and the linked documents._"
)


@dataclass(frozen=True)
class Result:
    outcome: Literal["answered", "no_supporting_text", "sources_only"]
    markdown: str
    sources: list[Source]
    #: Credits the API reported for this run, as text ("20", "not reported", ...).
    credits: str
    answer: GroundedAnswer | None = None


def _notices(notices: list[str]) -> str:
    return "## Notices\n\n" + "\n".join(f"- {n}" for n in notices) + "\n\n" if notices else ""


def answer_question(question: str, *, client: LexAPI, llm: Llm, as_of: str | None = None) -> Result:
    # 1. Retrieve ------------------------------------------------------------
    found = retrieve(client, question, as_of)
    sources = found.sources
    scope = f" (as of {as_of})" if as_of else ""
    header = f"# GDPR research assistant\n\n**Question{scope}:** {question}\n\nModel: {llm.label}.\n\n"

    if not sources:
        # The "no supporting text found" path: say so; do not ask the model to fill the gap.
        body = (
            "## No supporting text found\n\nThe searches returned no GDPR article and no judgment paragraph "
            "for this question, so there is nothing to quote and no answer is given.\n\n"
        )
        return Result("no_supporting_text", header + body + _notices(found.notices), sources, found.describe_credits())

    # 2 + 3. Draft, then check (one retry with the violations fed back) --------
    user = (
        f"Question: {question}\n\nSources:\n\n{sources_for_prompt(sources)}\n\n"
        "Answer from these sources only. Copy quotes exactly, one line each. Write each sentence as one line "
        "of plain text and give it its source ids."
    )
    answer: GroundedAnswer | None = None
    feedback = ""
    for _attempt in range(2):
        raw = llm.generate(
            LlmRequest(
                system=SYSTEM,
                user=user + feedback,
                tool_name="submit_answer",
                tool_description="Submit quotes from the sources and a cited summary.",
                schema=DRAFT_SCHEMA,
            )
        )
        try:
            answer = validate_draft(raw, sources)
            break
        except GroundingError as error:
            rejected = "\n- ".join(error.violations)
            feedback = f"\n\nYour previous draft was rejected:\n- {rejected}\nFix these and submit again."

    # 4. Render --------------------------------------------------------------
    if answer is None:
        listing = "\n".join(f"- [{s.id}] {citation(s)}" + "".join(f"\n  - Note: {n}" for n in s.notes) for s in sources)
        body = (
            "## Retrieved sources (no model text)\n\nThe model's draft failed the grounding check twice, so none "
            f"of it is shown. These are the sources the searches returned:\n\n{listing}\n\n"
        )
        return Result("sources_only", header + body + _notices(found.notices), sources, found.describe_credits())
    if answer.status == "no_support":
        body = (
            f"## No supporting text found\n\nThe searches returned {len(sources)} source(s), but the model found "
            "nothing in them that answers the question. Nothing is quoted and no answer is given.\n\n"
        )
        return Result(
            "no_supporting_text", header + body + _notices(found.notices), sources, found.describe_credits(), answer
        )

    markdown = header + render_answer(answer, sources) + "\n" + _notices(found.notices) + NOT_ADVICE + "\n"
    return Result("answered", markdown, sources, found.describe_credits(), answer)
