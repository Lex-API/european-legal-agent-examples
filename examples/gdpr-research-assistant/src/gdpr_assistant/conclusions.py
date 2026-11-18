"""The conclusion filter.

Wording that tells the reader what the law requires of THEM, or whether THEY
comply. This example shows sources and leaves those conclusions to a
qualified person, so a statement that matches is removed (only a count is shown).

A blunt lexical filter for English output. It catches common phrasings, not
every one, and nothing in other languages. A match is ignored when an earlier
word of the same sentence is ``whether``, ``if``, ``how``, ``which``, ``may``,
``might`` or ``could``: such a sentence frames a question, it does not answer it.

The TypeScript twin is ``shared/conclusions.ts`` (which adds AI Act patterns);
both run the cases in ``shared/grounding-vectors.json``.
"""

from __future__ import annotations

import re

_DUTY = "(?:must|shall|should|has to|have to|needs? to|(?:is|are) (?:required|obliged|obligated) to)"

#: Addressed to the reader: what "you" must do, whether "you" comply.
READER_PATTERNS: tuple[str, ...] = (
    rf"\byou {_DUTY}\b",
    r"\byour (?:system|tool|product|company|organisation|organization|processing|transfer|use case) "
    rf"(?:{_DUTY}|is|are)\b",
    r"\b(?:is|are) (?:not )?(?:compliant|non-compliant|in compliance)\b",
    r"\bappl(?:y|ies) to (?:you|your)\b",
)

_HEDGE = re.compile(r"\b(?:whether|if|how|which|may|might|could)\b", re.IGNORECASE)


def states_conclusion(text: str, patterns: tuple[str, ...] = READER_PATTERNS) -> bool:
    """Does ``text`` (already normalised) state a conclusion under one of ``patterns``?"""
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            before = text[: match.start()]
            sentence_start = max(before.rfind(". "), before.rfind("; "), before.rfind("! "), before.rfind("? ")) + 1
            if not _HEDGE.search(before[sentence_start:]):
                return True
    return False
