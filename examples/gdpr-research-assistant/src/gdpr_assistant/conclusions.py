"""The conclusion filter.

Wording that tells the reader what the law requires of THEM, whether THEY
comply, or what they should do. This example shows sources and leaves those
conclusions to a qualified person, so a statement that matches is removed
(only a count is shown).

A blunt lexical filter for English output. It catches common phrasings, not
every one, and nothing in other languages.

A match is ignored only when it sits in a clause that frames a question
("whether you must notify is for a qualified person to decide"). All of these
must hold:

- the nearest earlier ``whether``, ``if``, ``when``, ``how`` or ``which`` of
  the same sentence is not followed by ``means``, ``is why``, ``makes`` or
  ``shows``;
- between that word and the match there is no clause break (``, ; : ( )``, a
  dash) and no ``you``, ``your`` or ``yours``;
- for ``if`` and ``when``, a clause break follows the match in the sentence.

Quotations already verified as verbatim from a cited source are masked before
the filter runs, so the law's own words do not get a statement removed.

The TypeScript twin is ``shared/conclusions.ts`` (which adds AI Act patterns);
both run the cases in ``shared/grounding-vectors.json``.
"""

from __future__ import annotations

import re

_DUTY = (
    "(?:must|shall|should|has to|have to|needs? to|will need to|will have to"
    "|(?:is|are) (?:required|obliged|obligated) to)"
)

#: Addressed to the reader: what "you" must do, whether "you" comply, advice.
READER_PATTERNS: tuple[str, ...] = (
    rf"\byou {_DUTY}\b",
    r"\byour (?:system|tool|product|company|organisation|organization|processing|transfer|use case) "
    rf"(?:{_DUTY}|is|are)\b",
    r"\byour [a-z-]+(?:'s)? (?:[a-z-]+ )?(?:is|are) (?:not )?"
    r"(?:lawful|unlawful|legal|illegal|allowed|permitted|prohibited|covered)\b",
    r"\b(?:is|are) (?:not )?(?:compliant|non-compliant|in compliance)\b",
    r"\b(?:is|are) not (?:needed|required|necessary) for your\b",
    r"\bappl(?:y|ies) to (?:you|your)\b",
    r"\bwe (?:recommend|advise|suggest)\b",
    r"\bthe safest course\b",
    r"\bin your (?:position|case|situation)\b",
)

# re.ASCII keeps \\b and case-insensitive matching on ASCII, as in JavaScript without the `u` flag.
_FLAGS = re.IGNORECASE | re.ASCII
_HEDGE = re.compile(r"\b(whether|if|when|how|which)\b", _FLAGS)
_HEDGE_AS_ASSERTION = re.compile(r" (?:means|is why|makes|shows)\b", _FLAGS)
_CLAUSE_BREAK = re.compile("[,;:()\u2013\u2014]| - ")
_SECOND_PERSON = re.compile(r"\b(?:you|your|yours)\b", _FLAGS)
_SENTENCE_ENDS = (". ", "! ", "? ")


def _sentence_start(text: str, index: int) -> int:
    return max(text.rfind(mark, 0, index) for mark in _SENTENCE_ENDS) + 1


def _sentence_end(text: str, index: int) -> int:
    ends = [at for at in (text.find(mark, index) for mark in _SENTENCE_ENDS) if at != -1]
    return min(ends) if ends else len(text)


def _framed_as_question(text: str, start: int, end: int) -> bool:
    """Is the match at [start, end) inside a clause that frames a question rather than answering it?"""
    begin = _sentence_start(text, start)
    hedges = list(_HEDGE.finditer(text, begin, start))
    if not hedges:
        return False
    hedge = hedges[-1]
    between = text[hedge.end() : start]
    if _HEDGE_AS_ASSERTION.match(between) or _CLAUSE_BREAK.search(between) or _SECOND_PERSON.search(between):
        return False
    if hedge.group(0).lower() in ("if", "when"):
        return bool(_CLAUSE_BREAK.search(text[end : _sentence_end(text, end)]))
    return True


def states_conclusion(text: str, patterns: tuple[str, ...] = READER_PATTERNS) -> bool:
    """Does ``text`` (already normalised) state a conclusion under one of ``patterns``?"""
    for pattern in patterns:
        for match in re.finditer(pattern, text, _FLAGS):
            if not _framed_as_question(text, match.start(), match.end()):
                return True
    return False
