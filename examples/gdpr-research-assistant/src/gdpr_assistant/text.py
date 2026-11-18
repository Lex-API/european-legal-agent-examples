"""Text handling for the grounding check.

Normalisation, the rules for what model-written text may contain, quotation
detection, word-boundary matching and Markdown escaping.

Everything is defined on explicit code-point sets (no regex ``\\s`` / ``\\d``
defaults), so this module and its TypeScript twin (``shared/text.ts``) behave
the same. ``shared/grounding-vectors.json`` holds the cases both test suites run.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit

#: Characters treated as whitespace.
WHITESPACE = frozenset(
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F]
    + [0x3000]
)
#: Whitespace that ends a line.
LINE_BREAKS = frozenset([0x0A, 0x0B, 0x0C, 0x0D, 0x85, 0x2028, 0x2029])
SOFT_HYPHEN = 0xAD
TAB = 0x09

#: Quotation-mark characters, mapped to ``'`` or ``"`` by ``normalise``.
SINGLE_MARKS = "‘’‚‛‹›`´′＇❛❜"
DOUBLE_MARKS = "“”„‟«»＂❝❞「」『』″〝〞〟"

HYPHENS = "-‐‑"
END_PUNCTUATION = ".,;:!?…"
#: Endings after an apostrophe that make it part of a word: provider's, don't, they're, we've, it'll, I'd, I'm.
CONTRACTIONS = frozenset(["s", "t", "re", "ve", "ll", "d", "m"])

MAX_STATEMENT_LENGTH = 600
MAX_QUOTE_LENGTH = 1200

_CHARACTER_REFERENCE = re.compile(r"&(#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]{1,31});")
_OPENING_BRACKETS = "[\uff3b\u27e6\u3010\u3014"
_CLOSING_BRACKETS = "]\uff3d\u27e7\u3011\u3015"
_MARKER_CONTENT = re.compile(r"s[0-9]+([,;]s[0-9]+)*", re.ASCII)
_MARKDOWN_CONTROL = re.compile(r"[\\`*_#>|\[\]<~&]")
_LIST_NUMBER = re.compile(r"([0-9]{1,9})([.)]) ")


def is_letter(ch: str) -> bool:
    return ch != "" and unicodedata.category(ch)[0] in "LM"


def is_letter_or_digit(ch: str) -> bool:
    return ch != "" and unicodedata.category(ch)[0] in "LNM"


def _is_ascii_digit(ch: str) -> bool:
    return ch != "" and "0" <= ch <= "9"


def _is_wordish(ch: str) -> bool:
    return is_letter_or_digit(ch) or (ch != "" and ch in HYPHENS)


def normalise(text: str) -> str:
    """Canonical form used for every comparison.

    Soft hyphens dropped, every quotation-mark style mapped to ``'`` or ``"``
    (a doubled ``''`` and an opening ``,,`` count as ``"``), every whitespace
    character turned into a space, runs of spaces collapsed, ends trimmed.
    """
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == SOFT_HYPHEN:
            continue
        if code in WHITESPACE:
            if not out or out[-1] != " ":
                out.append(" ")
        elif ch in SINGLE_MARKS:
            out.append("'")
        elif ch in DOUBLE_MARKS:
            out.append('"')
        else:
            out.append(ch)
    joined = "".join(out).strip(" ").replace("''", '"')
    # ``,,quoted text''`` - two commas standing in for a low opening quotation mark.
    result: list[str] = []
    i = 0
    while i < len(joined):
        opens_low_quote = (
            joined[i] == ","
            and joined[i + 1 : i + 2] == ","
            and (i == 0 or joined[i - 1] == " ")
            and i + 2 < len(joined)
            and joined[i + 2] not in " ,"
        )
        if opens_low_quote:
            result.append('"')
            i += 2
        else:
            result.append(joined[i])
            i += 1
    return "".join(result)


def model_text_problem(raw: str, kind: str) -> str | None:
    """Why a piece of model-written text may not be used, or ``None`` if it may.

    ``raw`` is the text as the model sent it; ``kind`` is ``statement``,
    ``quote`` or ``query``.

    - every kind: no line break; no control or format character (terminal
      escapes, zero-width and bidi characters); no leading heading,
      blockquote, table or code-fence marker
    - statement: also at most 600 characters; no HTML character reference
      (``&quot;``, ``&#34;``); no citation marker of its own (``[S1]``)
    - quote: also at most 1,200 characters
    """
    for ch in raw:
        code = ord(ch)
        if code in LINE_BREAKS:
            return "contains a line break"
        if code not in (TAB, SOFT_HYPHEN) and unicodedata.category(ch) in ("Cc", "Cf"):
            return "contains a control or format character"
    start = 0
    while start < len(raw) and ord(raw[start]) in WHITESPACE:
        start += 1
    if raw[start:].startswith(("#", ">", "|", "```", "~~~")):
        return "starts with a Markdown block marker"
    text = normalise(raw)
    if kind == "statement":
        if len(text) > MAX_STATEMENT_LENGTH:
            return f"is longer than {MAX_STATEMENT_LENGTH} characters"
        if _CHARACTER_REFERENCE.search(text):
            return "contains an HTML character reference"
        if has_citation_marker(text):
            return "contains a citation marker; cite through the sources list instead"
    if kind == "quote" and len(text) > MAX_QUOTE_LENGTH:
        return f"is longer than {MAX_QUOTE_LENGTH} characters"
    return None


def has_citation_marker(text: str) -> bool:
    """Does the text contain something shaped like one of our citation markers?

    Any bracketed token - ``[ ]`` or the look-alikes U+FF3B/U+FF3D, U+27E6/U+27E7,
    U+3010/U+3011, U+3014/U+3015 - whose content, after Unicode NFKC normalisation
    and with spaces removed, is ``S<digits>`` (or a list such as ``S1,S2``) or
    starts with ``Source``.
    """
    for i, ch in enumerate(text):
        if ch not in _OPENING_BRACKETS:
            continue
        end = i + 1
        while end < len(text) and text[end] not in _CLOSING_BRACKETS:
            end += 1
        if end >= len(text):
            continue
        inner = unicodedata.normalize("NFKC", text[i + 1 : end])
        content = "".join(c for c in inner if ord(c) not in WHITESPACE).lower()
        if _MARKER_CONTENT.fullmatch(content) or content.startswith("source"):
            return True
    return False


def urls_in(text: str) -> list[str]:
    """URLs written in a text (anything around ``://`` or starting ``www.``), without trailing punctuation."""
    urls = []
    for token in normalise(text).split(" "):
        if "://" in token or "www." in token.lower():
            urls.append(token.rstrip(END_PUNCTUATION + ")]\"'").lstrip("([\"'"))
    return urls


def _letters_before(chars: str, index: int) -> int:
    count = 0
    i = index - 1
    while i >= 0 and is_letter(chars[i]):
        count += 1
        i -= 1
    return count


def _letters_after(chars: str, index: int) -> str:
    i = index + 1
    while i < len(chars) and is_letter(chars[i]):
        i += 1
    return chars[index + 1 : i]


def _is_apostrophe(chars: str, index: int) -> bool:
    """Is the ``'`` at ``index`` part of a word (provider's, don't, l'Union, 50(1)'s) rather than a quotation mark?"""
    before = chars[index - 1] if index > 0 else ""
    after = chars[index + 1] if index + 1 < len(chars) else ""
    if not is_letter(after):
        return False
    tail = _letters_after(chars, index).lower()
    if is_letter(before):
        return tail in CONTRACTIONS or _letters_before(chars, index) <= 2
    # A possessive after a digit or a closing bracket: "Article 50(1)'s", "2016's".
    return tail == "s" and (_is_ascii_digit(before) or before in (")", "]"))


@dataclass(frozen=True)
class Quotation:
    text: str  # the quoted words, without the marks
    open: int  # position of the opening mark in the normalised statement
    close: int  # position of the closing mark


def quotation_spans(statement: str) -> list[Quotation] | None:
    """The quotations inside a statement.

    The text is normalised first, so only ``"`` and ``'`` remain as quotation
    marks. A ``'`` that is part of a word, that follows a word without an
    opening mark ("subjects' rights"), or that precedes a digit ("the '95
    Directive") is not a quotation mark. Returns ``None`` when a quotation
    mark is left open.
    """
    chars = normalise(statement)
    spans: list[Quotation] = []
    opened: tuple[str, int] | None = None
    for i, ch in enumerate(chars):
        if ch not in "\"'":
            continue
        if opened and ch != opened[0]:
            continue  # the other mark inside a quotation is content
        if ch == "'" and _is_apostrophe(chars, i):
            continue
        if opened:
            spans.append(Quotation(chars[opened[1] + 1 : i].strip(" "), opened[1], i))
            opened = None
            continue
        if ch == "'":
            before = chars[i - 1] if i > 0 else ""
            after = chars[i + 1] if i + 1 < len(chars) else ""
            if _is_ascii_digit(after):
                continue  # the '95 Directive
            if is_letter_or_digit(before) and not is_letter_or_digit(after):
                continue  # subjects' rights
        opened = (ch, i)
    return None if opened else spans


def quotations_in(statement: str) -> list[str] | None:
    spans = quotation_spans(statement)
    return None if spans is None else [q.text for q in spans]


def mask_quotations(statement: str, quotations: list[Quotation]) -> str:
    """The normalised statement with the words of the given quotations removed (the marks stay)."""
    chars = list(normalise(statement))
    for q in quotations:
        for i in range(q.open + 1, q.close):
            chars[i] = ""
    return "".join(chars)


def has_letter(text: str) -> bool:
    return any(is_letter(ch) for ch in text)


def trim_quotation(text: str) -> str:
    """Remove punctuation and ellipses (``...``, ``[...]``) from the two ends only."""
    current = text.strip(" ")
    while True:
        nxt = current
        for ellipsis in ("[...]", "[…]", "..."):
            if nxt.startswith(ellipsis):
                nxt = nxt[len(ellipsis) :]
            if nxt.endswith(ellipsis):
                nxt = nxt[: len(nxt) - len(ellipsis)]
        nxt = nxt.strip(" " + END_PUNCTUATION)
        if nxt == current:
            return current
        current = nxt


def word_count(text: str) -> int:
    return len([word for word in text.split(" ") if word])


def _index_on_word_boundary(haystack: str, needle: str) -> int:
    """Find ``needle`` in ``haystack`` (both normalised) as whole words.

    Where the needle starts or ends with a letter, digit or hyphen, the
    neighbouring character in the haystack must not be one. So "lawful" is not
    found in "unlawful", nor "compliance" in "Non-compliance".
    """
    if not needle:
        return -1
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            return -1
        before = haystack[index - 1] if index > 0 else ""
        end = index + len(needle)
        after = haystack[end] if end < len(haystack) else ""
        start_ok = not _is_wordish(needle[0]) or not _is_wordish(before)
        end_ok = not _is_wordish(needle[-1]) or not _is_wordish(after)
        if start_ok and end_ok:
            return index
        start = index + 1


def find_verbatim(haystack: str, wanted: str) -> str | None:
    """The excerpt of ``haystack`` that ``wanted`` quotes, or ``None``.

    Verbatim on word boundaries; only the case of the first character may
    differ (a quotation that starts a sentence is often capitalised).
    """
    if not wanted:
        return None
    candidates = [wanted]
    first = wanted[0]
    if is_letter(first):
        swapped = first.upper() if first == first.lower() else first.lower()
        if swapped != first and len(swapped) == 1:
            candidates.append(swapped + wanted[1:])
    for candidate in candidates:
        index = _index_on_word_boundary(haystack, candidate)
        if index != -1:
            return haystack[index : index + len(candidate)]
    return None


def plain_text(text: str) -> str:
    """Render ANY text that did not come from this program as literal characters.

    Control and format characters removed (a tab or line break becomes a
    space), whitespace collapsed, Markdown control characters and ``&`` escaped,
    and a leading list marker (``-``, ``+``, ``1.``, ``1)``) escaped. This is
    the ONE function through which model-written text, user input and strings
    from the API reach the output.
    """
    visible = []
    for ch in text:
        if ord(ch) in WHITESPACE:
            visible.append(" ")
        elif unicodedata.category(ch) not in ("Cc", "Cf"):
            visible.append(ch)
    escaped = _MARKDOWN_CONTROL.sub(lambda m: "\\" + m.group(0), normalise("".join(visible)))
    marker = _LIST_NUMBER.match(escaped)
    if marker:
        return f"{marker.group(1)}\\{marker.group(2)}{escaped[marker.end() - 1 :]}"
    if escaped.startswith(("- ", "+ ")):
        return "\\" + escaped
    return escaped


def printable_url(url: str) -> str:
    """A URL as printable text, or ``""`` unless it parses as an http(s) URL."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    return plain_text(url) if parts.scheme in ("http", "https") and parts.netloc else ""
