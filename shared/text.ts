/**
 * Text handling shared by the grounding check: normalisation, the rules for
 * what model-written text may contain, quotation detection, word-boundary
 * matching and Markdown escaping.
 *
 * Everything is defined on explicit code-point sets (no regex `\s` / `\d`
 * defaults), so the Python twin (`gdpr_assistant/text.py`) behaves the same.
 * `shared/grounding-vectors.json` holds the cases both test suites run.
 */

/** Characters treated as whitespace. */
const WHITESPACE = new Set<number>([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
  0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);
/** Whitespace that ends a line. */
const LINE_BREAKS = new Set<number>([0x0a, 0x0b, 0x0c, 0x0d, 0x85, 0x2028, 0x2029]);
const SOFT_HYPHEN = 0xad;
const TAB = 0x09;

/** Quotation-mark characters, mapped to `'` or `"` by `normalise`. */
const SINGLE_MARKS = "‘’‚‛‹›`´′＇❛❜";
const DOUBLE_MARKS = "“”„‟«»＂❝❞「」『』″〝〞〟";

const HYPHENS = "-‐‑";
const END_PUNCTUATION = ".,;:!?…";
/** Endings after an apostrophe that make it part of a word: provider's, don't, they're, we've, it'll, I'd, I'm. */
const CONTRACTIONS = new Set(["s", "t", "re", "ve", "ll", "d", "m"]);

/** Trim ASCII spaces only (not `String.prototype.trim`, which also strips U+FEFF and other characters). */
export function trimSpaces(text: string): string {
  let start = 0;
  let end = text.length;
  while (start < end && text[start] === " ") start++;
  while (end > start && text[end - 1] === " ") end--;
  return text.slice(start, end);
}

export const isLetterOrDigit = (ch: string): boolean => ch !== "" && /^[\p{L}\p{N}\p{M}]$/u.test(ch);
const isLetter = (ch: string): boolean => ch !== "" && /^[\p{L}\p{M}]$/u.test(ch);
const isAsciiDigit = (ch: string): boolean => ch.length === 1 && ch >= "0" && ch <= "9";
const isWordish = (ch: string): boolean => isLetterOrDigit(ch) || (ch !== "" && HYPHENS.includes(ch));

/** Length in code points (not UTF-16 units). */
export function length(text: string): number {
  return Array.from(text).length;
}

/**
 * Canonical form used for every comparison: soft hyphens dropped, every
 * quotation-mark style mapped to `'` or `"` (a doubled `''` and an opening
 * `,,` count as `"`), every whitespace character turned into a space, runs
 * of spaces collapsed, ends trimmed.
 */
export function normalise(text: string): string {
  const parts: string[] = [];
  let lastWasSpace = false;
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    if (code === SOFT_HYPHEN) continue;
    if (WHITESPACE.has(code)) {
      if (!lastWasSpace) parts.push(" ");
      lastWasSpace = true;
      continue;
    }
    lastWasSpace = false;
    if (SINGLE_MARKS.includes(ch)) parts.push("'");
    else if (DOUBLE_MARKS.includes(ch)) parts.push('"');
    else parts.push(ch);
  }
  const joined = trimSpaces(parts.join("")).split("''").join('"');
  if (!joined.includes(",,")) return joined;
  // `,,quoted text''` - two commas standing in for a low opening quotation mark.
  const out: string[] = [];
  for (let i = 0; i < joined.length; i++) {
    const opensLowQuote =
      joined[i] === "," &&
      joined[i + 1] === "," &&
      (i === 0 || joined[i - 1] === " ") &&
      i + 2 < joined.length &&
      joined[i + 2] !== " " &&
      joined[i + 2] !== ",";
    if (opensLowQuote) {
      out.push('"');
      i++;
    } else out.push(joined[i] ?? "");
  }
  return out.join("");
}

export const MAX_STATEMENT_LENGTH = 600;
export const MAX_QUOTE_LENGTH = 1200;

/**
 * Why a piece of model-written text may not be used, or `undefined` if it may.
 * `raw` is the text as the model sent it.
 *
 *   every kind   no line break; no control or format character (terminal
 *                escapes, zero-width and bidi characters); no leading heading,
 *                blockquote, table or code-fence marker
 *   statement    also: at most 600 characters; no HTML character reference
 *                (`&quot;`, `&#34;`); no citation marker of its own (`[S1]`)
 *   quote        also: at most 1,200 characters
 */
export function modelTextProblem(raw: string, kind: "statement" | "quote" | "query"): string | undefined {
  for (const ch of raw) {
    const code = ch.codePointAt(0) ?? 0;
    if (LINE_BREAKS.has(code)) return "contains a line break";
    if (code !== TAB && code !== SOFT_HYPHEN && /^[\p{Cc}\p{Cf}]$/u.test(ch)) return "contains a control or format character";
  }
  let start = 0;
  while (start < raw.length && WHITESPACE.has(raw.charCodeAt(start))) start++;
  const lead = raw.slice(start, start + 3);
  if (lead.startsWith("#") || lead.startsWith(">") || lead.startsWith("|") || lead.startsWith("```") || lead.startsWith("~~~")) {
    return "starts with a Markdown block marker";
  }
  const text = normalise(raw);
  if (kind === "statement") {
    if (length(text) > MAX_STATEMENT_LENGTH) return `is longer than ${MAX_STATEMENT_LENGTH} characters`;
    if (/&(#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]{1,31});/.test(text)) return "contains an HTML character reference";
    if (hasCitationMarker(text)) return "contains a citation marker; cite through the sources list instead";
  }
  if (kind === "quote" && length(text) > MAX_QUOTE_LENGTH) return `is longer than ${MAX_QUOTE_LENGTH} characters`;
  return undefined;
}

const OPENING_BRACKETS = "[\uFF3B\u27E6\u3010\u3014";
const CLOSING_BRACKETS = "]\uFF3D\u27E7\u3011\u3015";

/**
 * Does the text contain something shaped like one of our citation markers?
 * Any bracketed token - `[ ]` or the look-alikes `\uFF3B \uFF3D`, `\u27E6 \u27E7`, `\u3010 \u3011`,
 * `\u3014 \u3015` - whose content, after Unicode NFKC normalisation and with spaces removed, is
 * `S<digits>` (or a list such as `S1,S2`) or starts with `Source`.
 */
export function hasCitationMarker(text: string): boolean {
  const chars = Array.from(text);
  for (let i = 0; i < chars.length; i++) {
    if (!OPENING_BRACKETS.includes(chars[i] ?? " ")) continue;
    let end = i + 1;
    while (end < chars.length && !CLOSING_BRACKETS.includes(chars[end] ?? " ")) end++;
    if (end >= chars.length) continue;
    const content = Array.from(chars.slice(i + 1, end).join("").normalize("NFKC"))
      .filter((ch) => !WHITESPACE.has(ch.codePointAt(0) ?? 0))
      .join("")
      .toLowerCase();
    if (/^s[0-9]+([,;]s[0-9]+)*$/.test(content) || content.startsWith("source")) return true;
  }
  return false;
}

/** URLs written in a text (anything around `://` or starting `www.`), without trailing punctuation. */
export function urlsIn(text: string): string[] {
  return normalise(text)
    .split(" ")
    .filter((token) => token.includes("://") || token.toLowerCase().includes("www."))
    .map((token) => {
      let url = token;
      while (url.length > 0 && (END_PUNCTUATION + ")]\"'").includes(url[url.length - 1] ?? "")) url = url.slice(0, -1);
      while (url.length > 0 && "([\"'".includes(url[0] ?? "")) url = url.slice(1);
      return url;
    });
}

function lettersBefore(chars: string[], index: number): number {
  let count = 0;
  for (let i = index - 1; i >= 0 && isLetter(chars[i] ?? ""); i--) count++;
  return count;
}

function lettersAfter(chars: string[], index: number): string {
  let run = "";
  for (let i = index + 1; i < chars.length && isLetter(chars[i] ?? ""); i++) run += chars[i];
  return run;
}

/** Is the `'` at `index` part of a word (provider's, don't, l'Union, 50(1)'s) rather than a quotation mark? */
function isApostrophe(chars: string[], index: number): boolean {
  const before = chars[index - 1] ?? "";
  const after = chars[index + 1] ?? "";
  if (!isLetter(after)) return false;
  const tail = lettersAfter(chars, index).toLowerCase();
  if (isLetter(before)) return CONTRACTIONS.has(tail) || lettersBefore(chars, index) <= 2;
  // A possessive after a digit or a closing bracket: "Article 50(1)'s", "2016's".
  return tail === "s" && (isAsciiDigit(before) || before === ")" || before === "]");
}

export interface Quotation {
  /** The quoted words, without the marks. */
  text: string;
  /** Positions of the opening and closing mark, in code points of the normalised statement. */
  open: number;
  close: number;
}

/**
 * The quotations inside a statement. The text is normalised first, so only
 * `"` and `'` remain as quotation marks. A `'` that is part of a word, that
 * follows a word without an opening mark ("subjects' rights"), or that
 * precedes a digit ("the '95 Directive") is not a quotation mark.
 * Returns `undefined` when a quotation mark is left open.
 */
export function quotationSpans(statement: string): Quotation[] | undefined {
  const chars = Array.from(normalise(statement));
  const spans: Quotation[] = [];
  let open: { mark: string; at: number } | undefined;
  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i];
    if (ch !== '"' && ch !== "'") continue;
    if (open && ch !== open.mark) continue; // the other mark inside a quotation is content
    if (ch === "'" && isApostrophe(chars, i)) continue;
    if (open) {
      spans.push({ text: trimSpaces(chars.slice(open.at + 1, i).join("")), open: open.at, close: i });
      open = undefined;
      continue;
    }
    if (ch === "'") {
      const before = chars[i - 1] ?? "";
      const after = chars[i + 1] ?? "";
      if (isAsciiDigit(after)) continue; // the '95 Directive
      if (isLetterOrDigit(before) && !isLetterOrDigit(after)) continue; // subjects' rights
    }
    open = { mark: ch, at: i };
  }
  return open ? undefined : spans;
}

export function quotationsIn(statement: string): string[] | undefined {
  return quotationSpans(statement)?.map((q) => q.text);
}

/** The normalised statement with the words of the given quotations removed (the marks stay). */
export function maskQuotations(statement: string, quotations: Quotation[]): string {
  const chars = Array.from(normalise(statement));
  for (const q of quotations) for (let i = q.open + 1; i < q.close; i++) chars[i] = "";
  return chars.join("");
}

export function hasLetter(text: string): boolean {
  return Array.from(text).some(isLetter);
}

/** Remove punctuation and ellipses (`...`, `[...]`) from the two ends only. */
export function trimQuotation(text: string): string {
  let current = trimSpaces(text);
  for (;;) {
    let next = current;
    for (const ellipsis of ["[...]", "[\u2026]", "..."]) {
      if (next.startsWith(ellipsis)) next = next.slice(ellipsis.length);
      if (next.endsWith(ellipsis)) next = next.slice(0, -ellipsis.length);
    }
    next = trimSpaces(next);
    while (next.length > 0 && END_PUNCTUATION.includes(next[0] ?? "")) next = trimSpaces(next.slice(1));
    while (next.length > 0 && END_PUNCTUATION.includes(next[next.length - 1] ?? "")) next = trimSpaces(next.slice(0, -1));
    if (next === current) return current;
    current = next;
  }
}

export function wordCount(text: string): number {
  return text.split(" ").filter((word) => word.length > 0).length;
}

function charBefore(text: string, index: number): string {
  if (index <= 0) return "";
  const unit = text.charCodeAt(index - 1);
  const isLowSurrogate = unit >= 0xdc00 && unit <= 0xdfff;
  return isLowSurrogate && index >= 2 ? String.fromCodePoint(text.codePointAt(index - 2) ?? 0) : (text[index - 1] ?? "");
}

function charAt(text: string, index: number): string {
  return index >= text.length ? "" : String.fromCodePoint(text.codePointAt(index) ?? 0);
}

/**
 * Find `needle` in `haystack` (both normalised) as whole words: where the
 * needle starts or ends with a letter, digit or hyphen, the neighbouring
 * character in the haystack must not be one. So "lawful" is not found in
 * "unlawful", nor "compliance" in "Non-compliance". Returns the index or -1.
 */
function indexOnWordBoundary(haystack: string, needle: string): number {
  if (needle.length === 0) return -1;
  const first = charAt(needle, 0);
  const last = charBefore(needle, needle.length);
  for (let from = 0; ; ) {
    const index = haystack.indexOf(needle, from);
    if (index === -1) return -1;
    const startOk = !isWordish(first) || !isWordish(charBefore(haystack, index));
    const endOk = !isWordish(last) || !isWordish(charAt(haystack, index + needle.length));
    if (startOk && endOk) return index;
    from = index + 1;
  }
}

/**
 * The excerpt of `haystack` that `wanted` quotes, or `undefined`. Verbatim on
 * word boundaries; only the case of the first character may differ (a
 * quotation that starts a sentence is often capitalised).
 */
export function findVerbatim(haystack: string, wanted: string): string | undefined {
  const candidates = [wanted];
  const first = charAt(wanted, 0);
  if (isLetter(first)) {
    const rest = wanted.slice(first.length);
    const swapped = first === first.toLowerCase() ? first.toUpperCase() : first.toLowerCase();
    if (swapped !== first && length(swapped) === 1) candidates.push(swapped + rest);
  }
  for (const candidate of candidates) {
    const index = indexOnWordBoundary(haystack, candidate);
    if (index !== -1) return haystack.slice(index, index + candidate.length);
  }
  return undefined;
}

/**
 * Render ANY text that did not come from this program as literal characters:
 * control and format characters removed (a tab or line break becomes a
 * space), whitespace collapsed, Markdown control characters and `&` escaped,
 * and a leading list marker (`-`, `+`, `1.`, `1)`) escaped. This is the ONE
 * function through which model-written text, user input and strings from the
 * API or the configuration reach the output.
 */
export function plainText(text: string): string {
  let visible = "";
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    if (WHITESPACE.has(code)) visible += " ";
    else if (!/^[\p{Cc}\p{Cf}]$/u.test(ch)) visible += ch;
  }
  const escaped = normalise(visible).replace(/[\\`*_#>|[\]<~&]/g, "\\$&");
  const marker = /^([0-9]{1,9})([.)]) /.exec(escaped);
  if (marker) return `${marker[1]}\\${marker[2]}${escaped.slice(marker[0].length - 1)}`;
  if (escaped.startsWith("- ") || escaped.startsWith("+ ")) return `\\${escaped}`;
  return escaped;
}

/** `plainText` inside double quotation marks, with any `"` in the text escaped, so the end of the quoted text is unambiguous. */
export function quotedPlainText(text: string): string {
  return `"${plainText(text).replace(/"/g, '\\"')}"`;
}

/** A URL as printable text, or `""` unless it parses as an http(s) URL. */
export function printableUrl(url: string): string {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:" ? plainText(url) : "";
  } catch {
    return "";
  }
}
