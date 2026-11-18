/**
 * Grounding: the rule that keeps model-written text tied to retrieved sources.
 *
 * The model never writes the final answer directly. It returns a `Draft`:
 *   - `quotes`     - excerpts it wants shown, each naming the source it came from;
 *   - `statements` - its own sentences, each naming the sources that support it.
 *
 * `validateDraft` then enforces, in code:
 *   1. every source id in the draft was actually retrieved in this run        -> otherwise REJECT;
 *   2. every quote is at least three words long and a verbatim excerpt of the
 *      source it names                                                        -> otherwise REJECT;
 *   3. every quotation inside a statement (two words or more, or eight
 *      characters or more, in any style of quotation marks) is verbatim in a
 *      source that statement cites; quotation marks must be balanced          -> otherwise REJECT;
 *   4. quotes and statements are single lines of plain text: no line breaks,
 *      no leading Markdown block marker (heading, blockquote, list, table,
 *      code fence)                                                            -> otherwise REJECT;
 *   5. a statement that cites no source is removed and reported;
 *   6. a statement matching a `forbidden` pattern is removed and reported.
 *
 * "Verbatim" is checked after collapsing whitespace, unifying quotation-mark
 * styles and dropping soft hyphens, on both sides.
 *
 * "Reject" throws `GroundingError`; the agent loop decides what to do next
 * (the examples retry once, then fall back to showing the sources only).
 *
 * This checks that citations are real and excerpts are verbatim. It cannot
 * check that an excerpt is quoted in context, or that a model-written
 * statement is a correct reading of the source - that is why the rendered
 * output keeps quoted text and model text visibly apart, and escapes
 * Markdown in everything the model wrote.
 */

/** One retrieved piece of text the model may quote or cite. */
export interface Source {
  /** Run-local id (`S1`, `S2`, ...). The only handle the model is given. */
  id: string;
  kind: "legislation" | "case-law" | "document";
  /** CELEX number of the document the text comes from. */
  celex: string;
  /** Article id as returned by the API (e.g. `art_9`), when the source is an article. */
  articleRef?: string;
  /** Paragraph number, when the source is a paragraph of a judgment. */
  paragraph?: number;
  /** ECLI, for case law. */
  ecli?: string;
  title: string;
  /** EUR-Lex link. */
  url: string;
  /** The retrieved text, exactly as the API returned it. */
  text: string;
  /** Caveats the output must carry (consolidated-text note, version status, ...). */
  notes: string[];
}

export interface DraftQuote {
  source: string;
  text: string;
}

export interface DraftStatement {
  text: string;
  sources: string[];
}

/** What the model returns. */
export interface Draft {
  /** `no_support`: the model says the sources do not support an answer. */
  status: "answered" | "no_support";
  quotes: DraftQuote[];
  statements: DraftStatement[];
}

export interface RemovedStatement {
  text: string;
  reason: "uncited" | "forbidden";
}

/** A draft that passed validation. */
export interface GroundedAnswer {
  status: "answered" | "no_support";
  quotes: DraftQuote[];
  statements: DraftStatement[];
  removed: RemovedStatement[];
}

export class GroundingError extends Error {
  readonly violations: string[];
  constructor(violations: string[]) {
    super(`Draft rejected by the grounding check:\n- ${violations.join("\n- ")}`);
    this.name = "GroundingError";
    this.violations = violations;
  }
}

export interface ValidateOptions {
  /** Statements matching any of these are removed (e.g. wording that states a legal conclusion). */
  forbidden?: RegExp[];
}

/**
 * Collapse whitespace, unify quotation-mark styles and drop soft hyphens, so
 * that line wrapping and typography do not break a verbatim match.
 */
export function normalise(text: string): string {
  return text
    .replace(/\u00AD/g, "")
    .replace(/[\u2018\u2019\u201A\u201B\u2039\u203A]/g, "'")
    .replace(/[\u201C\u201D\u201E\u201F\u00AB\u00BB]/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}

/** Why a piece of model text may not be rendered as one plain line, or `undefined` if it may. */
function shapeProblem(text: string, kind: "quote" | "statement"): string | undefined {
  if (/[\n\r\u2028\u2029]/.test(text)) return "contains a line break";
  const start = text.trimStart();
  // A quote may open with a paragraph number ("1. The controller ..."): laws are written that way.
  const markers = kind === "quote" ? /^(#|>|\||```|~~~|[-*+]\s)/ : /^(#|>|\||```|~~~|[-*+]\s|\d+[.)]\s)/;
  if (markers.test(start)) return "starts with a Markdown block marker";
  return undefined;
}

const WORD_CHAR = /[\p{L}\p{N}]/u;

/**
 * The quotations inside a statement, in any style of quotation marks (the
 * text is normalised first, so only `"` and `'` remain). A `'` between two
 * letters, or closing nothing, is an apostrophe ("provider's", "subjects'").
 * Returns `undefined` when a quotation mark is left open.
 */
export function embeddedQuotations(statement: string): string[] | undefined {
  const text = normalise(statement);
  const spans: string[] = [];
  let open: { mark: string; at: number } | undefined;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch !== '"' && ch !== "'") continue;
    const before = text[i - 1] ?? " ";
    const after = text[i + 1] ?? " ";
    if (open) {
      if (ch !== open.mark) continue;
      if (ch === "'" && WORD_CHAR.test(before) && WORD_CHAR.test(after)) continue; // apostrophe inside a word
      spans.push(text.slice(open.at + 1, i).trim());
      open = undefined;
    } else if (ch === '"') {
      open = { mark: ch, at: i };
    } else if (!WORD_CHAR.test(before) && after !== " ") {
      open = { mark: ch, at: i }; // a single quote opens only at the start of a word
    }
  }
  return open ? undefined : spans;
}

/** Quotations long enough to carry meaning: two words or more, or eight characters or more. */
function mustBeVerbatim(quotation: string): boolean {
  return quotation.length >= 8 || quotation.split(" ").filter(Boolean).length >= 2;
}

const MIN_QUOTE_WORDS = 3;

export function validateDraft(draft: Draft, sources: Source[], options: ValidateOptions = {}): GroundedAnswer {
  if (draft.status === "no_support") {
    return { status: "no_support", quotes: [], statements: [], removed: [] };
  }

  // Normalise each source once; drafts can carry many quotes.
  const byId = new Map(sources.map((s) => [s.id, normalise(s.text)]));
  const violations: string[] = [];

  const quotes: DraftQuote[] = [];
  for (const quote of draft.quotes) {
    const sourceText = byId.get(quote.source);
    if (sourceText === undefined) {
      violations.push(`quote cites "${quote.source}", which was not retrieved`);
      continue;
    }
    const wanted = normalise(quote.text);
    const shape = shapeProblem(quote.text.trim(), "quote");
    if (shape) {
      violations.push(`quote from ${quote.source} ${shape}; a quote is one line of plain text`);
    } else if (wanted.split(" ").filter(Boolean).length < MIN_QUOTE_WORDS) {
      violations.push(`quote from ${quote.source} is shorter than ${MIN_QUOTE_WORDS} words`);
    } else if (!sourceText.includes(wanted)) {
      violations.push(`quote from ${quote.source} is not a verbatim excerpt of that source`);
    } else {
      quotes.push({ source: quote.source, text: wanted });
    }
  }

  const kept: DraftStatement[] = [];
  const removed: RemovedStatement[] = [];
  for (const statement of draft.statements) {
    const ids = [...new Set(statement.sources)];
    const unknown = ids.filter((id) => !byId.has(id));
    if (unknown.length > 0) {
      violations.push(`statement cites ${unknown.map((id) => `"${id}"`).join(", ")}, which was not retrieved`);
      continue;
    }
    const shape = shapeProblem(statement.text.trim(), "statement");
    if (shape) {
      violations.push(`a statement ${shape}; each statement is one line of plain text`);
      continue;
    }
    const text = normalise(statement.text);
    if (ids.length === 0) {
      removed.push({ text, reason: "uncited" });
      continue;
    }
    if (options.forbidden?.some((pattern) => pattern.test(text))) {
      removed.push({ text, reason: "forbidden" });
      continue;
    }
    const quotations = embeddedQuotations(text);
    if (quotations === undefined) {
      violations.push("a statement has an unbalanced quotation mark");
      continue;
    }
    const cited = ids.map((id) => byId.get(id) ?? "");
    const invented = quotations.filter(mustBeVerbatim).find((q) => !cited.some((source) => source.includes(q)));
    if (invented !== undefined) {
      violations.push(`statement contains a quotation that is not in the sources it cites: "${invented}"`);
      continue;
    }
    kept.push({ text, sources: ids });
  }

  if (violations.length > 0) throw new GroundingError(violations);

  const status = kept.length === 0 && quotes.length === 0 ? "no_support" : "answered";
  return { status, quotes, statements: kept, removed };
}

/** JSON Schema for `Draft`, used as the model's tool input schema. */
export const DRAFT_SCHEMA = {
  type: "object",
  properties: {
    status: {
      type: "string",
      enum: ["answered", "no_support"],
      description: "no_support when the sources do not contain text that supports an answer.",
    },
    quotes: {
      type: "array",
      description: "Verbatim excerpts copied character for character from one source: one line each, at least three words.",
      items: {
        type: "object",
        properties: {
          source: { type: "string", description: "Source id, e.g. S1." },
          text: { type: "string", description: "The excerpt, copied exactly." },
        },
        required: ["source", "text"],
        additionalProperties: false,
      },
    },
    statements: {
      type: "array",
      description: "Your own sentences, one plain-text line each (no Markdown, no line breaks). Each lists the source ids that support it.",
      items: {
        type: "object",
        properties: {
          text: { type: "string" },
          sources: { type: "array", items: { type: "string" } },
        },
        required: ["text", "sources"],
        additionalProperties: false,
      },
    },
  },
  required: ["status", "quotes", "statements"],
  additionalProperties: false,
} as const;

/** `DRAFT_SCHEMA` without `quotes`, for callers that only want cited statements. */
export const STATEMENTS_SCHEMA = {
  type: "object",
  properties: { status: DRAFT_SCHEMA.properties.status, statements: DRAFT_SCHEMA.properties.statements },
  required: ["status", "statements"],
  additionalProperties: false,
} as const;

/** Narrow an unknown tool input to a `Draft` (the schema is not trusted blindly). */
export function parseDraft(value: unknown): Draft {
  const fail = (): never => {
    throw new GroundingError(["the model's reply does not have the expected shape"]);
  };
  if (typeof value !== "object" || value === null) return fail();
  const v = value as Record<string, unknown>;
  if (v.status !== "answered" && v.status !== "no_support") return fail();
  const rawQuotes = v.quotes ?? []; // absent when the caller used STATEMENTS_SCHEMA
  if (!Array.isArray(rawQuotes) || !Array.isArray(v.statements)) return fail();
  const quotes = rawQuotes.map((q: unknown) => {
    const o = q as Record<string, unknown> | null;
    if (!o || typeof o.source !== "string" || typeof o.text !== "string") return fail();
    return { source: o.source, text: o.text };
  });
  const statements = v.statements.map((s: unknown) => {
    const o = s as Record<string, unknown> | null;
    if (!o || typeof o.text !== "string" || !Array.isArray(o.sources)) return fail();
    if (!o.sources.every((id): id is string => typeof id === "string")) return fail();
    return { text: o.text, sources: o.sources };
  });
  return { status: v.status, quotes, statements };
}

// -- Rendering -----------------------------------------------------------

/** `https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:<celex>` */
export function eurLexUrl(celex: string): string {
  return `https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:${encodeURIComponent(celex)}`;
}

/** `art_9` -> `Article 9`; anything else is shown as returned. */
export function articleLabel(articleRef: string): string {
  const match = /^art_(\w+)$/.exec(articleRef);
  return match ? `Article ${match[1]}` : articleRef;
}

/** The citation line for one source: title, pinpoint, CELEX, ECLI, link. */
export function citation(source: Source): string {
  const parts = [source.title];
  if (source.articleRef) parts.push(`${articleLabel(source.articleRef)} (article id ${source.articleRef})`);
  if (source.paragraph !== undefined) parts.push(`paragraph ${source.paragraph}`);
  if (source.ecli) parts.push(source.ecli);
  parts.push(`CELEX ${source.celex}`);
  parts.push(source.url);
  return parts.join(" - ");
}

/**
 * Escape Markdown control characters in text the model wrote, so it renders
 * as literal characters: it cannot open a heading, a blockquote, a link or a
 * citation marker of its own.
 */
export function escapeMarkdown(text: string): string {
  return normalise(text).replace(/[\\`*_#>|[\]<~]/g, "\\$&");
}

/** One model-written statement as a list item, followed by its citation markers. */
export function renderStatement(statement: DraftStatement): string {
  return `- ${escapeMarkdown(statement.text)} ${statement.sources.map((id) => `[${id}]`).join("")}`;
}

export interface RenderHeadings {
  quotes: string;
  statements: string;
}

/**
 * Render a grounded answer as Markdown. Quoted source text and model-written
 * text go in separate sections, and every line carries its source ids.
 */
export function renderAnswer(answer: GroundedAnswer, sources: Source[], headings: RenderHeadings): string {
  const byId = new Map(sources.map((s) => [s.id, s]));
  const used = new Set<string>();
  const lines: string[] = [];

  lines.push(`## ${headings.quotes}`, "");
  if (answer.quotes.length === 0) lines.push("_The model selected no excerpts._", "");
  for (const quote of answer.quotes) {
    used.add(quote.source);
    lines.push(`> "${escapeMarkdown(quote.text)}" [${quote.source}]`, "");
  }

  lines.push(`## ${headings.statements}`, "");
  if (answer.statements.length === 0) lines.push("_None._", "");
  for (const statement of answer.statements) {
    statement.sources.forEach((id) => used.add(id));
    lines.push(renderStatement(statement));
  }
  if (answer.statements.length > 0) lines.push("");

  if (answer.removed.length > 0) {
    lines.push("## Removed by the grounding check", "");
    for (const item of answer.removed) {
      const why = item.reason === "uncited" ? "cited no source" : "stated a conclusion this example does not make";
      lines.push(`- (${why}) ${escapeMarkdown(item.text)}`);
    }
    lines.push("");
  }

  lines.push("## Sources", "");
  for (const id of [...used].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)))) {
    const source = byId.get(id);
    if (!source) continue;
    lines.push(`- [${id}] ${citation(source)}`);
    for (const note of source.notes) lines.push(`  - Note: ${note}`);
  }
  return lines.join("\n") + "\n";
}

/** The sources as shown to the model: id, citation line, caveats, then the text. */
export function sourcesForPrompt(sources: Source[]): string {
  return sources
    .map((s) => [`<source id="${s.id}">`, citation(s), ...s.notes.map((n) => `Note: ${n}`), "", s.text, "</source>"].join("\n"))
    .join("\n\n");
}
