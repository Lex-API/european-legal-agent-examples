/**
 * Grounding: the rule that keeps model-written text tied to retrieved sources.
 *
 * The model never writes the final answer directly. It returns a `Draft`:
 *   - `quotes`     - excerpts it wants shown, each naming the source it came from;
 *   - `statements` - its own sentences, each naming the sources that support it.
 *
 * `validateDraft` REJECTS the draft (throws `GroundingError`) when:
 *   1. a quote or statement cites a source id that was not retrieved in this run;
 *   2. a quote is not a verbatim excerpt of the source it names, or is shorter
 *      than three words;
 *   3. a quotation inside a statement (any quotation that contains a letter)
 *      is not verbatim in a source that statement cites, or a quotation mark
 *      is left open;
 *   4. a quote or statement breaks the rules for model text in `text.ts`
 *      (line breaks, control characters, leading block markers, length; for
 *      statements also HTML character references and bracketed citation
 *      markers such as `[S1]`);
 *   5. a statement contains a URL that is not the URL of a source it cites;
 *   6. there are more than 12 quotes or more than 12 statements.
 *
 * It REMOVES, and counts without repeating them:
 *   7. statements that cite no source;
 *   8. statements that match the conclusion filter (`conclusions.ts`), when
 *      the caller passes patterns. Verified quotations are masked first.
 *
 * A statement that is empty or only whitespace is dropped silently.
 *
 * "Verbatim" means: after normalisation (see `normalise`), the words appear
 * in the source in the same order, starting and ending on a word boundary.
 * Punctuation and ellipses at the two ends of a quotation are ignored, and
 * the case of its first character may differ. A quote is printed with the
 * source's own characters, not the model's.
 *
 * The examples retry a rejected draft once, then show the sources only.
 *
 * What this does not check: that an excerpt is quoted in context, or that a
 * model-written statement is a correct reading of the source it cites. That
 * is why the output keeps quoted text and model text in separate sections.
 */
import { statesConclusion } from "./conclusions.ts";
import {
  findVerbatim,
  hasLetter,
  maskQuotations,
  modelTextProblem,
  normalise,
  plainText,
  printableUrl,
  quotationSpans,
  trimQuotation,
  urlsIn,
  wordCount,
} from "./text.ts";

export { normalise, plainText } from "./text.ts";

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

/** A draft that passed validation. */
export interface GroundedAnswer {
  status: "answered" | "no_support";
  /** `text` is the excerpt as it stands in the source. */
  quotes: DraftQuote[];
  statements: DraftStatement[];
  /** How many statements were removed, by reason. Their text is not kept. */
  removed: { uncited: number; conclusion: number };
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
  /** Conclusion patterns (see `conclusions.ts`); statements matching one are removed. */
  conclusionPatterns?: string[];
}

export const MAX_QUOTES = 12;
export const MAX_STATEMENTS = 12;
export const MIN_QUOTE_WORDS = 3;

export function validateDraft(draft: Draft, sources: Source[], options: ValidateOptions = {}): GroundedAnswer {
  const none = { uncited: 0, conclusion: 0 };
  if (draft.status === "no_support") return { status: "no_support", quotes: [], statements: [], removed: none };

  // Normalise each source once; drafts can carry many quotes.
  const texts = new Map(sources.map((s) => [s.id, normalise(s.text)]));
  const urls = new Map(sources.map((s) => [s.id, s.url]));
  const violations: string[] = [];
  if (draft.quotes.length > MAX_QUOTES) violations.push(`more than ${MAX_QUOTES} quotes`);
  if (draft.statements.length > MAX_STATEMENTS) violations.push(`more than ${MAX_STATEMENTS} statements`);

  const quotes: DraftQuote[] = [];
  for (const quote of draft.quotes.slice(0, MAX_QUOTES)) {
    const sourceText = texts.get(quote.source);
    if (sourceText === undefined) {
      violations.push(`quote cites "${quote.source}", which was not retrieved`);
      continue;
    }
    const problem = modelTextProblem(quote.text, "quote");
    if (problem) {
      violations.push(`quote from ${quote.source} ${problem}`);
      continue;
    }
    // A model often wraps the excerpt in quotation marks or ends it with "..."; neither is part of the excerpt.
    let wanted = trimQuotation(normalise(quote.text));
    if (wanted.length >= 2 && wanted.startsWith('"') && wanted.endsWith('"')) wanted = trimQuotation(wanted.slice(1, -1));
    if (wordCount(wanted) < MIN_QUOTE_WORDS) {
      violations.push(`quote from ${quote.source} is shorter than ${MIN_QUOTE_WORDS} words`);
      continue;
    }
    const excerpt = findVerbatim(sourceText, wanted);
    if (excerpt === undefined) {
      violations.push(`quote from ${quote.source} is not a verbatim excerpt of that source (whole words, in order)`);
      continue;
    }
    quotes.push({ source: quote.source, text: excerpt });
  }

  const kept: DraftStatement[] = [];
  const removed = { ...none };
  for (const statement of draft.statements.slice(0, MAX_STATEMENTS)) {
    const ids = [...new Set(statement.sources)];
    const unknown = ids.filter((id) => !texts.has(id));
    if (unknown.length > 0) {
      violations.push(`statement cites ${unknown.map((id) => `"${id}"`).join(", ")}, which was not retrieved`);
      continue;
    }
    const text = normalise(statement.text);
    if (text === "") continue; // an empty statement is dropped: there is nothing to print
    if (ids.length === 0) {
      removed.uncited++;
      continue;
    }
    const problem = modelTextProblem(statement.text, "statement");
    if (problem) {
      violations.push(`a statement ${problem}`);
      continue;
    }
    const foreignUrl = urlsIn(text).find((url) => !ids.some((id) => urls.get(id) === url));
    if (foreignUrl !== undefined) {
      violations.push("a statement contains a URL that is not the URL of a source it cites");
      continue;
    }
    const quotations = quotationSpans(text);
    if (quotations === undefined) {
      violations.push("a statement has an unbalanced quotation mark");
      continue;
    }
    // Every quotation that contains a letter must be verbatim in a source this statement cites.
    const cited = ids.map((id) => texts.get(id) ?? "");
    const invented = quotations
      .map((q) => trimQuotation(q.text))
      .find((q) => hasLetter(q) && !cited.some((source) => findVerbatim(source, q) !== undefined));
    if (invented !== undefined) {
      violations.push(`a statement contains a quotation that is not in the sources it cites (whole words, in order): "${invented}"`);
      continue;
    }
    // The verified quotations are the source's words, not the model's: mask them before the conclusion filter.
    if (options.conclusionPatterns && statesConclusion(maskQuotations(text, quotations), options.conclusionPatterns)) {
      removed.conclusion++;
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
      description:
        "At most 12 verbatim excerpts, each copied character for character from one source: one line, whole words, 3 words to 1,200 characters.",
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
      description:
        "At most 12 of your own sentences, one plain-text line of at most 600 characters each (no Markdown, no line breaks, no URLs, no [S1] markers in the text; quotation marks only around words copied exactly from a cited source). Each lists the source ids that support it.",
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

/**
 * The citation line for one source: title, pinpoint, ECLI, CELEX, link.
 * Every part comes from the API, so every part is printed through
 * `plainText`; a link is printed only if it parses as an http(s) URL.
 */
export function citation(source: Source): string {
  const parts = [plainText(source.title)];
  if (source.articleRef) parts.push(`${plainText(articleLabel(source.articleRef))} (article id ${plainText(source.articleRef)})`);
  if (source.paragraph !== undefined) parts.push(`paragraph ${source.paragraph}`);
  if (source.ecli) parts.push(plainText(source.ecli));
  parts.push(`CELEX ${plainText(source.celex)}`);
  const url = printableUrl(source.url);
  if (url) parts.push(url);
  return parts.join(" - ");
}

/** A source's citation line followed by its caveats, as a list item. */
export function sourceListItem(source: Source): string {
  return [`- [${source.id}] ${citation(source)}`, ...source.notes.map((note) => `  - Note: ${plainText(note)}`)].join("\n");
}

/** One model-written statement as a list item, followed by its citation markers. */
export function renderStatement(statement: DraftStatement): string {
  return `- ${plainText(statement.text)} ${statement.sources.map((id) => `[${id}]`).join("")}`;
}

/** What the check removed: counts and reasons only - the removed text is never repeated. */
export function renderRemoved(removed: GroundedAnswer["removed"]): string[] {
  const lines: string[] = [];
  if (removed.uncited > 0) lines.push(`- ${removed.uncited} statement(s) removed: cited no source.`);
  if (removed.conclusion > 0) lines.push(`- ${removed.conclusion} statement(s) removed: matched the conclusion filter.`);
  return lines;
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
    lines.push(`> "${plainText(quote.text)}" [${quote.source}]`, "");
  }

  lines.push(`## ${headings.statements}`, "");
  if (answer.statements.length === 0) lines.push("_None._", "");
  for (const statement of answer.statements) {
    statement.sources.forEach((id) => used.add(id));
    lines.push(renderStatement(statement));
  }
  if (answer.statements.length > 0) lines.push("");

  const removed = renderRemoved(answer.removed);
  if (removed.length > 0) lines.push("## Removed by the grounding check", "", ...removed, "");

  lines.push("## Sources", "");
  for (const id of [...used].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)))) {
    const source = byId.get(id);
    if (!source) continue;
    lines.push(sourceListItem(source));
  }
  return lines.join("\n") + "\n";
}

/** The sources as shown to the MODEL (not printed): id, citation line, caveats, then the text. */
export function sourcesForPrompt(sources: Source[]): string {
  return sources
    .map((s) => [`<source id="${s.id}">`, citation(s), ...s.notes.map((n) => `Note: ${n}`), "", s.text, "</source>"].join("\n"))
    .join("\n\n");
}
