/**
 * Grounding: the rule that keeps model-written text tied to retrieved sources.
 *
 * The model never writes the final answer directly. It returns a `Draft`:
 *   - `quotes`     - excerpts it wants shown, each naming the source it came from;
 *   - `statements` - its own sentences, each naming the sources that support it.
 *
 * `validateDraft` then enforces, in code:
 *   1. every source id in the draft was actually retrieved in this run   -> otherwise REJECT;
 *   2. every quote is a verbatim excerpt of the source it names           -> otherwise REJECT;
 *   3. a quotation embedded in a statement is verbatim in a cited source  -> otherwise REJECT;
 *   4. a statement that cites no source is removed and reported;
 *   5. a statement matching a `forbidden` pattern is removed and reported.
 *
 * "Reject" throws `GroundingError`; the agent loop decides what to do next
 * (the examples retry once, then fall back to showing the sources only).
 *
 * This checks that citations are real and quotes are verbatim. It cannot check
 * that a model-written statement is a correct reading of the source - that is
 * why the rendered output keeps quoted text and model text visibly apart.
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
  /** Quotations inside statements shorter than this are not checked (default 20 characters). */
  minEmbeddedQuoteLength?: number;
}

/** Collapse whitespace and unify quote characters so line wrapping does not break a verbatim match. */
export function normalise(text: string): string {
  return text
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}

function embeddedQuotations(text: string, minLength: number): string[] {
  const found: string[] = [];
  for (const match of normalise(text).matchAll(/"([^"]+)"/g)) {
    const inner = match[1]?.trim() ?? "";
    if (inner.length >= minLength) found.push(inner);
  }
  return found;
}

export function validateDraft(draft: Draft, sources: Source[], options: ValidateOptions = {}): GroundedAnswer {
  if (draft.status === "no_support") {
    return { status: "no_support", quotes: [], statements: [], removed: [] };
  }

  const byId = new Map(sources.map((s) => [s.id, s]));
  const minLength = options.minEmbeddedQuoteLength ?? 20;
  const violations: string[] = [];

  for (const quote of draft.quotes) {
    const source = byId.get(quote.source);
    if (!source) {
      violations.push(`quote cites "${quote.source}", which was not retrieved`);
      continue;
    }
    const wanted = normalise(quote.text);
    if (wanted.length === 0) {
      violations.push(`quote from ${quote.source} is empty`);
    } else if (!normalise(source.text).includes(wanted)) {
      violations.push(`quote from ${quote.source} is not a verbatim excerpt of that source`);
    }
  }

  const kept: DraftStatement[] = [];
  const removed: RemovedStatement[] = [];
  for (const statement of draft.statements) {
    const unknown = statement.sources.filter((id) => !byId.has(id));
    if (unknown.length > 0) {
      violations.push(`statement cites ${unknown.map((id) => `"${id}"`).join(", ")}, which was not retrieved`);
      continue;
    }
    if (statement.sources.length === 0) {
      removed.push({ text: statement.text, reason: "uncited" });
      continue;
    }
    if (options.forbidden?.some((pattern) => pattern.test(statement.text))) {
      removed.push({ text: statement.text, reason: "forbidden" });
      continue;
    }
    const cited = statement.sources.map((id) => normalise(byId.get(id)?.text ?? ""));
    const invented = embeddedQuotations(statement.text, minLength).filter(
      (quotation) => !cited.some((text) => text.includes(quotation)),
    );
    if (invented.length > 0) {
      violations.push(`statement contains a quotation that is not in the sources it cites: "${invented[0]}"`);
      continue;
    }
    kept.push(statement);
  }

  if (violations.length > 0) throw new GroundingError(violations);

  const status = kept.length === 0 && draft.quotes.length === 0 ? "no_support" : "answered";
  return { status, quotes: draft.quotes, statements: kept, removed };
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
      description: "Verbatim excerpts copied character for character from one source.",
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
      description: "Your own sentences. Each one lists the source ids that support it.",
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

/** Narrow an unknown tool input to a `Draft` (the schema is not trusted blindly). */
export function parseDraft(value: unknown): Draft {
  const fail = (): never => {
    throw new GroundingError(["the model's reply does not have the expected shape"]);
  };
  if (typeof value !== "object" || value === null) return fail();
  const v = value as Record<string, unknown>;
  if (v.status !== "answered" && v.status !== "no_support") return fail();
  if (!Array.isArray(v.quotes) || !Array.isArray(v.statements)) return fail();
  const quotes = v.quotes.map((q: unknown) => {
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
    lines.push(`> "${normalise(quote.text)}" [${quote.source}]`, "");
  }

  lines.push(`## ${headings.statements}`, "");
  if (answer.statements.length === 0) lines.push("_None._", "");
  for (const statement of answer.statements) {
    statement.sources.forEach((id) => used.add(id));
    lines.push(`- ${statement.text} ${statement.sources.map((id) => `[${id}]`).join("")}`);
  }
  if (answer.statements.length > 0) lines.push("");

  if (answer.removed.length > 0) {
    lines.push("## Removed by the grounding check", "");
    for (const item of answer.removed) {
      const why = item.reason === "uncited" ? "cited no source" : "stated a conclusion this example does not make";
      lines.push(`- (${why}) ${item.text}`);
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
