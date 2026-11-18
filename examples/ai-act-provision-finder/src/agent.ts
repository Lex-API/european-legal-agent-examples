/**
 * AI Act provision finder - the agent loop.
 *
 *   1. plan      the model turns the system description into search queries
 *   2. retrieve  LexAPI semantic search; keep AI Act articles only
 *   3. draft     the model picks excerpts and writes notes, citing source ids
 *   4. check     the grounding validator (one retry, then sources only)
 *   5. render    quoted text and model text in separate sections, with citations
 *
 * It finds provisions to read. It does not find the obligations that apply.
 *
 * The loop is a fixed pipeline on purpose: the number of API calls - and so
 * the credit cost of a run - is bounded by `maxQueries`.
 */
import {
  DRAFT_SCHEMA,
  GroundingError,
  citation,
  parseDraft,
  renderAnswer,
  sourcesForPrompt,
  validateDraft,
  type GroundedAnswer,
  type Source,
} from "../../../shared/grounding.ts";
import { CreditTally } from "../../../shared/env.ts";
import type { Llm } from "../../../shared/llm.ts";
import { toSources, type LegislationHit, type LegislationSearch } from "./retrieve.ts";

export interface FinderOptions {
  /** Semantic searches per run (5 credits each). Default 3. */
  maxQueries?: number;
  /** Articles handed to the model. Default 8. */
  maxSources?: number;
}

export interface FinderResult {
  outcome: "answered" | "no_supporting_text" | "sources_only";
  markdown: string;
  queries: string[];
  sources: Source[];
  answer?: GroundedAnswer;
  /** Credits the API reported for this run, as text ("15", "not reported", ...). */
  credits: string;
}

const PLAN_TOOL = {
  name: "plan_searches",
  description: "Propose semantic search queries for finding relevant articles of the EU AI Act.",
  schema: {
    type: "object",
    properties: {
      queries: {
        type: "array",
        description: "One to three short natural-language queries, each describing a kind of provision to look for.",
        items: { type: "string" },
      },
    },
    required: ["queries"],
    additionalProperties: false,
  },
};

const DRAFT_TOOL = {
  name: "submit_findings",
  description: "Submit excerpts from the retrieved provisions and notes on why each may be relevant.",
  schema: DRAFT_SCHEMA,
};

const SYSTEM = `You help a reader find provisions of Regulation (EU) 2024/1689 (the AI Act) that may be relevant to an AI system they describe.

You do not classify the system, you do not say which obligations apply, and you do not give legal advice. Whether a provision applies is for a qualified person to decide. Your notes say why a provision may be worth reading for this system and which question that person has to answer - phrased as a question or a possibility, never as a conclusion.

You only know the provisions given to you as sources. Do not rely on anything else you may remember about the regulation.`;

const RISK_CLASS = String.raw`(high[- ]risk|prohibited|limited[- ]risk|minimal[- ]risk|low[- ]risk)`;
const SUBJECT = String.raw`(system|systems|tool|tools|product|provider|providers|deployer|deployers|company|operator|operators)`;
const DUTY = String.raw`(must|shall|should|has to|have to|needs? to|(is|are) (required|obliged|obligated) to)`;

/**
 * Wording that states a legal conclusion: a classification ("is high-risk",
 * "is not high risk", "is classified as ...", "falls under Annex III"), a
 * duty ("you must", "the provider has to", "must comply", "is mandatory",
 * "are obliged to") or applicability ("applies to your system").
 *
 * A blunt lexical filter for English output: it catches common phrasings,
 * not every one, and nothing in other languages.
 */
export const CONCLUSION_PATTERNS: RegExp[] = [
  new RegExp(String.raw`\b(is|are|was|will be|would be)\s+(not\s+)?(an?\s+)?((classified|considered|regarded|deemed|treated)\s+(as\s+)?)?(an?\s+)?${RISK_CLASS}\b`, "i"),
  new RegExp(String.raw`\b(qualif(y|ies)|counts?|classif(y|ies)|classified)\s+as\s+(an?\s+)?${RISK_CLASS}\b`, "i"),
  /\bfalls?\s+(under|within)\s+(annex|article|the\s+(scope|category|definition))\b/i,
  /\b(is|are)\s+(not\s+)?(compliant|non-compliant|in compliance|exempt|subject to)\b/i,
  /\b(is|are)\s+(mandatory|obligatory|compulsory)\b/i,
  /\b(is|are)\s+(obliged|obligated|required)\s+to\b/i,
  new RegExp(String.raw`\b${DUTY}\s+comply\b`, "i"),
  new RegExp(String.raw`\byou\s+${DUTY}\b`, "i"),
  new RegExp(String.raw`\b${SUBJECT}\s+${DUTY}\b`, "i"),
  /\bappl(y|ies)\s+to\s+(your|the|this)\s+(system|tool|product|company)\b/i,
];

const HUMAN_DECIDES = `## What a human must decide

This output is a reading list, not an assessment. It does not say whether the described system is covered by the AI Act, which category it falls into, or which obligations apply. Those are legal questions for a qualified person, who should read the provisions above in full, in the authentic Official Journal text, together with the rest of the regulation.`;

export async function findProvisions(
  description: string,
  deps: { llm: Llm; search: LegislationSearch },
  options: FinderOptions = {},
): Promise<FinderResult> {
  const maxQueries = options.maxQueries ?? 3;
  const maxSources = options.maxSources ?? 8;

  // 1. Plan ---------------------------------------------------------------
  const plan = (await deps.llm.generate({
    system: SYSTEM,
    user: `AI system description:\n${description}\n\nPropose up to ${maxQueries} search queries.`,
    tool: PLAN_TOOL,
  })) as { queries?: unknown };
  const queries = (Array.isArray(plan.queries) ? plan.queries : [])
    .filter((q): q is string => typeof q === "string")
    .map((q) => q.trim())
    .filter((q) => q.length >= 3 && q.length <= 1000) // the API's limits on `query`
    .slice(0, maxQueries);
  if (queries.length === 0) queries.push(description.trim().slice(0, 1000));

  // 2. Retrieve -----------------------------------------------------------
  const hits: LegislationHit[] = [];
  const tally = new CreditTally();
  for (const query of queries) {
    const outcome = await deps.search.search(query);
    hits.push(...outcome.hits);
    tally.add(outcome.credits);
  }
  const credits = tally.describe();
  const { sources, otherActs } = toSources(hits, maxSources);
  const header = renderHeader(description, queries, deps, hits.length, otherActs);

  if (sources.length === 0) {
    // The "no supporting text found" path: say so, and do not ask the model to fill the gap.
    const markdown = `${header}\n## No supporting text found\n\nThe searches returned no AI Act article for this description, so there is nothing to quote. This says nothing about whether the regulation is relevant to the system: try a more specific description, or search the regulation directly.\n`;
    return { outcome: "no_supporting_text", markdown, queries, sources, credits };
  }

  // 3 + 4. Draft, then check (one retry with the violations fed back) -------
  const baseUser = `AI system description:\n${description}\n\nRetrieved provisions:\n\n${sourcesForPrompt(sources)}\n\nSelect the excerpts most worth reading for this system and add a short note for each. Copy excerpts exactly, one line each. Write each note as one line of plain text and give it the source ids it relies on. If none of the provisions is relevant, set status to no_support.`;
  let answer: GroundedAnswer | undefined;
  let feedback = "";
  for (let attempt = 0; attempt < 2 && !answer; attempt++) {
    const raw = await deps.llm.generate({ system: SYSTEM, user: baseUser + feedback, tool: DRAFT_TOOL });
    try {
      answer = validateDraft(parseDraft(raw), sources, { forbidden: CONCLUSION_PATTERNS });
    } catch (error) {
      if (!(error instanceof GroundingError)) throw error;
      feedback = `\n\nYour previous draft was rejected:\n- ${error.violations.join("\n- ")}\nFix these and submit again.`;
    }
  }

  // 5. Render -------------------------------------------------------------
  if (!answer) {
    const list = sources.map((s) => `- [${s.id}] ${citation(s)}${s.notes.map((n) => `\n  - Note: ${n}`).join("")}`).join("\n");
    const markdown = `${header}\n## Retrieved provisions (no model text)\n\nThe model's draft failed the grounding check twice, so none of it is shown. These are the provisions the search returned:\n\n${list}\n\n${HUMAN_DECIDES}\n`;
    return { outcome: "sources_only", markdown, queries, sources, credits };
  }
  if (answer.status === "no_support") {
    const markdown = `${header}\n## No supporting text found\n\nThe search returned ${sources.length} AI Act article(s), but the model found nothing in them that it could tie to this description. Nothing is quoted.\n\n${HUMAN_DECIDES}\n`;
    return { outcome: "no_supporting_text", markdown, queries, sources, answer, credits };
  }

  const body = renderAnswer(answer, sources, {
    quotes: "Provisions that may be relevant (quoted from the source text)",
    statements: "Why they may be relevant (written by the model - not legal text)",
  });
  return { outcome: "answered", markdown: `${header}\n${body}\n${HUMAN_DECIDES}\n`, queries, sources, answer, credits };
}

function renderHeader(
  description: string,
  queries: string[],
  deps: { llm: Llm; search: LegislationSearch },
  hitCount: number,
  otherActs: number,
): string {
  return [
    "# AI Act provision finder",
    "",
    `**System described:** ${description}`,
    "",
    `Searches run (${deps.search.label}): ${queries.map((q) => `"${q}"`).join("; ")}`,
    `Hits: ${hitCount} in total, ${otherActs} from other acts (not shown). Model: ${deps.llm.label}.`,
    "",
  ].join("\n");
}
