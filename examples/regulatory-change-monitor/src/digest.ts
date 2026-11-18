/**
 * Turn the new items into a cited digest.
 *
 * The list of documents is printed by code, straight from the API records.
 * The model only adds a short grouped summary on top, and every sentence of
 * it must cite the items it is about (the grounding check enforces that).
 */
import {
  GroundingError,
  STATEMENTS_SCHEMA,
  citation,
  escapeMarkdown,
  parseDraft,
  renderStatement,
  sourcesForPrompt,
  validateDraft,
  type GroundedAnswer,
} from "../../../shared/grounding.ts";
import type { Llm } from "../../../shared/llm.ts";
import { itemsToSources, type Item } from "./items.ts";

const DIGEST_TOOL = {
  name: "submit_digest",
  description: "Submit a short digest of the listed documents.",
  // Statements only: the records are printed by code, so the model has nothing to quote.
  schema: STATEMENTS_SCHEMA,
};

const SYSTEM = `You write a short digest of new or changed EU legal documents for a reader who tracks them.

You are given one record per document: its title, type, date and why it was picked up. That is all you know about each document - you have not read its text. Group related documents and say what each record shows. Do not describe what a document changes, requires or means beyond what its title states, and do not give advice. Write each sentence as one line of plain text and cite the source ids it is about.`;

/** Items handed to the model in one digest; the rest are still listed. */
const MAX_FOR_MODEL = 30;

export interface DigestInput {
  items: Item[];
  notices: string[];
  /** Shown in the heading, e.g. `polling` or `webhook delivery dl_...`. */
  origin: string;
  generatedAt: Date;
}

export interface Digest {
  outcome: "nothing_new" | "summarised" | "list_only";
  markdown: string;
  answer?: GroundedAnswer;
}

export async function buildDigest(input: DigestInput, llm: Llm): Promise<Digest> {
  const heading = `# Regulatory change digest\n\nGenerated ${input.generatedAt.toISOString()} from ${input.origin}. Model: ${llm.label}.\n`;
  const notices = input.notices.length > 0 ? `\n## Notices\n\n${input.notices.map((n) => `- ${n}`).join("\n")}\n` : "";

  if (input.items.length === 0) {
    // Nothing retrieved, nothing to summarise: the model is not called.
    return { outcome: "nothing_new", markdown: `${heading}\n## No new or changed documents\n\nNothing new was found for the watch list in this run.\n${notices}` };
  }

  const sources = itemsToSources(input.items);
  const list = sources.map((s) => `- [${s.id}] ${citation(s)}${s.notes.map((n) => `\n  - Note: ${n}`).join("")}\n  - Listed because: ${input.items[Number(s.id.slice(1)) - 1]?.reason}`).join("\n");
  const records = `\n## Documents (records returned by LexAPI)\n\n${list}\n`;

  const forModel = sources.slice(0, MAX_FOR_MODEL);
  const user = `Documents:\n\n${sourcesForPrompt(forModel)}\n\nWrite the digest: a few sentences, each citing the source ids it is about.`;
  let answer: GroundedAnswer | undefined;
  let feedback = "";
  for (let attempt = 0; attempt < 2 && !answer; attempt++) {
    const raw = await llm.generate({ system: SYSTEM, user: user + feedback, tool: DIGEST_TOOL });
    try {
      answer = validateDraft(parseDraft(raw), forModel);
    } catch (error) {
      if (!(error instanceof GroundingError)) throw error;
      feedback = `\n\nYour previous draft was rejected:\n- ${error.violations.join("\n- ")}\nFix these and submit again.`;
    }
  }

  if (!answer || answer.statements.length === 0) {
    const why = answer ? "The model produced no cited summary." : "The model's summary failed the grounding check twice, so it is not shown.";
    return { outcome: "list_only", markdown: `${heading}${records}\n## Summary\n\n${why}\n${notices}`, answer };
  }

  const summary = answer.statements.map(renderStatement).join("\n");
  const removed = answer.removed.length > 0 ? `\nRemoved by the grounding check (cited no source): ${answer.removed.map((r) => `"${escapeMarkdown(r.text)}"`).join("; ")}\n` : "";
  const overflow = sources.length > forModel.length ? `\nThe summary covers the first ${forModel.length} of ${sources.length} documents.\n` : "";
  return {
    outcome: "summarised",
    markdown: `${heading}${records}\n## Summary (written by the model from the records above - it has not read the documents)\n\n${summary}\n${removed}${overflow}${notices}`,
    answer,
  };
}
