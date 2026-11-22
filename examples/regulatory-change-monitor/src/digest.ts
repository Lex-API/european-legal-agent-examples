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
  sourceListItem,
  parseDraft,
  renderRemoved,
  renderStatement,
  sourcesForPrompt,
  validateDraft,
  type GroundedAnswer,
} from "../../../shared/grounding.ts";
import { READER_PATTERNS } from "../../../shared/conclusions.ts";
import type { Llm } from "../../../shared/llm.ts";
import { plainText } from "../../../shared/text.ts";
import { itemsToSources, type Item } from "./items.ts";

const DIGEST_TOOL = {
  name: "submit_digest",
  description: "Submit a short digest of the listed documents.",
  // Statements only: the records are printed by code, so the model has nothing to quote.
  schema: STATEMENTS_SCHEMA,
};

const SYSTEM = `You write a short digest of new or changed EU legal documents for a reader who tracks them.

You are given one record per document: its title, type, date and why it was picked up. That is all you know about each document - you have not read its text. Group related documents and say what each record shows. Do not describe what a document changes, requires or means beyond what its title states, and do not give advice. Write each sentence as one line of plain text and cite the source ids it is about. Put quotation marks only around words copied exactly from a record you cite. Do not tell the reader what to do.`;

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
  // Everything below that comes from the API, the watch list or a webhook payload is printed through plainText.
  const heading = `# Regulatory change digest\n\nGenerated ${input.generatedAt.toISOString()} from ${plainText(input.origin)}. Model: ${plainText(llm.label)}.\n`;
  const notices = input.notices.length > 0 ? `\n## Notices\n\n${input.notices.map((n) => `- ${plainText(n)}`).join("\n")}\n` : "";

  if (input.items.length === 0) {
    // Nothing retrieved, nothing to summarise: the model is not called.
    return { outcome: "nothing_new", markdown: `${heading}\n## No new or changed documents\n\nNothing new was found for the watch list in this run.\n${notices}` };
  }

  const sources = itemsToSources(input.items);
  const list = sources.map((s, index) => `${sourceListItem(s)}\n  - Listed because: ${plainText(input.items[index]?.reason ?? "")}`).join("\n");
  const records = `\n## Documents (records returned by LexAPI)\n\n${list}\n`;

  const forModel = sources.slice(0, MAX_FOR_MODEL);
  const user = `Documents:\n\n${sourcesForPrompt(forModel)}\n\nWrite the digest: a few sentences, each citing the source ids it is about.`;
  let answer: GroundedAnswer | undefined;
  let feedback = "";
  for (let attempt = 0; attempt < 2 && !answer; attempt++) {
    const raw = await llm.generate({ system: SYSTEM, user: user + feedback, tool: DIGEST_TOOL });
    try {
      // The reader patterns of the conclusion filter apply here too: a digest reports, it does not advise.
      answer = validateDraft(parseDraft(raw), forModel, { conclusionPatterns: READER_PATTERNS });
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
  const removedLines = renderRemoved(answer.removed);
  const removed = removedLines.length > 0 ? `\nRemoved by the grounding check:\n${removedLines.join("\n")}\n` : "";
  const overflow = sources.length > forModel.length ? `\nThe summary covers the first ${forModel.length} of ${sources.length} documents.\n` : "";
  return {
    outcome: "summarised",
    markdown: `${heading}${records}\n## Summary (written by the model from the records above - it has not read the documents)\n\n${summary}\n${removed}${overflow}${notices}`,
    answer,
  };
}
