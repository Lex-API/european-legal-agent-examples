import { eurLexUrl, type Source } from "../../../shared/grounding.ts";

/** One new or changed document the monitor found, from polling or from a webhook delivery. */
export interface Item {
  /** Stable key used to remember what was already reported. */
  key: string;
  /** Why it is in the digest, e.g. `amendment of GDPR (32016R0679)` or `topic: New regulations`. */
  reason: string;
  celex: string;
  /** Title as returned by the API; `null` when the API has none yet. */
  title: string | null;
  documentType?: string | null;
  /** Document date (ISO), when known. */
  date?: string | null;
  url: string;
  /** Caveats to print with the item. */
  notes: string[];
}

export const CONSOLIDATED_NOTE =
  "Consolidated version: a documentation tool without legal force. The authentic texts are the act and its amending acts as published in the Official Journal.";

/**
 * Each item becomes a source. Its "text" is the record the API returned
 * (title, type, date) - the digest may quote it and nothing else: the
 * monitor does not fetch document bodies.
 */
export function itemsToSources(items: Item[]): Source[] {
  return items.map((item, index) => ({
    id: `S${index + 1}`,
    kind: "document",
    celex: item.celex,
    title: item.title ?? "(no title available yet)",
    url: item.url || eurLexUrl(item.celex),
    text: [item.title ?? "(no title available yet)", item.documentType ? `Type: ${item.documentType}` : "", item.date ? `Date: ${item.date}` : "", `Reason: ${item.reason}`]
      .filter(Boolean)
      .join("\n"),
    notes: item.notes,
  }));
}
