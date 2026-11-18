import { readFileSync } from "node:fs";
import type { DocumentType } from "@lexapi/client";

export interface WatchedAct {
  /** Base-act CELEX number (e.g. `32016R0679`), not a consolidated version. */
  celex: string;
  label: string;
}

export interface Topic {
  name: string;
  /** Keyword query for `POST /search`. Without it the topic lists recent documents of `documentType`. */
  query?: string;
  /** One document type. */
  documentType?: DocumentType;
}

export interface Watchlist {
  acts: WatchedAct[];
  topics: Topic[];
  /** Look-back window for topics, in days. */
  days: number;
}

export function loadWatchlist(path: string): Watchlist {
  const raw = JSON.parse(readFileSync(path, "utf8")) as Partial<Watchlist>;
  const acts = (raw.acts ?? []).filter((a) => typeof a?.celex === "string" && a.celex.length > 0);
  const topics = (raw.topics ?? []).filter((t) => typeof t?.name === "string" && (t.query || t.documentType));
  if (acts.length === 0 && topics.length === 0) throw new Error(`${path} lists no acts and no topics.`);
  const days = Number.isInteger(raw.days) && (raw.days as number) > 0 ? (raw.days as number) : 7;
  return { acts: acts.map((a) => ({ celex: a.celex, label: a.label ?? a.celex })), topics, days };
}
