import { readFileSync } from "node:fs";
import type { DocumentType } from "@lexapi/client";
import { ConfigError } from "../../../shared/cli.ts";

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

const SHAPE =
  'expected {"acts": [{"celex": "<CELEX number>", "label": "<text>"}], "topics": [{"name": "<text>", "query": "<keywords>", "documentType": "<type>"}], "days": <whole number>}';

export function loadWatchlist(path: string): Watchlist {
  const invalid = (what: string) => new ConfigError(`The watch list ${path} is not valid: ${what}; ${SHAPE}.`);
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    const code = (error as { code?: unknown }).code;
    throw new ConfigError(`The watch list ${path} could not be read${typeof code === "string" ? ` (${code})` : " as JSON"}.`);
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw invalid("it is not a JSON object");
  const list = raw as Record<string, unknown>;
  const text = (value: unknown) => typeof value === "string" && value.trim().length > 0;
  const optionalText = (value: unknown) => value === undefined || text(value);
  for (const name of ["acts", "topics"]) {
    if (list[name] !== undefined && !Array.isArray(list[name])) throw invalid(`"${name}" is not a list`);
  }
  const acts = (list.acts ?? []) as unknown[];
  const topics = (list.topics ?? []) as unknown[];
  for (const act of acts) {
    const entry = act as Record<string, unknown> | null;
    if (!entry || typeof entry !== "object" || !text(entry.celex) || !optionalText(entry.label)) {
      throw invalid('every act needs a "celex" (and may have a "label"), as text');
    }
  }
  for (const topic of topics) {
    const entry = topic as Record<string, unknown> | null;
    if (!entry || typeof entry !== "object" || !text(entry.name) || !optionalText(entry.query) || !optionalText(entry.documentType) || !(entry.query || entry.documentType)) {
      throw invalid('every topic needs a "name" and a "query" and/or a "documentType", as text');
    }
  }
  if (list.days !== undefined && !(Number.isInteger(list.days) && (list.days as number) > 0)) throw invalid('"days" is not a whole number above 0');
  if (acts.length === 0 && topics.length === 0) throw new ConfigError(`${path} lists no acts and no topics.`);
  return {
    acts: (acts as Array<{ celex: string; label?: string }>).map((a) => ({ celex: a.celex, label: a.label ?? a.celex })),
    topics: topics as Topic[],
    days: (list.days as number | undefined) ?? 7,
  };
}
