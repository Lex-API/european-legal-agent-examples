/**
 * Polling mode: ask LexAPI what changed. Works on every plan.
 *
 *   - per watched act: `getCitedBy` filtered to amendment / corrigendum /
 *     repeal edges (1 credit per act);
 *   - per topic: `search` when the topic has a query, otherwise
 *     `getRecentDocuments` (3 credits per topic - one result page).
 *
 * Anything not seen on an earlier run is new. The first time an act is
 * polled, its existing relations are recorded as a baseline and not reported.
 */
import type { LexAPI, SearchResult } from "@lexapi/client";
import { CreditTally, creditsCharged } from "../../../shared/env.ts";
import { eurLexUrl } from "../../../shared/grounding.ts";
import type { Item } from "./items.ts";
import { takeNew, type MonitorState } from "./state.ts";
import type { Topic, WatchedAct, Watchlist } from "./watchlist.ts";

export interface PollResult {
  /** New since the last run. */
  items: Item[];
  /** Things the reader should know about this run (stale data, truncated lists, baselines). */
  notices: string[];
  /** Credits the API reported for this run, as text ("8", "not reported", ...). */
  credits: string;
}

const RELATIONS = ["amendment", "corrigendum", "repeal"] as const;

export async function poll(client: LexAPI, watchlist: Watchlist, state: MonitorState, now: Date = new Date()): Promise<PollResult> {
  const items: Item[] = [];
  const notices: string[] = [];
  const tally = new CreditTally();
  const firstSeen = now.toISOString();

  const report = (candidates: Item[]): void => {
    items.push(...takeNew(state, candidates, now));
  };

  for (const act of watchlist.acts) {
    const { found, notice, credits } = await pollAct(client, act);
    tally.add(credits);
    if (notice) notices.push(notice);
    if (!state.baselined.includes(act.celex)) {
      // First run for this act: remember what already exists, report nothing.
      for (const item of found) state.seen[item.key] = firstSeen;
      state.baselined.push(act.celex);
      notices.push(`${act.label} (${act.celex}): baseline recorded (${found.length} existing relation(s)); changes are reported from the next run.`);
      continue;
    }
    report(found);
  }

  const dateFrom = new Date(now.getTime() - watchlist.days * 86_400_000).toISOString().slice(0, 10);
  for (const topic of watchlist.topics) {
    const { found, notice, credits } = await pollTopic(client, topic, watchlist.days, dateFrom);
    tally.add(credits);
    if (notice) notices.push(notice);
    report(found);
  }

  return { items, notices, credits: tally.describe() };
}

async function pollAct(client: LexAPI, act: WatchedAct): Promise<{ found: Item[]; notice?: string; credits: number | undefined }> {
  const response = await client.getCitedBy(act.celex, { citationType: [...RELATIONS], limit: 100 });
  const found: Item[] = [];
  for (const group of response.citedBy) {
    const celex = group.celexNumber;
    if (!celex) continue;
    for (const type of new Set((group.citations ?? []).map((c) => c.citationType))) {
      if (!type) continue;
      found.push({
        key: `act|${act.celex}|${type}|${celex}`,
        reason: `${type} of ${act.label} (${act.celex})`,
        celex,
        title: group.title ?? null,
        url: eurLexUrl(celex),
        notes: [],
      });
    }
  }

  const notes: string[] = [];
  // `relationSync` says how fresh these edges are; anything but fresh/refreshed means "may be behind".
  const sync = response.relationSync?.status;
  if (sync && sync !== "fresh" && sync !== "refreshed") {
    notes.push(`relations were served with sync status "${sync}" and may be incomplete; run again later`);
  }
  if ((response.totalEdges ?? 0) > (response.limit ?? 100)) {
    notes.push(`${response.totalEdges} relations exist; only the newest ${response.limit ?? 100} were read`);
  }
  return {
    found,
    notice: notes.length > 0 ? `${act.label} (${act.celex}): ${notes.join("; ")}.` : undefined,
    credits: creditsCharged(response),
  };
}

async function pollTopic(
  client: LexAPI,
  topic: Topic,
  days: number,
  dateFrom: string,
): Promise<{ found: Item[]; notice?: string; credits: number | undefined }> {
  let results: SearchResult[];
  let credits: number | undefined;
  const notes: string[] = [];

  if (topic.query) {
    const response = await client.search({ query: topic.query, dateFrom, documentType: topic.documentType, maxPages: 1 });
    results = response.results ?? [];
    credits = creditsCharged(response);
    if (response.truncated) notes.push(`more results exist than one page (${response.truncatedReason ?? "plan page limit"})`);
    if (response.partial) notes.push(`results may be incomplete (${response.partialReason ?? "partial answer"})`);
  } else {
    const response = await client.getRecentDocuments({ days, documentType: topic.documentType, limit: 10 });
    results = response.results;
    credits = creditsCharged(response);
    if ((response.totalResults ?? 0) > results.length) notes.push(`${response.totalResults} documents match; only ${results.length} were read`);
  }

  const found = results
    .filter((row): row is SearchResult & { celex: string } => typeof row.celex === "string" && row.celex.length > 0)
    .map((row): Item => ({
      key: `topic|${topic.name}|${row.celex}`,
      reason: `topic: ${topic.name}`,
      celex: row.celex,
      title: row.title ?? null,
      documentType: row.documentType,
      date: row.dateOfDocumentISO,
      url: row.url ?? eurLexUrl(row.celex),
      notes: [],
    }));
  return { found, notice: notes.length > 0 ? `Topic "${topic.name}": ${notes.join("; ")}.` : undefined, credits };
}
