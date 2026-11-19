/**
 * Regulatory change monitor - polling mode (the default; works on every plan).
 *
 *   npm run monitor                 (your keys; reads watchlist.json next to this example)
 *   npm run monitor -- --dry-run    (no keys, fixtures; state is kept in memory)
 *
 * Run it on a schedule (cron, a CI job). Each run reports what is new since
 * the previous one, using the state file to remember what it has seen.
 */
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { DRY_RUN_BANNER, startDryRun } from "../../../offline/dry-run.ts";
import { createLexApiClient, loadEnvFile } from "../../../shared/env.ts";
import { reportLexapiError } from "../../../shared/lexapi-error.ts";
import { createLlmFromEnv, type Llm } from "../../../shared/llm.ts";
import { buildDigest } from "./digest.ts";
import { poll } from "./poll.ts";
import { emptyState, loadState, saveState } from "./state.ts";
import { loadWatchlist } from "./watchlist.ts";

const here = join(dirname(fileURLToPath(import.meta.url)), "..");
const { values } = parseArgs({
  options: {
    "dry-run": { type: "boolean", default: false },
    watchlist: { type: "string" },
    state: { type: "string", default: join(here, ".monitor-state.json") },
  },
});

loadEnvFile(join(here, ".env"));
const ownList = join(here, "watchlist.json");
const watchlist = loadWatchlist(values.watchlist ?? (existsSync(ownList) ? ownList : join(here, "watchlist.example.json")));

let llm: Llm;
let closeDryRun: (() => Promise<void>) | undefined;
if (values["dry-run"]) {
  const dryRun = await startDryRun("monitor");
  llm = dryRun.llm;
  closeDryRun = dryRun.close;
} else {
  llm = await createLlmFromEnv();
}

try {
  const client = createLexApiClient();
  // A dry run starts with every watched act already baselined, so the fixtures' relations show up.
  const state = values["dry-run"] ? { ...emptyState(), baselined: watchlist.acts.map((a) => a.celex) } : loadState(values.state);
  const now = new Date();
  const result = await poll(client, watchlist, state, now);
  const digest = await buildDigest({ items: result.items, notices: result.notices, origin: "polling", generatedAt: now }, llm);
  if (!values["dry-run"]) saveState(values.state, state);

  if (values["dry-run"]) console.log(DRY_RUN_BANNER);
  console.log(digest.markdown);
  console.error(`LexAPI credits reported for this run: ${result.credits}.`);
} catch (error) {
  reportLexapiError(error); // a refused LexAPI call: one line, exit code 1; the state file is not changed
} finally {
  await closeDryRun?.();
}
