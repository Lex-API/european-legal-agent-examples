/**
 * Regulatory change monitor - polling mode (the default; works on every plan).
 *
 *   npm run monitor                 (your keys; reads watchlist.json next to this example)
 *   npm run monitor -- --dry-run    (no keys, fixtures; state is kept in memory)
 *
 * Run it on a schedule (cron, a CI job). Each run reports what is new since
 * the previous one, using the state file to remember what it has seen. The
 * state file is written (atomically) only after a run printed its digest. The
 * monitor is meant to run one at a time.
 *
 * Exit codes: 0 printed; 1 LexAPI or the model failed; 2 usage or configuration error.
 */
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { DRY_RUN_BANNER, startDryRun } from "../../../offline/dry-run.ts";
import { requireSettings, runCli } from "../../../shared/cli.ts";
import { CreditTally, createLexApiClient, loadEnvFile } from "../../../shared/env.ts";
import { createLlmFromEnv, type Llm } from "../../../shared/llm.ts";
import { buildDigest } from "./digest.ts";
import { poll } from "./poll.ts";
import { emptyState, loadState, saveState, stateNotSaved } from "./state.ts";
import { loadWatchlist } from "./watchlist.ts";

const here = join(dirname(fileURLToPath(import.meta.url)), "..");
const tally = new CreditTally();

await runCli(
  async () => {
    const { values } = parseArgs({
      options: {
        "dry-run": { type: "boolean", default: false },
        watchlist: { type: "string" },
        state: { type: "string", default: join(here, ".monitor-state.json") },
      },
    });

    loadEnvFile(join(here, ".env"));
    // Every key this run needs, before any request.
    if (!values["dry-run"]) requireSettings(process.env, { lexapi: true, model: true });
    const ownList = join(here, "watchlist.json");
    const watchlist = loadWatchlist(values.watchlist ?? (existsSync(ownList) ? ownList : join(here, "watchlist.example.json")));
    // A dry run starts with every watched act already baselined, so the fixtures' relations show up.
    const state = values["dry-run"] ? { ...emptyState(), baselined: watchlist.acts.map((a) => a.celex) } : loadState(values.state);

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
      const now = new Date();
      const result = await poll(client, watchlist, state, now, tally);
      const digest = await buildDigest({ items: result.items, notices: result.notices, origin: "polling", generatedAt: now }, llm);

      if (values["dry-run"]) console.log(DRY_RUN_BANNER);
      console.log(digest.markdown);
      console.error(`LexAPI credits reported for this run: ${result.credits}.`);
      // The state is written after the digest is printed, so a run whose digest was lost reports the items again.
      if (!values["dry-run"]) {
        try {
          saveState(values.state, state);
        } catch (error) {
          console.error(stateNotSaved(values.state, error));
          process.exitCode = 1;
        }
      }
    } finally {
      await closeDryRun?.();
    }
  },
  () => (tally.count > 0 ? `LexAPI credits reported for this run: ${tally.describe()}.` : undefined),
);
