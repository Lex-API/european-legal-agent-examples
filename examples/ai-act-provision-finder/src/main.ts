/**
 * AI Act provision finder - command line.
 *
 *   npm run ai-act -- "A tool that ranks job applicants' CVs"            (your keys)
 *   npm run ai-act -- --dry-run "A tool that ranks job applicants' CVs"  (no keys, fixtures)
 *   npm run ai-act -- --via mcp "..."                                    (retrieve through @lexapi/mcp)
 *
 * Exit codes: 0 printed; 1 LexAPI, the MCP tool or the model failed; 2 usage or configuration error.
 */
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { DRY_RUN_BANNER, startDryRun } from "../../../offline/dry-run.ts";
import { ConfigError, markInterrupted, requireSettings, runCli } from "../../../shared/cli.ts";
import { CreditTally, createLexApiClient, loadEnvFile } from "../../../shared/env.ts";
import { createLlmFromEnv, type Llm } from "../../../shared/llm.ts";
import { findProvisions } from "./agent.ts";
import { McpLegislationSearch } from "./retrieve-mcp.ts";
import { SdkLegislationSearch, type LegislationSearch } from "./retrieve.ts";

/** SIGTERM the MCP server at once; SIGKILL it if it is still there after 300 ms; then `done`. */
function terminate(pid: number, done: () => void): void {
  const alive = () => {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  };
  try {
    process.kill(pid, "SIGTERM");
  } catch {
    done();
    return;
  }
  const started = Date.now();
  const poll = setInterval(() => {
    if (!alive()) {
      clearInterval(poll);
      done();
    } else if (Date.now() - started >= 300) {
      clearInterval(poll);
      try {
        process.kill(pid, "SIGKILL");
      } catch {
        // gone in the meantime
      }
      done();
    }
  }, 10);
}

const USAGE = 'Usage: npm run ai-act -- [--dry-run] [--via sdk|mcp] [--max-queries N] "<description of the AI system>"';
const tally = new CreditTally();
let via = "sdk";

/** The credits line: what the API reported; through MCP, also what the searches cost. */
function creditsLine(reported: string, searches: number): string {
  const mcp = via === "mcp" ? ` The MCP tool result does not carry them; ${searches} search(es) cost 5 credits each.` : "";
  return `LexAPI credits reported for this run: ${reported}.${mcp}`;
}

await runCli(
  async () => {
    const { values, positionals } = parseArgs({
      allowPositionals: true,
      options: {
        "dry-run": { type: "boolean", default: false },
        via: { type: "string", default: "sdk" },
        "max-queries": { type: "string", default: "3" },
      },
    });
    const description = positionals.join(" ").trim();
    if (!description) throw new ConfigError(USAGE);
    if (values.via !== "sdk" && values.via !== "mcp") throw new ConfigError(`--via is "sdk" or "mcp".\n${USAGE}`);
    via = values.via;
    const maxQueries = Math.min(Math.max(Number.parseInt(values["max-queries"], 10) || 3, 1), 3);

    loadEnvFile(join(dirname(fileURLToPath(import.meta.url)), "..", ".env"));
    // Every key this run needs, before any request.
    if (!values["dry-run"]) requireSettings(process.env, { lexapi: true, model: true });

    let llm: Llm;
    let closeDryRun: (() => Promise<void>) | undefined;
    if (values["dry-run"]) {
      const dryRun = await startDryRun("ai-act");
      llm = dryRun.llm;
      closeDryRun = dryRun.close;
    } else {
      llm = await createLlmFromEnv();
    }

    let search: LegislationSearch | undefined;
    // Ctrl-C or a termination signal: stop the MCP server process first, so it does not outlive this one.
    // The handlers stay installed, so a second signal cannot reach the default handler and orphan the server.
    let stopping = false;
    const stop = (signal: NodeJS.Signals, code: number) =>
      process.on(signal, () => {
        if (stopping) return;
        stopping = true;
        markInterrupted();
        console.error("Interrupted.");
        const pid = search instanceof McpLegislationSearch ? search.pid : null;
        if (!pid) process.exit(code);
        terminate(pid, () => process.exit(code));
      });
    stop("SIGINT", 130);
    stop("SIGTERM", 143);
    try {
      search = values.via === "mcp" ? await McpLegislationSearch.start() : new SdkLegislationSearch(createLexApiClient());
      const result = await findProvisions(description, { llm, search }, { maxQueries, tally });
      if (values["dry-run"]) console.log(DRY_RUN_BANNER);
      console.log(result.markdown);
      console.error(creditsLine(result.credits, result.queries.length));
    } finally {
      await search?.close();
      await closeDryRun?.();
    }
  },
  () => (tally.count > 0 ? creditsLine(tally.describe(), tally.count) : undefined),
);
