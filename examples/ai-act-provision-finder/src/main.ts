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
import { ConfigError, requireSettings, runCli } from "../../../shared/cli.ts";
import { CreditTally, createLexApiClient, loadEnvFile } from "../../../shared/env.ts";
import { createLlmFromEnv, type Llm } from "../../../shared/llm.ts";
import { findProvisions } from "./agent.ts";
import { McpLegislationSearch } from "./retrieve-mcp.ts";
import { SdkLegislationSearch, type LegislationSearch } from "./retrieve.ts";

const USAGE = 'Usage: npm run ai-act -- [--dry-run] [--via sdk|mcp] [--max-queries N] "<description of the AI system>"';
const tally = new CreditTally();

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
    try {
      search = values.via === "mcp" ? await McpLegislationSearch.start() : new SdkLegislationSearch(createLexApiClient());
      const result = await findProvisions(description, { llm, search }, { maxQueries, tally });
      if (values["dry-run"]) console.log(DRY_RUN_BANNER);
      console.log(result.markdown);
      console.error(
        `LexAPI credits reported for this run: ${result.credits}.` +
          (values.via === "mcp" ? ` The MCP tool result does not carry them; ${result.queries.length} search(es) cost 5 credits each.` : ""),
      );
    } finally {
      await search?.close();
      await closeDryRun?.();
    }
  },
  () => (tally.count > 0 ? `LexAPI credits reported for this run: ${tally.describe()}.` : undefined),
);
