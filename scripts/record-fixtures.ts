/**
 * Record real response shapes for the calls the examples make.
 *
 *   LEXAPI_API_KEY=lex_... npm run record-fixtures -- --i-have-a-key
 *
 * THIS SCRIPT CALLS THE REAL LexAPI SERVICE AND SPENDS CREDITS (about 30-50,
 * see `PLAN` below). It is for the repository owner or a contributor who
 * wants to compare the synthetic fixtures in `offline/fixtures` with what the
 * API returns today. It is never run by the tests or by CI, and it refuses
 * to start without the explicit flag, inside a test run, or on a CI runner.
 *
 * Output goes to `fixtures-recorded/`, which is git-ignored: the files hold
 * real legal texts and your account's plan and usage figures. Review them
 * before sharing, and do not commit them as test fixtures.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { LexAPI } from "@lexapi/client";

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures-recorded");

const { values } = parseArgs({ options: { "i-have-a-key": { type: "boolean", default: false } } });

if (!values["i-have-a-key"]) {
  console.error("This script calls the real LexAPI service and spends credits. Re-run with --i-have-a-key to confirm.");
  process.exit(2);
}
if (process.env.NODE_TEST_CONTEXT || process.env.CI) {
  console.error("Refusing to record fixtures inside a test run or on a CI runner.");
  process.exit(2);
}
if (!process.env.LEXAPI_API_KEY) {
  console.error("LEXAPI_API_KEY is not set.");
  process.exit(2);
}

// The default base URL: the real API. LEXAPI_BASE_URL is deliberately ignored here.
const client = new LexAPI({ apiKey: process.env.LEXAPI_API_KEY, userAgentSuffix: "european-legal-agent-examples-recorder" });

function save(name: string, value: unknown): void {
  writeFileSync(join(OUT_DIR, `${name}.json`), JSON.stringify(value, null, 2) + "\n");
  console.log(`recorded ${name}.json`);
}

/** One entry per call an example makes, with its documented credit cost. */
const PLAN: Array<{ name: string; credits: string; run: () => Promise<unknown> }> = [
  { name: "info", credits: "0", run: () => client.getInfo() },
  {
    name: "legislation-semantic.ai-act",
    credits: "5",
    run: () => client.semanticLegislationSearch({ query: "risk management requirements for AI systems", limit: 10, includeText: true }),
  },
  {
    name: "legislation-semantic.gdpr",
    credits: "5",
    run: () => client.semanticLegislationSearch({ query: "transfer of personal data to a third country", limit: 10, includeText: true }),
  },
  {
    name: "legislation-semantic.gdpr-asof",
    credits: "5 + up to 15",
    run: () =>
      client.semanticLegislationSearch({ query: "transfer of personal data to a third country", limit: 10, includeText: true, asOf: "2019-06-01" }),
  },
  { name: "cited-by.32016R0679", credits: "1", run: () => client.getCitedBy("32016R0679", { citationType: ["amendment", "corrigendum", "repeal"], limit: 100 }) },
  { name: "cited-by.32024R1689", credits: "1", run: () => client.getCitedBy("32024R1689", { citationType: ["amendment", "corrigendum", "repeal"], limit: 100 }) },
  { name: "recent.regulation", credits: "3", run: () => client.getRecentDocuments({ days: 7, documentType: "regulation", limit: 10 }) },
  {
    name: "search.artificial-intelligence",
    credits: "3",
    run: () => client.search({ query: '"artificial intelligence"', dateFrom: new Date(Date.now() - 7 * 86_400_000).toISOString().slice(0, 10), maxPages: 1 }),
  },
];

mkdirSync(OUT_DIR, { recursive: true });
for (const step of PLAN) {
  console.log(`${step.name} (${step.credits} credit(s)) ...`);
  save(step.name, await step.run());
}

// The case-law search, then the judgment the GDPR assistant would read first.
console.log("case-law-semantic.gdpr (5 credit(s)) ...");
const cases = await client.semanticSearch({ query: "transfer of personal data to a third country", limit: 5 });
save("case-law-semantic.gdpr", cases);
const celex = cases.results[0]?.metadata?.celex_id;
// The CELEX comes from a response and ends up in a file name: accept only the characters a CELEX has.
if (typeof celex === "string" && /^[0-9A-Z()]{6,24}$/.test(celex)) {
  console.log(`document.${celex} (1-5 credit(s)) ...`);
  save(`document.${celex}`, await client.getDocument({ celexNumber: celex, include: ["metadata", "paragraphs"] }));
}

console.log(`\nDone. Files are in ${OUT_DIR} (git-ignored). Webhook calls are not recorded: they create resources on your account.`);
