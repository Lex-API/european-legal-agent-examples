/**
 * `--dry-run`: run an example with no keys and no network beyond 127.0.0.1.
 * LexAPI is replaced by the local stub, the model by scripted replies from
 * `offline/fixtures/llm/<script>.json`. The output is illustrative only.
 */
import { ScriptedLlm } from "../shared/llm.ts";
import { assertLocalBaseUrl } from "./guard.ts";
import { readFixture, startStubServer } from "./stub-server.ts";

export const DRY_RUN_BANNER =
  "> DRY RUN - LexAPI was replaced by a local stub and the model by scripted replies.\n" +
  "> The legal texts below are synthetic placeholders, not the wording of any act or judgment.\n";

export async function startDryRun(script: string): Promise<{ llm: ScriptedLlm; close: () => Promise<void> }> {
  const stub = await startStubServer();
  process.env.LEXAPI_BASE_URL = assertLocalBaseUrl(stub.baseUrl);
  process.env.LEXAPI_API_KEY = "lex_dry_run_no_key";
  const llm = new ScriptedLlm(JSON.parse(readFixture(`llm/${script}.json`)) as Record<string, unknown>);
  return { llm, close: stub.close };
}
