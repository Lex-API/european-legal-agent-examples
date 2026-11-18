import { existsSync } from "node:fs";
import { LexAPI } from "@lexapi/client";

/** Load `KEY=value` lines from a file into `process.env` (existing variables win). */
export function loadEnvFile(path: string): void {
  if (existsSync(path)) process.loadEnvFile(path);
}

/**
 * The LexAPI client, configured from the environment:
 *   LEXAPI_API_KEY   - your key (`lex_...`); never hard-code it.
 *   LEXAPI_BASE_URL  - optional; the tests and `--dry-run` point it at a local stub.
 */
export function createLexApiClient(env: NodeJS.ProcessEnv = process.env): LexAPI {
  if (!env.LEXAPI_API_KEY) {
    throw new Error("LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard or run with --dry-run.");
  }
  return new LexAPI({
    apiKey: env.LEXAPI_API_KEY,
    ...(env.LEXAPI_BASE_URL ? { baseUrl: env.LEXAPI_BASE_URL } : {}),
    userAgentSuffix: "european-legal-agent-examples",
  });
}

/** Credits a response reports for its own call (`credits.operation_weight`); 0 when absent. */
export function creditsCharged(response: { credits?: { operation_weight?: number } }): number {
  return response.credits?.operation_weight ?? 0;
}
