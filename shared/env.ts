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
    throw new Error(
      "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples or run with --dry-run.",
    );
  }
  return new LexAPI({
    apiKey: env.LEXAPI_API_KEY,
    ...(env.LEXAPI_BASE_URL ? { baseUrl: env.LEXAPI_BASE_URL } : {}),
    userAgentSuffix: "european-legal-agent-examples",
  });
}

/** Credits a response reports for its own call (`credits.operation_weight`); `undefined` when it reports none. */
export function creditsCharged(response: { credits?: { operation_weight?: number } }): number | undefined {
  return response.credits?.operation_weight;
}

/** Adds up reported credits without turning "not reported" into 0. */
export class CreditTally {
  private reported = 0;
  private calls = 0;
  private silent = 0;

  add(credits: number | undefined): void {
    this.calls++;
    if (credits === undefined) this.silent++;
    else this.reported += credits;
  }

  /** The sum of what was reported, or `undefined` when no call reported anything. */
  get total(): number | undefined {
    return this.calls > 0 && this.silent === this.calls ? undefined : this.reported;
  }

  describe(): string {
    if (this.total === undefined) return "not reported";
    return this.silent > 0 ? `${this.reported} (${this.silent} call(s) did not report credits)` : String(this.reported);
  }
}
