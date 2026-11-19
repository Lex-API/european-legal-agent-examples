import { LexAPIError } from "@lexapi/client";

/**
 * One line for a refused or failed LexAPI call: the SDK's own text,
 * `[CODE] message (HTTP status)`. Control characters become spaces and the
 * key is masked; nothing is added (no wait, no link).
 */
export function lexapiErrorLine(error: LexAPIError, apiKey: string | undefined = process.env.LEXAPI_API_KEY?.trim()): string {
  let text = "";
  for (const ch of error.message) text += /^[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]$/u.test(ch) ? " " : ch;
  if (apiKey) text = text.split(apiKey).join("[key]");
  return `LexAPI error: ${text.split(/\s+/).filter(Boolean).join(" ")}`;
}

/** Print the line for a LexAPI error and set exit code 1; anything else is rethrown. */
export function reportLexapiError(error: unknown): void {
  if (!(error instanceof LexAPIError)) throw error;
  console.error(lexapiErrorLine(error));
  process.exitCode = 1;
}
