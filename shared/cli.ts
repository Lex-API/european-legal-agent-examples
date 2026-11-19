/**
 * What the TypeScript command lines share: the key check before any request,
 * and one top-level handler that turns an expected failure into clean lines on
 * stderr and an exit code (0 printed; 1 LexAPI or the model failed; 2 usage or
 * configuration error). Anything else is a bug and keeps its stack trace.
 */
import { LexAPIError } from "@lexapi/client";
import { DASHBOARD_URL, PrintedError, clean, lexapiErrorLines, mask } from "./lexapi-error.ts";
import { LlmError, llmProvider } from "./llm.ts";

/** A usage or configuration error (exit code 2), found before any request. */
export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigError";
  }
}

/** The key values this repository's `.env.example` files shipped, or a `<...>` placeholder, count as not set. */
const PLACEHOLDERS = new Set(["lex_your_key_here"]);

/** The key in `name`, trimmed; `undefined` when unset, blank or a placeholder. */
export function envKey(env: NodeJS.ProcessEnv, name: string): string | undefined {
  const value = env[name]?.trim();
  if (!value || PLACEHOLDERS.has(value.toLowerCase()) || (value.startsWith("<") && value.endsWith(">"))) return undefined;
  return value;
}

export const LEXAPI_KEY_MISSING = `LEXAPI_API_KEY is not set. Create a key at ${DASHBOARD_URL} or use --dry-run.`;

/** One line per missing setting the run needs; empty when nothing is missing. */
export function missingSettings(env: NodeJS.ProcessEnv, needs: { lexapi: boolean; model: boolean }): string[] {
  const lines: string[] = [];
  if (needs.lexapi && !envKey(env, "LEXAPI_API_KEY")) lines.push(LEXAPI_KEY_MISSING);
  if (!needs.model) return lines;
  const provider = llmProvider(env);
  if (provider === "anthropic") {
    if (!envKey(env, "ANTHROPIC_API_KEY")) lines.push("ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run.");
  } else if (provider === "openai") {
    if (!envKey(env, "OPENAI_API_KEY")) lines.push("OPENAI_API_KEY is not set. Add an OpenAI API key to .env or use --dry-run.");
    if (!env.OPENAI_MODEL?.trim()) lines.push("OPENAI_MODEL is not set. Name the OpenAI model to use (there is no default).");
  } else {
    lines.push(`LLM_PROVIDER is "${clean(env.LLM_PROVIDER, 200) ?? ""}"; it must be "anthropic" (the default) or "openai".`);
  }
  return lines;
}

/** Throw a ConfigError naming every missing setting, before any request is made. */
export function requireSettings(env: NodeJS.ProcessEnv, needs: { lexapi: boolean; model: boolean }): void {
  const lines = missingSettings(env, needs);
  if (lines.length > 0) throw new ConfigError(lines.join("\n"));
}

/** The lines and exit code for an expected failure; `undefined` for anything else. */
export function describeFailure(error: unknown): { lines: string[]; exitCode: number } | undefined {
  if (error instanceof ConfigError) return { lines: error.message.split("\n"), exitCode: 2 };
  if (error instanceof LexAPIError) return { lines: lexapiErrorLines(error), exitCode: 1 };
  if (error instanceof PrintedError) return { lines: [error.line], exitCode: 1 };
  if (error instanceof LlmError) return { lines: [`Model error: ${mask(clean(error.message, 1000) ?? "no message")}`], exitCode: 1 };
  if (error instanceof Error) {
    const code = (error as Error & { code?: unknown }).code;
    if (typeof code === "string" && code.startsWith("ERR_PARSE_ARGS")) {
      return { lines: [clean(error.message, 1000) ?? "Invalid arguments."], exitCode: 2 };
    }
  }
  return undefined;
}

/**
 * Run a command line: an expected failure prints its lines on stderr (after the
 * credits line of the calls already charged, when there were any) and sets the
 * exit code; nothing else is printed for it.
 */
export async function runCli(body: () => Promise<void>, creditsLine: () => string | undefined = () => undefined): Promise<void> {
  try {
    await body();
  } catch (error) {
    const failure = describeFailure(error);
    if (!failure) throw error;
    const credits = failure.exitCode === 1 ? creditsLine() : undefined;
    if (credits) console.error(credits);
    for (const line of failure.lines) console.error(line);
    process.exitCode = failure.exitCode;
  }
}
