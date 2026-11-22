/**
 * The one error format of this repository, the same as the LexAPI adapters for
 * LangChain and LlamaIndex print for a failed call:
 *
 *   LexAPI error CODE (HTTP n): <message>.
 *     then, each only when the error carries it, in this order:
 *     Credits reset at <t>.                          credits exhausted
 *     This is an exhausted quota, not a rate limit.  an exhausted quota
 *     Quota resets at <t>.                           the same, when the time is known
 *     Access ended at <t>.                           payment past due
 *     Billing: <url>                                 the same, unless the message holds the link
 *     Retry after <n> s.                             only for an error the SDK retries
 *
 * Strings from the server are put on one line (control characters and line
 * breaks become spaces) and capped: the message at 1,000 characters, the code,
 * a time or a link at 200 (`…` marks a cut). Anything shaped like a LexAPI key
 * is masked. Built on what `@lexapi/client` 0.5.0 exposes.
 */
import { CreditsExhaustedError, type LexAPIError, RateLimitedError } from "@lexapi/client";

export const DASHBOARD_URL = "https://lex-api.com/dashboard?ref=github-examples";
const MESSAGE_CAP = 1000;
const VALUE_CAP = 200;
const KEY_PATTERN = /lex_[A-Za-z0-9_-]+/g;
/** Cloudflare could not reach LexAPI: the request never arrived, so the SDK retries it. */
const NOT_REACHED = new Set([521, 522, 523]);
const DASHBOARD_CODES = new Set(["UNAUTHORIZED", "INVALID_API_KEY", "CREDITS_EXHAUSTED"]);
const UNREADABLE = "The answer could not be read.";

/** An error whose printed form is fixed by the example; printed as is, exit code 1. */
export class PrintedError extends Error {
  constructor(readonly line: string) {
    super(line);
    this.name = "PrintedError";
  }
}

/** A 2xx answer without the fields the example reads. */
export class UnreadableAnswerError extends PrintedError {
  constructor(status?: number) {
    super(`LexAPI error INVALID_RESPONSE${status === undefined ? "" : ` (HTTP ${status})`}: ${UNREADABLE}`);
    this.name = "UnreadableAnswerError";
  }
}

/** A string fit for one line, cut at `cap` code points; `undefined` when not a string or blank. */
export function clean(value: unknown, cap: number): string | undefined {
  if (typeof value !== "string") return undefined;
  let text = "";
  for (const ch of value) text += /^[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]$/u.test(ch) ? " " : ch;
  text = text.replace(/\s+/g, " ").trim();
  if (!text) return undefined;
  const chars = [...text];
  return chars.length > cap ? `${chars.slice(0, cap).join("").trimEnd()}…` : text;
}

/** Every key set in the environment, masked. */
export function maskSecrets(text: string, env: NodeJS.ProcessEnv = process.env): string {
  let masked = text;
  for (const name of ["LEXAPI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]) {
    const secret = env[name]?.trim();
    if (secret && secret.length >= 8) masked = masked.split(secret).join("[key]");
  }
  return masked;
}

/** Anything shaped like a LexAPI key, then every key set in the environment, masked. */
export function mask(text: string, env: NodeJS.ProcessEnv = process.env): string {
  return maskSecrets(text.replace(KEY_PATTERN, "lex_***"), env);
}

function sentence(message: string): string {
  const trimmed = message.replace(/[: ]+$/, "");
  return !trimmed || /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

/** The API's own message: the SDK builds `error.message` as `[CODE] <message> (HTTP n)`. */
function apiMessage(error: LexAPIError): string {
  let message = error.message;
  const head = `[${error.code}] `;
  if (message.startsWith(head)) message = message.slice(head.length);
  const tail = ` (HTTP ${error.status})`;
  if (message.endsWith(tail)) message = message.slice(0, -tail.length);
  return message;
}

/** A string field of a legacy-shape body, read from `details` first, then the body's top level. */
function field(error: LexAPIError, name: string): string | undefined {
  for (const container of [error.details, error.body]) {
    if (container && typeof container === "object" && !Array.isArray(container)) {
      const value = (container as Record<string, unknown>)[name];
      if (typeof value === "string") return value;
    }
  }
  return undefined;
}

function quotaExhausted(error: LexAPIError): boolean {
  return error.code === "QUOTA_EXHAUSTED" || (error instanceof RateLimitedError && error.quotaExhausted);
}

function sdkRetries(error: LexAPIError): boolean {
  if (error.status !== 429 && !NOT_REACHED.has(error.status)) return false;
  return error.code !== "PAYMENT_PAST_DUE" && !quotaExhausted(error) && !(error instanceof CreditsExhaustedError);
}

function waitText(seconds: unknown): string | undefined {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds <= 0) return undefined;
  return Number.isInteger(seconds) ? String(seconds) : String(Number(seconds.toFixed(6)));
}

/** The one-line text for a LexAPI error, in the adapters' form. */
export function lexapiErrorText(error: LexAPIError): string {
  const status = error.status > 0 ? error.status : undefined; // 0: no answer arrived
  const http = status === undefined ? "" : ` (HTTP ${status})`;
  if (error.code === "INVALID_RESPONSE" || (status !== undefined && status >= 200 && status < 300)) {
    return `LexAPI error INVALID_RESPONSE${http}: ${UNREADABLE}`;
  }
  const message = clean(apiMessage(error), MESSAGE_CAP);
  const said = message ? sentence(message) : "";
  const parts = [`LexAPI error ${clean(error.code, VALUE_CAP) ?? "UNKNOWN"}${http}${said ? `: ${said}` : "."}`];
  if (error instanceof CreditsExhaustedError) {
    const resetsAt = clean(error.resetsAt, VALUE_CAP);
    if (resetsAt) parts.push(`Credits reset at ${resetsAt}.`);
  }
  if (quotaExhausted(error)) {
    parts.push("This is an exhausted quota, not a rate limit.");
    const resetsAt = clean(error instanceof RateLimitedError ? error.resetsAt : undefined, VALUE_CAP);
    if (resetsAt) parts.push(`Quota resets at ${resetsAt}.`);
  }
  if (error.code === "PAYMENT_PAST_DUE") {
    const ended = clean(field(error, "accessEndedAt"), VALUE_CAP);
    if (ended) parts.push(`Access ended at ${ended}.`);
    const url = field(error, "billingUrl");
    const fullUrl = clean(url, Number.MAX_SAFE_INTEGER);
    const fullMessage = clean(apiMessage(error), Number.MAX_SAFE_INTEGER);
    if (fullUrl && !(fullMessage && fullMessage.includes(fullUrl))) parts.push(`Billing: ${clean(url, VALUE_CAP)}`);
  }
  if (sdkRetries(error)) {
    const wait = waitText(error.retryAfter);
    if (wait) parts.push(`Retry after ${wait} s.`);
  }
  return mask(parts.join(" "));
}

/** The lines printed for a LexAPI error: its text, then the dashboard link after a key or credits error. */
export function lexapiErrorLines(error: LexAPIError): string[] {
  const lines = [lexapiErrorText(error)];
  if (error.status === 401 || DASHBOARD_CODES.has(error.code)) lines.push(`Dashboard: ${DASHBOARD_URL}`);
  return lines;
}
