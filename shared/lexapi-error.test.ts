import assert from "node:assert/strict";
import { test } from "node:test";
import { LexAPIError } from "@lexapi/client";
import { DASHBOARD_LINE, REFUSAL_BODIES, REFUSAL_TEXTS } from "../offline/test-harness.ts";
import { describeFailure, missingSettings } from "./cli.ts";
import { UnreadableAnswerError, lexapiErrorLines, lexapiErrorText } from "./lexapi-error.ts";
import { LlmError } from "./llm.ts";

// The errors the SDK builds for these answers (no request is made).
const STATUS = { credits: 402, quota: 429, "past due": 403, "rate limit": 429 } as const;
const errorFor = (name: keyof typeof REFUSAL_BODIES, retryAfter?: number) =>
  LexAPIError.fromResponse(STATUS[name], REFUSAL_BODIES[name], retryAfter);

test("the four answers that stop a run read exactly as the adapters print them", () => {
  assert.equal(lexapiErrorText(errorFor("credits")), REFUSAL_TEXTS.credits);
  assert.equal(lexapiErrorText(errorFor("quota")), REFUSAL_TEXTS.quota);
  assert.equal(lexapiErrorText(errorFor("past due")), REFUSAL_TEXTS["past due"]);
  assert.equal(lexapiErrorText(errorFor("rate limit", 7)), REFUSAL_TEXTS["rate limit"]);
});

test("the dashboard link follows a key or credits error on a line of its own, and nothing else", () => {
  assert.deepEqual(lexapiErrorLines(errorFor("credits")), [REFUSAL_TEXTS.credits, DASHBOARD_LINE]);
  const key = LexAPIError.fromResponse(401, { error: "Invalid API key" });
  assert.deepEqual(lexapiErrorLines(key), ["LexAPI error UNAUTHORIZED (HTTP 401): Invalid API key.", DASHBOARD_LINE]);
  for (const name of ["quota", "past due", "rate limit"] as const) assert.equal(lexapiErrorLines(errorFor(name, 7)).length, 1);
});

test("a failed payment never reads as a plan to upgrade, and is never given a wait", () => {
  const text = lexapiErrorText(errorFor("past due", 30));
  assert.doesNotMatch(text, /upgrade|Retry after/i);
});

test("an exhausted quota of the legacy shape (no code, no retry hint) is told from a rate limit", () => {
  const legacy = LexAPIError.fromResponse(429, { error: "Daily limit reached", usage: { resetsAt: "2099-07-02T00:00:00.000Z" } });
  assert.equal(
    lexapiErrorText(legacy),
    "LexAPI error RATE_LIMITED (HTTP 429): Daily limit reached. This is an exhausted quota, not a rate limit. Quota resets at 2099-07-02T00:00:00.000Z.",
  );
});

test("a wait is given only for an error the SDK retries", () => {
  assert.match(lexapiErrorText(LexAPIError.fromResponse(522, { error: "Origin unreachable" }, 3)), / Retry after 3 s\.$/);
  assert.doesNotMatch(lexapiErrorText(LexAPIError.fromResponse(500, { error: "Internal" }, 3)), /Retry/);
  assert.doesNotMatch(lexapiErrorText(errorFor("credits", 3)), /Retry/);
  assert.doesNotMatch(lexapiErrorText(errorFor("quota", 3)), /Retry/);
});

test("a link already in the message is not repeated", () => {
  const body = { ...REFUSAL_BODIES["past due"], message: "Pay at https://lex-api.com/dashboard#billing" };
  const text = lexapiErrorText(LexAPIError.fromResponse(403, body));
  assert.equal(text.split("https://lex-api.com/dashboard#billing").length, 2);
});

test("server strings are put on one line, capped, and keys are masked", () => {
  const bidi = String.fromCharCode(0x202e);
  const long = LexAPIError.fromResponse(400, { success: false, error: { code: `ODD${String.fromCharCode(0x1b)}[2J${bidi}`, message: `line\none ${"x".repeat(2000)}` } });
  const text = lexapiErrorText(long);
  assert.ok(![...text].some((ch) => ch.charCodeAt(0) < 0x20 || ch === bidi));
  assert.match(text, /^LexAPI error ODD \[2J \(HTTP 400\): line one x+…\.$/);
  assert.equal([...text.split(": ")[1]!].length, 1000 + "….".length); // the message is cut at 1,000 characters
  const echoed = LexAPIError.fromResponse(401, { error: "Invalid key lex_abcdefghij0123456789" });
  assert.equal(lexapiErrorText(echoed), "LexAPI error UNAUTHORIZED (HTTP 401): Invalid key lex_***.");
});

test("an unreadable answer has one fixed text", () => {
  assert.equal(new UnreadableAnswerError(200).line, "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read.");
  assert.equal(new UnreadableAnswerError().line, "LexAPI error INVALID_RESPONSE: The answer could not be read.");
});

test("missing settings are named one per line; a model error is one masked line", () => {
  assert.deepEqual(missingSettings({}, { lexapi: true, model: true }), [
    "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples or use --dry-run.",
    "ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run.",
  ]);
  assert.deepEqual(missingSettings({ LEXAPI_API_KEY: "lex_your_key_here", LLM_PROVIDER: "openai" }, { lexapi: true, model: true }).length, 3);
  assert.deepEqual(missingSettings({ LEXAPI_API_KEY: "lex_x", ANTHROPIC_API_KEY: "k" }, { lexapi: true, model: true }), []);
  const failure = describeFailure(new LlmError("invalid x-api-key lex_abcdefghij0123456789\nmore"));
  assert.deepEqual(failure, { lines: ["Model error: invalid x-api-key lex_*** more"], exitCode: 1 });
});
