/**
 * Shared setup for the TypeScript tests: start the stub, point the
 * environment at it, and refuse to continue if anything could reach a real
 * service.
 */
import { assertLocalBaseUrl, blockExternalFetch } from "./guard.ts";
import { startStubServer, type StubOptions, type StubReply, type StubServer } from "./stub-server.ts";

blockExternalFetch();

export async function startOfflineApi(options: StubOptions = {}): Promise<StubServer> {
  const stub = await startStubServer(options);
  process.env.LEXAPI_BASE_URL = stub.baseUrl;
  process.env.LEXAPI_API_KEY = "lex_offline_test_key";
  assertLocalBaseUrl(process.env.LEXAPI_BASE_URL);
  return stub;
}

/** For tests that build their own client: the environment, checked. */
export function offlineEnv(): NodeJS.ProcessEnv {
  assertLocalBaseUrl(process.env.LEXAPI_BASE_URL);
  return process.env;
}

/** The error bodies the API sends for the four answers that stop a run for a while (synthetic values). */
export const REFUSAL_BODIES = {
  credits: {
    success: false,
    error: {
      code: "CREDITS_EXHAUSTED",
      message: "Credit pool exhausted",
      details: { operation: "semantic_legislation", weight: 5, available: 2, resetsAt: "2099-07-01T00:00:00.000Z" },
    },
  },
  quota: { error: "Daily quota exhausted", code: "QUOTA_EXHAUSTED", usage: { resetsAt: "2099-07-02T00:00:00.000Z" } },
  "past due": {
    code: "PAYMENT_PAST_DUE",
    error: "Payment past due",
    message: "Your subscription payment failed, so API requests are refused.",
    accessEndedAt: "2099-06-25T09:14:03.000Z",
    billingUrl: "https://lex-api.com/dashboard#billing",
  },
  "rate limit": { error: "Rate limit exceeded", code: "RATE_LIMITED", retryAfter: 7 },
} as const;

/** The text for each, as the adapters print it (the rate limit with the SDK's retry wait of 7 s). */
export const REFUSAL_TEXTS: Record<keyof typeof REFUSAL_BODIES, string> = {
  credits: "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Credit pool exhausted. Credits reset at 2099-07-01T00:00:00.000Z.",
  quota:
    "LexAPI error QUOTA_EXHAUSTED (HTTP 429): Daily quota exhausted. This is an exhausted quota, not a rate limit. Quota resets at 2099-07-02T00:00:00.000Z.",
  "past due":
    "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due: Your subscription payment failed, so API requests are refused. Access ended at 2099-06-25T09:14:03.000Z. Billing: https://lex-api.com/dashboard#billing",
  "rate limit": "LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded. Retry after 7 s.",
};

export const DASHBOARD_LINE = "Dashboard: https://lex-api.com/dashboard?ref=github-examples";

/**
 * The same four answers from the stub, with the lines a run prints for them. The
 * rate limit names a wait of 0 s here, so the SDK's three retries take no time
 * (and no wait is printed).
 */
export const REFUSALS: Record<string, StubReply & { lines: string[]; attempts: number }> = {
  credits: { status: 402, body: REFUSAL_BODIES.credits, lines: [REFUSAL_TEXTS.credits, DASHBOARD_LINE], attempts: 1 },
  quota: { status: 429, body: REFUSAL_BODIES.quota, lines: [REFUSAL_TEXTS.quota], attempts: 1 },
  "past due": { status: 403, body: REFUSAL_BODIES["past due"], lines: [REFUSAL_TEXTS["past due"]], attempts: 1 },
  "rate limit": {
    status: 429,
    headers: { "retry-after": "0", "ratelimit-remaining": "0", "ratelimit-reset": "0" },
    body: { ...REFUSAL_BODIES["rate limit"], retryAfter: 0 },
    lines: ["LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded."],
    attempts: 4, // the SDK retries a rate limit three times
  },
};
