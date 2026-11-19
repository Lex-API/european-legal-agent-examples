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

/**
 * The four answers that stop a run for a while, with the fields the API sends
 * for them (synthetic), and the line the examples print: the SDK's own text.
 * The rate limit names a wait of 0 s, so the SDK's retries take no time.
 */
export const REFUSALS: Record<string, StubReply & { printed: string; attempts: number }> = {
  credits: {
    status: 402,
    body: {
      success: false,
      error: {
        code: "CREDITS_EXHAUSTED",
        message: "Credit pool exhausted",
        details: { operation: "semantic_legislation", weight: 5, available: 2, resetsAt: "2026-07-01T00:00:00Z" },
      },
    },
    printed: "LexAPI error: [CREDITS_EXHAUSTED] Credit pool exhausted (HTTP 402)",
    attempts: 1,
  },
  quota: {
    status: 429,
    body: {
      success: false,
      code: "QUOTA_EXHAUSTED",
      error: "Daily quota exceeded",
      message: "The daily call quota of this plan is used up.",
      usage: { resetsAt: "2026-07-01T00:00:00Z" },
    },
    printed: "LexAPI error: [QUOTA_EXHAUSTED] Daily quota exceeded: The daily call quota of this plan is used up. (HTTP 429)",
    attempts: 1,
  },
  "past due": {
    status: 403,
    body: {
      success: false,
      code: "PAYMENT_PAST_DUE",
      error: "Payment past due",
      message: "Your subscription payment failed, so API requests are refused.",
      accessEndedAt: "2026-06-20T00:00:00Z",
      billingUrl: "https://lex-api.com/dashboard#billing",
    },
    printed:
      "LexAPI error: [PAYMENT_PAST_DUE] Payment past due: Your subscription payment failed, so API requests are refused. (HTTP 403)",
    attempts: 1,
  },
  "rate limit": {
    status: 429,
    headers: { "retry-after": "0", "ratelimit-remaining": "0", "ratelimit-reset": "0" },
    body: { success: false, error: { code: "RATE_LIMITED", message: "Too many requests" } },
    printed: "LexAPI error: [RATE_LIMITED] Too many requests (HTTP 429)",
    attempts: 4, // the SDK retries a rate limit three times
  },
};
