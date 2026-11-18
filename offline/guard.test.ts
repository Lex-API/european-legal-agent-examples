import assert from "node:assert/strict";
import { test } from "node:test";
import { LexAPI } from "@lexapi/client";
import { assertLocalBaseUrl } from "./guard.ts";
import { startOfflineApi } from "./test-harness.ts";

test("the harness refuses any base URL that is not a 127.0.0.1 stub", () => {
  for (const url of [undefined, "", "https://lex-api.com/api/v1", "http://localhost:3000", "http://127.0.0.1.example.com:80", "https://127.0.0.1:443"]) {
    assert.throws(() => assertLocalBaseUrl(url), /Refusing to run/);
  }
  assert.equal(assertLocalBaseUrl("http://127.0.0.1:4010"), "http://127.0.0.1:4010");
});

test("a client left on the SDK's default base URL is blocked before any request is sent", async () => {
  const client = new LexAPI({ apiKey: "lex_offline_test_key", retry: { maxRetries: 0 } });
  await assert.rejects(client.getInfo(), (error: unknown) => {
    const text = `${String(error)} ${String((error as { cause?: unknown }).cause ?? "")}`;
    return /Offline tests blocked a request to https:\/\/lex-api\.com/.test(text);
  });
});

test("the stub serves fixtures, records requests and answers 404 for unknown routes", async () => {
  const stub = await startOfflineApi();
  try {
    const client = new LexAPI({ apiKey: process.env.LEXAPI_API_KEY, baseUrl: stub.baseUrl, retry: { maxRetries: 0 } });
    const info = await client.getInfo();
    assert.equal(info.subscription?.tier, "FREE");
    assert.equal(stub.requests[0]?.headers["x-api-key"], "lex_offline_test_key");
    await assert.rejects(client.resolve("ECLI:EU:C:2099:901"), /no fixture/);
  } finally {
    await stub.close();
  }
});
