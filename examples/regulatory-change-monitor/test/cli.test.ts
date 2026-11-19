/**
 * The command lines as processes: `src/main.ts` and `src/webhook-setup.ts` run
 * as children against the local stub and a local stand-in for the model API.
 */
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { STACK_TRACE, hasOwnEnv, realRunEnv, runExample, startFakeModel } from "../../../offline/process-harness.ts";
import { startStubServer, type StubReply } from "../../../offline/stub-server.ts";
import { REFUSALS } from "../../../offline/test-harness.ts";

const MONITOR = "examples/regulatory-change-monitor/src/main.ts";
const SETUP = "examples/regulatory-change-monitor/src/webhook-setup.ts";
const WATCHLIST = "examples/regulatory-change-monitor/watchlist.example.json";
const dir = mkdtempSync(join(tmpdir(), "monitor-cli-"));
after(() => rmSync(dir, { recursive: true, force: true }));
let n = 0;
const statePath = () => join(dir, `state-${++n}.json`);

async function run(entry: string, reply: StubReply | undefined, nth: number, args: string[], extraEnv: Record<string, string> = {}, model?: { status: number; message: string }) {
  const stub = await startStubServer({ override: () => (reply && stub.requests.length >= nth ? reply : undefined) });
  const fake = await startFakeModel("monitor", model);
  try {
    const result = await runExample(entry, args, { ...realRunEnv(stub.baseUrl, fake), ...extraEnv });
    return { ...result, requests: stub.requests.map((r) => `${r.method} ${r.path}`) };
  } finally {
    await stub.close();
    await fake.close();
  }
}
const poll = (reply: StubReply | undefined, nth: number, state: string, model?: { status: number; message: string }) =>
  run(MONITOR, reply, nth, ["--watchlist", WATCHLIST, "--state", state], {}, model);

test("a real poll prints the digest and writes the state file", async () => {
  const state = statePath();
  const result = await poll(undefined, 1, state);
  assert.equal(result.code, 0, result.stderr);
  assert.match(result.stdout, /# Regulatory change digest/);
  assert.equal(result.stderr.trim(), "LexAPI credits reported for this run: 8.");
  assert.ok(existsSync(state));
});

for (const [name, refusal] of Object.entries(REFUSALS)) {
  test(`a ${name} answer on a first run: its lines, nothing on stdout, exit 1, no state file created`, async () => {
    const state = statePath();
    const result = await poll(refusal, 1, state);
    assert.equal(result.code, 1);
    assert.equal(result.stdout, "");
    assert.deepEqual(result.stderr.trimEnd().split("\n"), refusal.lines);
    assert.doesNotMatch(result.stderr, STACK_TRACE);
    assert.equal(existsSync(state), false);
    assert.equal(result.requests.length, refusal.attempts);
  });
}

test("a refusal on a later run leaves the state file byte for byte as it was, after the credits already charged", async () => {
  const state = statePath();
  assert.equal((await poll(undefined, 1, state)).code, 0);
  const before = readFileSync(state);
  const result = await poll(REFUSALS.credits, 2, state);
  assert.equal(result.code, 1);
  assert.equal(result.stdout, "");
  assert.deepEqual(result.stderr.trimEnd().split("\n"), ["LexAPI credits reported for this run: 1.", ...REFUSALS.credits!.lines]);
  assert.deepEqual(readFileSync(state), before);
});

test("an unreadable answer is an error line, and the state file is not written", async () => {
  const state = statePath();
  const result = await poll({ status: 200, body: { success: true } }, 1, state);
  assert.equal(result.code, 1);
  assert.equal(result.stdout, "");
  assert.equal(result.stderr.trim(), "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read.");
  assert.equal(existsSync(state), false);
});

test("a model-provider error is one line after the credits line, and the state file is not written", async () => {
  const state = statePath();
  const result = await poll(undefined, 1, state, { status: 401, message: "invalid x-api-key" });
  assert.equal(result.code, 1);
  assert.equal(result.stdout, "");
  assert.deepEqual(result.stderr.trimEnd().split("\n"), ["LexAPI credits reported for this run: 8.", "Model error: invalid x-api-key"]);
  assert.equal(existsSync(state), false);
});

test("a malformed state file: one line naming it, exit 2, no request, the file untouched", async () => {
  const state = statePath();
  writeFileSync(state, "{ not json");
  const result = await poll(undefined, 1, state);
  assert.equal(result.code, 2);
  assert.equal(result.stdout, "");
  assert.equal(result.stderr.trimEnd().split("\n").length, 1);
  assert.ok(result.stderr.includes(state));
  assert.deepEqual(result.requests, []);
  assert.equal(readFileSync(state, "utf8"), "{ not json");
});

test("missing keys: one line each, exit 2, no request", { skip: hasOwnEnv("regulatory-change-monitor") && "this example has its own .env" }, async () => {
  const stub = await startStubServer();
  try {
    const result = await runExample(MONITOR, ["--watchlist", WATCHLIST, "--state", statePath()], { LEXAPI_BASE_URL: stub.baseUrl });
    assert.equal(result.code, 2);
    assert.equal(result.stdout, "");
    assert.deepEqual(result.stderr.trimEnd().split("\n"), [
      "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples or use --dry-run.",
      "ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run.",
    ]);
    assert.equal(stub.requests.length, 0);
  } finally {
    await stub.close();
  }
});

// ── webhook setup ───────────────────────────────────────────────────────

const HOOK_URL = { LEXAPI_WEBHOOK_URL: "https://example.com/webhooks/lexapi" };

test("webhook setup: a refused create prints only the error and exits 1", async () => {
  const result = await run(SETUP, REFUSALS["past due"], 1, [], HOOK_URL);
  assert.equal(result.code, 1);
  assert.equal(result.stdout, "");
  assert.deepEqual(result.stderr.trimEnd().split("\n"), REFUSALS["past due"]!.lines);
  assert.deepEqual(result.requests, ["POST /webhooks"]);
});

test("webhook setup: a refused test delivery still shows the webhook's id and secret, and how to re-test or remove it", async () => {
  const result = await run(SETUP, REFUSALS.quota, 2, [], HOOK_URL);
  assert.equal(result.code, 1);
  assert.match(result.stdout, /^Webhook wh_fixture_0001 created\.\n/);
  assert.match(result.stdout, /fixture-secret-not-a-real-secret/);
  assert.deepEqual(result.stderr.trimEnd().split("\n"), [
    'The webhook wh_fixture_0001 was created, but its test delivery failed. Re-test it with client.webhooks.test("wh_fixture_0001") (POST /webhooks/wh_fixture_0001/test), or remove it with client.webhooks.delete("wh_fixture_0001") (DELETE /webhooks/wh_fixture_0001) before running this again.',
    ...REFUSALS.quota!.lines,
  ]);
  assert.deepEqual(result.requests, ["POST /webhooks", "POST /webhooks/wh_fixture_0001/test"]);
});

test("webhook setup: both calls succeed", async () => {
  const result = await run(SETUP, undefined, 1, [], HOOK_URL);
  assert.equal(result.code, 0, result.stderr);
  assert.match(result.stdout, /Test delivery: SUCCESS \(HTTP 200\)/);
});

test("webhook setup: missing settings exit 2 with no request", { skip: hasOwnEnv("regulatory-change-monitor") && "this example has its own .env" }, async () => {
  const stub = await startStubServer();
  try {
    const result = await runExample(SETUP, [], { LEXAPI_BASE_URL: stub.baseUrl });
    assert.equal(result.code, 2);
    assert.deepEqual(result.stderr.trimEnd().split("\n"), [
      "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples or use --dry-run.",
      "LEXAPI_WEBHOOK_URL is not set. Set it to the public HTTPS URL of your receiver.",
    ]);
    assert.equal(stub.requests.length, 0);
  } finally {
    await stub.close();
  }
});
