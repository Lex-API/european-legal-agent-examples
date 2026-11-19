/**
 * The command lines as processes: `src/main.ts` and `src/webhook-setup.ts` run
 * as children against the local stub and a local stand-in for the model API.
 */
import assert from "node:assert/strict";
import { chmodSync, existsSync, lstatSync, mkdtempSync, readFileSync, readdirSync, readlinkSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { STACK_TRACE, realRunEnv, runExample, startFakeModel } from "../../../offline/process-harness.ts";
import { readFixture, startStubServer, type StubReply } from "../../../offline/stub-server.ts";
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
  assert.deepEqual(result.stderr.trimEnd().split("\n"), ["LexAPI credits reported for this run: 8.", "Model error: invalid x-api-key. Check ANTHROPIC_API_KEY."]);
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

test("missing keys: one line each, exit 2, no request", async () => {
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

test("webhook setup: missing settings exit 2 with no request", async () => {
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

// ── state file and watch list ───────────────────────────────────────────

const notSaved = (path: string, code: string) =>
  `The state file ${path} could not be written (${code}). The digest above was produced, but the state was not saved, so the next run will report the same items again.`;

test("a state path whose directory is a regular file: one line, exit 2, no request", async () => {
  const parent = join(dir, `not-a-directory-${++n}`);
  writeFileSync(parent, "");
  const state = join(parent, "state.json");
  const result = await poll(undefined, 1, state);
  assert.equal(result.code, 2);
  assert.equal(result.stdout, "");
  assert.equal(result.stderr.trim(), `The state file ${state} cannot be written: ${parent} is not a directory.`);
  assert.deepEqual(result.requests, []);
});

test("a state file in a directory that cannot be searched: one line, exit 2, no request", { skip: process.getuid?.() === 0 && "root can search any directory" }, async () => {
  const closed = mkdtempSync(join(dir, "closed-"));
  const state = join(closed, "state.json");
  chmodSync(closed, 0o000);
  try {
    const result = await poll(undefined, 1, state);
    assert.equal(result.code, 2);
    assert.equal(result.stdout, "");
    assert.equal(result.stderr.trim(), `The state file ${state} cannot be checked (EACCES).`);
    assert.deepEqual(result.requests, []);
  } finally {
    chmodSync(closed, 0o755);
  }
});

test("a symlinked state path stays a link: the target is written", async () => {
  const target = statePath();
  assert.equal((await poll(undefined, 1, target)).code, 0);
  const link = join(dir, `link-${++n}.json`);
  symlinkSync(target, link);
  const result = await poll(undefined, 1, link);
  assert.equal(result.code, 0, result.stderr);
  assert.ok(lstatSync(link).isSymbolicLink());
  assert.equal(readlinkSync(link), target);
  assert.ok(JSON.parse(readFileSync(target, "utf8")).baselined.length > 0);
});

test("a new state file is readable by its owner only; an existing file keeps its mode", async () => {
  const fresh = statePath();
  assert.equal((await poll(undefined, 1, fresh)).code, 0);
  assert.equal(statSync(fresh).mode & 0o777, 0o600);
  chmodSync(fresh, 0o640);
  assert.equal((await poll(undefined, 1, fresh)).code, 0);
  assert.equal(statSync(fresh).mode & 0o777, 0o640);
});

test("the state is written by a rename (never in place), and a killed run's temporary files are removed", async () => {
  const state = statePath();
  assert.equal((await poll(undefined, 1, state)).code, 0);
  const before = statSync(state).ino;
  const name = state.split("/").pop()!;
  const leftover = join(dir, `.${name}.99999.tmp`);
  const unrelated = [join(dir, `.${name}.tmp`), join(dir, `.other-${name}.99999.tmp`), join(dir, `x.${name}.99999.tmp`)];
  for (const path of [leftover, ...unrelated]) writeFileSync(path, "{ partial");
  assert.equal((await poll(undefined, 1, state)).code, 0);
  assert.notEqual(statSync(state).ino, before); // a new file took its place: the old one was never rewritten
  assert.equal(existsSync(leftover), false);
  for (const path of unrelated) assert.ok(existsSync(path), path);
  assert.deepEqual(readdirSync(dir).filter((file) => file.startsWith(`.${name}.`) && file.endsWith(".tmp") && /\.\d+\.tmp$/.test(file)), []);
});

test("a read-only state directory: the digest is printed, the state is not saved, exit 1", { skip: process.getuid?.() === 0 && "root can write to a read-only directory" }, async () => {
  const readOnly = mkdtempSync(join(dir, "read-only-"));
  const state = join(readOnly, "state.json");
  chmodSync(readOnly, 0o555);
  try {
    const result = await poll(undefined, 1, state);
    assert.equal(result.code, 1);
    assert.match(result.stdout, /# Regulatory change digest/);
    assert.deepEqual(result.stderr.trimEnd().split("\n"), ["LexAPI credits reported for this run: 8.", notSaved(state, "EACCES")]);
    assert.deepEqual(readdirSync(readOnly), []); // no temporary file left behind
  } finally {
    chmodSync(readOnly, 0o755);
  }
});

test("a state path that is a directory: one line saying so, exit 2, no request", async () => {
  const state = mkdtempSync(join(dir, "state-dir-"));
  const result = await poll(undefined, 1, state);
  assert.equal(result.code, 2);
  assert.equal(result.stdout, "");
  assert.equal(result.stderr.trim(), `The state file ${state} is a directory; give --state a file path.`);
  assert.deepEqual(result.requests, []);
});

for (const [name, content] of Object.entries({
  "acts not a list": '{"acts": "x"}',
  "an act without a celex": '{"acts": [{"label": "GDPR"}]}',
  "a celex that is not text": '{"acts": [{"celex": 5}]}',
  "a topic without a query or type": '{"topics": [{"name": "AI"}]}',
  "not an object": "[]",
  "days too large": '{"acts": [{"celex": "32016R0679"}], "days": 1e308}',
  "days 0": '{"acts": [{"celex": "32016R0679"}], "days": 0}',
  "days 366": '{"acts": [{"celex": "32016R0679"}], "days": 366}',
  "days not whole": '{"acts": [{"celex": "32016R0679"}], "days": 1.5}',
})) {
  test(`a watch list with ${name}: one line naming the file and what is expected, exit 2, no request`, async () => {
    const path = join(dir, `watchlist-${++n}.json`);
    writeFileSync(path, content);
    const result = await run(MONITOR, undefined, 1, ["--watchlist", path, "--state", statePath()]);
    assert.equal(result.code, 2);
    assert.equal(result.stdout, "");
    const lines = result.stderr.trimEnd().split("\n");
    assert.equal(lines.length, 1);
    assert.ok(lines[0]!.startsWith(`The watch list ${path} is not valid: `), lines[0]);
    assert.match(lines[0]!, /expected \{"acts": \[\{"celex"/);
    assert.deepEqual(result.requests, []);
  });
}

// ── webhook setup: the other two failures ───────────────────────────────

test("webhook setup: a test delivery that FAILED is a failed test: guidance and the delivery's own status, exit 1", async () => {
  const failed = { status: 200, body: JSON.parse(readFixture("webhook-test.failed.json")) as unknown };
  const result = await run(SETUP, failed, 2, [], HOOK_URL);
  assert.equal(result.code, 1);
  assert.match(result.stdout, /^Webhook wh_fixture_0001 created\.\n/);
  assert.match(result.stdout, /fixture-secret-not-a-real-secret/);
  assert.doesNotMatch(result.stdout, /Test delivery/);
  assert.deepEqual(result.stderr.trimEnd().split("\n"), [
    'The webhook wh_fixture_0001 was created, but its test delivery failed. Re-test it with client.webhooks.test("wh_fixture_0001") (POST /webhooks/wh_fixture_0001/test), or remove it with client.webhooks.delete("wh_fixture_0001") (DELETE /webhooks/wh_fixture_0001) before running this again.',
    "Test delivery: FAILED (HTTP 500) - connect ECONNREFUSED 203.0.113.7:443 [2Jretry later lex_***",
  ]);
});

test("webhook setup: a create answer without an id prints what came back and says a webhook may exist, exit 1", async () => {
  const noId = { status: 201, body: { success: true, webhook: { name: "Regulatory change monitor", secret: "fixture-secret-not-a-real-secret" } } };
  const result = await run(SETUP, noId, 1, [], HOOK_URL);
  assert.equal(result.code, 1);
  assert.match(result.stdout, /^The answer to the create \(its body and the SDK's response metadata, API keys masked\):\n\{/);
  assert.match(result.stdout, /"secret": "fixture-secret-not-a-real-secret"/);
  assert.equal(
    result.stderr.trim(),
    "LexAPI error INVALID_RESPONSE: The answer to the create has no webhook id, so a webhook may have been created. List your webhooks with client.webhooks.list() (GET /webhooks) before running this again.",
  );
  assert.deepEqual(result.requests, ["POST /webhooks"]);
});


test("a CELEX number listed twice is polled once, with one note", async () => {
  const path = join(dir, `watchlist-${++n}.json`);
  writeFileSync(path, JSON.stringify({ acts: [{ celex: "32016R0679" }, { celex: " 32016r0679 ", label: "again" }], topics: [], days: 365 }));
  const result = await run(MONITOR, undefined, 1, ["--watchlist", path, "--state", statePath()]);
  assert.equal(result.code, 0, result.stderr);
  assert.deepEqual(result.requests, ["GET /citations/cited-by/32016R0679"]);
  assert.deepEqual(result.stderr.trimEnd().split("\n"), [
    `The watch list ${path} names CELEX 32016R0679 more than once; it is polled once.`,
    "LexAPI credits reported for this run: 1.",
  ]);
});

for (const [name, body, printed] of [
  ["success false with status SUCCESS", { success: false, delivery: { status: "SUCCESS", responseStatus: 200 } }, "Test delivery: not confirmed (success: false, status: SUCCESS, HTTP 200)"],
  ["success missing", { delivery: { status: "SUCCESS", responseStatus: 200 } }, "Test delivery: not confirmed (success: missing, status: SUCCESS, HTTP 200)"],
] as const) {
  test(`webhook setup: a test answer with ${name} is not shown as a success`, async () => {
    const result = await run(SETUP, { status: 200, body }, 2, [], HOOK_URL);
    assert.equal(result.code, 1);
    const lines = result.stderr.trimEnd().split("\n");
    assert.match(lines[0]!, /^The webhook wh_fixture_0001 was created, but its test delivery failed\./);
    assert.equal(lines[1], printed);
    assert.doesNotMatch(result.stdout, /Test delivery/);
  });
}
