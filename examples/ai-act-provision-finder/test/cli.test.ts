/**
 * The command line as a process: `src/main.ts` runs as a child against the
 * local stub and a local stand-in for the model API; the test reads its exit
 * code, stdout and stderr.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { STACK_TRACE, hasOwnEnv, realRunEnv, runExample, startFakeModel } from "../../../offline/process-harness.ts";
import { startStubServer, type StubReply } from "../../../offline/stub-server.ts";
import { REFUSALS } from "../../../offline/test-harness.ts";

const ENTRY = "examples/ai-act-provision-finder/src/main.ts";
const DESCRIPTION = "A SaaS tool that ranks job applicants' CVs and chats with candidates";
const CREDITS_AFTER_ONE = "LexAPI credits reported for this run: 5.";

/** Run the CLI against a stub whose `nth` request (1-based) and every later one get `reply`. */
async function runWithFault(reply: StubReply | undefined, nth: number, args: string[] = [], model?: { status: number; message: string }) {
  const stub = await startStubServer({ override: () => (reply && stub.requests.length >= nth ? reply : undefined) });
  const fake = await startFakeModel("ai-act", model);
  try {
    const result = await runExample(ENTRY, [...args, DESCRIPTION], realRunEnv(stub.baseUrl, fake));
    return { ...result, requests: stub.requests.length, modelRequests: fake.requests };
  } finally {
    await stub.close();
    await fake.close();
  }
}

test("a real run against the stand-ins prints the answer on stdout and exits 0", async () => {
  const run = await runWithFault(undefined, 1);
  assert.equal(run.code, 0, run.stderr);
  assert.match(run.stdout, /## What a human must decide/);
  assert.equal(run.stderr.trim(), "LexAPI credits reported for this run: 15.");
  assert.equal(run.requests, 3);
});

for (const [name, refusal] of Object.entries(REFUSALS)) {
  test(`a ${name} answer on the first search: its lines on stderr, nothing on stdout, exit 1`, async () => {
    const run = await runWithFault(refusal, 1);
    assert.equal(run.code, 1);
    assert.equal(run.stdout, "");
    assert.deepEqual(run.stderr.trimEnd().split("\n"), refusal.lines);
    assert.doesNotMatch(run.stderr, STACK_TRACE);
    assert.equal(run.requests, refusal.attempts);
  });
}

test("a refusal after a charged search prints the credits of that search first", async () => {
  const run = await runWithFault(REFUSALS.credits, 2);
  assert.equal(run.code, 1);
  assert.equal(run.stdout, "");
  assert.deepEqual(run.stderr.trimEnd().split("\n"), [CREDITS_AFTER_ONE, ...REFUSALS.credits!.lines]);
});

test("an unreadable answer is an error, not 'no supporting text'", async () => {
  const run = await runWithFault({ status: 200, body: { success: true } }, 1);
  assert.equal(run.code, 1);
  assert.equal(run.stdout, "");
  assert.equal(run.stderr.trim(), "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read.");
});

test("a model-provider error is one line, after the credits of the searches made", async () => {
  // The plan is the first model call, before any search: no LexAPI request is made.
  const run = await runWithFault(undefined, 1, [], { status: 401, message: "invalid x-api-key" });
  assert.equal(run.code, 1);
  assert.equal(run.stdout, "");
  assert.equal(run.stderr.trim(), "Model error: invalid x-api-key");
  assert.equal(run.requests, 0);
});

test("through the MCP server, a refusal prints the tool's own text on one line and exits 1", async () => {
  const run = await runWithFault(REFUSALS["past due"], 1, ["--via", "mcp"]);
  assert.equal(run.code, 1, run.stderr);
  assert.equal(run.stdout, "");
  const lines = run.stderr.trimEnd().split("\n");
  assert.equal(lines.length, 1, run.stderr);
  assert.match(lines[0]!, /PAYMENT_PAST_DUE/);
  assert.doesNotMatch(run.stderr, STACK_TRACE);
});

test("through the MCP server, an unreadable answer is an error, not 'no supporting text'", async () => {
  const run = await runWithFault({ status: 200, body: { success: true } }, 1, ["--via", "mcp"]);
  assert.equal(run.code, 1);
  assert.equal(run.stdout, "");
  assert.equal(run.stderr.trimEnd().split("\n").length, 1, run.stderr);
  assert.doesNotMatch(run.stderr, /No supporting text/);
  assert.doesNotMatch(run.stderr, STACK_TRACE);
});

test("missing keys: one line each, exit 2, and no request reaches LexAPI or the model", { skip: hasOwnEnv("ai-act-provision-finder") && "this example has its own .env" }, async () => {
  for (const args of [[], ["--via", "mcp"]]) {
    const stub = await startStubServer();
    const fake = await startFakeModel("ai-act");
    try {
      const run = await runExample(ENTRY, [...args, DESCRIPTION], { LEXAPI_BASE_URL: stub.baseUrl, ANTHROPIC_BASE_URL: fake.baseUrl });
      assert.equal(run.code, 2);
      assert.equal(run.stdout, "");
      assert.deepEqual(run.stderr.trimEnd().split("\n"), [
        "LEXAPI_API_KEY is not set. Create a key at https://lex-api.com/dashboard?ref=github-examples or use --dry-run.",
        "ANTHROPIC_API_KEY is not set. Add an Anthropic API key to .env or use --dry-run.",
      ]);
      assert.equal(stub.requests.length, 0);
      assert.equal(fake.requests, 0);
    } finally {
      await stub.close();
      await fake.close();
    }
  }
});

test("a usage error exits 2 without a stack trace", async () => {
  const run = await runExample(ENTRY, ["--no-such-option", "x"], {});
  assert.equal(run.code, 2);
  assert.doesNotMatch(run.stderr, STACK_TRACE);
});
