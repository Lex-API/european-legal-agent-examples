/**
 * The command line as a process: `src/main.ts` runs as a child against the
 * local stub and a local stand-in for the model API; the test reads its exit
 * code, stdout and stderr.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { test } from "node:test";
import { STACK_TRACE, realRunEnv, runExample, startFakeModel } from "../../../offline/process-harness.ts";
import { startStubServer, type StubReply } from "../../../offline/stub-server.ts";
import { assertLocalBaseUrl } from "../../../offline/guard.ts";
import { REFUSALS } from "../../../offline/test-harness.ts";
import { describeFailure } from "../../../shared/cli.ts";
import { McpLegislationSearch } from "../src/retrieve-mcp.ts";

const MCP_ENTRY = createRequire(import.meta.url).resolve("@lexapi/mcp");

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
  assert.equal(run.stderr.trim(), "Model error: invalid x-api-key. Check ANTHROPIC_API_KEY.");
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

test("missing keys: one line each, exit 2, and no request reaches LexAPI or the model", async () => {
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

/** True while a process with this id exists. */
function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

test("through the MCP server, a refusal after a charged search prints the credits line in the same form as a finished run", async () => {
  const run = await runWithFault(REFUSALS.credits, 2, ["--via", "mcp"]);
  assert.equal(run.code, 1);
  assert.equal(run.stdout, "");
  const lines = run.stderr.trimEnd().split("\n");
  assert.equal(lines.length, 2, run.stderr);
  assert.equal(lines[0], "LexAPI credits reported for this run: not reported. The MCP tool result does not carry them; 1 search(es) cost 5 credits each.");
  assert.match(lines[1]!, /CREDITS_EXHAUSTED/);
});

/** The MCP server process the command line started: its child running the `@lexapi/mcp` entry point, and nothing else. */
function mcpChildOf(cliPid: number): number[] {
  try {
    const out = execFileSync("pgrep", ["-P", String(cliPid), "-f", MCP_ENTRY], { encoding: "utf8" });
    return out.trim().split("\n").filter(Boolean).map(Number);
  } catch {
    return []; // pgrep exits 1 when nothing matches
  }
}

async function waitUntilGone(pids: number[]): Promise<void> {
  for (let i = 0; i < 100 && pids.some(alive); i++) await new Promise((resolve) => setTimeout(resolve, 20));
}

test("through the MCP server, a server that dies mid-call is one line, exit 1, and leaves no process behind", async () => {
  let killed: number[] = [];
  let cliPid = 0;
  const stub = await startStubServer({
    override: () => {
      // While the MCP server waits for this answer, end it (and only it).
      killed = mcpChildOf(cliPid);
      for (const pid of killed) process.kill(pid, "SIGKILL");
      return "hang";
    },
  });
  const fake = await startFakeModel("ai-act");
  try {
    const run = await runExample(ENTRY, ["--via", "mcp", DESCRIPTION], realRunEnv(stub.baseUrl, fake), (pid) => (cliPid = pid));
    assert.equal(run.code, 1, run.stderr);
    assert.equal(run.stdout, "");
    assert.equal(run.stderr.trim(), "MCP server error: MCP error -32000: Connection closed");
    assert.equal(killed.length, 1);
    assert.equal(alive(killed[0]!), false);
  } finally {
    await stub.close();
    await fake.close();
  }
});

test("through the MCP server, a server killed before the first call is one line, exit 1, no stack trace", async () => {
  let killed: number[] = [];
  let cliPid = 0;
  const stub = await startStubServer();
  // The plan (the first model call) comes after the MCP server started and before any tool call.
  const fake = await startFakeModel("ai-act", undefined, () => {
    if (killed.length > 0) return;
    killed = mcpChildOf(cliPid);
    for (const pid of killed) process.kill(pid, "SIGKILL");
    execFileSync(process.execPath, ["-e", "setTimeout(() => {}, 300)"]); // let the transport see it go
  });
  try {
    const run = await runExample(ENTRY, ["--via", "mcp", DESCRIPTION], realRunEnv(stub.baseUrl, fake), (pid) => (cliPid = pid));
    assert.equal(run.code, 1, run.stderr);
    assert.equal(run.stdout, "");
    assert.match(run.stderr.trim(), /^MCP server error: /);
    assert.equal(run.stderr.trim().split("\n").length, 1);
    assert.doesNotMatch(run.stderr, STACK_TRACE);
    assert.equal(killed.length, 1);
    assert.equal(stub.requests.length, 0);
  } finally {
    await stub.close();
    await fake.close();
  }
});

test("through the MCP server, a server gone between two calls: the second call is one line", async () => {
  const stub = await startStubServer();
  try {
    const search = await McpLegislationSearch.start({ LEXAPI_API_KEY: "lex_offline_test_key", LEXAPI_BASE_URL: assertLocalBaseUrl(stub.baseUrl) });
    const pid = search.pid!;
    assert.ok((await search.search("classification of AI systems")).hits.length > 0);
    process.kill(pid, "SIGKILL");
    await waitUntilGone([pid]);
    await new Promise((resolve) => setTimeout(resolve, 100));
    const error = await search.search("risk management").then(() => undefined, (thrown: unknown) => thrown);
    await search.close().catch(() => {});
    assert.deepEqual(describeFailure(error)?.exitCode, 1);
    assert.match(describeFailure(error)?.lines[0] ?? "", /^MCP server error: /);
  } finally {
    await stub.close();
  }
});

for (const [signal, code] of [["SIGINT", 130], ["SIGTERM", 143]] as const) {
  test(`${signal} to the command line alone stops the MCP server process too`, async () => {
    let child: number[] = [];
    let cliPid = 0;
    const stub = await startStubServer({
      override: () => {
        child = mcpChildOf(cliPid);
        process.kill(cliPid, signal); // the parent only, not its process group
        return "hang";
      },
    });
    const fake = await startFakeModel("ai-act");
    try {
      const run = await runExample(ENTRY, ["--via", "mcp", DESCRIPTION], realRunEnv(stub.baseUrl, fake), (pid) => (cliPid = pid));
      assert.equal(run.code, code, run.stderr);
      assert.equal(run.stderr.trim(), "Interrupted.");
      assert.equal(child.length, 1);
      await waitUntilGone(child);
      assert.equal(alive(child[0]!), false);
    } finally {
      await stub.close();
      await fake.close();
    }
  });
}

test("through the OpenAI client: a real run with LLM_PROVIDER=openai against the local stand-in", async () => {
  const stub = await startStubServer();
  const fake = await startFakeModel("ai-act");
  try {
    const env = { ...realRunEnv(stub.baseUrl, fake), LLM_PROVIDER: "openai", OPENAI_API_KEY: "sk-fake-offline-test", OPENAI_MODEL: "fake-model", OPENAI_BASE_URL: `${fake.baseUrl}/v1` };
    const run = await runExample(ENTRY, [DESCRIPTION], env);
    assert.equal(run.code, 0, run.stderr);
    assert.match(run.stdout, /Model: openai:fake-model/);
    assert.ok(fake.requests >= 2); // the plan and the draft
  } finally {
    await stub.close();
    await fake.close();
  }
});

test("through the MCP server, a call that is not answered in time is one line, and the server process is stopped", async () => {
  const stub = await startStubServer({ override: () => "hang" });
  try {
    const search = await McpLegislationSearch.start(
      { LEXAPI_API_KEY: "lex_offline_test_key", LEXAPI_BASE_URL: assertLocalBaseUrl(stub.baseUrl) },
      { requestTimeoutMs: 300 },
    );
    const pid = search.pid;
    assert.ok(pid);
    const error = await search.search("query").then(() => undefined, (thrown: unknown) => thrown);
    await search.close();
    assert.deepEqual(describeFailure(error)?.exitCode, 1);
    assert.match(describeFailure(error)?.lines[0] ?? "", /^MCP server error: .*Request timed out/);
    await new Promise((resolve) => setTimeout(resolve, 200));
    assert.equal(alive(pid), false);
  } finally {
    await stub.close();
  }
});

test("LLM_PROVIDER: an unknown value is a configuration error before any request; empty means the default", async () => {
  const stub = await startStubServer();
  const fake = await startFakeModel("ai-act");
  try {
    const bogus = await runExample(ENTRY, [DESCRIPTION], { ...realRunEnv(stub.baseUrl, fake), LLM_PROVIDER: " bogus " });
    assert.equal(bogus.code, 2);
    assert.equal(bogus.stderr.trim(), 'LLM_PROVIDER is "bogus"; it must be "anthropic" (the default) or "openai".');
    assert.equal(stub.requests.length, 0);
    assert.equal(fake.requests, 0);
    for (const value of ["", "  ", " Anthropic "]) {
      const run = await runExample(ENTRY, [DESCRIPTION], { ...realRunEnv(stub.baseUrl, fake), LLM_PROVIDER: value });
      assert.equal(run.code, 0, `${JSON.stringify(value)}: ${run.stderr}`);
    }
  } finally {
    await stub.close();
    await fake.close();
  }
});

test("two Ctrl-C in a row during an MCP call: exit 130, one line, and no MCP server process left", async () => {
  let child: number[] = [];
  let cliPid = 0;
  const stub = await startStubServer({
    override: () => {
      child = mcpChildOf(cliPid);
      process.kill(cliPid, "SIGINT");
      setTimeout(() => process.kill(cliPid, "SIGINT"), 5);
      return "hang";
    },
  });
  const fake = await startFakeModel("ai-act");
  try {
    const run = await runExample(ENTRY, ["--via", "mcp", DESCRIPTION], realRunEnv(stub.baseUrl, fake), (pid) => (cliPid = pid));
    assert.equal(run.code, 130, run.stderr);
    assert.equal(run.stderr, "Interrupted.\n");
    assert.equal(child.length, 1);
    assert.equal(alive(child[0]!), false); // gone when the example exits: terminated, not left to a graceful close
  } finally {
    await stub.close();
    await fake.close();
  }
});

test("Ctrl-C during the plan (the model call) on the MCP route prints only Interrupted.", async () => {
  let cliPid = 0;
  let child: number[] = [];
  const stub = await startStubServer();
  const fake = await startFakeModel("ai-act", undefined, () => {
    if (child.length > 0) return;
    child = mcpChildOf(cliPid);
    process.kill(cliPid, "SIGINT");
    execFileSync(process.execPath, ["-e", "setTimeout(() => {}, 200)"]); // answer the plan after the signal
  });
  try {
    const run = await runExample(ENTRY, ["--via", "mcp", DESCRIPTION], realRunEnv(stub.baseUrl, fake), (pid) => (cliPid = pid));
    assert.equal(run.code, 130, run.stderr);
    assert.equal(run.stderr, "Interrupted.\n");
    assert.equal(child.length, 1);
    await waitUntilGone(child);
    assert.equal(alive(child[0]!), false);
  } finally {
    await stub.close();
    await fake.close();
  }
});
