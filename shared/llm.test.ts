import assert from "node:assert/strict";
import { test } from "node:test";
import type Anthropic from "@anthropic-ai/sdk";
import type OpenAI from "openai";
import "../offline/test-harness.ts";
import { AnthropicLlm, DEFAULT_ANTHROPIC_MODEL } from "./llm-anthropic.ts";
import { OpenAiLlm } from "./llm-openai.ts";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { startFakeModel, type ModelFault } from "../offline/process-harness.ts";
import { blockExternalFetch } from "../offline/guard.ts";
import { TEST_IGNORE_DOTENV, loadEnvFile } from "./env.ts";
import { LlmError, ScriptedLlm, createLlmFromEnv, type LlmRequest } from "./llm.ts";

blockExternalFetch(); // the real clients below may reach 127.0.0.1 only

const request: LlmRequest = {
  system: "system",
  user: "user",
  tool: { name: "submit", description: "d", schema: { type: "object", properties: {}, required: [], additionalProperties: false } },
};

/** A stand-in for the provider SDK client: records the request, returns canned responses. No network. */
function fakeAnthropic(responses: unknown[]): { client: Anthropic; calls: Anthropic.MessageCreateParamsNonStreaming[] } {
  const calls: Anthropic.MessageCreateParamsNonStreaming[] = [];
  const client = {
    messages: {
      create: async (params: Anthropic.MessageCreateParamsNonStreaming) => {
        calls.push(structuredClone(params));
        return responses[Math.min(calls.length - 1, responses.length - 1)];
      },
    },
  } as unknown as Anthropic;
  return { client, calls };
}

test("ScriptedLlm replies by tool name and consumes a list in order", async () => {
  const llm = new ScriptedLlm({ submit: [{ n: 1 }, { n: 2 }] });
  assert.deepEqual(await llm.generate(request), { n: 1 });
  assert.deepEqual(await llm.generate(request), { n: 2 });
  assert.deepEqual(await llm.generate(request), { n: 2 });
  await assert.rejects(llm.generate({ ...request, tool: { ...request.tool, name: "other" } }), LlmError);
});

test("AnthropicLlm sends a strict tool, leaves tool_choice unset and returns the tool input", async () => {
  const { client, calls } = fakeAnthropic([
    { stop_reason: "tool_use", content: [{ type: "text", text: "ok" }, { type: "tool_use", id: "t1", name: "submit", input: { a: 1 } }] },
  ]);
  const llm = new AnthropicLlm({ client });
  assert.deepEqual(await llm.generate(request), { a: 1 });
  assert.equal(llm.label, `anthropic:${DEFAULT_ANTHROPIC_MODEL}`);
  assert.equal(calls[0]?.model, "claude-sonnet-5-5");
  assert.equal(calls[0]?.tool_choice, undefined);
  assert.deepEqual(calls[0]?.tools, [{ name: "submit", description: "d", input_schema: request.tool.schema, strict: true }]);
});

test("AnthropicLlm re-prompts once when the model answers without the tool, then gives up", async () => {
  const text = { stop_reason: "end_turn", content: [{ type: "text", text: "Here is my answer." }] };
  const tool = { stop_reason: "tool_use", content: [{ type: "tool_use", id: "t1", name: "submit", input: { a: 2 } }] };

  const recovered = fakeAnthropic([text, tool]);
  assert.deepEqual(await new AnthropicLlm({ client: recovered.client }).generate(request), { a: 2 });
  assert.equal(recovered.calls.length, 2);
  assert.equal(recovered.calls[1]?.messages.length, 3);

  const stuck = fakeAnthropic([text]);
  await assert.rejects(new AnthropicLlm({ client: stuck.client }).generate(request), /did not call the submit tool/);
  assert.equal(stuck.calls.length, 2);
});

test("AnthropicLlm surfaces a refusal and a truncated reply as errors", async () => {
  const refusal = fakeAnthropic([{ stop_reason: "refusal", content: [] }]);
  await assert.rejects(new AnthropicLlm({ client: refusal.client }).generate(request), /declined/);
  const cutOff = fakeAnthropic([{ stop_reason: "max_tokens", content: [] }]);
  await assert.rejects(new AnthropicLlm({ client: cutOff.client }).generate(request), /cut off/);
});

test("OpenAiLlm forces the function call and parses its arguments", async () => {
  const calls: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming[] = [];
  const reply = (message: unknown) =>
    ({
      chat: {
        completions: {
          create: async (params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming) => {
            calls.push(params);
            return { choices: [{ message }] };
          },
        },
      },
    }) as unknown as OpenAI;

  const ok = reply({ tool_calls: [{ id: "c1", type: "function", function: { name: "submit", arguments: '{"a":3}' } }] });
  assert.deepEqual(await new OpenAiLlm({ model: "test-model", client: ok }).generate(request), { a: 3 });
  assert.deepEqual(calls[0]?.tool_choice, { type: "function", function: { name: "submit" } });
  assert.equal(calls[0]?.model, "test-model");

  await assert.rejects(new OpenAiLlm({ model: "m", client: reply({ refusal: "no" }) }).generate(request), /declined/);
  await assert.rejects(new OpenAiLlm({ model: "m", client: reply({ content: "text" }) }).generate(request), /did not call/);
  const broken = reply({ tool_calls: [{ id: "c1", type: "function", function: { name: "submit", arguments: "{" } }] });
  await assert.rejects(new OpenAiLlm({ model: "m", client: broken }).generate(request), /not valid JSON/);
});

test("createLlmFromEnv requires an explicit OpenAI model and rejects unknown providers", async () => {
  await assert.rejects(createLlmFromEnv({ LLM_PROVIDER: "openai" }), /OPENAI_MODEL/);
  await assert.rejects(createLlmFromEnv({ LLM_PROVIDER: "other" }), /Unknown LLM_PROVIDER/);
});

// ── the real provider clients, against a local stand-in on 127.0.0.1 ───

async function withFake<T>(fail: ModelFault | undefined, run: (baseUrl: string) => Promise<T>): Promise<T> {
  const fake = await startFakeModel("ai-act", fail);
  const saved = { a: process.env.ANTHROPIC_BASE_URL, o: process.env.OPENAI_BASE_URL };
  process.env.ANTHROPIC_BASE_URL = fake.baseUrl;
  process.env.OPENAI_BASE_URL = `${fake.baseUrl}/v1`;
  try {
    return await run(fake.baseUrl);
  } finally {
    if (saved.a === undefined) delete process.env.ANTHROPIC_BASE_URL;
    else process.env.ANTHROPIC_BASE_URL = saved.a;
    if (saved.o === undefined) delete process.env.OPENAI_BASE_URL;
    else process.env.OPENAI_BASE_URL = saved.o;
    await fake.close();
  }
}

const PLAN_REQUEST: LlmRequest = {
  system: "s",
  user: "u",
  tool: { name: "plan_searches", description: "d", schema: { type: "object", properties: {}, additionalProperties: false } },
};
const REAL = {
  anthropic: { LLM_PROVIDER: "anthropic", ANTHROPIC_API_KEY: "sk-fake-offline-test" },
  openai: { LLM_PROVIDER: "openai", OPENAI_API_KEY: "sk-fake-offline-test", OPENAI_MODEL: "fake-model" },
};

for (const [provider, env] of Object.entries(REAL)) {
  test(`the real ${provider} client sends the request to the stand-in and its answer is parsed`, async () => {
    await withFake(undefined, async () => {
      const llm = await createLlmFromEnv(env);
      const plan = (await llm.generate(PLAN_REQUEST)) as { queries?: unknown };
      assert.ok(Array.isArray(plan.queries) && plan.queries.length > 0);
    });
  });

  const keyVariable = provider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY";
  test(`${provider}: a rejected key names ${keyVariable}; a page that is not JSON reads "HTTP <status>: <text>"`, async () => {
    await withFake({ status: 401, message: "invalid x-api-key" }, async () => {
      const error = await (await createLlmFromEnv(env)).generate(PLAN_REQUEST).then(() => undefined, (thrown: unknown) => thrown);
      assert.ok(error instanceof LlmError);
      assert.equal(error.message, `invalid x-api-key. Check ${keyVariable}.`);
    });
    await withFake({ status: 400, raw: "<html>bad gateway</html>" }, async () => {
      const error = await (await createLlmFromEnv(env)).generate(PLAN_REQUEST).then(() => undefined, (thrown: unknown) => thrown);
      assert.ok(error instanceof LlmError);
      assert.equal(error.message, "HTTP 400: <html>bad gateway</html>");
    });
  });
}

test("the .env test hook is honoured only for the exact value 1", () => {
  const folder = mkdtempSync(join(tmpdir(), "dotenv-"));
  const file = join(folder, ".env");
  writeFileSync(file, "EXAMPLES_DOTENV_PROBE=loaded\n");
  try {
    for (const [value, loaded] of [["1", false], ["0", true], ["", true], ["true", true], [" 1", true]] as const) {
      delete process.env.EXAMPLES_DOTENV_PROBE;
      process.env[TEST_IGNORE_DOTENV] = value;
      loadEnvFile(file);
      assert.equal(process.env.EXAMPLES_DOTENV_PROBE === "loaded", loaded, JSON.stringify(value));
    }
  } finally {
    delete process.env[TEST_IGNORE_DOTENV];
    delete process.env.EXAMPLES_DOTENV_PROBE;
    rmSync(folder, { recursive: true, force: true });
  }
});
