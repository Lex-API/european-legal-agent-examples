/**
 * Process-level tests: start an example's real command line as a child
 * process against the local stub and a local stand-in for the Anthropic API,
 * and read its exit code, stdout and stderr. The child gets a clean, explicit
 * environment (an empty HOME, LLM_PROVIDER and the model URL pinned, the
 * example's .env ignored), and `offline/no-network.ts` blocks every request
 * that is not to 127.0.0.1 inside it.
 */
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { assertLocalBaseUrl } from "./guard.ts";
import { TEST_IGNORE_DOTENV } from "../shared/env.ts";
import { readFixture } from "./stub-server.ts";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

export interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

export function runExample(entry: string, args: string[], env: Record<string, string>, onStart?: (pid: number) => void): Promise<RunResult> {
  if (env.LEXAPI_BASE_URL) assertLocalBaseUrl(env.LEXAPI_BASE_URL);
  if (env.ANTHROPIC_BASE_URL) assertLocalBaseUrl(env.ANTHROPIC_BASE_URL);
  // An empty home: no model-provider configuration of this machine is read.
  const home = mkdtempSync(join(tmpdir(), "example-home-"));
  const child = spawn(
    process.execPath,
    ["--import", "tsx", "--import", join(ROOT, "offline", "no-network.ts"), join(ROOT, entry), ...args],
    {
      cwd: ROOT,
      env: {
        PATH: process.env.PATH ?? "",
        HOME: home,
        // Hermetic: the example's own .env is not read, and the provider and its URL are pinned.
        [TEST_IGNORE_DOTENV]: "1",
        LLM_PROVIDER: "anthropic",
        ANTHROPIC_BASE_URL: "http://127.0.0.1:9", // nothing listens there; tests that need a model pass their own
        // Keep the MCP route's guard on: under a test context it starts the MCP server only against the stub.
        NODE_TEST_CONTEXT: process.env.NODE_TEST_CONTEXT ?? "child",
        ...env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  if (child.pid !== undefined) onStart?.(child.pid);
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString("utf8")));
  child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString("utf8")));
  return new Promise((resolve) =>
    child.on("close", (code) => {
      rmSync(home, { recursive: true, force: true });
      resolve({ code, stdout, stderr });
    }),
  );
}

export interface FakeModel {
  /** `http://127.0.0.1:<port>` - pass as `ANTHROPIC_BASE_URL`, or with `/v1` as `OPENAI_BASE_URL`. */
  baseUrl: string;
  requests: number;
  close(): Promise<void>;
}

/** An error answer: a provider error body with `message`, or `raw` text (an HTML page from a proxy). */
export type ModelFault = { status: number; message: string } | { status: number; raw: string };

/**
 * A local stand-in for the Anthropic Messages API (`POST /v1/messages`) and the OpenAI Chat
 * Completions API (`POST /v1/chat/completions`): it answers every request with a call of the
 * requested tool, taking the input from `offline/fixtures/llm/<script>.json`, or with `fail`.
 * `onRequest` runs before each answer.
 */
export async function startFakeModel(script: string, fail?: ModelFault, onRequest?: () => void): Promise<FakeModel> {
  const replies = JSON.parse(readFixture(`llm/${script}.json`)) as Record<string, unknown>;
  const fake = { requests: 0 };
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      fake.requests++;
      onRequest?.();
      if (fail) {
        if ("raw" in fail) {
          res.writeHead(fail.status, { "content-type": "text/html" }).end(fail.raw);
        } else {
          const body = { type: "error", error: { type: "invalid_request_error", message: fail.message } };
          res.writeHead(fail.status, { "content-type": "application/json" }).end(JSON.stringify(body));
        }
        return;
      }
      res.setHeader("content-type", "application/json");
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
        tools?: Array<{ name?: string; function?: { name: string } }>;
        model?: string;
      };
      const name = body.tools?.[0]?.function?.name ?? body.tools?.[0]?.name ?? "";
      const scripted = replies[name];
      const input = Array.isArray(scripted) ? scripted[scripted.length - 1] : scripted;
      if (req.url?.endsWith("/chat/completions")) {
        res.writeHead(200).end(
          JSON.stringify({
            id: `chatcmpl-fake-${fake.requests}`,
            object: "chat.completion",
            created: 0,
            model: body.model ?? "fake",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: null, tool_calls: [{ id: `call_fake_${fake.requests}`, type: "function", function: { name, arguments: JSON.stringify(input) } }] },
                finish_reason: "tool_calls",
              },
            ],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          }),
        );
        return;
      }
      res.writeHead(200).end(
        JSON.stringify({
          id: `msg_fake_${fake.requests}`,
          type: "message",
          role: "assistant",
          model: body.model ?? "fake",
          content: [{ type: "tool_use", id: `toolu_fake_${fake.requests}`, name, input }],
          stop_reason: "tool_use",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
      );
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    get requests() {
      return fake.requests;
    },
    close: () => new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve()))),
  };
}

/** The keys of a real run, pointed at the local stand-ins. */
export function realRunEnv(lexapiBaseUrl: string, model: FakeModel): Record<string, string> {
  return {
    LEXAPI_API_KEY: "lex_offline_test_key",
    LEXAPI_BASE_URL: lexapiBaseUrl,
    ANTHROPIC_API_KEY: "sk-fake-offline-test-not-a-key",
    ANTHROPIC_BASE_URL: model.baseUrl,
  };
}

/** A Node stack trace line, which no expected failure may print. */
export const STACK_TRACE = /^\s+at |Error: .*\n\s+at /m;
