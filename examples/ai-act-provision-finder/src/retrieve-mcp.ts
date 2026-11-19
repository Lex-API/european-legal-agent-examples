/**
 * The same retrieval step through the LexAPI MCP server instead of the SDK.
 *
 * This starts `@lexapi/mcp` as a child process over stdio and calls its
 * `lex_semantic_legislation` tool. The tool answers with one text block
 * holding JSON; its hits use the same field names as the REST API.
 *
 * One difference from the SDK route: the tool shares a character budget
 * (`maxChars`) between the hit texts and cuts a longer article, appending a
 * marker. `stripTruncation` removes that marker - it is not part of the law -
 * and records how much is missing, so the source is labelled as truncated.
 */
import { createRequire } from "node:module";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { assertLocalBaseUrl } from "../../../offline/guard.ts";
import type { LegislationHit, LegislationSearch, SearchOutcome } from "./retrieve.ts";

/** The largest text budget the tool accepts. */
const MAX_CHARS = 200_000;
/** What `@lexapi/mcp` 0.5.1 appends to a hit text it cut. */
const TRUNCATION_MARKER = / \u2026\[truncated: (\d+) more characters\]$/;

export function stripTruncation(hit: LegislationHit): LegislationHit {
  const match = typeof hit.text === "string" ? TRUNCATION_MARKER.exec(hit.text) : null;
  if (!match || typeof hit.text !== "string") return hit;
  return { ...hit, text: hit.text.slice(0, match.index), truncatedBy: Number(match[1]) };
}

export class McpLegislationSearch implements LegislationSearch {
  readonly label = "@lexapi/mcp (lex_semantic_legislation)";
  private constructor(private readonly client: Client) {}

  static async start(env: NodeJS.ProcessEnv = process.env): Promise<McpLegislationSearch> {
    const apiKey = env.LEXAPI_API_KEY?.trim();
    if (!apiKey) throw new Error("LEXAPI_API_KEY is not set.");
    // The child process is outside the test harness's fetch guard, so under `node --test`
    // refuse to start it unless it will talk to the local stub.
    if (process.env.NODE_TEST_CONTEXT) assertLocalBaseUrl(env.LEXAPI_BASE_URL);
    const serverEntry = createRequire(import.meta.url).resolve("@lexapi/mcp");
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [serverEntry],
      env: {
        LEXAPI_API_KEY: apiKey,
        // Only the read-only tools are needed here.
        LEXAPI_MCP_READ_ONLY: "true",
        ...(env.LEXAPI_BASE_URL ? { LEXAPI_BASE_URL: env.LEXAPI_BASE_URL } : {}),
      },
    });
    const client = new Client({ name: "ai-act-provision-finder", version: "0.0.0" });
    await client.connect(transport);
    return new McpLegislationSearch(client);
  }

  async search(query: string): Promise<SearchOutcome> {
    const result = await this.client.callTool({
      name: "lex_semantic_legislation",
      arguments: { query, limit: 10, include: ["text"], maxChars: MAX_CHARS },
    });
    const blocks = Array.isArray(result.content) ? (result.content as Array<{ type: string; text?: string }>) : [];
    const text = blocks.find((block) => block.type === "text")?.text ?? "";
    if (result.isError) throw new Error(`lex_semantic_legislation failed: ${text}`);
    const parsed = JSON.parse(text) as { results?: LegislationHit[] };
    // The tool result does not include the credits charged for the call.
    return { hits: (parsed.results ?? []).map(stripTruncation), credits: undefined };
  }

  async close(): Promise<void> {
    await this.client.close();
  }
}
