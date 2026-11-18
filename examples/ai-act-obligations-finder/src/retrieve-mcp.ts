/**
 * The same retrieval step through the LexAPI MCP server instead of the SDK.
 *
 * This starts `@lexapi/mcp` as a child process over stdio and calls its
 * `lex_semantic_legislation` tool. The tool answers with one text block
 * holding JSON; its hits use the same field names as the REST API.
 */
import { createRequire } from "node:module";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { LegislationHit, LegislationSearch, SearchOutcome } from "./retrieve.ts";

export class McpLegislationSearch implements LegislationSearch {
  readonly label = "@lexapi/mcp (lex_semantic_legislation)";
  private constructor(private readonly client: Client) {}

  static async start(env: NodeJS.ProcessEnv = process.env): Promise<McpLegislationSearch> {
    if (!env.LEXAPI_API_KEY) throw new Error("LEXAPI_API_KEY is not set.");
    const serverEntry = createRequire(import.meta.url).resolve("@lexapi/mcp");
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [serverEntry],
      env: {
        LEXAPI_API_KEY: env.LEXAPI_API_KEY,
        // Only the read-only tools are needed here.
        LEXAPI_MCP_READ_ONLY: "true",
        ...(env.LEXAPI_BASE_URL ? { LEXAPI_BASE_URL: env.LEXAPI_BASE_URL } : {}),
      },
    });
    const client = new Client({ name: "ai-act-obligations-finder", version: "0.0.0" });
    await client.connect(transport);
    return new McpLegislationSearch(client);
  }

  async search(query: string): Promise<SearchOutcome> {
    const result = await this.client.callTool({
      name: "lex_semantic_legislation",
      arguments: { query, limit: 10, include: ["text"] },
    });
    const blocks = Array.isArray(result.content) ? (result.content as Array<{ type: string; text?: string }>) : [];
    const text = blocks.find((block) => block.type === "text")?.text ?? "";
    if (result.isError) throw new Error(`lex_semantic_legislation failed: ${text}`);
    const parsed = JSON.parse(text) as { results?: LegislationHit[] };
    // The tool result does not include the credits charged for the call.
    return { hits: parsed.results ?? [], credits: undefined };
  }

  async close(): Promise<void> {
    await this.client.close();
  }
}
