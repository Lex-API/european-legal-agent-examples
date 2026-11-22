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
import { ConfigError, LEXAPI_KEY_MISSING, envKey } from "../../../shared/cli.ts";
import { PrintedError, UnreadableAnswerError, clean, maskSecrets } from "../../../shared/lexapi-error.ts";
import { isHitList, type LegislationHit, type LegislationSearch, type SearchOutcome } from "./retrieve.ts";

/** The largest text budget the tool accepts. */
const MAX_CHARS = 200_000;
/** What `@lexapi/mcp` 0.6.0 appends to a hit text it cut. */
const TRUNCATION_MARKER = / \u2026\[truncated: (\d+) more characters\]$/;

/** The MCP tool reported an error: its own text is printed, on one line (the key masked; tool names are kept). */
export class McpToolError extends PrintedError {
  constructor(text: string) {
    super(maskSecrets(clean(text, 2000) ?? "The MCP tool reported an error without a message."));
    this.name = "McpToolError";
  }
}

/** The MCP server process failed (it closed the connection, or did not answer in time). */
export class McpServerError extends PrintedError {
  constructor(message: string) {
    super(`MCP server error: ${maskSecrets(clean(message, 1000) ?? "no message")}`);
    this.name = "McpServerError";
  }
}

export function stripTruncation(hit: LegislationHit): LegislationHit {
  const match = typeof hit.text === "string" ? TRUNCATION_MARKER.exec(hit.text) : null;
  if (!match || typeof hit.text !== "string") return hit;
  return { ...hit, text: hit.text.slice(0, match.index), truncatedBy: Number(match[1]) };
}

export class McpLegislationSearch implements LegislationSearch {
  readonly label = "@lexapi/mcp (lex_semantic_legislation)";
  private constructor(
    private readonly client: Client,
    private readonly transport: StdioClientTransport,
    private readonly requestTimeoutMs: number | undefined,
  ) {}

  /** The MCP server's process id while it runs. */
  get pid(): number | null {
    return this.transport.pid;
  }

  /** `requestTimeoutMs`: how long to wait for each tool call (the MCP SDK's default when not given). */
  static async start(env: NodeJS.ProcessEnv = process.env, options: { requestTimeoutMs?: number } = {}): Promise<McpLegislationSearch> {
    const apiKey = envKey(env, "LEXAPI_API_KEY");
    if (!apiKey) throw new ConfigError(LEXAPI_KEY_MISSING);
    // The child process is outside the test harness's fetch guard, so under `node --test`
    // refuse to start it unless it will talk to the local stub.
    if (process.env.NODE_TEST_CONTEXT) assertLocalBaseUrl(env.LEXAPI_BASE_URL);
    const serverEntry = createRequire(import.meta.url).resolve("@lexapi/mcp");
    const transport = new StdioClientTransport({
      command: process.execPath,
      args: [serverEntry],
      // The server's own log lines (its start banner) would mix with this program's output.
      stderr: "ignore",
      env: {
        LEXAPI_API_KEY: apiKey,
        // Only the read-only tools are needed here.
        LEXAPI_MCP_READ_ONLY: "true",
        ...(env.LEXAPI_BASE_URL ? { LEXAPI_BASE_URL: env.LEXAPI_BASE_URL } : {}),
      },
    });
    const client = new Client({ name: "ai-act-provision-finder", version: "0.0.0" });
    try {
      await client.connect(transport);
    } catch (error) {
      await transport.close().catch(() => {});
      throw new McpServerError(error instanceof Error ? error.message : String(error));
    }
    return new McpLegislationSearch(client, transport, options.requestTimeoutMs);
  }

  async search(query: string): Promise<SearchOutcome> {
    let result: Awaited<ReturnType<Client["callTool"]>>;
    try {
      result = await this.client.callTool(
        { name: "lex_semantic_legislation", arguments: { query, limit: 10, include: ["text"], maxChars: MAX_CHARS } },
        undefined,
        this.requestTimeoutMs === undefined ? undefined : { timeout: this.requestTimeoutMs },
      );
    } catch (error) {
      // The server process closed the connection (-32000), did not answer in time (-32001), or was
      // already gone ("Not connected"): every failure of the MCP client or its transport.
      throw new McpServerError(error instanceof Error ? error.message : String(error));
    }
    const blocks = Array.isArray(result.content) ? (result.content as Array<{ type: string; text?: string }>) : [];
    const text = blocks.find((block) => block.type === "text")?.text ?? "";
    if (result.isError) throw new McpToolError(text);
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = undefined;
    }
    const hits = parsed && typeof parsed === "object" ? (parsed as { results?: unknown }).results : undefined;
    // An answer without its list of hits is not "no hits".
    if (!isHitList(hits)) throw new UnreadableAnswerError();
    // The tool result does not include the credits charged for the call.
    return { hits: hits.map(stripTruncation), credits: undefined };
  }

  /** Stop the MCP server process. */
  async close(): Promise<void> {
    try {
      await this.client.close();
    } catch (error) {
      throw new McpServerError(error instanceof Error ? error.message : String(error));
    }
  }
}
