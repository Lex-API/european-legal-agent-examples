/**
 * A local stand-in for the LexAPI REST API, for tests and `--dry-run`.
 *
 * It listens on 127.0.0.1 only and replays the files in `offline/fixtures`
 * (see `routes.json`). The fixtures are synthetic: they follow the response
 * shapes the SDKs type, but were not recorded from the live API.
 */
import { readFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), "fixtures");

export function readFixture(name: string): string {
  return readFileSync(join(FIXTURES_DIR, name), "utf8");
}

interface Route {
  method: string;
  path: string;
  /** Substring that must appear in the request body (POST) or query string (GET). */
  includes?: string;
  status?: number;
  fixture: string;
}

export interface RecordedRequest {
  method: string;
  path: string;
  query: string;
  headers: Record<string, string | string[] | undefined>;
  body: string;
}

export interface StubReply {
  status: number;
  body: unknown;
  /** Response headers that replace or add to the defaults below (for example `retry-after`). */
  headers?: Record<string, string>;
}

export interface StubServer {
  /** `http://127.0.0.1:<port>` - pass as `LEXAPI_BASE_URL`. */
  baseUrl: string;
  requests: RecordedRequest[];
  close(): Promise<void>;
}

export interface StubOptions {
  /** Checked before `routes.json`; return `undefined` to fall through, or `"hang"` to never answer. */
  override?: (request: RecordedRequest) => StubReply | "hang" | undefined;
}

export async function startStubServer(options: StubOptions = {}): Promise<StubServer> {
  const routes = JSON.parse(readFixture("routes.json")) as Route[];
  const requests: RecordedRequest[] = [];

  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      const recorded: RecordedRequest = {
        method: req.method ?? "GET",
        path: decodeURIComponent(url.pathname),
        query: url.search,
        headers: req.headers,
        body: Buffer.concat(chunks).toString("utf8"),
      };
      requests.push(recorded);

      const custom = options.override?.(recorded);
      if (custom === "hang") return; // no answer; the connection is closed with the server
      const reply = custom ?? replyFromRoutes(routes, recorded);
      res.writeHead(reply.status, {
        "content-type": "application/json",
        // The real API sends these on every /api/v1 response; the SDK reads them.
        "x-lexapi-version": "stub",
        "ratelimit-limit": "15",
        "ratelimit-remaining": "14",
        "ratelimit-reset": "60",
        ...reply.headers,
      });
      res.end(typeof reply.body === "string" ? reply.body : JSON.stringify(reply.body));
    });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
        server.closeAllConnections();
      }),
  };
}

function replyFromRoutes(routes: Route[], request: RecordedRequest): StubReply {
  const haystack = request.method === "GET" ? decodeURIComponent(request.query) : request.body;
  const route = routes.find(
    (r) => r.method === request.method && r.path === request.path && (!r.includes || haystack.includes(r.includes)),
  );
  if (!route) {
    return {
      status: 404,
      body: { success: false, error: { code: "NOT_FOUND", message: `The stub has no fixture for ${request.method} ${request.path}` } },
    };
  }
  return { status: route.status ?? 200, body: readFixture(route.fixture) };
}
