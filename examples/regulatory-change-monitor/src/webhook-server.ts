/**
 * Webhook mode receiver: a plain node:http server.
 *
 *   LEXAPI_WEBHOOK_SECRET=... npm run monitor:webhook-server
 *
 * It listens on 127.0.0.1 only (set HOST to change that). Expose it through
 * a reverse proxy or a tunnel that terminates HTTPS - do not bind it to a
 * public interface directly - and register the public URL with
 * `npm run monitor:webhook-setup`. The handler answers within LexAPI's
 * 10-second delivery timeout and writes the digest afterwards.
 */
import { createServer, type Server } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { WEBHOOK_SIGNATURE_HEADER, type VerifyWebhookOptions } from "@lexapi/client";
import { loadEnvFile } from "../../../shared/env.ts";
import { ConfigError, missingSettings, runCli } from "../../../shared/cli.ts";
import { createLlmFromEnv, type Llm } from "../../../shared/llm.ts";
import { buildDigest, type Digest } from "./digest.ts";
import { loadState, saveState, takeNew, type MonitorState } from "./state.ts";
import { receiveDelivery } from "./webhook.ts";

export interface ReceiverOptions {
  secret: string;
  llm: Llm;
  state: MonitorState;
  onDigest: (digest: Digest) => void;
  /** Called after the state changed, to persist it. */
  onState?: (state: MonitorState) => void;
  verify?: VerifyWebhookOptions;
}

export const MAX_BODY_BYTES = 5 * 1024 * 1024;

export function createReceiver(options: ReceiverOptions): Server {
  return createServer((req, res) => {
    if (req.method !== "POST") {
      res.writeHead(405).end();
      return;
    }
    // Keep the RAW bytes: the signature is over the exact body sent.
    const chunks: Buffer[] = [];
    let size = 0;
    // An oversized body is not buffered: it is read to its end and answered with 413.
    let oversized = Number(req.headers["content-length"] ?? 0) > MAX_BODY_BYTES;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (!oversized && size <= MAX_BODY_BYTES) {
        chunks.push(chunk);
        return;
      }
      oversized = true;
      chunks.length = 0;
      if (size > 4 * MAX_BODY_BYTES) req.destroy(); // far past the limit: stop reading altogether
    });
    req.on("end", () => {
      if (oversized) {
        res.writeHead(413).end();
        return;
      }
      const header = req.headers[WEBHOOK_SIGNATURE_HEADER];
      const signature = Array.isArray(header) ? header[0] : header;
      receiveDelivery(Buffer.concat(chunks), signature, options.secret, options.state, options.verify)
        .then((receipt) => {
          res.writeHead(receipt.status).end();
          if (receipt.status !== 200 || receipt.kind !== "items") return;
          const now = new Date();
          const items = takeNew(options.state, receipt.items, now);
          options.onState?.(options.state);
          // Respond first, summarise after: the model call can take longer than the delivery timeout.
          return buildDigest({ items, notices: receipt.notices, origin: receipt.origin, generatedAt: now }, options.llm).then(options.onDigest);
        })
        .catch((error: unknown) => {
          if (!res.headersSent) res.writeHead(500).end();
          console.error(error);
        });
    });
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await runCli(async () => {
    const here = join(dirname(fileURLToPath(import.meta.url)), "..");
    loadEnvFile(join(here, ".env"));
    const secret = process.env.LEXAPI_WEBHOOK_SECRET?.trim();
    const missing = missingSettings(process.env, { lexapi: false, model: true });
    if (!secret) missing.unshift("LEXAPI_WEBHOOK_SECRET is not set. Use the secret printed by `npm run monitor:webhook-setup`.");
    if (missing.length > 0) throw new ConfigError(missing.join("\n"));
    const statePath = join(here, ".monitor-state.json");
    const port = Number.parseInt(process.env.PORT ?? "8787", 10);
    const host = process.env.HOST || "127.0.0.1";
    const server = createReceiver({
      secret: secret as string,
      llm: await createLlmFromEnv(),
      state: loadState(statePath),
      onDigest: (digest) => console.log(digest.markdown),
      onState: (state) => saveState(statePath, state),
    });
    server.listen(port, host, () => console.error(`Listening for LexAPI webhook deliveries on ${host}:${port}.`));
  });
}
