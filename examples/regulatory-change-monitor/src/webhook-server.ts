/**
 * Webhook mode receiver: a plain node:http server.
 *
 *   LEXAPI_WEBHOOK_SECRET=... npm run monitor:webhook-server
 *
 * Put it behind a public HTTPS endpoint (a reverse proxy or a tunnel) and
 * register that URL with `npm run monitor:webhook-setup`. The handler
 * answers within LexAPI's 10-second delivery timeout and writes the digest
 * afterwards.
 */
import { createServer, type Server } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { WEBHOOK_SIGNATURE_HEADER, type VerifyWebhookOptions } from "@lexapi/client";
import { loadEnvFile } from "../../../shared/env.ts";
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

const MAX_BODY_BYTES = 5 * 1024 * 1024;

export function createReceiver(options: ReceiverOptions): Server {
  return createServer((req, res) => {
    if (req.method !== "POST") {
      res.writeHead(405).end();
      return;
    }
    // Keep the RAW bytes: the signature is over the exact body sent.
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) req.destroy();
      else chunks.push(chunk);
    });
    req.on("end", () => {
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
  const here = join(dirname(fileURLToPath(import.meta.url)), "..");
  loadEnvFile(join(here, ".env"));
  const secret = process.env.LEXAPI_WEBHOOK_SECRET;
  if (!secret) {
    console.error("Set LEXAPI_WEBHOOK_SECRET to the secret printed by `npm run monitor:webhook-setup`.");
    process.exit(2);
  }
  const statePath = join(here, ".monitor-state.json");
  const port = Number.parseInt(process.env.PORT ?? "8787", 10);
  const server = createReceiver({
    secret,
    llm: await createLlmFromEnv(),
    state: loadState(statePath),
    onDigest: (digest) => console.log(digest.markdown),
    onState: (state) => saveState(statePath, state),
  });
  server.listen(port, () => console.error(`Listening for LexAPI webhook deliveries on port ${port}.`));
}
