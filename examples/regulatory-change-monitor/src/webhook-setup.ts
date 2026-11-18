/**
 * Create the webhook for webhook mode and send a test delivery.
 *
 *   LEXAPI_WEBHOOK_URL=https://your-host.example/webhooks/lexapi npm run monitor:webhook-setup
 *
 * Webhooks need a paid plan (the FREE plan allows none) and a public HTTPS
 * URL on port 443 or 8443. The secret is printed ONCE: store it as
 * LEXAPI_WEBHOOK_SECRET for `npm run monitor:webhook-server`.
 */
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { LexAPI, WebhookCreateRequest } from "@lexapi/client";
import { createLexApiClient, loadEnvFile } from "../../../shared/env.ts";
import { loadWatchlist, type Watchlist } from "./watchlist.ts";

/** One webhook: the watched acts as a watch list, plus the first topic as `documents.new` criteria. */
export function webhookRequest(watchlist: Watchlist, url: string): WebhookCreateRequest {
  const topic = watchlist.topics[0];
  return {
    name: "Regulatory change monitor",
    url,
    ...(watchlist.acts.length > 0 ? { watchCelex: watchlist.acts.map((a) => a.celex) } : {}),
    // Webhook criteria match document TITLES only (see the API reference).
    ...(topic ? { searchCriteria: { ...(topic.query ? { query: topic.query } : {}), ...(topic.documentType ? { documentType: topic.documentType } : {}) } } : {}),
  };
}

export async function setUpWebhook(client: LexAPI, watchlist: Watchlist, url: string): Promise<{ id: string; secret: string | undefined; test: string }> {
  const created = await client.webhooks.create(webhookRequest(watchlist, url));
  const test = await client.webhooks.test(created.webhook.id);
  return {
    id: created.webhook.id,
    secret: created.webhook.secret,
    test: `${test.delivery.status ?? "unknown"}${test.delivery.responseStatus ? ` (HTTP ${test.delivery.responseStatus})` : ""}${test.delivery.errorMessage ? ` - ${test.delivery.errorMessage}` : ""}`,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const here = join(dirname(fileURLToPath(import.meta.url)), "..");
  loadEnvFile(join(here, ".env"));
  const url = process.env.LEXAPI_WEBHOOK_URL;
  if (!url) {
    console.error("Set LEXAPI_WEBHOOK_URL to the public HTTPS URL of your receiver.");
    process.exit(2);
  }
  const ownList = join(here, "watchlist.json");
  const watchlist = loadWatchlist(existsSync(ownList) ? ownList : join(here, "watchlist.example.json"));
  const result = await setUpWebhook(createLexApiClient(), watchlist, url);
  console.log(`Webhook ${result.id} created.`);
  console.log(`Test delivery: ${result.test}`);
  console.log(`\nSecret (shown once - store it as LEXAPI_WEBHOOK_SECRET, never commit it):\n${result.secret ?? "(not returned)"}`);
}
