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
import { ConfigError, missingSettings, runCli } from "../../../shared/cli.ts";
import { createLexApiClient, loadEnvFile } from "../../../shared/env.ts";
import { PrintedError, UnreadableAnswerError, clean, mask, maskSecrets } from "../../../shared/lexapi-error.ts";
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

export interface CreatedWebhook {
  id: string;
  secret: string | undefined;
}

/** Said on stderr when the webhook exists but its test delivery could not be made. */
export function createdButNotTested(id: string): string {
  return (
    `The webhook ${id} was created, but its test delivery failed. ` +
    `Re-test it with client.webhooks.test("${id}") (POST /webhooks/${id}/test), ` +
    `or remove it with client.webhooks.delete("${id}") (DELETE /webhooks/${id}) before running this again.`
  );
}

/** A test delivery whose result is not a success, with the delivery's own status and message. */
export function deliveryFailed(delivery: { status?: unknown; responseStatus?: unknown; errorMessage?: unknown } | undefined): string {
  const status = clean(delivery?.status, 200) ?? "unknown";
  const http = typeof delivery?.responseStatus === "number" ? ` (HTTP ${delivery.responseStatus})` : "";
  const message = clean(delivery?.errorMessage, 300);
  return mask(`Test delivery: ${status}${http}${message ? ` - ${message}` : ""}`);
}

/**
 * Create the webhook, report it through `onCreated` at once (its secret is shown only
 * in this answer), then send a test delivery. A failed test - the call refused, or a
 * delivery that did not succeed - leaves the webhook in place and is an error.
 */
export async function setUpWebhook(
  client: LexAPI,
  watchlist: Watchlist,
  url: string,
  onCreated: (webhook: CreatedWebhook) => void = () => {},
  onUnidentified: (answer: unknown) => void = () => {},
): Promise<CreatedWebhook & { test: string }> {
  const created = await client.webhooks.create(webhookRequest(watchlist, url));
  const id: unknown = created?.webhook?.id;
  if (typeof id !== "string" || !id) {
    onUnidentified(created);
    throw new PrintedError(
      "LexAPI error INVALID_RESPONSE: The answer to the create has no webhook id, so a webhook may have been created. " +
        "List your webhooks with client.webhooks.list() (GET /webhooks) before running this again.",
    );
  }
  const webhook = { id, secret: typeof created.webhook.secret === "string" ? created.webhook.secret : undefined };
  onCreated(webhook);
  let test: Awaited<ReturnType<LexAPI["webhooks"]["test"]>>;
  try {
    test = await client.webhooks.test(id);
    if (!test?.delivery || typeof test.delivery !== "object") throw new UnreadableAnswerError(200);
    // Success, as the SDK types it: `success: true` and a delivery with status SUCCESS.
    if (test.success !== true || test.delivery.status !== "SUCCESS") throw new PrintedError(deliveryFailed(test.delivery));
  } catch (error) {
    console.error(createdButNotTested(id));
    throw error;
  }
  const delivery = test.delivery;
  return {
    ...webhook,
    test: `${delivery.status}${delivery.responseStatus ? ` (HTTP ${delivery.responseStatus})` : ""}`,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await runCli(async () => {
    const here = join(dirname(fileURLToPath(import.meta.url)), "..");
    loadEnvFile(join(here, ".env"));
    const url = process.env.LEXAPI_WEBHOOK_URL?.trim();
    const missing = missingSettings(process.env, { lexapi: true, model: false });
    if (!url) missing.push("LEXAPI_WEBHOOK_URL is not set. Set it to the public HTTPS URL of your receiver.");
    if (missing.length > 0) throw new ConfigError(missing.join("\n"));
    const ownList = join(here, "watchlist.json");
    const watchlist = loadWatchlist(existsSync(ownList) ? ownList : join(here, "watchlist.example.json"));
    const result = await setUpWebhook(
      createLexApiClient(),
      watchlist,
      url as string,
      (webhook) => {
        console.log(`Webhook ${webhook.id} created.`);
        console.log(`\nSecret (shown once - store it as LEXAPI_WEBHOOK_SECRET, never commit it):\n${webhook.secret ?? "(not returned)"}\n`);
      },
      (answer) => {
        // Whatever came back, the secret included: it may belong to a webhook that now exists.
        console.log(`The answer to the create, as returned:\n${maskSecrets(JSON.stringify(answer, null, 2) ?? String(answer))}`);
      },
    );
    console.log(`Test delivery: ${result.test}`);
  });
}
