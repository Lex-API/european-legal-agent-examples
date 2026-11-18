/**
 * Webhook mode: LexAPI pushes changes to you (paid plans; needs a public HTTPS URL).
 *
 * `receiveDelivery` is the part that matters: verify the signature over the
 * RAW request body with the SDK's `verifyWebhook` (HMAC-SHA256 with the
 * webhook secret, constant-time comparison, 5-minute replay window), drop
 * deliveries already handled (`delivery.id` is stable across retries), and
 * map the payload to the same `Item`s polling produces.
 */
import { verifyWebhook, WebhookSignatureError, type VerifyWebhookOptions, type WebhookChangeDocument, type WebhookPayload } from "@lexapi/client";
import { eurLexUrl } from "../../../shared/grounding.ts";
import { CONSOLIDATED_NOTE, type Item } from "./items.ts";
import { rememberDelivery, type MonitorState } from "./state.ts";

/** Webhook relation names -> the citation types polling uses. */
const POLL_RELATION: Record<string, string> = { amended_by: "amendment", corrected_by: "corrigendum", repealed_by: "repeal" };

export type Receipt =
  | { status: 401; reason: WebhookSignatureError["reason"] }
  | { status: 200; kind: "duplicate" | "test" }
  | { status: 200; kind: "items"; items: Item[]; origin: string; notices: string[] };

export async function receiveDelivery(
  rawBody: Uint8Array | string,
  signature: string | undefined,
  secret: string,
  state: MonitorState,
  options: VerifyWebhookOptions = {},
): Promise<Receipt> {
  let event: WebhookPayload;
  try {
    event = await verifyWebhook(rawBody, signature, secret, options);
  } catch (error) {
    // An empty secret is a configuration error, not a bad delivery: let it surface.
    if (error instanceof WebhookSignatureError && error.reason !== "missing_secret") return { status: 401, reason: error.reason };
    throw error;
  }

  if (event.event === "webhook.test") return { status: 200, kind: "test" };

  const deliveryId = event.delivery?.id;
  if (deliveryId && !rememberDelivery(state, deliveryId)) return { status: 200, kind: "duplicate" };

  const notices: string[] = [];
  let items: Item[];
  if (event.event === "watchlist.changes") {
    items = event.data.changes.flatMap((change) =>
      toItem(
        change.document,
        `${change.relation.replaceAll("_", " ")} - watched act ${change.watchedCelex}`,
        // Same keys as polling, so a change seen in both modes is reported once.
        `act|${change.watchedCelex}|${POLL_RELATION[change.relation] ?? change.relation}`,
        change.relation === "consolidated_version",
      ),
    );
  } else {
    items = event.data.documents.flatMap((doc) => toItem(doc, "new document matching the webhook's search criteria", "new"));
    const total = event.data.totalAvailable ?? event.data.count;
    if (total > event.data.count) notices.push(`This delivery carries ${event.data.count} of ${total} new documents; the rest arrive in further deliveries.`);
  }
  return { status: 200, kind: "items", items, origin: `webhook delivery ${deliveryId ?? "(no id)"} (${event.event})`, notices };
}

function toItem(doc: WebhookChangeDocument, reason: string, keyPrefix: string, consolidated = false): Item[] {
  if (!doc.celex) return [];
  const notes: string[] = [];
  if (consolidated) {
    notes.push(CONSOLIDATED_NOTE);
    if (doc.consolidationDate) notes.push(`This version applies from ${doc.consolidationDate}.`);
  }
  return [
    {
      key: `${keyPrefix}|${doc.celex}`,
      reason,
      celex: doc.celex,
      title: doc.title ?? null,
      documentType: doc.documentType,
      date: doc.dateOfDocumentISO,
      url: doc.url ?? eurLexUrl(doc.celex),
      notes,
    },
  ];
}
