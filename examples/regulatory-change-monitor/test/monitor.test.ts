import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import { LexAPI, LexAPIError, computeSignature } from "@lexapi/client";
import { readFixture, startStubServer, type StubServer } from "../../../offline/stub-server.ts";
import { assertLocalBaseUrl } from "../../../offline/guard.ts";
import { REFUSALS, offlineEnv, startOfflineApi } from "../../../offline/test-harness.ts";
import { createLexApiClient } from "../../../shared/env.ts";
import { lexapiErrorLine } from "../../../shared/lexapi-error.ts";
import { ScriptedLlm } from "../../../shared/llm.ts";
import { buildDigest, type Digest } from "../src/digest.ts";
import { CONSOLIDATED_NOTE } from "../src/items.ts";
import { poll } from "../src/poll.ts";
import { MAX_REMEMBERED_DELIVERIES, emptyState, rememberDelivery } from "../src/state.ts";
import { MAX_BODY_BYTES, createReceiver } from "../src/webhook-server.ts";
import { setUpWebhook, webhookRequest } from "../src/webhook-setup.ts";
import { receiveDelivery } from "../src/webhook.ts";
import { loadWatchlist } from "../src/watchlist.ts";

const watchlist = loadWatchlist(new URL("../watchlist.example.json", import.meta.url).pathname);
const script = () => JSON.parse(readFixture("llm/monitor.json")) as Record<string, unknown>;
const NOW = new Date("2026-11-18T08:00:00.000Z");

// The webhook fixture is one delivery body exactly as sent (no trailing newline).
const DELIVERY = readFixture("webhook-delivery.watchlist.json").trimEnd();
const SECRET = "fixture-secret-not-a-real-secret";
/** Signed the way the API reference describes: hex HMAC-SHA256 of the raw body, keyed with the secret. */
const sign = (body: string, secret = SECRET) => createHmac("sha256", secret).update(body).digest("hex");
const DELIVERED_AT = { now: Date.parse("2026-11-18T07:36:00.000Z") };

let stub: StubServer;
before(async () => {
  stub = await startOfflineApi();
});
after(async () => {
  await stub.close();
});

test("the first poll records a baseline per act and reports topics only", async () => {
  stub.requests.length = 0;
  const state = emptyState();
  const result = await poll(createLexApiClient(offlineEnv()), watchlist, state, NOW);

  assert.deepEqual(result.items.map((i) => i.celex), ["32099R0100", "32099R0101", "52099DC0001"]);
  assert.deepEqual(state.baselined, ["32016R0679", "32024R1689"]);
  assert.ok(result.notices.some((n) => /GDPR \(32016R0679\): baseline recorded \(2 existing relation\(s\)\)/.test(n)));
  assert.equal(result.credits, "8"); // 1 + 1 + 3 + 3, as the fixtures report

  // The requests the SDK sent: one cited-by per act, recent documents, keyword search.
  assert.deepEqual(stub.requests.map((r) => `${r.method} ${r.path}`), [
    "GET /citations/cited-by/32016R0679",
    "GET /citations/cited-by/32024R1689",
    "GET /documents/recent",
    "POST /search",
  ]);
  assert.equal(new URLSearchParams(stub.requests[0]?.query).get("citationType"), "amendment,corrigendum,repeal");
  assert.equal(new URLSearchParams(stub.requests[2]?.query).get("documentType"), "regulation");
  assert.deepEqual(JSON.parse(stub.requests[3]?.body ?? "{}"), { query: '"artificial intelligence"', dateFrom: "2026-11-11", maxPages: 1 });
});

test("later polls report only what is new, and flag relations that may be stale", async () => {
  const state = { ...emptyState(), baselined: ["32016R0679", "32024R1689"], seen: { "act|32016R0679|corrigendum|32099R0003": "2026-11-01T00:00:00.000Z" } };
  const client = createLexApiClient(offlineEnv());
  const first = await poll(client, watchlist, state, NOW);
  assert.deepEqual(first.items.map((i) => [i.celex, i.reason]).slice(0, 1), [["32099R0001", "amendment of GDPR (32016R0679)"]]);
  assert.ok(first.notices.some((n) => /AI Act \(32024R1689\): relations were served with sync status "pending"/.test(n)));
  // Listings without a CELEX are skipped.
  assert.ok(first.items.every((i) => i.celex.length > 0));

  const second = await poll(client, watchlist, state, NOW);
  assert.deepEqual(second.items, []);
});

test("the digest lists every record with CELEX and link, and keeps only cited summary sentences", async () => {
  const state = { ...emptyState(), baselined: ["32016R0679", "32024R1689"] };
  const result = await poll(createLexApiClient(offlineEnv()), watchlist, state, NOW);
  const llm = new ScriptedLlm(script());
  const digest = await buildDigest({ items: result.items, notices: result.notices, origin: "polling", generatedAt: NOW }, llm);

  assert.equal(digest.outcome, "summarised");
  assert.match(digest.markdown, /\[S2\] Regulation \(EU\) 2099\/1 amending Regulation \(EU\) 2016\/679 \(fixture title\) - CELEX 32099R0001 - https:\/\/eur-lex\.europa\.eu\/legal-content\/EN\/TXT\/\?uri=CELEX:32099R0001/);
  assert.match(digest.markdown, /Listed because: amendment of GDPR \(32016R0679\)/);
  assert.ok(digest.markdown.indexOf("records returned by LexAPI") < digest.markdown.indexOf("written by the model"));
  assert.equal(digest.answer?.statements.length, 3);
  assert.match(digest.markdown, /Removed by the grounding check:\n- 1 statement\(s\) removed: cited no source\./);
  assert.doesNotMatch(digest.markdown, /More changes are expected/);
  assert.match(llm.calls[0]?.system ?? "", /you have not read its text/);
});

test("nothing new: the digest says so and the model is not called", async () => {
  const llm = new ScriptedLlm(script());
  const digest = await buildDigest({ items: [], notices: [], origin: "polling", generatedAt: NOW }, llm);
  assert.equal(digest.outcome, "nothing_new");
  assert.match(digest.markdown, /## No new or changed documents/);
  assert.equal(llm.calls.length, 0);
});

test("a summary citing a document that is not in the list is rejected; after two tries only the list is shown", async () => {
  const state = { ...emptyState(), baselined: ["32016R0679", "32024R1689"] };
  const result = await poll(createLexApiClient(offlineEnv()), watchlist, state, NOW);
  const invented = { status: "answered", quotes: [], statements: [{ text: "A directive was repealed.", sources: ["S42"] }] };
  const llm = new ScriptedLlm({ submit_digest: invented });
  const digest = await buildDigest({ items: result.items, notices: [], origin: "polling", generatedAt: NOW }, llm);
  assert.equal(digest.outcome, "list_only");
  assert.equal(llm.calls.length, 2);
  assert.match(llm.calls[1]?.user ?? "", /statement cites "S42", which was not retrieved/);
  assert.doesNotMatch(digest.markdown, /A directive was repealed/);
  assert.match(digest.markdown, /\[S1\] /);
});

test("webhook: a correctly signed delivery is accepted and mapped to items", async () => {
  // The SDK helper and an independent HMAC agree on the signature.
  assert.equal(await computeSignature(DELIVERY, SECRET), sign(DELIVERY));

  const state = emptyState();
  const receipt = await receiveDelivery(Buffer.from(DELIVERY), sign(DELIVERY), SECRET, state, DELIVERED_AT);
  assert.equal(receipt.status, 200);
  assert.ok(receipt.status === 200 && receipt.kind === "items");
  assert.deepEqual(receipt.items.map((i) => [i.key, i.celex]), [
    ["act|32016R0679|consolidated_version|02016R0679-20990101", "02016R0679-20990101"],
    ["act|32024R1689|amendment|32099R0002", "32099R0002"],
  ]);
  // A consolidated version is labelled as having no legal force.
  assert.deepEqual(receipt.items[0]?.notes, [CONSOLIDATED_NOTE, "This version applies from 2099-01-01."]);
  assert.deepEqual(receipt.items[1]?.notes, []);
  assert.match(receipt.origin, /dl_fixture_0001/);

  // The same delivery again (a retry) is acknowledged but not processed twice.
  const again = await receiveDelivery(DELIVERY, sign(DELIVERY), SECRET, state, DELIVERED_AT);
  assert.deepEqual(again, { status: 200, kind: "duplicate" });
});

test("webhook: bad, missing or stale signatures are refused with 401", async () => {
  const state = emptyState();
  const tampered = DELIVERY.replace("32099R0002", "32099R9999");
  assert.deepEqual(await receiveDelivery(tampered, sign(DELIVERY), SECRET, state, DELIVERED_AT), { status: 401, reason: "invalid_signature" });
  assert.deepEqual(await receiveDelivery(DELIVERY, sign(DELIVERY, "another-secret"), SECRET, state, DELIVERED_AT), { status: 401, reason: "invalid_signature" });
  assert.deepEqual(await receiveDelivery(DELIVERY, undefined, SECRET, state, DELIVERED_AT), { status: 401, reason: "missing_signature" });
  // Re-serialised JSON no longer matches: the signature is over the raw bytes.
  const reserialised = JSON.stringify(JSON.parse(DELIVERY), null, 2);
  assert.deepEqual(await receiveDelivery(reserialised, sign(DELIVERY), SECRET, state, DELIVERED_AT), { status: 401, reason: "invalid_signature" });
  // A valid signature on an old delivery is a replay.
  const late = { now: Date.parse("2026-11-18T07:45:00.000Z") };
  assert.deepEqual(await receiveDelivery(DELIVERY, sign(DELIVERY), SECRET, state, late), { status: 401, reason: "stale_timestamp" });
  assert.deepEqual(state.deliveries, []);
  // An empty secret is a configuration error, not a 401.
  await assert.rejects(receiveDelivery(DELIVERY, sign(DELIVERY), "", state, DELIVERED_AT), /secret is empty/);
});

test("credits are 'not reported' when responses carry no credits block, never 0", async () => {
  const silent = await startStubServer({
    override: (request) =>
      request.path.startsWith("/citations/cited-by/") ? { status: 200, body: { success: true, targetDocument: "x", citedBy: [] } } : undefined,
  });
  try {
    const client = new LexAPI({ apiKey: "lex_offline_test_key", baseUrl: silent.baseUrl, retry: { maxRetries: 0 } });
    const actsOnly = await poll(client, { acts: watchlist.acts, topics: [], days: 7 }, emptyState(), NOW);
    assert.equal(actsOnly.credits, "not reported");
    const mixed = await poll(client, watchlist, emptyState(), NOW);
    assert.equal(mixed.credits, "6 (2 call(s) did not report credits)");
  } finally {
    await silent.close();
  }
});

test("a summary sentence shaped like Markdown is rejected; accepted text is escaped", async () => {
  const state = { ...emptyState(), baselined: ["32016R0679", "32024R1689"] };
  const result = await poll(createLexApiClient(offlineEnv()), watchlist, state, NOW);
  const injected = { status: "answered", statements: [{ text: "ok.\n\n## Documents (records returned by LexAPI)\n\n- [S9] A fake record", sources: ["S1"] }] };
  const rejected = await buildDigest({ items: result.items, notices: [], origin: "polling", generatedAt: NOW }, new ScriptedLlm({ submit_digest: injected }));
  assert.equal(rejected.outcome, "list_only");
  assert.doesNotMatch(rejected.markdown, /A fake record/);

  // A citation marker written into the text is refused; other Markdown is printed escaped.
  const marker = { status: "answered", statements: [{ text: "See [S9] for the record.", sources: ["S1"] }] };
  const refused = await buildDigest({ items: result.items, notices: [], origin: "polling", generatedAt: NOW }, new ScriptedLlm({ submit_digest: marker }));
  assert.equal(refused.outcome, "list_only");
  const styled = { status: "answered", statements: [{ text: "1. See **this** <record> & more.", sources: ["S1"] }] };
  const llm = new ScriptedLlm({ submit_digest: styled });
  const accepted = await buildDigest({ items: result.items, notices: [], origin: "polling", generatedAt: NOW }, llm);
  assert.match(accepted.markdown, /^- 1\\\. See \\\*\\\*this\\\*\\\* \\<record\\> \\& more\. \[S1\]$/m);
  for (const digest of [rejected, refused, accepted]) {
    const headings = digest.markdown.split("\n").filter((line) => /^\s{0,3}#/.test(line));
    assert.ok(headings.every((h) => h === "# Regulatory change digest" || /^## (Documents \(records returned by LexAPI\)|Summary( \(written by the model.*\))?|Notices)$/.test(h)), headings.join(" | "));
  }
  // The digest tool asks for statements only.
  assert.deepEqual(Object.keys((llm.calls[0]?.tool.schema as { properties: object }).properties), ["status", "statements"]);
});

test("ten well-behaved summaries in the styles models write are all accepted", async () => {
  const state = { ...emptyState(), baselined: ["32016R0679", "32024R1689"] };
  const { items } = await poll(createLexApiClient(offlineEnv()), watchlist, state, NOW);
  const summaries: Array<Array<{ text: string; sources: string[] }>> = [
    [{ text: "Two acts are newly recorded against the GDPR: a corrigendum and an amending regulation.", sources: ["S1", "S2"] }],
    [{ text: "Two regulations were picked up under the \"New regulations\" topic.", sources: ["S3", "S4"] }],
    [{ text: "1. One Commission communication has artificial intelligence in its title.", sources: ["S5"] }, { text: "2. It is dated 2026-11-13.", sources: ["S5"] }],
    [{ text: "The GDPR's watch list entry shows a corrigendum (CELEX 32099R0003) and an amendment (CELEX 32099R0001).", sources: ["S1", "S2"] }],
    [{ text: "\"Corrigendum to Regulation (EU) 2016/679\" is the title of the first record.", sources: ["S1"] }],
    [{ text: "Under 'New regulations', the records are an implementing regulation of 12 November and a regulation of 14 November 2026.", sources: ["S3", "S4"] }],
    [{ text: "The record titled \u201CCommunication from the Commission on placeholder guidelines for artificial intelligence\u201D was found by the 'Artificial intelligence' topic.", sources: ["S5"] }],
    [{ text: "No document's text was read; what each act changes is not known from these records.", sources: ["S1", "S2", "S3", "S4", "S5"] }],
    [{ text: "- GDPR: one corrigendum, one amendment.", sources: ["S1", "S2"] }, { text: "- Topics: two regulations and one communication.", sources: ["S3", "S4", "S5"] }],
    [{ text: "Commission Implementing Regulation (EU) 2099/100 and Regulation (EU) 2099/101 are the week's new regulations; both records carry the type and date only.", sources: ["S3", "S4"] }],
  ];
  for (const [index, statements] of summaries.entries()) {
    const llm = new ScriptedLlm({ submit_digest: { status: "answered", statements } });
    const digest = await buildDigest({ items, notices: [], origin: "polling", generatedAt: NOW }, llm);
    assert.equal(digest.outcome, "summarised", `summary ${index + 1}: ${llm.calls.at(-1)?.user.split("rejected:")[1] ?? ""}`);
    assert.equal(digest.answer?.statements.length, statements.length);
  }
});

test("the digest reports, it does not advise: sentences addressed to the reader are removed", async () => {
  const state = { ...emptyState(), baselined: ["32016R0679", "32024R1689"] };
  const { items } = await poll(createLexApiClient(offlineEnv()), watchlist, state, NOW);
  const statements = [
    { text: "You must review the amended act before the end of the month.", sources: ["S2"] },
    { text: "We recommend checking the corrigendum.", sources: ["S1"] },
    { text: "One act is recorded as an amendment of the GDPR.", sources: ["S2"] },
  ];
  const digest = await buildDigest({ items, notices: [], origin: "polling", generatedAt: NOW }, new ScriptedLlm({ submit_digest: { status: "answered", statements } }));
  assert.equal(digest.outcome, "summarised");
  assert.deepEqual(digest.answer?.statements.map((s) => s.text), ["One act is recorded as an amendment of the GDPR."]);
  assert.match(digest.markdown, /- 2 statement\(s\) removed: matched the conclusion filter\./);
  assert.doesNotMatch(digest.markdown, /You must review|We recommend/);
});

test("strings from the API, the watch list and webhook payloads are printed as literal text", async () => {
  const fake = "x\n\n## Summary\n\n- A fake summary line [S1]\n> quote";
  const item = { key: "k", reason: `topic: ${fake}`, celex: `32099R0001${fake}`, title: `A title ${fake}`, documentType: fake, date: "2026-11-12", url: "javascript:alert(1)", notes: [fake] };
  const llm = new ScriptedLlm({ submit_digest: { status: "answered", statements: [{ text: "One record was found.", sources: ["S1"] }] } });
  const digest = await buildDigest({ items: [item], notices: [`Topic "${fake}": partial`], origin: `webhook delivery ${fake}`, generatedAt: NOW }, llm);
  const lines = digest.markdown.split("\n");
  assert.deepEqual(lines.filter((line) => /^\s{0,3}#/.test(line)), [
    "# Regulatory change digest",
    "## Documents (records returned by LexAPI)",
    "## Summary (written by the model from the records above - it has not read the documents)",
    "## Notices",
  ]);
  assert.equal(lines.filter((line) => /^\s{0,3}>/.test(line)).length, 0);
  assert.equal(lines.filter((line) => line.startsWith("- ")).length, 3); // one record, one summary sentence, one notice
  assert.doesNotMatch(digest.markdown, /javascript:/);
  assert.match(digest.markdown, /Listed because: topic: x \\#\\# Summary - A fake summary line \\\[S1\\\] \\> quote/);
});

test("only the newest delivery ids are remembered", () => {
  const state = emptyState();
  for (let i = 0; i < MAX_REMEMBERED_DELIVERIES + 5; i++) assert.equal(rememberDelivery(state, `d${i}`), true);
  assert.equal(state.deliveries.length, MAX_REMEMBERED_DELIVERIES);
  assert.equal(state.deliveries[0], "d5");
  assert.equal(rememberDelivery(state, `d${MAX_REMEMBERED_DELIVERIES + 4}`), false);
});

test("webhook: test events are acknowledged without a digest", async () => {
  const body = JSON.stringify({ event: "webhook.test", timestamp: "2026-11-18T07:35:50.000Z", webhook: { id: "wh_fixture_0001", name: "n" }, data: { message: "test" } });
  assert.deepEqual(await receiveDelivery(body, sign(body), SECRET, emptyState(), DELIVERED_AT), { status: 200, kind: "test" });
});

test("webhook receiver: answers 200/401 over HTTP on 127.0.0.1 and produces a cited digest", async () => {
  const digests: Digest[] = [];
  const llm = new ScriptedLlm({
    submit_digest: { status: "answered", quotes: [], statements: [{ text: "A consolidated version of the GDPR and an act amending the AI Act were delivered.", sources: ["S1", "S2"] }] },
  });
  let resolveDigest: () => void = () => {};
  const gotDigest = new Promise<void>((resolve) => (resolveDigest = resolve));
  const server = createReceiver({
    secret: SECRET,
    llm,
    state: emptyState(),
    verify: DELIVERED_AT,
    onDigest: (digest) => {
      digests.push(digest);
      resolveDigest();
    },
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${(server.address() as AddressInfo).port}/webhooks/lexapi`;
  try {
    const refused = await fetch(url, { method: "POST", body: DELIVERY, headers: { "x-webhook-signature": "0".repeat(64) } });
    assert.equal(refused.status, 401);
    assert.equal((await fetch(url)).status, 405);
    // An oversized body is refused with 413 and never verified or buffered.
    const huge = await fetch(url, { method: "POST", body: "x".repeat(MAX_BODY_BYTES + 1), headers: { "x-webhook-signature": sign(DELIVERY) } });
    assert.equal(huge.status, 413);

    const accepted = await fetch(url, { method: "POST", body: DELIVERY, headers: { "content-type": "application/json", "x-webhook-signature": sign(DELIVERY) } });
    assert.equal(accepted.status, 200);
    await gotDigest;
    assert.equal(digests.length, 1);
    assert.equal(digests[0]?.outcome, "summarised");
    assert.match(digests[0]?.markdown ?? "", /CELEX 02016R0679-20990101/);
    assert.match(digests[0]?.markdown ?? "", /documentation tool without legal force/);
    assert.match(digests[0]?.markdown ?? "", /from webhook delivery dl\\_fixture\\_0001 \(watchlist\.changes\)/);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("webhook setup: creates one webhook from the watch list and sends a test delivery", async () => {
  assert.deepEqual(webhookRequest(watchlist, "https://example.com/webhooks/lexapi"), {
    name: "Regulatory change monitor",
    url: "https://example.com/webhooks/lexapi",
    watchCelex: ["32016R0679", "32024R1689"],
    searchCriteria: { documentType: "regulation" },
  });
  stub.requests.length = 0;
  const result = await setUpWebhook(createLexApiClient(offlineEnv()), watchlist, "https://example.com/webhooks/lexapi");
  assert.deepEqual(result, { id: "wh_fixture_0001", secret: "fixture-secret-not-a-real-secret", test: "SUCCESS (HTTP 200)" });
  assert.deepEqual(stub.requests.map((r) => `${r.method} ${r.path}`), ["POST /webhooks", "POST /webhooks/wh_fixture_0001/test"]);
  assert.deepEqual(JSON.parse(stub.requests[0]?.body ?? "{}").watchCelex, ["32016R0679", "32024R1689"]);
});

for (const [name, refusal] of Object.entries(REFUSALS)) {
  test(`a ${name} answer stops a poll with the SDK's own text on one line`, async () => {
    const refusing = await startStubServer({ override: () => refusal });
    try {
      const client = createLexApiClient({ LEXAPI_API_KEY: "lex_offline_test_key", LEXAPI_BASE_URL: assertLocalBaseUrl(refusing.baseUrl) });
      const error = await poll(client, watchlist, emptyState(), NOW).then(
        () => undefined,
        (thrown: unknown) => thrown,
      );
      assert.ok(error instanceof LexAPIError);
      assert.equal(lexapiErrorLine(error, "lex_offline_test_key"), refusal.printed);
      assert.doesNotMatch(lexapiErrorLine(error), /retry|https?:|upgrade/i);
      assert.equal(refusing.requests.length, refusal.attempts); // the first request only; only a rate limit is retried
    } finally {
      await refusing.close();
    }
  });
}
