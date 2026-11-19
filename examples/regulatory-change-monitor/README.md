# Regulatory change monitor (TypeScript)

Watches a list of EU acts and topics for new or changed documents and writes a cited digest.
Polling is the default and works on every plan; webhook mode is included for paid plans.

## What it does not do

- It does not read the documents it lists. The digest is built from the records LexAPI returns
  (title, type, date, relation) and says so; it does not describe what an act changes or requires.
- It is not legal advice and does not assess what a change means for you.
- Polling reads one result page per topic and the newest 100 relations per act. When more exist,
  the digest says so in its notices.

## Configure

Copy `watchlist.example.json` to `watchlist.json` (git-ignored) and edit it:

- `acts` - base-act CELEX numbers to watch, with a label.
- `topics` - each has a `name` and a `query` (keyword search) and/or one `documentType`.
- `days` - look-back window for topics.

## Polling mode (default)

`src/poll.ts`:

- per act: `getCitedBy(celex, { citationType: ["amendment", "corrigendum", "repeal"] })`.
  The response's `relationSync.status` says how fresh those relations are; anything other than
  `fresh` / `refreshed` becomes a notice in the digest.
- per topic: `search({ query, dateFrom, documentType, maxPages: 1 })` when the topic has a query,
  otherwise `getRecentDocuments({ days, documentType, limit: 10 })`.

A state file (`.monitor-state.json`, git-ignored) remembers what was reported. The first time an
act is polled its existing relations are stored as a baseline and not reported.

```bash
npm run monitor -- --dry-run     # no keys: local stub + scripted model reply

cp examples/regulatory-change-monitor/.env.example examples/regulatory-change-monitor/.env   # then fill it in
npm run monitor                  # for real, with your own keys; schedule it with cron or CI
```

Polling does not detect new consolidated versions; webhook watch lists do.

If LexAPI refuses a request (for example 402, credits exhausted; 403, payment past due; or 429, a
rate limit or an exhausted quota), the run stops, prints one line with the SDK's own error text,
`LexAPI error: [<code>] <message> (HTTP <status>)`, leaves the state file unchanged and exits with
code 1. The SDK retries a rate limit before that; nothing else is added to the line.
`npm run monitor:webhook-setup` does the same.

## The digest and its grounding

`src/digest.ts` prints the list of documents in code, straight from the API records, each with
its CELEX number and EUR-Lex link. The model only adds a short summary on top: statements with
source ids, nothing else. The same check as in the other examples (`shared/grounding.ts`) applies.
It **rejects** the summary if:

- a statement cites a document that is not in the list;
- a quotation inside a statement - any quotation that contains a letter - is not verbatim, on
  word boundaries, in a record that statement cites, or a quotation mark is left open;
- a statement contains a line break or a control or format character, or starts with a heading,
  blockquote, table or code-fence marker;
- a statement contains an HTML character reference, a bracketed citation marker of its own
  (`[S1]` and its look-alikes), or a URL that is not the URL of a record it cites;
- there are more than 12 statements, or one is longer than 600 characters.

It **removes**, and only counts, statements without a source and statements that match the
reader patterns of the conclusion filter (`shared/conclusions.ts`): wording addressed to you
("you must review", "we recommend", "you will need to", "in your position"). A digest reports; it
does not advise. The filter is blunt and English-only; the
[AI Act example's README](../ai-act-provision-finder/README.md) gives its rules and the
quotation-mark styles recognised.

A rejected summary is retried once; after that the list is shown without a summary. Everything
printed that this program did not write itself - the summary, and every string from the API, the
watch list and webhook payloads (titles, CELEX numbers, labels, topic names, delivery ids,
notices) - goes through one function that removes control characters and escapes Markdown. A
link is printed only if it parses as an http(s) URL. When nothing is new, the digest says so and
the model is not called.

The check cannot tell whether a sentence describes its cited records correctly.

## Webhook mode (paid plans)

LexAPI can push changes instead: `documents.new` for new documents matching title criteria and
`watchlist.changes` when a watched act is amended, corrected, repealed or gets a new consolidated
version. Webhooks are not available on the FREE plan, and the receiver needs a public HTTPS URL
(port 443 or 8443).

```bash
LEXAPI_WEBHOOK_URL=https://your-host.example/webhooks/lexapi npm run monitor:webhook-setup
LEXAPI_WEBHOOK_SECRET=... npm run monitor:webhook-server   # listens on 127.0.0.1:8787
```

- `src/webhook-setup.ts` creates one webhook (`webhooks.create`: the acts as `watchCelex`, the
  first topic as `searchCriteria`), sends a test delivery (`webhooks.test`) and prints the secret
  once. Webhook criteria match document titles only.
- `src/webhook.ts` verifies each delivery with the SDK's `verifyWebhook` over the **raw** request
  body (HMAC-SHA256 with the secret, constant-time comparison, 5-minute replay window), ignores
  deliveries whose `delivery.id` was already handled, and maps the payload to the same items
  polling produces. A `consolidated_version` item is labelled as a documentation tool without
  legal force.
- `src/webhook-server.ts` is a plain `node:http` receiver: it answers 200 or 401 first and builds
  the digest afterwards, because a model call can outlast the 10-second delivery timeout. It
  listens on `127.0.0.1` by default (`HOST` and `PORT` change that). Expose it only through a
  reverse proxy or a tunnel that terminates HTTPS; do not bind it to a public interface directly.
  Bodies over 5 MB are answered with 413, and the last 10,000 delivery ids are remembered.

Webhook mode is tested offline only, with a signed fixture delivery. It has not been run against
real deliveries.

## Cost

LexAPI credits per polling run, from the [documented weights](https://lex-api.com/docs?ref=github-examples):
1 per watched act (cited-by) + 3 per topic (one page of search or recent documents). The example
watch list (2 acts, 2 topics) is **8 credits** and 4 requests per run. Run daily, that is about
240 credits a month - more than the FREE plan's monthly pool after the first cycle; a weekly run
is about 35. Webhook management calls and deliveries cost no credits. Model usage is billed
separately by your model provider.

## Tests

`npm test` runs `test/monitor.test.ts` offline: baseline on first poll, only-new on later polls,
stale-relation notices, the cited digest, the nothing-new path, rejection of an ungrounded summary
and of a summary shaped like Markdown, removal of advice, hostile strings in records and
payloads, ten well-behaved summaries, "not reported" credits, signature verification (valid,
tampered, wrong secret, missing, re-serialised body, stale timestamp, empty secret), duplicate
deliveries and the bound on remembered ids, the HTTP receiver on 127.0.0.1 (200, 401, 405, 413),
webhook setup against the stub, and the line printed when LexAPI refuses a poll (credits exhausted,
an exhausted quota, payment past due, a rate limit). Not covered by any test: the live API, real deliveries and a real model.

## Sample output (illustrative)

Generated with `--dry-run`. Titles and CELEX numbers are synthetic fixtures, and the summary is
scripted, not written by a model.

```markdown
> DRY RUN - LexAPI was replaced by a local stub and the model by scripted replies.
> The legal texts below are synthetic placeholders, not the wording of any act or judgment.

# Regulatory change digest

Generated 2026-11-18T08:00:00.000Z from polling. Model: scripted (no model was called).

## Documents (records returned by LexAPI)

- [S1] Corrigendum to Regulation (EU) 2016/679 (fixture title) - CELEX 32099R0003 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32099R0003
  - Listed because: corrigendum of GDPR (32016R0679)
- [S2] Regulation (EU) 2099/1 amending Regulation (EU) 2016/679 (fixture title) - CELEX 32099R0001 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32099R0001
  - Listed because: amendment of GDPR (32016R0679)
- [S3] Commission Implementing Regulation (EU) 2099/100 on placeholder subject matter (fixture title) - CELEX 32099R0100 - https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32099R0100
  - Listed because: topic: New regulations
- [S4] Regulation (EU) 2099/101 of the European Parliament and of the Council on placeholder subject matter (fixture title) - CELEX 32099R0101 - https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32099R0101
  - Listed because: topic: New regulations
- [S5] Communication from the Commission on placeholder guidelines for artificial intelligence (fixture title) - CELEX 52099DC0001 - https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:52099DC0001
  - Listed because: topic: Artificial intelligence

## Summary (written by the model from the records above - it has not read the documents)

- Two acts are newly recorded against the GDPR: a corrigendum and an amending regulation. [S1][S2]
- Two regulations were picked up under the "New regulations" topic. [S3][S4]
- One Commission communication has artificial intelligence in its title. [S5]

Removed by the grounding check:
- 1 statement(s) removed: cited no source.

## Notices

- AI Act (32024R1689): relations were served with sync status "pending" and may be incomplete; run again later.
```
