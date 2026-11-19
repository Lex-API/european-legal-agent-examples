import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { LexAPI, LexAPIError } from "@lexapi/client";
import { readFixture, startStubServer, type StubServer } from "../../../offline/stub-server.ts";
import { assertLocalBaseUrl } from "../../../offline/guard.ts";
import { REFUSALS, offlineEnv, startOfflineApi } from "../../../offline/test-harness.ts";
import { createLexApiClient } from "../../../shared/env.ts";
import { lexapiErrorLine } from "../../../shared/lexapi-error.ts";
import { statesConclusion } from "../../../shared/conclusions.ts";
import { ScriptedLlm } from "../../../shared/llm.ts";
import { CONCLUSION_PATTERNS, findProvisions } from "../src/agent.ts";
import { McpLegislationSearch, stripTruncation } from "../src/retrieve-mcp.ts";
import { SdkLegislationSearch, toSources } from "../src/retrieve.ts";

const DESCRIPTION = "A SaaS tool that ranks job applicants' CVs and chats with candidates";
const script = () => JSON.parse(readFixture("llm/ai-act.json")) as { plan_searches: unknown; submit_findings: { quotes: unknown[]; statements: unknown[] } };

let stub: StubServer;
before(async () => {
  stub = await startOfflineApi();
});
after(async () => {
  await stub.close();
});

const sdkSearch = () => new SdkLegislationSearch(createLexApiClient(offlineEnv()));

test("answers with quoted provisions, cited notes and the human-decision section", async () => {
  stub.requests.length = 0;
  const llm = new ScriptedLlm(script());
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });

  assert.equal(result.outcome, "answered");
  assert.equal(result.credits, "15");
  // Three searches, each asking for full article text, with the wire names the API documents.
  assert.equal(stub.requests.length, 3);
  assert.deepEqual(JSON.parse(stub.requests[0]?.body ?? "{}"), {
    query: "classification of AI systems used to screen job applicants",
    limit: 10,
    include: ["text"],
  });
  assert.equal(stub.requests[0]?.path, "/legislation/semantic");

  // Only AI Act articles become sources; the GDPR hit in the fixture is dropped.
  assert.deepEqual(result.sources.map((s) => [s.id, s.articleRef]), [["S1", "art_6"], ["S2", "art_9"], ["S3", "art_50"]]);
  assert.match(result.markdown, /3 from other acts \(not shown\)/);

  // Citations: CELEX, article id, EUR-Lex link; the consolidated hit carries its caveat and base act.
  assert.match(result.markdown, /Article 9 \(article id art\\_9\) - CELEX 32024R1689 - https:\/\/eur-lex\.europa\.eu\/legal-content\/EN\/TXT\/\?uri=CELEX:32024R1689/);
  assert.match(result.markdown, /CELEX 02024R1689-20990101/);
  assert.match(result.markdown, /not the authentic text.*Authentic act: CELEX 32024R1689 as published in the Official Journal/);

  // Quoted text and model text are separate; the uncited sentence is removed and reported.
  assert.ok(result.markdown.indexOf("quoted from the source text") < result.markdown.indexOf("written by the model"));
  assert.equal(result.answer?.statements.length, 3);
  assert.match(result.markdown, /- 1 statement\(s\) removed: cited no source\./);
  assert.doesNotMatch(result.markdown, /Screening tools are widely used/);
  assert.match(result.markdown, /## What a human must decide/);
});

test("the model only sees the retrieved sources, and the prompt forbids conclusions", async () => {
  const llm = new ScriptedLlm(script());
  await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  const draftCall = llm.calls.find((call) => call.tool.name === "submit_findings");
  assert.match(draftCall?.user ?? "", /<source id="S1">/);
  assert.doesNotMatch(draftCall?.user ?? "", /32016R0679/);
  assert.match(draftCall?.system ?? "", /do not classify the system/);
});

test("reports 'no supporting text found' without asking the model to draft", async () => {
  const llm = new ScriptedLlm({ plan_searches: { queries: ["no-such-topic placeholder query"] } });
  const result = await findProvisions("no-such-topic", { llm, search: sdkSearch() });
  assert.equal(result.outcome, "no_supporting_text");
  assert.match(result.markdown, /## No supporting text found/);
  assert.equal(llm.calls.length, 1);
});

test("a model reply of no_support is reported as such", async () => {
  const llm = new ScriptedLlm({ ...script(), submit_findings: { status: "no_support", quotes: [], statements: [] } });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.equal(result.outcome, "no_supporting_text");
  assert.match(result.markdown, /found nothing in them that it could tie to this description/);
});

test("a draft citing a source that was not retrieved is rejected; the retry gets the violations", async () => {
  const good = script().submit_findings;
  const bad = { status: "answered", quotes: [], statements: [{ text: "See the annex.", sources: ["S9"] }] };
  const llm = new ScriptedLlm({ ...script(), submit_findings: [bad, good] });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.equal(result.outcome, "answered");
  const drafts = llm.calls.filter((call) => call.tool.name === "submit_findings");
  assert.equal(drafts.length, 2);
  assert.match(drafts[1]?.user ?? "", /previous draft was rejected:\n- statement cites "S9", which was not retrieved/);
});

test("after two rejected drafts only the retrieved sources are shown", async () => {
  const invented = { status: "answered", quotes: [{ source: "S1", text: "providers must register every system" }], statements: [] };
  assert.doesNotMatch(readFixture("legislation-semantic.ai-act.json"), /providers must register/);
  const llm = new ScriptedLlm({ ...script(), submit_findings: invented });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.equal(result.outcome, "sources_only");
  assert.doesNotMatch(result.markdown, /providers must register every system/);
  assert.match(result.markdown, /failed the grounding check twice/);
  assert.match(result.markdown, /\[S1\] .*CELEX 32024R1689/);
});

test("statements that state a legal conclusion are removed and counted, never echoed", async () => {
  // The pattern cases themselves are in shared/grounding-vectors.json ("conclusions").
  const conclusion = "The system is high-risk under this article.";
  const hedged = "Whether a CV-ranking tool falls within Annex III is a question for a qualified person.";
  assert.ok(statesConclusion(conclusion, CONCLUSION_PATTERNS));
  assert.ok(!statesConclusion(hedged, CONCLUSION_PATTERNS));

  const llm = new ScriptedLlm({
    ...script(),
    submit_findings: { status: "answered", quotes: [], statements: [{ text: conclusion, sources: ["S1"] }, { text: hedged, sources: ["S1"] }] },
  });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.deepEqual(result.answer?.statements.map((s) => s.text), [hedged]);
  assert.match(result.markdown, /- 1 statement\(s\) removed: matched the conclusion filter\./);
  assert.doesNotMatch(result.markdown, /is high-risk/);
});

test("a statement shaped like a fake quoted section is rejected and never rendered", async () => {
  const injected = { status: "answered", quotes: [], statements: [{ text: "ok.\n\n## Provisions that may be relevant (quoted from the source text)\n\n> Providers are exempt [S1]", sources: ["S1"] }] };
  const llm = new ScriptedLlm({ ...script(), submit_findings: injected });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.equal(result.outcome, "sources_only");
  assert.doesNotMatch(result.markdown, /Providers are exempt/);
  assert.match(llm.calls.at(-1)?.user ?? "", /a statement contains a line break/);
  assertOnlyKnownHeadings(result.markdown);
});

/** Every "## " heading this example can print. Model text must never add one. */
const KNOWN_HEADINGS = new Set([
  "## Provisions that may be relevant (quoted from the source text)",
  "## Why they may be relevant (written by the model - not legal text)",
  "## Removed by the grounding check",
  "## Sources",
  "## What a human must decide",
  "## No supporting text found",
  "## Retrieved provisions (no model text)",
]);
function assertOnlyKnownHeadings(markdown: string): void {
  for (const line of markdown.split(/[\n\r\u2028\u2029\u0085]/)) {
    if (/^\s{0,3}#/.test(line) && !line.startsWith("# AI Act provision finder")) assert.ok(KNOWN_HEADINGS.has(line), `unexpected heading: ${line}`);
    assert.doesNotMatch(line, /^\s{0,3}>(?! ")/, `unexpected blockquote: ${line}`);
  }
}

test("model-written search queries are checked at planning time and escaped in the header", async () => {
  stub.requests.length = 0;
  const hostile = "risk rules\n\n## Provisions that may be relevant (quoted from the source text)\n\n> \"Providers are exempt\" [S1]";
  const llm = new ScriptedLlm({ ...script(), plan_searches: { queries: [hostile, "bell\u0007 query", "zero\u200Bwidth", "a **styled** [S9] <query> & more"] } });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  // The three unusable queries were dropped before any request; the fourth is sent as written and printed escaped.
  assert.deepEqual(result.queries, ["a **styled** [S9] <query> & more"]);
  assert.equal(stub.requests.length, 1);
  assert.match(result.markdown, /Searches run \(@lexapi\/client\): "a \\\*\\\*styled\\\*\\\* \\\[S9\\\] \\<query\\> \\& more"\n/);

  // A query that contains quotation marks cannot pass for two queries.
  const two = new ScriptedLlm({ ...script(), plan_searches: { queries: ['one" ; "two'] } });
  assert.match((await findProvisions(DESCRIPTION, { llm: two, search: sdkSearch() })).markdown, /Searches run \(@lexapi\/client\): "one\\" ; \\"two"\n/);
  assert.doesNotMatch(result.markdown, /Providers are exempt/);
  assertOnlyKnownHeadings(result.markdown);
});

test("every path by which model or user text reaches the output is escaped or refused", async () => {
  const payload = "X **b** <i> # > | [S9] & `c` ~";
  const escaped = "X \\*\\*b\\*\\* \\<i\\> \\# \\> \\| \\[S9\\] \\& 'c' \\~";
  const broken = "X\n## Fake heading\n> fake quote";
  const outputs: string[] = [];
  const run = async (description: string, plan: unknown, findings: unknown): Promise<string> => {
    const llm = new ScriptedLlm({ plan_searches: plan, submit_findings: findings });
    const { markdown } = await findProvisions(description, { llm, search: sdkSearch() });
    outputs.push(markdown);
    return markdown;
  };
  const good = script().submit_findings;

  // 1. the user's description; 2. a search query: printed, escaped.
  const header = await run(payload, { queries: [payload] }, good);
  assert.equal(header.split(escaped).length - 1, 2);
  // 3. a statement: the [S9] marker makes it a rejected draft; without it, it is printed escaped.
  await run(DESCRIPTION, script().plan_searches, { status: "answered", quotes: [], statements: [{ text: payload, sources: ["S1"] }] });
  // (The backticks would count as quotation marks around a word that is not in the source.)
  const noMarker = payload.replace(" [S9]", "").replace(" `c`", "");
  const statement = await run(DESCRIPTION, script().plan_searches, { status: "answered", quotes: [], statements: [{ text: noMarker, sources: ["S1"] }] });
  assert.ok(statement.includes(`- ${escaped.replace(" \\[S9\\]", "").replace(" 'c'", "")} [S1]`));
  // 4. a quote: only text found verbatim in the source is printed, taken from the source.
  await run(DESCRIPTION, script().plan_searches, { status: "answered", quotes: [{ source: "S1", text: payload }], statements: [] });
  // 5. removed statements (uncited, conclusion): counted, never printed.
  const removed = await run(DESCRIPTION, script().plan_searches, {
    status: "answered",
    quotes: [],
    statements: [{ text: payload, sources: [] }, { text: `You must ${payload}`, sources: ["S1"] }, { text: "A note.", sources: ["S1"] }],
  });
  assert.doesNotMatch(removed, /X \*\*b/);
  // 6. source ids, 7. status and other fields of the wrong shape, 8. multi-line text in every field: refused.
  await run(DESCRIPTION, script().plan_searches, { status: "answered", quotes: [], statements: [{ text: "A note.", sources: [broken] }] });
  await run(DESCRIPTION, script().plan_searches, { status: broken, quotes: [], statements: [] });
  await run(broken, { queries: [broken] }, { status: "answered", quotes: [{ source: "S1", text: broken }], statements: [{ text: broken, sources: ["S1"] }] });

  assert.equal(outputs.length, 8);
  for (const markdown of outputs) {
    assertOnlyKnownHeadings(markdown);
    assert.doesNotMatch(markdown, /X \*\*b\*\*/); // the raw payload never appears
    assert.doesNotMatch(markdown, /^> fake quote/m);
  }
});

test("ten well-behaved drafts in the styles models write are all accepted", async () => {
  // Realistic habits: sentence-initial capitals, punctuation inside quotation marks, ellipses at the
  // ends, wrapped quotes, possessives, single-quoted terms, the description paraphrased,
  // numbered notes, and questions framed with "whether" / "if" / "which".
  const drafts = [
    { quotes: [{ source: "S1", text: "fixture wording about how an example system is sorted into a placeholder category" }], statements: [{ text: "This article is about how a system is sorted into a category, which is the first thing to read.", sources: ["S1"] }] },
    { quotes: [{ source: "S2", text: "Placeholder provision B1: fixture wording about keeping a placeholder risk process running for the example system." }], statements: [{ text: "Article 9's risk process may be worth reading if the tool were found to be in the placeholder category.", sources: ["S2"] }] },
    { quotes: [{ source: "S3", text: "\"fixture wording about telling people that they are dealing with an example system\"" }], statements: [{ text: "The description says the tool chats with candidates, so the provision on \"telling people that they are dealing with an example system.\" may be relevant.", sources: ["S3"] }] },
    { quotes: [{ source: "S2", text: "...fixture wording saying the process repeats over the lifetime of the example system..." }], statements: [{ text: "\"Placeholder risk process\" is the term this article uses; whether it applies to the system is for a qualified person to decide.", sources: ["S2"] }] },
    { quotes: [{ source: "S1", text: "placeholder provision A2: fixture wording about writing down that sorting" }], statements: [{ text: "1. The article's second provision concerns 'writing down that sorting' before the system is offered.", sources: ["S1"] }, { text: "2. Whether a CV-ranking tool falls within the scope of this category is a legal question.", sources: ["S1"] }] },
    { quotes: [], statements: [{ text: "Which obligations apply to the system depends on how it is classified; these two articles are where that question starts.", sources: ["S1", "S2"] }] },
    { quotes: [{ source: "S3", text: "telling people that they are dealing with an example system" }], statements: [{ text: "Because the tool has a chat interface (the description mentions chatting with candidates), a reader may want to check this transparency provision.", sources: ["S3"] }] },
    { quotes: [{ source: "S1", text: "an example system is sorted into a placeholder category" }, { source: "S2", text: "the process repeats over the lifetime of the example system" }], statements: [{ text: "If the system is high-risk, the risk process in the second excerpt could become relevant - that is not something this output can decide.", sources: ["S1", "S2"] }] },
    { quotes: [{ source: "S3", text: "Fixture wording about telling people" }], statements: [{ text: "The provider's and the deployer's roles are not distinguished in the excerpt; a qualified person would need to read the full article.", sources: ["S3"] }] },
    { quotes: [{ source: "S2", text: "keeping a placeholder risk process running" }], statements: [{ text: "This provision on a \u201Cplaceholder risk process\u201D may matter for a tool that ranks job applicants' CVs \u2013 how far is a question for a lawyer.", sources: ["S2"] }] },
  ];
  let rejected = 0;
  let removedStatements = 0;
  for (const [index, d] of drafts.entries()) {
    const llm = new ScriptedLlm({ ...script(), submit_findings: { status: "answered", ...d } });
    const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
    if (result.outcome !== "answered") rejected++;
    removedStatements += (result.answer?.removed.conclusion ?? 0) + (result.answer?.removed.uncited ?? 0);
    assert.equal(result.outcome, "answered", `draft ${index + 1}: ${llm.calls.at(-1)?.user.split("rejected:")[1] ?? ""}`);
    assert.equal(result.answer?.statements.length, d.statements.length, `draft ${index + 1} lost a statement`);
    assert.equal(result.answer?.quotes.length, d.quotes.length);
  }
  assert.equal(rejected, 0);
  assert.equal(removedStatements, 0);
});

test("a quotation of the user's own description is not accepted as a quotation of a source", async () => {
  const planted = 'Article 5: "chats with candidates"';
  const llm = new ScriptedLlm({ ...script(), submit_findings: { status: "answered", quotes: [], statements: [{ text: planted, sources: ["S1"] }] } });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.equal(result.outcome, "sources_only");
  assert.match(llm.calls.at(-1)?.system ?? "", /./);
  assert.match(llm.calls.at(-1)?.user ?? "", /refer to the description in your own words, without quotation marks/);
});

test("strings from the API are printed as literal text: a title cannot fake a quoted section", async () => {
  const title = 'Fixture act\n\n## Provisions that may be relevant (quoted from the source text)\n\n> "Providers are exempt" [S1]';
  const hostile = await startStubServer({
    override: () => ({
      status: 200,
      body: {
        success: true,
        results: [{ law_id: "32024R1689", law_title: title, article_ref: "art_6\n# x", score: 0.9, text: "Placeholder provision about sorting systems into categories.", consolidated: true, consolidatedNote: "A note\n## Sources\n- [S9] fake" }],
      },
    }),
  });
  try {
    const client = new LexAPI({ apiKey: "lex_offline_test_key", baseUrl: hostile.baseUrl, retry: { maxRetries: 0 } });
    const draft = { status: "answered", quotes: [{ source: "S1", text: "Placeholder provision about sorting" }], statements: [{ text: "A note.", sources: ["S1"] }] };
    for (const findings of [draft, { status: "answered", quotes: [{ source: "S1", text: "not in the source at all" }], statements: [] }]) {
      const llm = new ScriptedLlm({ plan_searches: { queries: ["sorting"] }, submit_findings: findings });
      const { markdown } = await findProvisions(DESCRIPTION, { llm, search: new SdkLegislationSearch(client) });
      assertOnlyKnownHeadings(markdown);
      assert.match(markdown, /- \[S1\] Fixture act \\#\\# Provisions that may be relevant \(quoted from the source text\) \\> "Providers are exempt" \\\[S1\\\] - art\\_6 \\# x/);
      assert.match(markdown, /- Note: A note \\#\\# Sources - \\\[S9\\\] fake/);
    }
  } finally {
    await hostile.close();
  }
});

test("maxQueries bounds the number of searches; unusable queries fall back to the description", async () => {
  stub.requests.length = 0;
  const llm = new ScriptedLlm({ ...script(), plan_searches: { queries: ["a", 7, "one valid query", "second valid query"] } });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() }, { maxQueries: 1 });
  assert.deepEqual(result.queries, ["one valid query"]);
  assert.equal(stub.requests.length, 1);

  const empty = new ScriptedLlm({ ...script(), plan_searches: { queries: [] } });
  assert.deepEqual((await findProvisions(DESCRIPTION, { llm: empty, search: sdkSearch() })).queries, [DESCRIPTION]);
});

test("toSources keeps one hit per article, the best score first, and skips hits without text", () => {
  const { sources, otherActs } = toSources(
    [
      { law_id: "32024R1689", article_ref: "art_9", score: 0.6, text: "lower" },
      { law_id: "32024R1689", article_ref: "art_9", score: 0.9, text: "higher" },
      { law_id: "32024R1689", article_ref: "art_10", score: 0.7 },
      { law_id: "02024R1689-20990101", baseCelex: "32024R1689", consolidated: true, article_ref: "art_13", score: 0.95, text: "consolidated" },
      { law_id: "02016R0679-20160504", baseCelex: "32016R0679", article_ref: "art_22", score: 0.99, text: "other act" },
    ],
    8,
  );
  assert.equal(otherActs, 1);
  assert.deepEqual(sources.map((s) => [s.id, s.articleRef, s.text]), [["S1", "art_13", "consolidated"], ["S2", "art_9", "higher"]]);
  assert.equal(sources[0]?.notes.length, 1);
});

test("the MCP route returns the same sources as the SDK route", async () => {
  const search = await McpLegislationSearch.start(offlineEnv());
  try {
    const llm = new ScriptedLlm(script());
    const viaMcp = await findProvisions(DESCRIPTION, { llm, search });
    assert.equal(viaMcp.outcome, "answered");
    assert.equal(viaMcp.credits, "not reported");
    const viaSdk = await findProvisions(DESCRIPTION, { llm: new ScriptedLlm(script()), search: sdkSearch() });
    assert.deepEqual(viaMcp.sources, viaSdk.sources);
    assert.equal(stub.requests.at(-4)?.headers["x-api-key"], "lex_offline_test_key");
  } finally {
    await search.close();
  }
});

test("the MCP route refuses to start its child process against a non-local base URL under test", async () => {
  await assert.rejects(McpLegislationSearch.start({ LEXAPI_API_KEY: "lex_offline_test_key" }), /Refusing to run/);
  await assert.rejects(
    McpLegislationSearch.start({ LEXAPI_API_KEY: "lex_offline_test_key", LEXAPI_BASE_URL: "https://lex-api.com/api/v1" }),
    /Refusing to run/,
  );
});

test("MCP route: text cut by the tool loses its marker, is labelled, and the cut part cannot be quoted", async () => {
  assert.deepEqual(stripTruncation({ text: "abc \u2026[truncated: 12 more characters]" }), { text: "abc", truncatedBy: 12 });
  assert.deepEqual(stripTruncation({ text: "abc" }), { text: "abc" });

  // One article longer than the tool's largest budget (200,000 characters).
  const sentence = (n: number) => `Placeholder sentence number ${n} of a long fixture article. `;
  const longText = Array.from({ length: 4500 }, (_, i) => sentence(i)).join("");
  assert.ok(longText.length > 200_000);
  const longStub = await startStubServer({
    override: () => ({
      status: 200,
      body: { success: true, results: [{ law_id: "32024R1689", law_title: "Fixture act", article_ref: "art_99", score: 0.9, text: longText }] },
    }),
  });
  const search = await McpLegislationSearch.start({ LEXAPI_API_KEY: "lex_offline_test_key", LEXAPI_BASE_URL: longStub.baseUrl });
  try {
    const { hits } = await search.search("a long article");
    assert.equal(JSON.parse(longStub.requests[0]?.body ?? "{}").include[0], "text");
    assert.equal(hits[0]?.text?.length, 200_000);
    assert.equal(hits[0]?.truncatedBy, longText.length - 200_000);
    assert.doesNotMatch(hits[0]?.text ?? "", /truncated/);

    const last = sentence(4499).trim();
    const draft = (text: string) => ({ status: "answered", quotes: [{ source: "S1", text }], statements: [] });
    const llm = new ScriptedLlm({
      plan_searches: { queries: ["a long article"] },
      // First the marker text, then a sentence from the part that was cut off; neither is in the source.
      submit_findings: [draft(`[truncated: ${longText.length - 200_000} more characters]`), draft(last)],
    });
    const result = await findProvisions("a long article", { llm, search });
    assert.equal(result.outcome, "sources_only");
    assert.match(result.markdown, /Text truncated by the tool: the last \d+ characters of this article were not retrieved/);

    const first = new ScriptedLlm({ plan_searches: { queries: ["a long article"] }, submit_findings: draft(sentence(0).trim()) });
    assert.equal((await findProvisions("a long article", { llm: first, search })).outcome, "answered");
  } finally {
    await search.close();
    await longStub.close();
  }
});

for (const [name, refusal] of Object.entries(REFUSALS)) {
  test(`a ${name} answer stops the run with the SDK's own text on one line`, async () => {
    const refusing = await startStubServer({ override: () => refusal });
    try {
      const client = createLexApiClient({ LEXAPI_API_KEY: "lex_offline_test_key", LEXAPI_BASE_URL: assertLocalBaseUrl(refusing.baseUrl) });
      const llm = new ScriptedLlm(script());
      const error = await findProvisions(DESCRIPTION, { llm, search: new SdkLegislationSearch(client) }).then(
        () => undefined,
        (thrown: unknown) => thrown,
      );
      assert.ok(error instanceof LexAPIError);
      assert.equal(lexapiErrorLine(error, "lex_offline_test_key"), refusal.printed);
      // Nothing is added: no wait, no link, and a failed payment is not a plan gate to upgrade.
      assert.doesNotMatch(lexapiErrorLine(error), /retry|https?:|upgrade/i);
      // The SDK's code, whichever class it uses for a failed payment (0.4.0: the base class; later: a tier-error subclass).
      if (name === "past due") assert.equal(error.code, "PAYMENT_PAST_DUE");
      assert.equal(refusing.requests.length, refusal.attempts); // the first search only; only a rate limit is retried
    } finally {
      await refusing.close();
    }
  });
}

test("the error line removes control characters and masks the key", () => {
  const error = new LexAPIError("INTERNAL", "placeholder\u001b[2J\u202e lex_offline_test_key\nmore", 500);
  assert.equal(lexapiErrorLine(error, "lex_offline_test_key"), "LexAPI error: [INTERNAL] placeholder [2J [key] more (HTTP 500)");
});
