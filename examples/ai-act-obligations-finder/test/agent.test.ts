import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { readFixture, type StubServer } from "../../../offline/stub-server.ts";
import { offlineEnv, startOfflineApi } from "../../../offline/test-harness.ts";
import { createLexApiClient } from "../../../shared/env.ts";
import { ScriptedLlm } from "../../../shared/llm.ts";
import { CONCLUSION_PATTERNS, findProvisions } from "../src/agent.ts";
import { McpLegislationSearch } from "../src/retrieve-mcp.ts";
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
  assert.equal(result.creditsReported, 15);
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
  assert.match(result.markdown, /Article 9 \(article id art_9\) - CELEX 32024R1689 - https:\/\/eur-lex\.europa\.eu\/legal-content\/EN\/TXT\/\?uri=CELEX:32024R1689/);
  assert.match(result.markdown, /CELEX 02024R1689-20990101/);
  assert.match(result.markdown, /not the authentic text.*Authentic act: CELEX 32024R1689 as published in the Official Journal/);

  // Quoted text and model text are separate; the uncited sentence is removed and reported.
  assert.ok(result.markdown.indexOf("quoted from the source text") < result.markdown.indexOf("written by the model"));
  assert.equal(result.answer?.statements.length, 3);
  assert.match(result.markdown, /\(cited no source\) Screening tools are widely used in recruitment\./);
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
  const llm = new ScriptedLlm({ ...script(), submit_findings: invented });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.equal(result.outcome, "sources_only");
  assert.doesNotMatch(result.markdown, /providers must register every system/);
  assert.match(result.markdown, /failed the grounding check twice/);
  assert.match(result.markdown, /\[S1\] .*CELEX 32024R1689/);
});

test("statements that state a legal conclusion are removed", async () => {
  const conclusions = [
    "The system is high-risk under this article.",
    "Your system qualifies as a high-risk AI system.",
    "The provider must establish a risk management system.",
    "You must inform candidates.",
    "The tool is compliant with this provision.",
  ];
  for (const text of conclusions) {
    assert.ok(CONCLUSION_PATTERNS.some((p) => p.test(text)), text);
  }
  const allowed = "Whether the system falls in this category is a question for a qualified person.";
  assert.ok(!CONCLUSION_PATTERNS.some((p) => p.test(allowed)));

  const llm = new ScriptedLlm({
    ...script(),
    submit_findings: { status: "answered", quotes: [], statements: [{ text: conclusions[0], sources: ["S1"] }, { text: allowed, sources: ["S1"] }] },
  });
  const result = await findProvisions(DESCRIPTION, { llm, search: sdkSearch() });
  assert.deepEqual(result.answer?.statements.map((s) => s.text), [allowed]);
  assert.match(result.markdown, /\(stated a conclusion this example does not make\) The system is high-risk/);
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
    assert.equal(viaMcp.creditsReported, undefined);
    const viaSdk = await findProvisions(DESCRIPTION, { llm: new ScriptedLlm(script()), search: sdkSearch() });
    assert.deepEqual(viaMcp.sources, viaSdk.sources);
    assert.equal(stub.requests.at(-4)?.headers["x-api-key"], "lex_offline_test_key");
  } finally {
    await search.close();
  }
});
