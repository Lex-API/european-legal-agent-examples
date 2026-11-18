import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { AI_ACT_PATTERNS, READER_PATTERNS, statesConclusion } from "./conclusions.ts";
import { GroundingError, parseDraft, renderAnswer, renderRemoved, validateDraft, type Draft, type GroundedAnswer, type Source } from "./grounding.ts";
import { modelTextProblem, normalise, plainText, printableUrl, quotationsIn, quotedPlainText } from "./text.ts";

// The cases in grounding-vectors.json are run by this suite AND by the Python suite.
interface VectorDraft {
  group: string;
  name: string;
  draft: Draft;
  accept: boolean;
  violation?: string;
  quotes?: string[];
  statements?: number;
  sources?: string[];
  patterns?: "reader";
  removed?: { uncited: number; conclusion: number };
}
interface Vectors {
  sources: Array<{ id: string; url: string; text: string }>;
  normalise: Array<[string, string]>;
  quotations: Array<[string, string[] | null]>;
  plainText: Array<[string, string]>;
  modelTextProblems: Array<[string, "statement" | "quote" | "query", string | null]>;
  conclusions: Array<{ text: string; reader: boolean; aiAct: boolean; kind: string }>;
  drafts: VectorDraft[];
}

/** `{ "repeat": [unit, n] }` stands for the unit repeated n times. */
function expand(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(expand);
  if (typeof value === "object" && value !== null) {
    const repeat = (value as { repeat?: [string, number] }).repeat;
    if (repeat) return repeat[0].repeat(repeat[1]);
    return Object.fromEntries(Object.entries(value).map(([key, inner]) => [key, expand(inner)]));
  }
  return value;
}

const vectors = expand(JSON.parse(readFileSync(new URL("./grounding-vectors.json", import.meta.url), "utf8"))) as Vectors;
const sources: Source[] = vectors.sources.map((s) => ({ ...s, kind: "legislation", celex: "39999R0001", title: "Test source", notes: [] }));

function check(draft: Draft, patterns?: "reader"): GroundedAnswer | string[] {
  try {
    return validateDraft(parseDraft(draft), sources, patterns ? { conclusionPatterns: READER_PATTERNS } : {});
  } catch (error) {
    assert.ok(error instanceof GroundingError);
    return error.violations;
  }
}

for (const group of ["review1", "review2", "review3", "relaxed", "equivalence"]) {
  test(`shared vectors: drafts (${group})`, () => {
    const cases = vectors.drafts.filter((c) => c.group === group);
    assert.ok(cases.length > 0);
    for (const c of cases) {
      const result = check(c.draft, c.patterns);
      if (!c.accept) {
        assert.ok(Array.isArray(result), `${c.name}: expected a rejection`);
        assert.ok(result.some((v) => v.includes(c.violation ?? "")), `${c.name}: got ${JSON.stringify(result)}`);
        continue;
      }
      assert.ok(!Array.isArray(result), `${c.name}: rejected with ${JSON.stringify(result)}`);
      if (c.quotes) assert.deepEqual(result.quotes.map((q) => q.text), c.quotes, c.name);
      if (c.statements !== undefined) assert.equal(result.statements.length, c.statements, c.name);
      if (c.sources) assert.deepEqual(result.statements[0]?.sources, c.sources, c.name);
      if (c.removed) assert.deepEqual(result.removed, c.removed, c.name);
    }
  });
}

test("shared vectors: normalise, quotations, plainText, modelTextProblem", () => {
  for (const [input, expected] of vectors.normalise) assert.equal(normalise(input), expected, JSON.stringify(input));
  for (const [input, expected] of vectors.quotations) assert.deepEqual(quotationsIn(input) ?? null, expected, input);
  for (const [input, expected] of vectors.plainText) assert.equal(plainText(input), expected, input);
  for (const [input, kind, expected] of vectors.modelTextProblems) {
    const problem = modelTextProblem(input, kind);
    if (expected === null) assert.equal(problem, undefined, JSON.stringify(input));
    else assert.ok(problem?.includes(expected), `${JSON.stringify(input)}: ${problem}`);
  }
});

test("shared vectors: conclusion filter (reader patterns, and reader + AI Act patterns)", () => {
  for (const c of vectors.conclusions) {
    assert.equal(statesConclusion(normalise(c.text), READER_PATTERNS), c.reader, `reader: ${c.text}`);
    assert.equal(statesConclusion(normalise(c.text), [...READER_PATTERNS, ...AI_ACT_PATTERNS]), c.aiAct, `ai-act: ${c.text}`);
  }
  // The third review's targets: every hedged attack is removed; at least 11 of every 13 legitimate sentences are kept.
  const of = (kind: string) => vectors.conclusions.filter((c) => c.kind === kind);
  assert.ok(of("hedged attack").length >= 14 && of("hedged attack").every((c) => c.aiAct));
  assert.ok(of("legitimate").length >= 13 && of("legitimate").every((c) => !c.aiAct));
  assert.equal(of("legitimate, removed").length, 1);
});

const draft = (overrides: Partial<Draft>): Draft => ({ status: "answered", quotes: [], statements: [], ...overrides });

test("a quote is returned with the source's own characters, not the model's", () => {
  const typeset: Source[] = [{ ...sources[0]!, text: "The keeper’s duty “at once” ends   here." }];
  const answer = validateDraft(draft({ quotes: [{ source: "S1", text: "the keeper's duty \"at once\" ends..." }] }), typeset);
  assert.deepEqual(answer.quotes, [{ source: "S1", text: "The keeper's duty \"at once\" ends" }]);
});

test("statements matching the conclusion filter are removed and counted, never echoed", () => {
  const answer = validateDraft(
    draft({
      statements: [
        { text: "You must register the system.", sources: ["S1"] },
        { text: 'An uncited claim with an invented "quotation of the law".', sources: [] },
        { text: "The article covers consent.", sources: ["S1"] },
      ],
    }),
    sources,
    { conclusionPatterns: READER_PATTERNS },
  );
  assert.deepEqual(answer.statements.map((s) => s.text), ["The article covers consent."]);
  assert.deepEqual(answer.removed, { uncited: 1, conclusion: 1 });
  assert.deepEqual(renderRemoved(answer.removed), [
    "- 1 statement(s) removed: cited no source.",
    "- 1 statement(s) removed: matched the conclusion filter.",
  ]);
  const out = renderAnswer(answer, sources, { quotes: "Quoted", statements: "Model-written" });
  assert.doesNotMatch(out, /register the system|quotation of the law/);
});

test("no_support drafts and drafts left empty by the check both end as no_support", () => {
  assert.equal(validateDraft(draft({ status: "no_support", quotes: [{ source: "S9", text: "x" }] }), sources).status, "no_support");
  assert.equal(validateDraft(draft({ statements: [{ text: "Uncited.", sources: [] }] }), sources).status, "no_support");
  assert.deepEqual(parseDraft({ status: "answered", statements: [] }).quotes, []); // statements-only schema
});

test("parseDraft rejects a reply with the wrong shape", () => {
  assert.throws(() => parseDraft({ status: "answered", quotes: "none", statements: [] }), GroundingError);
  assert.throws(() => parseDraft({ status: "maybe", quotes: [], statements: [] }), GroundingError);
  assert.throws(() => parseDraft({ status: "answered", quotes: [], statements: [{ text: "x", sources: [1] }] }), GroundingError);
  assert.throws(() => parseDraft(null), GroundingError);
});

test("the rendered answer separates quotes from model text and cites CELEX, article, ECLI and link", () => {
  const cited: Source[] = [
    { id: "S1", kind: "legislation", celex: "32024R1689", articleRef: "art_9", title: "Fixture act", url: "https://example.org/a", text: "A placeholder process shall be kept running.", notes: ["Consolidated text; not the authentic text."] },
    { id: "S2", kind: "case-law", celex: "62099CJ9901", ecli: "ECLI:EU:C:2099:901", paragraph: 3, title: "Fixture judgment", url: "https://example.org/b", text: "Placeholder ground three.", notes: [] },
  ];
  const answer = validateDraft(
    draft({
      quotes: [{ source: "S1", text: "process shall be kept" }],
      statements: [{ text: "1. A note with **bold**   spacing & an <i>tag</i>.", sources: ["S2"] }],
    }),
    cited,
  );
  const out = renderAnswer(answer, cited, { quotes: "Quoted", statements: "Model-written" });
  assert.ok(out.indexOf("## Quoted") < out.indexOf("## Model-written"));
  assert.match(out, /^> "process shall be kept" \[S1\]$/m);
  assert.match(out, /^- 1\\\. A note with \\\*\\\*bold\\\*\\\* spacing \\& an \\<i\\>tag\\<\/i\\>\. \[S2\]$/m);
  assert.match(out, /\[S1\] Fixture act - Article 9 \(article id art\\_9\) - CELEX 32024R1689 - https:\/\/example\.org\/a/);
  assert.match(out, /\[S2\] Fixture judgment - paragraph 3 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901/);
  assert.match(out, /Note: Consolidated text; not the authentic text\./);
});

test("validation stays fast on a large source with many quotes", () => {
  const big: Source[] = [{ ...sources[0]!, text: "lorem ipsum dolor sit amet ".repeat(15_000) }];
  const many = draft({
    quotes: Array.from({ length: 12 }, () => ({ source: "S1", text: "dolor sit amet lorem" })),
    statements: Array.from({ length: 12 }, () => ({ text: 'It says "lorem ipsum dolor sit amet lorem" and "no such words here at all".', sources: ["S1"] })),
  });
  const started = performance.now();
  assert.throws(() => validateDraft(many, big), GroundingError);
  assert.ok(performance.now() - started < 2000);
});

test("strings from the API are printed as literal text: a title cannot fake a section, a bad URL is omitted", () => {
  const hostile: Source[] = [
    {
      id: "S1",
      kind: "case-law",
      celex: "62099CJ9901\n## Fake",
      ecli: "ECLI:EU:C:2099:901 **x**",
      articleRef: "art_9 [S7]",
      title: "A title\n\n## Quoted from the sources\n\n> \"Transfers are always lawful\" [S1]",
      url: "javascript:alert(1)",
      text: "Placeholder ground three of the judgment.",
      notes: ["A note with `code`\n# heading and \u001b[31mcolour"],
    },
  ];
  const answer = validateDraft(draft({ quotes: [{ source: "S1", text: "Placeholder ground three" }] }), hostile);
  const out = renderAnswer(answer, hostile, { quotes: "Quoted", statements: "Model-written" });
  assert.deepEqual(out.split("\n").filter((line) => /^\s{0,3}#/.test(line)), ["## Quoted", "## Model-written", "## Sources"]);
  assert.equal(out.split("\n").filter((line) => /^\s{0,3}>/.test(line)).length, 1); // the one real quote
  assert.match(out, /- \[S1\] A title \\#\\# Quoted from the sources \\> "Transfers are always lawful" \\\[S1\\\] - art\\_9 \\\[S7\\\] \(article id art\\_9 \\\[S7\\\]\) - ECLI:EU:C:2099:901 \\\*\\\*x\\\*\\\* - CELEX 62099CJ9901 \\#\\# Fake$/m);
  assert.doesNotMatch(out, /javascript:/);
  assert.match(out, /^ {2}- Note: A note with 'code' \\# heading and \\\[31mcolour$/m);
  assert.equal(printableUrl("https://example.org/a_b?x=1&y=2"), "https://example.org/a\\_b?x=1\\&y=2");
  assert.equal(printableUrl("not a url"), "");
  assert.equal(quotedPlainText('one" ; "two'), '"one\\" ; \\"two"');
});
