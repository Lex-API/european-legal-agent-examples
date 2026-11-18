import assert from "node:assert/strict";
import { test } from "node:test";
import { GroundingError, parseDraft, renderAnswer, validateDraft, type Draft, type Source } from "./grounding.ts";

const sources: Source[] = [
  {
    id: "S1",
    kind: "legislation",
    celex: "32024R1689",
    articleRef: "art_9",
    title: "Fixture act",
    url: "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    text: "A placeholder risk management system shall be established,\n  implemented, documented and maintained.",
    notes: ["Consolidated text; not the authentic text."],
  },
  {
    id: "S2",
    kind: "case-law",
    celex: "62018CJ0311",
    ecli: "ECLI:EU:C:2020:559",
    paragraph: 3,
    title: "Fixture judgment",
    url: "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62018CJ0311",
    text: "The safeguards must ensure an essentially equivalent level of protection.",
    notes: [],
  },
];

const draft = (overrides: Partial<Draft>): Draft => ({ status: "answered", quotes: [], statements: [], ...overrides });

function violationsOf(fn: () => unknown): string[] {
  try {
    fn();
  } catch (error) {
    assert.ok(error instanceof GroundingError);
    return error.violations;
  }
  assert.fail("expected a GroundingError");
}

test("accepts a draft whose quotes are verbatim and whose statements cite retrieved sources", () => {
  const answer = validateDraft(
    draft({
      quotes: [{ source: "S1", text: "shall be established, implemented, documented and maintained" }],
      statements: [{ text: "The article describes a process.", sources: ["S1", "S2"] }],
    }),
    sources,
  );
  assert.equal(answer.status, "answered");
  assert.equal(answer.statements.length, 1);
  assert.deepEqual(answer.removed, []);
});

test("a quote may differ from the source in whitespace only", () => {
  validateDraft(draft({ quotes: [{ source: "S1", text: "established,   implemented,\ndocumented" }] }), sources);
  const violations = violationsOf(() =>
    validateDraft(draft({ quotes: [{ source: "S1", text: "established, implemented and documented" }] }), sources),
  );
  assert.match(violations[0] ?? "", /not a verbatim excerpt/);
});

test("rejects a quote or statement that cites a source id that was not retrieved", () => {
  const fromQuote = violationsOf(() => validateDraft(draft({ quotes: [{ source: "S9", text: "anything" }] }), sources));
  assert.match(fromQuote[0] ?? "", /"S9", which was not retrieved/);
  const fromStatement = violationsOf(() =>
    validateDraft(draft({ statements: [{ text: "A claim.", sources: ["S1", "S7"] }] }), sources),
  );
  assert.match(fromStatement[0] ?? "", /"S7", which was not retrieved/);
});

test("rejects a quote attributed to the wrong source", () => {
  const violations = violationsOf(() =>
    validateDraft(draft({ quotes: [{ source: "S2", text: "shall be established" }] }), sources),
  );
  assert.equal(violations.length, 1);
});

test("rejects an empty quote", () => {
  const violations = violationsOf(() => validateDraft(draft({ quotes: [{ source: "S1", text: "  " }] }), sources));
  assert.match(violations[0] ?? "", /is empty/);
});

test("removes and reports statements that cite nothing", () => {
  const answer = validateDraft(
    draft({
      statements: [
        { text: "Supported.", sources: ["S1"] },
        { text: "Unsupported general knowledge.", sources: [] },
      ],
    }),
    sources,
  );
  assert.deepEqual(answer.statements, [{ text: "Supported.", sources: ["S1"] }]);
  assert.deepEqual(answer.removed, [{ text: "Unsupported general knowledge.", reason: "uncited" }]);
});

test("removes statements that match a forbidden pattern", () => {
  const answer = validateDraft(
    draft({ statements: [{ text: "The system is high-risk.", sources: ["S1"] }] }),
    sources,
    { forbidden: [/is high-risk/i] },
  );
  assert.equal(answer.statements.length, 0);
  assert.equal(answer.removed[0]?.reason, "forbidden");
  assert.equal(answer.status, "no_support");
});

test("rejects a statement that embeds a quotation its sources do not contain", () => {
  const violations = violationsOf(() =>
    validateDraft(
      draft({ statements: [{ text: 'The article says "providers must register every system in a database".', sources: ["S1"] }] }),
      sources,
    ),
  );
  assert.match(violations[0] ?? "", /quotation that is not in the sources/);
  // The same quotation is fine when it really is in a cited source.
  validateDraft(
    draft({ statements: [{ text: 'It refers to an "essentially equivalent level of protection".', sources: ["S2"] }] }),
    sources,
  );
});

test("no_support drafts and drafts left empty by the check both end as no_support", () => {
  assert.equal(validateDraft(draft({ status: "no_support", quotes: [{ source: "S9", text: "x" }] }), sources).status, "no_support");
  assert.equal(validateDraft(draft({ statements: [{ text: "Uncited.", sources: [] }] }), sources).status, "no_support");
});

test("parseDraft rejects a reply with the wrong shape", () => {
  assert.throws(() => parseDraft({ status: "answered", quotes: "none", statements: [] }), GroundingError);
  assert.throws(() => parseDraft({ status: "maybe", quotes: [], statements: [] }), GroundingError);
  assert.throws(() => parseDraft({ status: "answered", quotes: [], statements: [{ text: "x", sources: [1] }] }), GroundingError);
  assert.throws(() => parseDraft(null), GroundingError);
});

test("the rendered answer separates quotes from model text and cites CELEX, article, ECLI and link", () => {
  const answer = validateDraft(
    draft({
      quotes: [{ source: "S1", text: "shall be established" }],
      statements: [{ text: "A note.", sources: ["S2"] }],
    }),
    sources,
  );
  const out = renderAnswer(answer, sources, { quotes: "Quoted", statements: "Model-written" });
  assert.ok(out.indexOf("## Quoted") < out.indexOf("## Model-written"));
  assert.match(out, /> "shall be established" \[S1\]/);
  assert.match(out, /- A note\. \[S2\]/);
  assert.match(out, /\[S1\] Fixture act - Article 9 \(article id art_9\) - CELEX 32024R1689 - https:\/\/eur-lex\.europa\.eu/);
  assert.match(out, /\[S2\] Fixture judgment - paragraph 3 - ECLI:EU:C:2020:559 - CELEX 62018CJ0311/);
  assert.match(out, /Note: Consolidated text; not the authentic text\./);
});
