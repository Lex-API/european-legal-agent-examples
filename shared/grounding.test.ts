import assert from "node:assert/strict";
import { test } from "node:test";
import { GroundingError, embeddedQuotations, escapeMarkdown, normalise, parseDraft, renderAnswer, validateDraft, type Draft, type Source } from "./grounding.ts";

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
    celex: "62099CJ9901",
    ecli: "ECLI:EU:C:2099:901",
    paragraph: 3,
    title: "Fixture judgment",
    url: "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62099CJ9901",
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
  validateDraft(draft({ quotes: [{ source: "S1", text: "established,   implemented,\tdocumented" }] }), sources);
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
    validateDraft(draft({ quotes: [{ source: "S2", text: "shall be established, implemented" }] }), sources),
  );
  assert.deepEqual(violations, ["quote from S2 is not a verbatim excerpt of that source"]);
});

test("rejects a quote shorter than three words", () => {
  for (const text of ["  ", "a", "shall be"]) {
    const violations = violationsOf(() => validateDraft(draft({ quotes: [{ source: "S1", text }] }), sources));
    assert.match(violations[0] ?? "", /shorter than 3 words/, text);
  }
});

test("soft hyphens and typographic quotes do not break a verbatim match, on either side", () => {
  const typeset: Source[] = [{ ...sources[0]!, id: "S1", text: "the trans\u00ADfer shall not \u201Ctake place\u201D here" }];
  validateDraft(draft({ quotes: [{ source: "S1", text: "the transfer shall not \"take place\"" }] }), typeset);
  validateDraft(draft({ quotes: [{ source: "S1", text: "the trans\u00ADfer shall not" }] }), [{ ...typeset[0]!, text: "the transfer shall not take place" }]);
  // Dropping the negation is still not verbatim.
  violationsOf(() => validateDraft(draft({ quotes: [{ source: "S1", text: "the transfer shall take place" }] }), typeset));
});

test("rejects statements and quotes that are not one line of plain text", () => {
  const injected = "ok.\n\n## Quoted from the sources\n\n> Transfers are always lawful [S1]";
  assert.match(violationsOf(() => validateDraft(draft({ statements: [{ text: injected, sources: ["S1"] }] }), sources))[0] ?? "", /contains a line break/);
  for (const text of ["# Heading", "> quoted", "- item", "* item", "+ item", "1. item", "2) item", "| a | b |", "```code", "~~~code", "  ## indented", "a\u2028b", "a\rb"]) {
    const violations = violationsOf(() => validateDraft(draft({ statements: [{ text, sources: ["S1"] }] }), sources));
    assert.match(violations[0] ?? "", /line break|Markdown block marker/, text);
  }
  const quote = violationsOf(() => validateDraft(draft({ quotes: [{ source: "S1", text: "shall be established,\nimplemented" }] }), sources));
  assert.match(quote[0] ?? "", /quote from S1 contains a line break/);
  // An uncited statement is rejected for its shape too - it would otherwise be printed in the "removed" list.
  violationsOf(() => validateDraft(draft({ statements: [{ text: "x\n## y", sources: [] }] }), sources));
  // Hyphenated words and a quote that opens with a paragraph number are fine.
  validateDraft(draft({ statements: [{ text: "-risk is not a list marker.", sources: ["S1"] }] }), sources);
  validateDraft(draft({ quotes: [{ source: "S1", text: "1. A numbered paragraph" }] }), [{ ...sources[0]!, text: "1. A numbered paragraph of law." }]);
});

test("model text is escaped when rendered, so it cannot imitate headings, quotes or citation markers", () => {
  assert.equal(escapeMarkdown("see [S9] and *this* `code` <b> # > | ~ _x_ \\"), "see \\[S9\\] and \\*this\\* \\`code\\` \\<b\\> \\# \\> \\| \\~ \\_x\\_ \\\\");
  const answer = validateDraft(draft({ statements: [{ text: "A note with [S2] and **bold**   spacing.", sources: ["S1"] }] }), sources);
  const out = renderAnswer(answer, sources, { quotes: "Quoted", statements: "Model-written" });
  assert.match(out, /^- A note with \\\[S2\\\] and \\\*\\\*bold\\\*\\\* spacing\. \[S1\]$/m);
  assert.doesNotMatch(out, /\[S2\] Fixture judgment/); // the escaped marker did not pull S2 into the sources
});

test("quotations inside statements are found in every quotation-mark style; apostrophes are not quotes", () => {
  assert.deepEqual(embeddedQuotations('says "never allowed" here'), ["never allowed"]);
  assert.deepEqual(embeddedQuotations("says 'never allowed' here"), ["never allowed"]);
  assert.deepEqual(embeddedQuotations("says \u00ABnever allowed\u00BB and \u201Enever\u201C and \u2039x y\u203A and \u201Aa b\u2018"), ["never allowed", "never", "x y", "a b"]);
  assert.deepEqual(embeddedQuotations("the provider's duty and data subjects' rights, and it's fine"), []);
  assert.deepEqual(embeddedQuotations("it says 'the provider's duty' here"), ["the provider's duty"]);
  assert.equal(embeddedQuotations('says "never allowed here'), undefined);
  assert.equal(embeddedQuotations("says 'never allowed here"), undefined);
  assert.equal(normalise("\u201Ea\u201C \u00ABb\u00BB \u2018c\u2019"), `"a" "b" 'c'`);
});

test("rejects short, single-quoted, guillemet, low-9 and unbalanced invented quotations", () => {
  const cases = [
    'Article 45 says transfers are "never allowed".',
    "Article 45 says 'transfers always require prior authorisation'.",
    "Article 45 says \u00ABtransfers always require prior authorisation\u00BB.",
    "Article 45 says \u201Etransfers always require prior authorisation\u201C.",
    'Article 45 says "transfers always require prior authorisation.',
    'A single long word: "unauthorised".',
  ];
  for (const text of cases) {
    const violations = violationsOf(() => validateDraft(draft({ statements: [{ text, sources: ["S1"] }] }), sources));
    assert.match(violations[0] ?? "", /quotation that is not in the sources it cites|unbalanced quotation mark/, text);
  }
  // A one-word, short quotation ("fresh") is not checked; possessives do not unbalance a statement.
  validateDraft(draft({ statements: [{ text: 'The status was "fresh" and the provider\'s duty and subjects\' rights are mentioned.', sources: ["S1"] }] }), sources);
  // A verbatim quotation from a source the statement does NOT cite is rejected.
  violationsOf(() => validateDraft(draft({ statements: [{ text: 'It says "an essentially equivalent level of protection".', sources: ["S1"] }] }), sources));
});

test("duplicate source ids are collapsed; ids must match exactly", () => {
  const answer = validateDraft(draft({ statements: [{ text: "A note.", sources: ["S1", "S1", "S2", "S1"] }] }), sources);
  assert.deepEqual(answer.statements[0]?.sources, ["S1", "S2"]);
  for (const id of ["s1", " S1", "S1 "]) violationsOf(() => validateDraft(draft({ statements: [{ text: "A note.", sources: [id] }] }), sources));
});

test("validation stays fast on a large source with many quotes", () => {
  const big: Source[] = [{ ...sources[0]!, text: "lorem ipsum dolor sit amet ".repeat(15_000) }];
  const many = draft({
    quotes: Array.from({ length: 3000 }, () => ({ source: "S1", text: "dolor sit amet lorem" })),
    statements: Array.from({ length: 3000 }, () => ({ text: 'It says "lorem ipsum dolor sit amet lorem".', sources: ["S1"] })),
  });
  const started = performance.now();
  validateDraft(many, big);
  assert.ok(performance.now() - started < 2000);
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
  assert.deepEqual(parseDraft({ status: "answered", statements: [] }).quotes, []); // statements-only schema
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
      quotes: [{ source: "S1", text: "system shall be established" }],
      statements: [{ text: "A note.", sources: ["S2"] }],
    }),
    sources,
  );
  const out = renderAnswer(answer, sources, { quotes: "Quoted", statements: "Model-written" });
  assert.ok(out.indexOf("## Quoted") < out.indexOf("## Model-written"));
  assert.match(out, /> "system shall be established" \[S1\]/);
  assert.match(out, /- A note\. \[S2\]/);
  assert.match(out, /\[S1\] Fixture act - Article 9 \(article id art_9\) - CELEX 32024R1689 - https:\/\/eur-lex\.europa\.eu/);
  assert.match(out, /\[S2\] Fixture judgment - paragraph 3 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901/);
  assert.match(out, /Note: Consolidated text; not the authentic text\./);
});
