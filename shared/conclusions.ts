/**
 * The conclusion filter: wording that tells the reader what the law requires
 * of THEM, or how THEIR system is classified. The examples show sources and
 * leave those conclusions to a qualified person, so a statement that matches
 * is removed (only a count is shown).
 *
 * A blunt lexical filter for English output. It catches common phrasings,
 * not every one, and nothing in other languages.
 *
 * A match is ignored only when it sits in a clause that frames a question
 * ("whether the tool falls within Annex III is for a qualified person to
 * decide"). All of these must hold:
 *   - the nearest earlier `whether`, `if`, `when`, `how` or `which` of the
 *     same sentence is not followed by `means`, `is why`, `makes` or `shows`;
 *   - between that word and the match there is no clause break
 *     (`, ; : ( )`, a dash) and no `you`, `your` or `yours`;
 *   - for `if` and `when`, a clause break follows the match in the sentence
 *     ("If the tool is high-risk, Article 9 describes ...").
 *
 * Quotations already verified as verbatim from a cited source are masked
 * before the filter runs, so the law's own words do not get a statement removed.
 *
 * The Python twin is `gdpr_assistant/conclusions.py`; both run the cases in
 * `shared/grounding-vectors.json`.
 */

const DUTY = "(?:must|shall|should|has to|have to|needs? to|will need to|will have to|(?:is|are) (?:required|obliged|obligated) to)";
const RISK_CLASS = "(?:high[- ]risk|prohibited|limited[- ]risk|minimal[- ]risk|low[- ]risk)";
const SUBJECT = "(?:system|systems|tool|tools|product|provider|providers|deployer|deployers|company|operator|operators)";

/** Addressed to the reader: what "you" must do, whether "you" comply, advice. Used by all three examples. */
export const READER_PATTERNS: string[] = [
  `\\byou ${DUTY}\\b`,
  `\\byour (?:system|tool|product|company|organisation|organization|processing|transfer|use case) (?:${DUTY}|is|are)\\b`,
  "\\byour [a-z-]+(?:'s)? (?:[a-z-]+ )?(?:is|are) (?:not )?(?:lawful|unlawful|legal|illegal|allowed|permitted|prohibited|covered)\\b",
  "\\b(?:is|are) (?:not )?(?:compliant|non-compliant|in compliance)\\b",
  "\\b(?:is|are) not (?:needed|required|necessary) for your\\b",
  "\\bappl(?:y|ies) to (?:you|your)\\b",
  "\\bwe (?:recommend|advise|suggest)\\b",
  "\\bthe safest course\\b",
  "\\bin your (?:position|case|situation)\\b",
];

/** AI Act classification and duty conclusions. Used by the AI Act provision finder on top of the reader patterns. */
export const AI_ACT_PATTERNS: string[] = [
  `\\b(?:is|are|was|will be|would be) (?:not )?(?:an? )?(?:(?:classified|considered|regarded|deemed|treated) (?:as )?)?(?:an? )?${RISK_CLASS}\\b`,
  `\\b(?:qualif(?:y|ies)|counts?|classif(?:y|ies)|classified) as (?:an? )?${RISK_CLASS}\\b`,
  "\\bfalls? (?:under|within) (?:annex|article|the (?:scope|category|definition))\\b",
  "\\b(?:is|are) (?:not )?exempt\\b",
  "\\b(?:is|are) (?:mandatory|obligatory|compulsory)\\b",
  "\\b(?:is|are) (?:obliged|obligated|required) to\\b",
  `\\b${DUTY} comply\\b`,
  `\\b${DUTY} be registered\\b`,
  `\\b${SUBJECT} ${DUTY}\\b`,
  "\\bappl(?:y|ies) to (?:the|this) (?:system|tool|product|company)\\b",
];

const HEDGE = /\b(whether|if|when|how|which)\b/gi;
const HEDGE_AS_ASSERTION = /^ (?:means|is why|makes|shows)\b/i;
const CLAUSE_BREAK = /[,;:()–—]| - /;
const SECOND_PERSON = /\b(?:you|your|yours)\b/i;

function sentenceStart(text: string, index: number): number {
  const before = text.slice(0, index);
  return Math.max(before.lastIndexOf(". "), before.lastIndexOf("! "), before.lastIndexOf("? ")) + 1;
}

function sentenceEnd(text: string, index: number): number {
  const ends = [". ", "! ", "? "].map((mark) => text.indexOf(mark, index)).filter((at) => at !== -1);
  return ends.length > 0 ? Math.min(...ends) : text.length;
}

/** Is the match at [start, end) inside a clause that frames a question rather than answering it? */
function framedAsQuestion(text: string, start: number, end: number): boolean {
  const from = sentenceStart(text, start);
  const hedges = [...text.slice(from, start).matchAll(HEDGE)];
  const hedge = hedges[hedges.length - 1];
  if (!hedge) return false;
  const hedgeEnd = from + hedge.index + hedge[0].length;
  const between = text.slice(hedgeEnd, start);
  if (HEDGE_AS_ASSERTION.test(between) || CLAUSE_BREAK.test(between) || SECOND_PERSON.test(between)) return false;
  const word = hedge[0].toLowerCase();
  if (word === "if" || word === "when") return CLAUSE_BREAK.test(text.slice(end, sentenceEnd(text, end)));
  return true;
}

/** Does `text` (already normalised) state a conclusion under one of `patterns`? */
export function statesConclusion(text: string, patterns: string[]): boolean {
  for (const pattern of patterns) {
    for (const match of text.matchAll(new RegExp(pattern, "gi"))) {
      if (!framedAsQuestion(text, match.index, match.index + match[0].length)) return true;
    }
  }
  return false;
}
