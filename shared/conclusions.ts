/**
 * The conclusion filter: wording that tells the reader what the law requires
 * of THEM, or how THEIR system is classified. The examples show sources and
 * leave those conclusions to a qualified person, so a statement that matches
 * is removed (only a count is shown).
 *
 * A blunt lexical filter for English output. It catches common phrasings,
 * not every one, and nothing in other languages. A match is ignored when an
 * earlier word of the same sentence is `whether`, `if`, `how`, `which`,
 * `may`, `might` or `could`: "whether the tool falls within Annex III is for
 * a qualified person to decide" frames a question, it does not answer it.
 *
 * The Python twin is `gdpr_assistant/conclusions.py`; both run the cases in
 * `shared/grounding-vectors.json`.
 */

const DUTY = "(?:must|shall|should|has to|have to|needs? to|(?:is|are) (?:required|obliged|obligated) to)";
const RISK_CLASS = "(?:high[- ]risk|prohibited|limited[- ]risk|minimal[- ]risk|low[- ]risk)";
const SUBJECT = "(?:system|systems|tool|tools|product|provider|providers|deployer|deployers|company|operator|operators)";

/** Addressed to the reader: what "you" must do, whether "you" comply. Used by every example with the filter on. */
export const READER_PATTERNS: string[] = [
  `\\byou ${DUTY}\\b`,
  `\\byour (?:system|tool|product|company|organisation|organization|processing|transfer|use case) (?:${DUTY}|is|are)\\b`,
  "\\b(?:is|are) (?:not )?(?:compliant|non-compliant|in compliance)\\b",
  "\\bappl(?:y|ies) to (?:you|your)\\b",
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
  `\\b${SUBJECT} ${DUTY}\\b`,
  "\\bappl(?:y|ies) to (?:the|this) (?:system|tool|product|company)\\b",
];

const HEDGE = /\b(?:whether|if|how|which|may|might|could)\b/i;

/** Does `text` (already normalised) state a conclusion under one of `patterns`? */
export function statesConclusion(text: string, patterns: string[]): boolean {
  for (const pattern of patterns) {
    for (const match of text.matchAll(new RegExp(pattern, "gi"))) {
      const before = text.slice(0, match.index);
      const sentenceStart = Math.max(before.lastIndexOf(". "), before.lastIndexOf("; "), before.lastIndexOf("! "), before.lastIndexOf("? ")) + 1;
      if (!HEDGE.test(before.slice(sentenceStart))) return true;
    }
  }
  return false;
}
