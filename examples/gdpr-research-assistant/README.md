# GDPR research assistant (Python)

Ask a question about the GDPR (Regulation (EU) 2016/679). It answers with quotations from the
regulation and from CJEU judgments, each with its CELEX number or ECLI and a EUR-Lex link, plus a
short model-written summary kept apart from the quotes.

## What it does not do

- It is research support, not legal advice. It does not say what you must do or whether you comply.
- It only sees what the searches returned: at most 6 GDPR articles and a few paragraphs from at
  most 2 judgments. Relevant law outside that set is not considered.
- It quotes judgments from their own numbered paragraphs only. The `summary` and `snippet` of a
  case-law search hit are machine-written digests, so they are never offered to the model as
  quotable text.

## How it works

`src/gdpr_assistant/agent.py` is the loop; `retrieve.py` does the API calls:

1. **Retrieve legislation** - `semantic_legislation_search(question, limit=10, include_text=True)`,
   filtered in code to GDPR articles.
2. **Retrieve case law** - `semantic_search(question, limit=5)`, then for the top 2 cases
   `get_document(celex, include=["metadata", "paragraphs"])`. Paragraphs are chosen around the
   matched paragraph when the hit reports one, otherwise by word overlap with the question. A
   judgment whose text is not available is skipped and reported with the API's reason.
3. **Draft** - the model gets the articles and paragraphs as numbered sources and returns quotes
   plus summary sentences, each naming its sources.
4. **Check** - code validates the draft before anything is shown.
   The check (`grounding.py`) rejects the draft if:
   - it cites a source id that was not retrieved in this run;
   - a quote is shorter than three words, or is not a verbatim excerpt of the source it names;
   - a quotation inside a sentence (two words or more, or eight characters or more, in any style of
     quotation marks) is not verbatim in a source that sentence cites, or a quotation mark is left open;
   - a quote or sentence is not one line of plain text (a line break, or a leading heading,
     blockquote, list, table or code-fence marker).

   "Verbatim" is compared after collapsing whitespace, unifying quotation-mark styles and dropping
   soft hyphens on both sides.

   Sentences without a source are removed and listed. A rejected draft is retried once; after a
   second failure only the retrieved sources are printed.
5. **Render** - "Quoted from the sources" and "Summary (written by the model)" are separate
   sections, and Markdown in anything the model wrote is escaped; every source line carries
   CELEX, article id or paragraph, ECLI and link.

If nothing is retrieved, or the model reports that the sources do not answer the question, the
output says "No supporting text found" and gives no answer.

The check confirms that citations are real and excerpts are verbatim. It does not confirm that
an excerpt is quoted in context, or that the summary reads the sources correctly.

### Versions and dates

- A **consolidated** hit is labelled with the API's `consolidatedNote` and the authentic act
  (`baseCelex`): consolidated texts have no legal force, only the Official Journal text does.
- `--as-of YYYY-MM-DD` passes `as_of` to the legislation search. A source is described as the text
  in force on that date only when the API returns `asOfResolved: true`. Otherwise its note says the
  version for that date is not established (with the API's reason) and that the text shown is the
  latest one held.
- Without `--as-of`, the output makes no statement about the text in force on any date.

## Run it

```bash
cd examples/gdpr-research-assistant
python -m venv .venv && . .venv/bin/activate      # Python 3.10+
pip install -e ".[anthropic]"                     # or ".[openai]"; keep -e: --dry-run reads the repository's fixtures

# No keys: local stub + scripted model replies (synthetic texts)
gdpr-assistant --dry-run "When may personal data be transferred to a third country?"

# For real, with your own keys
cp .env.example .env                              # then fill it in
gdpr-assistant "When may personal data be transferred to a third country?"
gdpr-assistant --as-of 2019-06-01 "When may personal data be transferred to a third country?"
```

## Cost

LexAPI credits per question, from the [documented weights](https://lex-api.com/docs?ref=github-examples):

| Step | Credits |
| --- | --- |
| Semantic legislation search | 5 |
| Semantic case-law search | 5 |
| Up to 2 judgment reads | 5 each, settled to 4 or 1 depending on where the text is served from (1 when it has no text) |
| With `--as-of`: up to 3 version fetches | 0-5 each |

So **12-20 credits** per question, up to 35 with `--as-of`, in 4 requests. The run prints the
figure the API reported, which is the upfront charge: refunds settle after the response. If a
response carries no credits figure, the run says "not reported" rather than 0. Model
usage is billed separately by your model provider.

## Tests

`pytest` runs offline (local stub, scripted model; name lookups and connections to anything but
127.0.0.1 are blocked): the answered path with the exact requests sent, the `--as-of` labels, both
"no supporting text" paths, rejection and retry of an ungrounded draft, a draft shaped like a fake
quoted section, the sources-only fallback, "not reported" credits, paragraph selection, the
grounding rules, and the model adapters against fake provider clients.
Not covered by any test: the live API and a real model.

## Sample output (illustrative)

Generated with `--dry-run`. The "legal texts" are synthetic placeholders from
`offline/fixtures`, and the summary is scripted, not written by a model.

```markdown
> DRY RUN - LexAPI was replaced by a local stub and the model by scripted replies.
> The legal texts below are synthetic placeholders, not the wording of any act or judgment.

# GDPR research assistant

**Question:** When may personal data be transferred to a third country?

Model: scripted (no model was called).

## Quoted from the sources

> "fixture wording about sending fixture data abroad when placeholder safeguards exist" [S1]

> "fixture wording saying no separate permission is needed in that case" [S2]

> "a level of protection comparable to the placeholder standard" [S4]

## Summary (written by the model - not legal text)

- The retrieved articles describe two routes for a transfer: an adequacy finding for the third country, or safeguards. [S1][S2]
- The retrieved judgment paragraphs address the level of protection the safeguards have to give and what is weighed. [S4][S5]

## Removed by the grounding check

- (cited no source) Most companies rely on standard contractual clauses.

## Sources

- [S1] Regulation (EU) 2016/679 (General Data Protection Regulation), consolidated [fixture title] - Article 46 (article id art_46) - CELEX 02016R0679-20990101 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02016R0679-20990101
  - Note: The act as amended, consolidated as of 2099-01-01; not the authentic text. Only the Official Journal text is authentic. Authentic act: CELEX 32016R0679 as published in the Official Journal (https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679).
- [S2] Regulation (EU) 2016/679 (General Data Protection Regulation) [fixture title] - Article 45 (article id art_45) - CELEX 32016R0679 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679
- [S4] Judgment of the Court of 1 January 2099 [fixture title] - paragraph 3 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901 - https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62099CJ9901
- [S5] Judgment of the Court of 1 January 2099 [fixture title] - paragraph 4 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901 - https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62099CJ9901

## Notices

- 1 legislation hit(s) from acts other than the GDPR were not used.
- ECLI:EU:T:2099:902: found by the search, but its text is not available (not_in_language).

_Research support, not legal advice. The summary is model-written and may misread the sources; check it against the quoted text and the linked documents._
```
