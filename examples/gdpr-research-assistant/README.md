# GDPR research assistant (Python)

Ask a question about the GDPR (Regulation (EU) 2016/679). It answers with quotations from the
regulation and from CJEU judgments, each with its CELEX number or ECLI and a EUR-Lex link, plus a
short model-written summary kept apart from the quotes.

Python 3.10 or later. The model is Anthropic's (default) or OpenAI's, through their SDKs.

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
   The check (`grounding.py`) **rejects** the draft if:
   - it cites a source id that was not retrieved in this run;
   - a quote is shorter than three words, or is not a verbatim excerpt of the source it names;
   - a quotation inside a sentence - any quotation that contains a letter - is not verbatim in a
     source that sentence cites, or a quotation mark is left open. Words from your question are not a
     source: the prompt tells the model to paraphrase them without quotation marks;
   - a quote or sentence contains a line break or a control or format character (terminal escapes,
     zero-width and bidi characters), or starts with a heading, blockquote, table or code-fence
     marker;
   - a sentence contains an HTML character reference (`&quot;`), a URL that is not the URL of a
     source it cites, or a citation marker of its own: a bracketed token (`[ ]`, `［ ］`, `⟦ ⟧`,
     `【 】`, `〔 〕`) whose content, after NFKC normalisation and with spaces removed, is
     `S<digits>` (or a list such as `S1,S2`) or starts with `Source`;
   - there are more than 12 quotes or 12 sentences, a quote is longer than 1,200 characters, or a
     sentence is longer than 600.

   "Verbatim" means that, after normalisation, the same words appear in the source in the same
   order, starting and ending on a word boundary ("lawful" is not found in "unlawful").
   Normalisation collapses whitespace, drops soft hyphens and maps these quotation marks to `"`
   or `'`: `“ ” „ ‟ « » ＂ ❝ ❞ 「 」 『 』 ″ 〝 〞 〟`, `‘ ’ ‚ ‛ ‹ › ′ ＇ ❛ ❜`, the backtick and the
   acute accent, a doubled `''`, and `,,` used as an opening mark. Punctuation and ellipses at the
   two ends of a quotation are ignored, and the case of its first letter may differ. An apostrophe
   inside a word (`provider's`, `don't`, `subjects'`) is not a quotation mark. A quote is printed
   with the source's own characters, without the model's end punctuation. Other quotation
   conventions are not recognised. An empty sentence is dropped.

   It **removes**, and only counts:
   - sentences that cite no source;
   - sentences that match the conclusion filter (`conclusions.py`): wording addressed to you
     ("you must", "you will need to", "your processing is lawful", "is compliant", "applies to
     you", "we recommend", "in your position").
     A match is ignored only when it sits in a clause that frames a question: the nearest
     earlier `whether`, `if`, `when`, `how` or `which` of the same sentence is not followed by
     `means`, `is why`, `makes` or `shows`; between that word and the match there is no clause
     break (`, ; : ( )` or a dash) and no `you`, `your` or `yours`; and, for `if` and `when`, a
     clause break follows the match ("If the tool is high-risk, Article 9 describes ..."). A
     quotation already verified as verbatim from a cited source is masked before the filter runs.
     This is a blunt lexical filter for English output: it does not catch every phrasing, it
     removes some legitimate sentences, and it does not cover other languages.
     Sentences that describe what the law requires of controllers in general are not filtered;
     keeping those apart from advice relies on the prompt.

   A rejected draft is retried once; after a second failure only the retrieved sources are printed.
5. **Render** - "Quoted from the sources" and "Summary (written by the model)" are separate
   sections. Everything printed that this program did not write itself goes through one function
   (`plain_text` in `text.py`): the model's sentences, your question, and every string from the
   API (titles, article ids, CELEX, ECLI, notes, hints). It removes control and format
   characters, collapses whitespace and escapes Markdown. A link is printed only if it parses as
   an http(s) URL. Removed sentences are counted, not printed. Every source line carries CELEX,
   article id or paragraph, ECLI and link.

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

## Install

`lexapi-client` 0.5.0 is not on PyPI yet. `pyproject.toml` installs it from its repository, pinned
to a commit, so installing this example needs `git`:

```
lexapi-client @ git+https://github.com/Lex-API/lexapi-python@e48dfdadf8fb4c2a05f4c6ec08f79146177a515e
```

## Run it

```bash
cd examples/gdpr-research-assistant
python3 -m venv .venv && . .venv/bin/activate     # Python 3.10+
pip install -e ".[anthropic]"                     # or ".[openai]"; keep -e: --dry-run reads the repository's fixtures

# No keys: local stub + scripted model replies (synthetic texts)
gdpr-assistant --dry-run "When may personal data be transferred to a third country?"

# For real, with your own keys
cp .env.example .env                              # then fill it in
gdpr-assistant "When may personal data be transferred to a third country?"
gdpr-assistant --as-of 2019-06-01 "When may personal data be transferred to a third country?"
```

### When a run fails

Nothing is printed on stdout; stderr gets one line per problem:

- **LexAPI refused a request** (for example 401, a key it does not accept; 402, credits exhausted;
  403, payment past due; 429, a rate limit or an exhausted quota): the line the LexAPI adapters
  for LangChain and LlamaIndex print, `LexAPI error <code> (HTTP <status>): <message>.`, followed,
  when the error carries them, by when the credits or the quota reset, when access ended and the
  billing link (payment past due), and the wait (a rate limit). After any 401 (a key it does not
  accept) and after credits exhausted, the dashboard link follows on a line of its own. An answer the example cannot read gives
  `LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read.` When earlier calls of
  the run were charged, their credits line comes first.
- **The model provider failed**: `Model error: <the provider's message>`, capped at 1,000
  characters; `Model error: HTTP <status>: <the response text>` when the answer carries no
  message; after a rejected key (401 or 403) it names the variable to check, never the key.
- **A key or setting is missing or invalid**: one line naming it, before any request. `LLM_PROVIDER`
  is trimmed and its case ignored; empty means `anthropic`; any value but `anthropic` or `openai` is
  a configuration error.

The SDK retries rate limits and gateway or connection errors by itself (up to 3 retries, each wait
capped at 30 s) and waits up to 95 s for each answer, so a refused or hanging call can take a
minute and a half or more before its line is printed.

Exit codes: 0 printed; 1 LexAPI or the model failed; 2 a usage or configuration error, found before
any request. Ctrl-C ends the run with exit code 130, without a traceback.

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
quoted section, every path by which model text reaches the output, ten well-behaved drafts, the
sources-only fallback, "not reported" credits, paragraph selection, the grounding rules (from
`shared/grounding-vectors.json`, which the TypeScript suite runs too), the model adapters
against fake provider clients and, in `tests/test_real_providers.py`, the real Anthropic and OpenAI
clients sending their requests to local stand-ins for both APIs, and `tests/test_errors.py`: the lines and exit code for each refusal
(credits exhausted, an exhausted quota, payment past due, a rate limit, a key error, a server error,
a timeout, an unreadable answer), a refusal after a charged call, a model-provider error, missing
keys and an unknown `LLM_PROVIDER` (exit 2, no request) and Ctrl-C, in process and with the command line run as a process
against the stub and a local stand-in for the model API.
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

- 1 statement(s) removed: cited no source.

## Sources

- [S1] Regulation (EU) 2016/679 (General Data Protection Regulation), consolidated (fixture title) - Article 46 (article id art\_46) - CELEX 02016R0679-20990101 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02016R0679-20990101
  - Note: The act as amended, consolidated as of 2099-01-01; not the authentic text. Only the Official Journal text is authentic. Authentic act: CELEX 32016R0679 as published in the Official Journal (https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679).
- [S2] Regulation (EU) 2016/679 (General Data Protection Regulation) (fixture title) - Article 45 (article id art\_45) - CELEX 32016R0679 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679
- [S4] Judgment of the Court of 1 January 2099 (fixture title) - paragraph 3 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901 - https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62099CJ9901
- [S5] Judgment of the Court of 1 January 2099 (fixture title) - paragraph 4 - ECLI:EU:C:2099:901 - CELEX 62099CJ9901 - https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62099CJ9901

## Notices

- 1 legislation hit(s) from acts other than the GDPR were not used.
- ECLI:EU:T:2099:902: found by the search, but its text is not available (not\_in\_language).

_Research support, not legal advice. The summary is model-written and may misread the sources; check it against the quoted text and the linked documents._
```
