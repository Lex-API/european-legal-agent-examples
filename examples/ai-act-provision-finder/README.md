# AI Act provision finder (TypeScript)

Give it a plain-language description of an AI system. It retrieves the articles of the EU AI Act
(Regulation (EU) 2024/1689) that may be relevant, quotes them with article-level citations, and
says what a human has to decide.

It finds provisions to read. It does not find the obligations that apply.

Node.js 20.12 or later. The model is Anthropic's (default) or OpenAI's, through their SDKs.

## What it does not do

- It does not classify the system (prohibited, high-risk, ...) and does not say which obligations
  apply. It shows text and frames open questions; a qualified person decides.
- It is not legal advice and does not assess compliance.
- It only sees the articles the search returned (at most 8). An article that is not in the output
  may still be relevant.

## How it works

`src/agent.ts` is the whole loop:

1. **Plan** - the model turns the description into up to three search queries.
2. **Retrieve** - `semanticLegislationSearch({ query, limit: 10, includeText: true })` per query.
   The search is not limited to the AI Act, so hits are filtered in code to the AI Act
   (`baseCelex ?? law_id` equal to `32024R1689`), one per article, best score first.
3. **Draft** - the model gets the retrieved articles as numbered sources (`S1`, `S2`, ...) and
   returns excerpts plus notes, each naming its sources.
4. **Check** - code validates the draft before anything is shown.
   The check (`shared/grounding.ts`) **rejects** the draft if:
   - it cites a source id that was not retrieved in this run;
   - a quote is shorter than three words, or is not a verbatim excerpt of the source it names;
   - a quotation inside a note - any quotation that contains a letter - is not verbatim in a
     source that note cites, or a quotation mark is left open. Words from the system description you gave are not a
     source: the prompt tells the model to paraphrase them without quotation marks;
   - a quote or note contains a line break or a control or format character (terminal escapes,
     zero-width and bidi characters), or starts with a heading, blockquote, table or code-fence
     marker;
   - a note contains an HTML character reference (`&quot;`), a URL that is not the URL of a
     source it cites, or a citation marker of its own: a bracketed token (`[ ]`, `［ ］`, `⟦ ⟧`,
     `【 】`, `〔 〕`) whose content, after NFKC normalisation and with spaces removed, is
     `S<digits>` (or a list such as `S1,S2`) or starts with `Source`;
   - there are more than 12 quotes or 12 notes, a quote is longer than 1,200 characters, or a
     note is longer than 600.

   "Verbatim" means that, after normalisation, the same words appear in the source in the same
   order, starting and ending on a word boundary ("lawful" is not found in "unlawful").
   Normalisation collapses whitespace, drops soft hyphens and maps these quotation marks to `"`
   or `'`: `“ ” „ ‟ « » ＂ ❝ ❞ 「 」 『 』 ″ 〝 〞 〟`, `‘ ’ ‚ ‛ ‹ › ′ ＇ ❛ ❜`, the backtick and the
   acute accent, a doubled `''`, and `,,` used as an opening mark. Punctuation and ellipses at the
   two ends of a quotation are ignored, and the case of its first letter may differ. An apostrophe
   inside a word (`provider's`, `don't`, `subjects'`) is not a quotation mark. A quote is printed
   with the source's own characters, without the model's end punctuation. Other quotation
   conventions are not recognised. An empty note is dropped.

   It **removes**, and only counts:
   - notes that cite no source;
   - notes that match the conclusion filter (`shared/conclusions.ts`): classifications ("is
     high-risk", "falls under Annex III"), duties ("the provider must", "you need to", "must
     comply"), applicability ("applies to your system") and advice ("we recommend").
     A match is ignored only when it sits in a clause that frames a question: the nearest
     earlier `whether`, `if`, `when`, `how` or `which` of the same sentence is not followed by
     `means`, `is why`, `makes` or `shows`; between that word and the match there is no clause
     break (`, ; : ( )` or a dash) and no `you`, `your` or `yours`; and, for `if` and `when`, a
     clause break follows the match ("If the tool is high-risk, Article 9 describes ..."). A
     quotation already verified as verbatim from a cited source is masked before the filter runs.
     This is a blunt lexical filter for English output: it does not catch every phrasing, it
     removes some legitimate sentences, and it does not cover other languages.

   A rejected draft is retried once with the violations; after a second failure only the
   retrieved sources are printed.
5. **Render** - quoted source text and model-written notes go in separate sections.
   Everything printed that this program did not write itself goes through one function
   (`plainText` in `shared/text.ts`): the model's notes and search queries, your description, and
   every string from the API (titles, article ids, CELEX, ECLI, notes). It removes control and
   format characters, collapses whitespace and escapes Markdown, so none of it can pass for a
   heading or a quote. A link is printed only if it parses as an http(s) URL.
   Removed notes are counted, not printed. Every source line carries the CELEX number, the
   article id and a EUR-Lex link.

If the search returns no AI Act article, the output says "No supporting text found" and the model
is not asked to draft anything.

A hit can be a **consolidated text** (`consolidated: true`). Consolidated texts have no legal
force; the output repeats the API's `consolidatedNote` and names the authentic act (`baseCelex`)
as published in the Official Journal.

The grounding check confirms that citations are real and excerpts are verbatim. It does not
confirm that an excerpt is quoted in context, or that a model-written note reads the article
correctly.

## Run it

From the repository root, after `npm ci`:

```bash
# No keys: local stub + scripted model replies (synthetic texts)
npm run ai-act -- --dry-run "A SaaS tool that ranks job applicants' CVs and chats with candidates"

# For real, with your own keys
cp examples/ai-act-provision-finder/.env.example examples/ai-act-provision-finder/.env   # then fill it in
npm run ai-act -- "A SaaS tool that ranks job applicants' CVs and chats with candidates"
```

Options: `--max-queries N` (1-3, default 3), `--via mcp` (below).

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
  characters.
- **A key or setting is missing or invalid**: one line naming it, before any request. `LLM_PROVIDER`
  is trimmed and its case ignored; empty means `anthropic`; any value but `anthropic` or `openai` is
  a configuration error.
- **The MCP server process ended or did not answer in time** (`--via mcp`):
  `MCP server error: <message>`; the server process is stopped.

The SDK retries rate limits and gateway or connection errors by itself (up to 3 retries, each wait
capped at 30 s) and waits up to 95 s for each answer, so a refused or hanging call can take a
minute and a half or more before its line is printed.

Exit codes: 0 printed; 1 LexAPI or the model failed; 2 a usage or configuration error, found before
any request. With `--via mcp`, a refused call prints the MCP tool's own error text instead,
on one line, with the same exit code; an answer from the tool that cannot be read is an error too,
not "no supporting text".

### The MCP route

`--via mcp` runs the same loop but retrieves through the LexAPI MCP server instead of the SDK:
`src/retrieve-mcp.ts` starts `@lexapi/mcp` (0.5.1) as a stdio child process and calls its
`lex_semantic_legislation` tool. The steps after retrieval are the same code, with two differences
in what they receive:

- The tool shares a character budget between the hit texts (the example asks for the maximum,
  200,000) and cuts a longer article, appending a marker. The example strips the marker, so it is
  never treated as source text, and labels that source "Text truncated by the tool". Nothing from
  the missing part can be quoted.
- The tool result does not include the credits charged, so the run prints "not reported" instead
  of a figure.

```bash
npm run ai-act -- --via mcp "A SaaS tool that ranks job applicants' CVs and chats with candidates"
```

## Cost

LexAPI credits per run, from the [documented weights](https://lex-api.com/docs?ref=github-examples):
one semantic legislation search is 5 credits, so a default run (3 queries) is **15 credits** and
`--max-queries 1` is 5. A run makes at most 3 API requests. Model usage is billed separately by
your model provider.

## Tests

`npm test` runs `test/agent.test.ts` offline (local stub, scripted model): the answered path
through the SDK and through the real `@lexapi/mcp` server process, the two "no supporting text"
paths, rejection and retry of an ungrounded draft, a draft and a search query shaped like a fake
quoted section, every path by which model text reaches the output, ten well-behaved drafts in
the styles models write, the sources-only fallback, removal of conclusion statements, the query
bound, the MCP truncation handling, and `test/cli.test.ts`: the command line run as a process
against the stub and a local stand-in for the model API (exit code, stdout empty, no stack trace)
for each refusal (credits exhausted, an exhausted quota, payment past due, a rate limit), a refusal
after a charged search, an unreadable answer, a model-provider error, missing keys (exit 2, no
request), an unknown `LLM_PROVIDER`, and through the MCP route a refusal (also after a charged
search), an unreadable answer, a server process that dies mid-call and one that does not answer. The grounding rules themselves are tested from
`shared/grounding-vectors.json`, which the Python suite runs too. Not covered by any test: the live API and a real model.

## Sample output (illustrative)

Generated with `--dry-run`. The "legal texts" are synthetic placeholders from
`offline/fixtures`, and the notes are scripted, not written by a model.

```markdown
> DRY RUN - LexAPI was replaced by a local stub and the model by scripted replies.
> The legal texts below are synthetic placeholders, not the wording of any act or judgment.

# AI Act provision finder

**System described:** A SaaS tool that ranks job applicants' CVs and chats with candidates

Searches run (@lexapi/client): "classification of AI systems used to screen job applicants"; "risk management requirements for AI systems"; "informing people that they are interacting with an AI system"
Hits: 12 in total, 3 from other acts (not shown). Model: scripted (no model was called).

## Provisions that may be relevant (quoted from the source text)

> "fixture wording about how an example system is sorted into a placeholder category" [S1]

> "fixture wording about keeping a placeholder risk process running for the example system" [S2]

> "fixture wording about telling people that they are dealing with an example system" [S3]

## Why they may be relevant (written by the model - not legal text)

- This article sets out how a system is assigned to a category; whether a CV-screening tool is covered by it is a question for a qualified person. [S1]
- If the system were found to fall in that category, this article describes a risk management process that may be worth reading. [S2]
- The tool includes a chat interface for candidates, so the provision on telling people they are interacting with an AI system may be worth reading. [S3]

## Removed by the grounding check

- 1 statement(s) removed: cited no source.

## Sources

- [S1] Regulation (EU) 2024/1689 (Artificial Intelligence Act) (fixture title) - Article 6 (article id art\_6) - CELEX 32024R1689 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
- [S2] Regulation (EU) 2024/1689 (Artificial Intelligence Act) (fixture title) - Article 9 (article id art\_9) - CELEX 32024R1689 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
- [S3] Regulation (EU) 2024/1689 (Artificial Intelligence Act), consolidated (fixture title) - Article 50 (article id art\_50) - CELEX 02024R1689-20990101 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02024R1689-20990101
  - Note: The act as amended, consolidated as of 2099-01-01; not the authentic text. Only the Official Journal text is authentic. Authentic act: CELEX 32024R1689 as published in the Official Journal (https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689).

## What a human must decide

This output is a reading list, not an assessment. It does not say whether the described system is covered by the AI Act, which category it falls into, or which obligations apply. Those are legal questions for a qualified person, who should read the provisions above in full, in the authentic Official Journal text, together with the rest of the regulation.
```
