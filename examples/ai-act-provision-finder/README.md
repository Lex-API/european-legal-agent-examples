# AI Act provision finder (TypeScript)

Give it a plain-language description of an AI system. It retrieves the articles of the EU AI Act
(Regulation (EU) 2024/1689) that may be relevant, quotes them with article-level citations, and
says what a human has to decide.

It finds provisions to read. It does not find the obligations that apply.

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
   The search covers all EU legislation, so hits are filtered in code to the AI Act
   (`baseCelex ?? law_id` equal to `32024R1689`), one per article, best score first.
3. **Draft** - the model gets the retrieved articles as numbered sources (`S1`, `S2`, ...) and
   returns excerpts plus notes, each naming its sources.
4. **Check** - code validates the draft before anything is shown.
   The check (`shared/grounding.ts`) rejects the draft if:
   - it cites a source id that was not retrieved in this run;
   - a quote is shorter than three words, or is not a verbatim excerpt of the source it names;
   - a quotation inside a note (two words or more, or eight characters or more, in any style of
     quotation marks) is not verbatim in a source that note cites, or a quotation mark is left open;
   - a quote or note is not one line of plain text (a line break, or a leading heading,
     blockquote, list, table or code-fence marker).

   "Verbatim" is compared after collapsing whitespace, unifying quotation-mark styles and dropping
   soft hyphens on both sides.

   Notes without a source are removed and listed. Notes that match a list of conclusion
   phrasings ("is high-risk", "falls under Annex III", "the provider must ...", "you need to
   ...") are removed too. That list is a blunt lexical filter for English output: it does not
   catch every phrasing, and it does not cover other languages. A rejected draft is retried once
   with the violations; after a second failure only the retrieved sources are printed.
5. **Render** - quoted source text and model-written notes go in separate sections, and Markdown
   in anything the model wrote is escaped, so a note cannot pass for a quote. Every source line
   carries the CELEX number, the article id and a EUR-Lex link.

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
paths, rejection and retry of an ungrounded draft, a draft shaped like a fake quoted section,
the sources-only fallback, removal of conclusion statements, the query bound, and the MCP
truncation handling. Not covered by any test: the live API and a real model.

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

- (cited no source) Screening tools are widely used in recruitment.

## Sources

- [S1] Regulation (EU) 2024/1689 (Artificial Intelligence Act) [fixture title] - Article 6 (article id art_6) - CELEX 32024R1689 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
- [S2] Regulation (EU) 2024/1689 (Artificial Intelligence Act) [fixture title] - Article 9 (article id art_9) - CELEX 32024R1689 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
- [S3] Regulation (EU) 2024/1689 (Artificial Intelligence Act), consolidated [fixture title] - Article 50 (article id art_50) - CELEX 02024R1689-20990101 - https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02024R1689-20990101
  - Note: The act as amended, consolidated as of 2099-01-01; not the authentic text. Only the Official Journal text is authentic. Authentic act: CELEX 32024R1689 as published in the Official Journal (https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689).

## What a human must decide

This output is a reading list, not an assessment. It does not say whether the described system is covered by the AI Act, which category it falls into, or which obligations apply. Those are legal questions for a qualified person, who should read the provisions above in full, in the authentic Official Journal text, together with the rest of the regulation.
```
