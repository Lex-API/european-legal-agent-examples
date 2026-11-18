# European legal agent examples

Small, runnable example agents that ground their answers in citable EU law using
[LexAPI](https://lex-api.com/?ref=github-examples) - EUR-Lex legislation parsed to article level,
CJEU case law, a citation graph and semantic search behind one REST API.

Each example is an explicit loop you can read top to bottom - retrieve, answer with citations
(the first three also check that any quotation is verbatim), and say so when the sources do not support an
answer - with a check in code that every citation points at a source that was actually retrieved.

## Examples

| Example | Language | What it shows | LexAPI credits per run |
| --- | --- | --- | --- |
| [AI Act provision finder](examples/ai-act-provision-finder) | TypeScript | Describe an AI system; get the AI Act articles that may be relevant, quoted with article-level citations, and what a human must decide. It finds provisions to read, not the obligations that apply. Also runs through the MCP server. | 15 (5 per search) |
| [GDPR research assistant](examples/gdpr-research-assistant) | Python | Ask a question; get quotations from the GDPR and CJEU judgments with CELEX / ECLI and links, optionally for the text in force on a date. | 12-20 (up to 35 with `--as-of`) |
| [Regulatory change monitor](examples/regulatory-change-monitor) | TypeScript | Watch acts and topics for new or changed documents; get a cited digest. Polling by default, signed webhooks on paid plans. | 8 for the example watch list |
| [DORA questions answered from cited passages](examples/langchain-eu-law-rag) | Python (LangChain) | RAG over EU law with LangChain: ask about DORA; the official retriever fetches the articles, each with its Source line (CELEX, article, EUR-Lex link), and the model answers with `[1]`-style citations. Optionally for a date: passages without text for that date arrive as notices. Prints the credits used. | 5 (up to 20 with `--as-of`) |

Credit figures are computed from the weights in the
[API reference](https://lex-api.com/docs?ref=github-examples); each example's README shows the
arithmetic.

## Try one without any keys

Every example has a `--dry-run` mode that replaces LexAPI with a local stub and the model with
scripted replies, so you can see the loop and the output format first. The texts in a dry run are
synthetic placeholders.

```bash
npm ci
npm run ai-act -- --dry-run "A SaaS tool that ranks job applicants' CVs and chats with candidates"
npm run monitor -- --dry-run
```

The Python examples have their own setup; see their READMEs.

## Run one for real

You need:

- Node.js 20.12 or later (TypeScript examples; CI runs Node 20 and 22) and/or Python 3.10 or
  later (Python examples; the LangChain example needs 3.11 or later; CI runs Python 3.12).
- A LexAPI key from the [dashboard](https://lex-api.com/dashboard?ref=github-examples).
- A model provider key. Adapters are included for Anthropic (default model `claude-sonnet-5-5`,
  override with `ANTHROPIC_MODEL`) and OpenAI (`LLM_PROVIDER=openai` and `OPENAI_MODEL`). The
  LangChain example uses Anthropic only, through `langchain-anthropic`.

Copy the example's `.env.example` to `.env`, fill it in, and run the command in its README.

### On the FREE plan

The examples work on LexAPI's FREE plan, which per the API reference gives 500 credits in the
first cycle and 150 a month after that, at 15 requests a minute. No example makes more than a
handful of requests per run. Two limits to know: semantic search returns at most 10 hits per query
on FREE, and webhooks need a paid plan (the change monitor's polling mode does not).

## How the examples stay grounded

This section describes the first three examples. The LangChain example is a plain retrieval
pipeline: the model writes free text citing passages as `[n]`, and code checks only its citations:
any bracketed token made only of numeric characters and separators must be exactly `[n]` naming a
retrieved passage, and the answer must cite at least one unless it says the passages do not answer
the question (see its README).

- **Sources get run-local ids.** Retrieved articles, judgment paragraphs and document records are
  numbered `S1`, `S2`, ... and those ids are the only way the model can refer to them.
- **The model returns structure, not prose.** It submits quotes (source id + excerpt) and
  statements (text + source ids) through a tool call.
- **Code checks the draft** ([`shared/grounding.ts`](shared/grounding.ts) and its Python twin;
  both run the cases in [`shared/grounding-vectors.json`](shared/grounding-vectors.json)). A draft
  is rejected if it cites an id that was not retrieved; if a quote is shorter than three words or
  is not a verbatim excerpt, on word boundaries, of its source; if any quotation inside a
  statement is not verbatim in a source that statement cites (the user's own input is not a
  source), or a quotation mark is left open; if a quote or statement contains a line break, a
  control or format character or a leading block marker; if a statement contains an HTML
  character reference, a bracketed citation marker or a foreign URL; or if it exceeds the size
  caps (12 quotes, 12 statements, 1,200 and 600 characters). Statements with no source, and
  statements that match the conclusion filter, are removed and counted. The example READMEs give
  the rules in full, including which quotation-mark styles are recognised.
- **Output keeps law and model text apart.** Quoted source text and model-written text are in
  separate, labelled sections. A quote is printed with the source's own characters. Everything
  printed that the program did not write itself - model text, user input, and strings from the
  API or the configuration - goes through one function that removes control characters and
  escapes Markdown, so none of it can imitate a heading or the quoted section; removed
  statements are never printed. Every source line has the CELEX number, article id or paragraph,
  ECLI for case law, and a EUR-Lex link.
- **Versions are labelled.** A consolidated text is marked as having no legal force, with the
  authentic act named. "The text in force on a date" is stated only when the API reports
  `asOfResolved: true`.

What the check does not do: it confirms that an excerpt is verbatim, not that it is quoted in
context, and it cannot tell whether a model-written sentence is a correct reading of the source
it cites. That is why the quotes are always shown next to it.

## What these examples are not

- **Not compliance tools.** LexAPI provides source texts and citations. It does not assess whether
  anyone complies with the law, and neither do these examples.
- **Not legal advice,** and not a substitute for a lawyer. They show text with its source and,
  where the example covers it, say what a human has to decide.
- **Not authoritative texts.** Consolidated versions are documentation tools; the authentic act is
  the one published in the Official Journal. Always check the linked source.
- **Not complete.** An example sees only what its searches returned.
- **Not for profiling or re-identification.** Do not adapt them to evaluate, compare or predict
  the conduct of individual judges, or to re-identify people in anonymised or pseudonymised
  decisions. If you republish retrieved content, keep the source attribution with it, as the
  [LexAPI terms](https://lex-api.com/terms?ref=github-examples) require.

## What is tested

The test suites run offline. They use a local stub server on 127.0.0.1 (in-process mock transport
for the LangChain example) that replays synthetic fixtures shaped like the API's responses, and a
scripted stand-in for the model. They refuse to run against any base URL that does not start with
`http://127.0.0.1:`.

That means the code paths, the request shapes the SDKs send, the grounding rules and webhook
signature verification are tested. The examples have **not** been run against the LexAPI
service or a real model at all, so real-world retrieval quality and model behaviour are not
covered. The sample outputs in the READMEs come from dry runs and are illustrative.

```bash
npm run lint && npm run typecheck && npm test
```

## Repository layout

```
examples/   one folder per example (README, .env.example, source, tests)
shared/     TypeScript pieces the examples share: grounding check, model adapters, env
offline/    stub server, synthetic fixtures, dry-run and test harness
scripts/    record-fixtures (calls the real API; never run by tests or CI)
```

### Recording real response shapes

The fixtures in `offline/fixtures` are synthetic: placeholder texts and identifiers that cannot be
mistaken for real ones, in the shapes the SDKs type. To compare them with what the API returns
today, `npm run record-fixtures -- --i-have-a-key` makes the calls the examples make with your own
`LEXAPI_API_KEY` (about 30-50 credits) and writes the responses to `fixtures-recorded/`. That
folder is git-ignored: the files contain real legal texts and your account's plan and usage
figures. The script refuses to run without the flag, inside a test run, or on a CI runner.

Built against `@lexapi/client` 0.4.0 (npm), `lexapi-client` 0.4.0 (PyPI) and `@lexapi/mcp` 0.5.1.
The LangChain example is built against `langchain-lexapi` 0.1.0 (installed from its repository at
commit `1680bc3`, which brings its own pinned `lexapi-client`), `langchain-core` 1.8.3 and
`langchain-anthropic` 1.7.11.

## Security

Keys come from the environment only; see [SECURITY.md](SECURITY.md).

## Licence

[MIT](LICENSE), the same licence as the official LexAPI SDKs.
