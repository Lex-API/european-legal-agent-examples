# DORA questions answered from cited passages (LangChain)

Ask a question about the Digital Operational Resilience Act (DORA, Regulation (EU) 2022/2554). A
LangChain retriever fetches matching DORA articles from LexAPI, and a chat model answers from those
passages only, citing each claim as `[1]`, `[2]`, ... The list of sources is printed after the
answer, one line per passage, with the CELEX number, article and EUR-Lex link.

It uses the official LangChain adapter, [`langchain-lexapi`](https://github.com/Lex-API/lexapi-langchain),
plus `langchain-core` and one chat model integration (`langchain-anthropic`). No vector store, no
agent loop.

## What it does not do

- LexAPI provides source texts and citations. It does not assess whether anyone complies with the
  law, and neither does this example. It is research support, not legal advice.
- It only sees what one search returned: at most 10 hits, of which only DORA articles are kept.
  Relevant text outside that set is not considered.
- It checks that every `[n]` in the answer is the number of a passage that was retrieved. It does
  not check quotations or whether a sentence reads its passage correctly, unlike the other examples
  in this repository: the answer is free text.

## How it works

Everything is in `src/dora_rag/rag.py`; the instructions to the model are in
`src/dora_rag/prompt.txt`.

```python
retriever = LexAPILegislationRetriever(client=client, k=10, celex="32022R2554")
documents, report = retriever.invoke_with_report(f"{DORA_CONTEXT}: {question}", as_of=as_of)
passages = "\n\n".join(f"[{n}]\n{doc.page_content}" for n, doc in enumerate(documents, start=1))
message = (PROMPT | model).invoke({"date": as_of or "none", "passages": passages, "question": question})
```

1. **Retrieve.** One semantic legislation search, with "Digital Operational Resilience Act (DORA,
   Regulation (EU) 2022/2554)" in front of the question. `celex="32022R2554"` keeps only DORA
   articles and DORA's consolidated texts. The API cannot limit a search to one act, so the adapter
   filters the hits after the search: the result can be shorter than `k` or empty, which is why
   `k` is 10 rather than a few (it costs the same). The output counts the hits from other acts that
   were left out.
2. **Citations travel with the text.** Each document's `page_content` is the article text followed
   by its `Source:` line (for example `Source: CELEX 32022R2554, art_5, https://eur-lex.europa.eu/...`).
   The passages are numbered `[1]`, `[2]`, ... in the prompt, and the same numbers head the source
   list printed after the answer.
3. **Answer.** The model is told to answer from the passages only, cite every claim by passage
   number, treat a "[No text: ...]" notice as no text for that date, name the act in the Source line
   rather than a consolidated text, and not give advice or say whether anyone complies.
4. **Check.** The reply is refused with an error, never printed as an empty or partial answer, when
   it has no text, was cut off or declined, or cites a number that is not a passage. If no passage
   has text, no model is called.
5. **Print.** The answer, the sources, how many passages were withheld (with `--as-of`), the hits
   left out, and the credits the call used and the credits remaining, from the adapter's
   `RetrievalReport`. Control and format characters are removed from everything printed that the
   program did not write itself.

### Versions and dates

- A **consolidated** passage is marked in its Source line as `consolidated text <CELEX> (no legal
  force)`, after the CELEX number of the act to cite. Consolidated texts are documentation; only the
  act as published in the Official Journal is authentic.
- `--as-of YYYY-MM-DD` asks for the text for that date. A passage whose text LexAPI did not
  establish for the date arrives as a notice without text (the adapter's default,
  `unresolved="withhold"`), for example
  `[No text: CELEX 32022R2554 art_6 is not established as the text for 2025-01-17 (reason: ...).]`.
  Its source line says `text for <date> not established`, and the output prints how many passages
  were withheld. A resolved passage's source line says `version for <date>: <CELEX>`.
- Without `--as-of`, nothing is said about the text for any date.

## Install

`langchain-lexapi` version 0.1.0 is not on PyPI yet. `pyproject.toml` installs it from its
repository, pinned to a commit:

```
langchain-lexapi @ git+https://github.com/Lex-API/lexapi-langchain@724a62debef42761362b5cea7bda6237ed5cd87a
```

## Run it

```bash
cd examples/langchain-eu-law-rag
python -m venv .venv && . .venv/bin/activate      # Python 3.11+
pip install -e .                                  # keep -e: --dry-run reads the repository's fixtures

# No keys: LexAPI is replaced by a local mock and the model by a scripted reply (synthetic texts)
dora-rag --dry-run "What does DORA say about ICT risk management frameworks?"
dora-rag --dry-run --as-of 2025-01-17 "What does DORA say about ICT risk management frameworks?"

# For real, with your own keys
cp .env.example .env                              # then fill in LEXAPI_API_KEY and ANTHROPIC_API_KEY
dora-rag "What does DORA say about ICT risk management frameworks?"
dora-rag --as-of 2025-01-17 "What does DORA say about ICT risk management frameworks?"
```

The model is Anthropic's `claude-sonnet-5-5` through `langchain-anthropic`; set `ANTHROPIC_MODEL`
to use another Claude model. To use another provider, replace `create_chat_model` in `rag.py` with
any LangChain chat model.

Exit codes: 0 when an answer (or "nothing found") is printed; 1 when LexAPI refuses the request,
for example 402 (credits exhausted) or 403 (payment past due), with the API's message; 2 for a
usage error such as a malformed `--as-of` date. An unusable model reply raises `AnswerError`.

## Cost

LexAPI credits per question, from the [documented weights](https://lex-api.com/docs?ref=github-examples):

| Step | Credits |
| --- | --- |
| Semantic legislation search | 5 |
| With `--as-of`: up to 3 version fetches | 0-5 each |

So **5 credits** per question, up to 20 with `--as-of`, in 1 request. The `celex` filter and `k=10`
do not change the price. The run prints the figures the API reported for the call; if a response
carries none, it says "not reported" rather than 0. Model usage is billed separately by your model
provider.

## Tests

`pytest` runs offline. LexAPI is answered in-process by an `httpx.MockTransport` on an
`http://127.0.0.1:` base URL, the model is LangChain's `GenericFakeChatModel`, socket connections
and name lookups for anything but 127.0.0.1 and httpx's real transports fail for the whole session,
and LangSmith tracing is off. The fixtures are synthetic placeholders, and no test asserts legal
content. Covered: the request sent (query, `limit`, `include`, `asOf`), `--as-of` validation, every
Source line in the prompt and no text of a withheld passage or of another act, the prompt's rules,
the source list against the passages, the withheld, left-out and credits lines (and "not
reported"), no model call when nothing has text, unusable replies, 402 and 403 exit codes and
messages, the API key never printed, the model settings, the retriever closed after the run, and the
dry run.
Not covered by any test: the live API and a real model.

```bash
pip install -e ".[dev]" && ruff check . && ruff format --check . && mypy && pytest -q
```

## Sample output (illustrative)

Generated with `--dry-run --as-of 2025-01-17`. The "legal texts" are synthetic placeholders from
`offline/fixtures`, and the answer is scripted, not written by a model.

```text
DRY RUN - LexAPI was replaced by a local mock and the model by a scripted reply.
The passages below are synthetic placeholders, not the wording of any act.

Question: What does DORA say about ICT risk management frameworks?
Text asked for: 2025-01-17
Model: scripted (no model was called)

Answer (written by the model from the passages below):
The first passage, in the version listed for the date asked, describes a placeholder body that keeps a fixture framework under review [1]. LexAPI has no text of the second provision for that date, so it is not described here [2].

Sources:
[1] CELEX 32022R2554, art_5, version for 2025-01-17: 32022R2554, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554
[2] CELEX 32022R2554, art_6, text for 2025-01-17 not established, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554 - notice only, no text was given to the model

Withheld for 2025-01-17: 1 passage(s) arrived as a notice without text (LexAPI did not establish their text for that date).
Left out: 1 hit(s) from acts other than DORA.
LexAPI credits used for this call: 5. Credits remaining: 495.

Research support, not legal advice. The answer is model-written and may misread the passages; check it against the sources.
```
