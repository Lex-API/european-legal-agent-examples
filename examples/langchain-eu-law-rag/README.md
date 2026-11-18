# DORA questions answered from cited passages (LangChain)

A minimal retrieval-augmented generation (RAG) example over EU legislation with LangChain.

Ask a question about the Digital Operational Resilience Act (DORA, Regulation (EU) 2022/2554). The
official LexAPI retriever for LangChain, [`langchain-lexapi`](https://github.com/Lex-API/lexapi-langchain),
fetches matching DORA articles, and a chat model answers from those passages only, citing each claim
as `[1]`, `[2]`, ... The sources are printed after the answer, one line per passage, with the CELEX
number, article and EUR-Lex link. It uses `langchain-core` and one chat model integration
(`langchain-anthropic`); no vector store, no agent loop.

## The core

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_lexapi import LexAPILegislationRetriever

system = "Answer only from the numbered passages; cite each claim as [n]. '[No text: ...]' means no text for that date."
prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Passages:\n\n{passages}\n\nQuestion: {question}")]
)
model = ChatAnthropic(model="claude-sonnet-5-5")  # reads ANTHROPIC_API_KEY
question = "What does DORA say about ICT risk management frameworks?"

# Reads LEXAPI_API_KEY; `with` closes the client the retriever built.
with LexAPILegislationRetriever(k=10, celex="32022R2554") as retriever:
    docs, report = retriever.invoke_with_report(
        f"Digital Operational Resilience Act (DORA): {question}",
        as_of="2025-01-17",  # optional, per call: the text for this date
    )
passages = "\n\n".join(f"[{n}]\n{doc.page_content}" for n, doc in enumerate(docs, start=1))
reply = (prompt | model).invoke({"passages": passages, "question": question})
print(reply.text)
for n, doc in enumerate(docs, start=1):
    print(f"[{n}] {doc.metadata['citation']}")  # CELEX, article, EUR-Lex link
print("Credits used:", report.units_charged, "remaining:", report.credits_remaining)
```

That is the whole idea. `src/dora_rag/rag.py` is the same pipeline with the full prompt
(`src/dora_rag/prompt.txt`) and a citation check; `cli.py` adds the command line, the output and
error handling, and `offline.py` the `--dry-run` stand-ins. A test runs the snippet above against a
mock LexAPI and a fake model, so it stays in step with the adapter.

## What it does not do

- LexAPI provides source texts and citations. It does not assess whether anyone complies with the
  law, and neither does this example. It is research support, not legal advice.
- It only sees what one search returned: at most 10 hits, of which only DORA articles are kept.
  Relevant text outside that set is not considered.
- The answer is free text. Code checks its citations (below), not whether a sentence reads its
  passage correctly; unlike the other examples in this repository, it does not check quotations.

## How it works

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
3. **Answer.** The model is called only if at least one passage has text; a notice (`[No text: ...]`
   or `[No text returned for ...]`) is not text. It is told to answer from the passages only, cite
   every claim by passage number, treat a notice as no text, name the act in the Source line rather
   than a consolidated text, and not give advice or say whether anyone complies.
4. **Check.** Every bracketed token in the answer that contains a digit must be a plain `[n]`
   (ASCII brackets and digits) with `n` from 1 to the number of passages, and the answer must cite
   at least one passage, unless it is exactly "The passages do not answer this question." A reply
   with no text, or that fails the check, is not printed as an answer. With the Anthropic model used
   here, a reply that was cut off or declined is never printed as an answer; with another provider,
   add that provider's check.
5. **Print.** The answer, the sources, how many passages were withheld (with `--as-of`) and why,
   the hits left out, and the credits the call used and the credits remaining, from the adapter's
   `RetrievalReport`. `--show-passages` also prints the passages given to the model. Control and
   format characters are removed from everything printed that the program did not write itself.

### Versions and dates

- A **consolidated** passage is marked in its Source line as `consolidated text <CELEX> (no legal
  force)`, after the CELEX number of the act to cite. Consolidated texts are documentation; only the
  act as published in the Official Journal is authentic.
- `--as-of YYYY-MM-DD` asks for the text for that date (the retriever's per-call `as_of=`). A
  passage whose text LexAPI did not establish for the date arrives as a notice without text (the
  adapter's default, `unresolved="withhold"`), for example
  `[No text: CELEX 32022R2554 art_6 is not established as the text for 2025-01-17 (reason: ...).]`.
  Its source line says `text for <date> not established` and gives the reason, and the output
  prints how many passages were withheld. A notice does not say whether the provision applied on
  that date. A resolved passage's source line says `version for <date>: <CELEX>`.
- Without `--as-of`, nothing is said about the text for any date.

## Install

`langchain-lexapi` version 0.1.0 is not on PyPI yet. `pyproject.toml` installs it from its
repository, pinned to a commit, so installing this example needs that repository to be reachable:

```
langchain-lexapi @ git+https://github.com/Lex-API/lexapi-langchain@724a62debef42761362b5cea7bda6237ed5cd87a
```

## Run it

```bash
cd examples/langchain-eu-law-rag
python3 -m venv .venv && . .venv/bin/activate     # Python 3.11+
pip install -e .                                  # keep -e: --dry-run reads the repository's fixtures

# No keys: LexAPI is replaced by a local mock and the model by a scripted reply (synthetic texts)
dora-rag --dry-run "What does DORA say about ICT risk management frameworks?"
dora-rag --dry-run --show-passages --as-of 2025-01-17 "What does DORA say about ICT risk management frameworks?"

# For real, with your own keys
cp .env.example .env                              # then fill in LEXAPI_API_KEY and ANTHROPIC_API_KEY
dora-rag "What does DORA say about ICT risk management frameworks?"
dora-rag --as-of 2025-01-17 "What does DORA say about ICT risk management frameworks?"
```

The model is Anthropic's `claude-sonnet-5-5` through `langchain-anthropic`; set `ANTHROPIC_MODEL`
to use another Claude model. To use another provider, replace `create_chat_model` in `cli.py` with
any LangChain chat model (and add that provider's stop-reason check to `read_answer`).

A dry run turns LangChain tracing off. A real run sends traces only if you have LangChain tracing
configured.

Exit codes:

- 0: an answer, or "no passages" / "no text", was printed.
- 1: LexAPI refused the request (for example 402, credits exhausted, or 403, payment past due; the
  API's message is printed), or the model call failed or its reply was not used. After a model
  failure, the credits line of the search, which was charged, is printed first.
- 2: a usage or configuration error: an empty question, a malformed `--as-of` date, a missing
  `LEXAPI_API_KEY` or `ANTHROPIC_API_KEY`, or `--dry-run` without the repository's fixtures.

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
content. Covered: the request sent (query, `limit`, `include`, `asOf`), `--as-of` and question
validation, every Source line in the prompt and no text of a withheld passage or of another act,
the prompt's rules, the source list against the passages, the withheld, reason, left-out and
credits lines (and "not reported"), no model call when no passage has text, the citation check,
unusable replies and model-provider errors, 402 and 403 exit codes and messages, sanitised output,
the API key never printed, the model settings, the client closed after the run, the dry run, and
the snippet in "The core".
Not covered by any test: the live API and a real model.
Until the adapter's repository is public, these tests run locally only (`pip install -e .` needs
access to it); the CI job checks whether it is reachable and skips with a notice when it is not.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q
```

## Sample output (illustrative)

Generated with `--dry-run --show-passages --as-of 2025-01-17`. The "legal texts" are synthetic
placeholders from `offline/fixtures`, and the answer is scripted, not written by a model. The
fixture forces the second passage to be withheld, to show what a notice looks like.

```text
DRY RUN - LexAPI was replaced by a local mock and the model by a scripted reply.
The passages given to the model are synthetic placeholders, not the wording of any act.

Question: What does DORA say about ICT risk management frameworks?
Text asked for: 2025-01-17
Model: scripted (no model was called)

Passages given to the model:

[1]
[SYNTHETIC FIXTURE - not the text of the act] Placeholder provision D1 (as published): fixture wording about a placeholder body that keeps a fixture framework under review.
Source: CELEX 32022R2554, art_5, version for 2025-01-17: 32022R2554, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554

[2]
[No text: CELEX 32022R2554 art_6 is not established as the text for 2025-01-17 (reason: fetch_timeout).]
Source: CELEX 32022R2554, art_6, text for 2025-01-17 not established, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554

Answer (written by the model from the passages above):
The first passage, in the version listed for the date asked, describes a placeholder body that keeps a fixture framework under review [1]. LexAPI has no text of the second provision for that date, so it is not described here [2].

Sources:
[1] CELEX 32022R2554, art_5, version for 2025-01-17: 32022R2554, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554
[2] CELEX 32022R2554, art_6, text for 2025-01-17 not established, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554 - notice only, no text was given to the model (reason: fetch_timeout)

Withheld for 2025-01-17: 1 passage(s) arrived as a notice without text (LexAPI did not establish their text for that date).
A notice says only that LexAPI did not establish the text for that date; it does not say whether the provision applied then.
Left out: 1 hit(s) from acts other than DORA.
LexAPI credits used for this call: 5. Credits remaining: 495.

Research support, not legal advice. The answer is model-written and may misread the passages; check it against the sources.
```
