# EU law for a date, from cited passages (LlamaIndex)

Ask about an EU act for a date: get the version LexAPI lists for that date and an answer from its passages with citations, and be told plainly where LexAPI has no text for that date.

Name an act (the GDPR, Regulation (EU) 2016/679, by default), a date and, optionally, a provision.
The official LexAPI tools for LlamaIndex, [`llama-index-tools-lexapi`](https://github.com/Lex-API/lexapi-llamaindex),
return the version LexAPI lists for that date and the provision's text for the date. The official
retriever, `llama-index-retrievers-lexapi`, then fetches the act's passages matching your question,
for the same date. A model answers from those passages only, citing each claim as `[1]`, `[2]`, ...
Passages whose text for that date is not established arrive as notices, without text. `--compare`
lists the version for two dates and says whether it is the same one. It is point-in-time (as-of-date)
retrieval-augmented generation (RAG) with `llama-index-core` and one LLM integration
(`llama-index-llms-anthropic`); no index, no embeddings, no agent loop.

Python 3.11 or later. The model is Anthropic's, through `llama-index-llms-anthropic`.

## The core

```python
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import MetadataMode
from llama_index.llms.anthropic import Anthropic
from llama_index.retrievers.lexapi import LexAPILegislationRetriever
from llama_index.tools.lexapi import LexAPIToolSpec

act, date, question = "32016R0679", "2019-06-01", "What did Article 17 of the GDPR say about erasure?"
llm = Anthropic(model="claude-sonnet-5-5")  # reads ANTHROPIC_API_KEY

# Both read LEXAPI_API_KEY; `with` closes the clients they built.
with LexAPIToolSpec() as tools, LexAPILegislationRetriever(similarity_top_k=10, as_of=date, celex=act) as retriever:
    version = tools.lexapi_legislation_as_of(act, date)  # the version listed for the date, and its Source line
    nodes, report, _ = retriever.retrieve_with_result(question)  # passages for the date, or notices
if not any(n.node.metadata["text_status"] == "provided" for n in nodes):
    raise SystemExit("No passage has text for that date, so no model is called.")
passages = "\n\n".join(f"[{i}]\n{n.node.get_content(metadata_mode=MetadataMode.LLM)}" for i, n in enumerate(nodes, 1))
system = "Answer only from the numbered passages; cite each claim as [n]. '[No text: ...]' means no text for that date."
user = f"Passages:\n\n{passages}\n\nQuestion: {question}"
reply = llm.chat([ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)])
print(version)
print(reply.message.content)
for i, n in enumerate(nodes, 1):
    print(f"[{i}] {n.node.metadata['citation']}")  # CELEX, article, version for the date, EUR-Lex link
print("Search credits used:", report.units_charged, "remaining:", report.credits_remaining)
```

That is the whole idea. `src/law_on_a_date/pipeline.py` is the same pipeline with the provision
step, the full prompt (`src/law_on_a_date/prompt.txt`), the same guard (`answer()` raises
`NoTextError` and calls no model when no passage has text) and a citation check; `cli.py` adds the
command line, `--compare`, the output and error handling, and `offline.py` the `--dry-run`
stand-ins. The full pipeline also reads the version and the lookup's credits from the tool's
`LexAPIToolEvent`. A test runs the snippet above against a mock LexAPI and a scripted LLM, so it
stays in step with the adapters. The first run can take up to about 20 seconds while Python
compiles the libraries; later runs take a few seconds.

It calls `llm.chat` itself rather than a query engine or response synthesizer, so that no model is
called when no passage has text and the `[n]` in the answer are the numbers in the printed source
list. To hand the same tools to an agent, pass `LexAPIToolSpec().to_tool_list()` to a LlamaIndex
agent; see the adapter's README.

## What it does not do

- LexAPI provides source texts and citations. It does not assess whether anyone complies with the
  law, and neither does this example. It is research support, not legal advice.
- It reports the version LexAPI lists for a date and the text for that date. It does not say whether
  a provision applied on that date: an act can be in force while some of its provisions apply later.
- When no consolidated version is listed for the date, the answer is the act as published in the
  Official Journal, without later corrigenda.
- `--provision` addresses whole articles, annexes and recitals, not paragraphs or points.
- For a date before the act existed, a real run prints `Version for <date>: not established by
  LexAPI (reason: not_yet_adopted)`, or `in_force_dates_unknown` when LexAPI does not know the
  act's dates.
- `--compare` compares the versions listed for two dates, not their texts. It says whether the same
  version is listed for both; it does not say what changed.
- It only sees the provision you name and what one search returned: at most 10 hits, of which only
  the act's passages are kept.
- The answer is free text. Code checks its citations (below), not whether a sentence reads its
  passage correctly.

## How it works

The pipeline is two explicit steps, not an agent: the model never chooses which tool to call.

1. **The version for the date.** `LexAPIToolSpec.lexapi_legislation_as_of(celex, date,
   provision=...)` asks LexAPI which version of the act it lists for the date: the newest
   consolidated version dated on or before it, or the act as published in the Official Journal. With
   `--provision` it also returns that provision's text for the date. When LexAPI cannot establish
   the version or the text, the answer is a notice with the reason and no text; the tool always
   withholds, whatever the search policy. The tool returns one string (the text or a notice, then
   a `Source:` line); the version and the credits are read from the `LexAPIToolEvent` the tool
   dispatches through LlamaIndex instrumentation. The provision becomes the first node, shaped like
   the retriever's.
2. **Passages for the question, for the date.** `LexAPILegislationRetriever(as_of=date,
   celex=act, similarity_top_k=10)` runs one semantic search with the date. The API cannot limit a
   search to one act, so the adapter keeps only the act's passages (and its consolidated texts) after
   the search (a hit is kept when its CELEX number, or the act it consolidates, is the one asked
   for); the output counts the hits left out, which includes a consolidated text that does not name
   its act. A hit whose text for
   the date LexAPI did not establish stays as a notice node (the adapter's defaults,
   `unresolved="withhold"`, `emit_withheld=True`). When the search returns the provision too, only
   one copy is kept, in the provision's place: the one with text, by the adapters' rule (a text over a
   labelled latest text over a notice), whichever way `--provision` was written. Two texts of the
   provision from different versions are both kept, and the output says so under the version line.
3. **What the model sees.** Each passage is `node.get_content(MetadataMode.LLM)`: the node's
   metadata lines (`text_status`, `cite_celex`, `consolidated`, `as_of`, `as_of_resolved`,
   `as_of_reason`, ...; the adapter hides the citation, ids and score from this view), a blank line,
   the text or the notice, and its `Source:` line. The prompt says what those lines are (see
   [What the model sees for a notice](#what-the-model-sees-for-a-notice)).
4. **Answer.** The model is called only if at least one passage has text; a notice or a blank text
   is not text. It is told to answer from the passages only, cite every claim by passage number, use
   square brackets for nothing else, treat a notice as no text, name the act in the Source line
   rather than a consolidated text, and not give advice, say whether anyone complies, or say whether
   a provision applied on the date.
5. **Check.** The same check as in the LangChain example (the code is kept identical, and a test
   compares the two): a bracketed token whose content is only numeric characters (of any script) and
   separators (`, ; - – — / .`, spaces), optionally after "passage", "source" or "ref", in square or
   other brackets, or a bracket left open on its line with a numeric character in it, must be exactly
   `[n]` with ASCII digits and `n` from 1 to the number of passages; other bracketed text (`[sic]`, an
   echoed notice) is ignored. The answer must cite at least one passage, unless it is only "The
   passages do not answer this question." (quotes and the full stop optional). A reply with no text,
   or that fails the check, is not printed as an answer. With the Anthropic model used here, a reply
   that was cut off or declined is never printed as an answer; with another provider, add that
   provider's check.
6. **Print.** The version listed for the date (as published, or a consolidated text with "no legal
   force" and the act to cite), the answer, the sources, how many passages have no text for the date
   and why, the hits left out, and the credits each call used, their total and the credits
   remaining. `--show-passages` also prints the passages exactly as the model was given them.
   Control and format characters are removed from everything printed that the program did not write
   itself.

### Versions and dates

- The version line uses the API's terms: "the version listed for" a date. A **consolidated** version
  is printed as `a consolidated text (no legal force); the act to cite is CELEX <act>`, with
  LexAPI's note; only the act as published in the Official Journal is authentic.
- A passage whose text LexAPI did not establish for the date is a notice, for example
  `[No text: CELEX 32016R0679 art_18 is not established as the text for 2019-06-01 (reason: ...).]`.
  Its source line says `text for <date> not established` and gives the reason, and the output prints
  how many passages have no text for that date. A notice gives only LexAPI's reason for having no
  text for that date; check the source for whether the provision applied then.
- Dates are checked before any request: `YYYY-MM-DD`, a real calendar day, from 1952-01-01 to today
  (UTC). `--compare` needs two different dates and prints the earlier one first.

### What the model sees for a notice

From `law-on-a-date --dry-run --show-passages --as-of 2019-06-01 --provision art_17 "What did Article
17 of the GDPR say about erasure?"` (synthetic fixture), the third passage as the model receives it. No text of
the provision is in it; the latest text LexAPI holds for that article is not passed on.

```text
[3]
text_status: withheld
consolidated: False
article_ref: art_18
title: Regulation (EU) 2016/679 (fixture title)
language: en
as_of: 2019-06-01
as_of_resolved: False
as_of_reason: fetch_timeout
cite_celex: 32016R0679

[No text: CELEX 32016R0679 art_18 is not established as the text for 2019-06-01 (reason: fetch_timeout).]
Source: CELEX 32016R0679, art_18, text for 2019-06-01 not established, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679
```

## Install

`pyproject.toml` installs `llama-index-retrievers-lexapi` and `llama-index-tools-lexapi` from PyPI
(0.1.0 or later, below 0.2).

`pyproject.toml` pins `anthropic` to 1.17.0: `llama-index-llms-anthropic` 0.13.0, its latest
release, fails with `anthropic` 2.x before any request is sent. On an Intel Mac, it also pins
`cryptography` below 49: `llama-index-llms-anthropic`
needs it, and from 49.0.0 it ships macOS wheels for Apple silicon only.

## Run it

```bash
cd examples/llamaindex-law-on-a-date
python3 -m venv .venv && . .venv/bin/activate     # Python 3.11+
pip install -e .                                  # keep -e (below)

# No keys: LexAPI is replaced by a local mock and the model by a scripted reply (synthetic texts)
law-on-a-date --dry-run --as-of 2019-06-01 --provision art_17 "What did Article 17 of the GDPR say about erasure?"
law-on-a-date --dry-run --show-passages --as-of 2024-01-01 "What does the GDPR say about objections?"
law-on-a-date --dry-run --compare 2018-05-25 2024-01-01

# For real, with your own keys
cp .env.example .env                              # then fill in LEXAPI_API_KEY and ANTHROPIC_API_KEY
law-on-a-date --as-of 2019-06-01 --provision art_17 "What did Article 17 of the GDPR say about erasure?"
law-on-a-date --celex 32022R2554 --as-of 2025-01-17 "What does DORA say about ICT risk management?"
law-on-a-date --compare 2018-05-25 2024-01-01     # needs LEXAPI_API_KEY only; no model is called
```

Keep `-e`: `--dry-run` reads the repository's fixtures and `.env` is read from this folder. The dry
run has fixtures for CELEX 32016R0679 only; with another `--celex` it exits with code 2.
`.env` takes `KEY=value` lines (`export KEY=value` too). A key left empty, or still a placeholder,
counts as not set: the run stops before any request and names every key that is missing.

Every run needs `--as-of` (with a question) or `--compare` (two dates, no question).

`--celex` takes the CELEX number of an EU regulation, directive or decision (default `32016R0679`).
`--provision` takes an article (`art_17`, `17`, `Article 17`), an annex (`anx_III`) or recitals
(`rct_26-28`). Name the act in your question: the search runs on the question as written.

The model is Anthropic's `claude-sonnet-5-5` through `llama-index-llms-anthropic`; set
`ANTHROPIC_MODEL` to use another Claude model. To use another provider, replace `create_llm` in
`cli.py` with any LlamaIndex LLM (and add that provider's stop-reason check to `read_answer`).

Exit codes:

- 0: an answer, "no passages" / "no text", or the `--compare` result was printed.
- 1: LexAPI refused a request (for example 401, a key it does not accept; 402, credits exhausted;
  403, payment past due; or 429, a rate limit or an exhausted quota), no answer arrived, or its
  answer could not be read. The tools' own error text is printed, on one line (`str()` of
  `LexAPIToolError`, also for an error from the retriever):
  `LexAPI error <code> (HTTP <status>): <message>.` (without `(HTTP <status>)` when no answer
  arrived, for example a connection error), then, each only when it applies and in this order,
  `Credits reset at <time>.`, `This is an exhausted quota, not a rate limit.` (and
  `Quota resets at <time>.`), `Access ended at <time>.`, `Billing: <link>` and
  `Retry after <n> s.` (only for an error LexAPI's client retries, such as a rate limit). After any
  401 (a key it does not accept) and after credits exhausted, the example adds the dashboard link on
  a line of its own. The credits line
  of any earlier call of the run, which was charged, comes first. Also when the model call failed
  (`Model error: <the provider's message>`, capped at 1,000 characters) or its reply was not used;
  then the credits line of the LexAPI calls, which were charged, is printed first.
  The LexAPI SDK retries rate limits and gateway or connection errors by itself (up to 3 retries,
  each wait capped at 30 s) and waits up to 95 s for each answer, so a refused or hanging call can
  take a minute and a half or more before its line is printed.
- 2: a usage or configuration error, before any request: an empty question, a date that is
  malformed, not a calendar date, in the future (UTC) or before 1952-01-01, the same date twice for
  `--compare`, a malformed `--celex`, a missing, empty or placeholder `LEXAPI_API_KEY` or
  `ANTHROPIC_API_KEY`, or `--dry-run` without the repository's fixtures or with another act.
- 130: interrupted (Ctrl-C), without a traceback.

## Cost

LexAPI credits per run, from the adapters' documentation and the
[documented weights](https://lex-api.com/docs?ref=github-examples):

| Step | Credits |
| --- | --- |
| Version lookup, without `--provision` | 1 |
| Version lookup with `--provision` | 1 when LexAPI already holds the text; 4 when it has to fetch it (5 when fetched live from EUR-Lex), also when the provision then turns out not to be in that version; 1 if the fetch fails or the version has no text |
| Semantic legislation search with a date | 5, plus up to 3 version fetches at 0-5 each |

The run prints what each call cost.

So a question costs **6-21 credits** without `--provision` and **6-25** with it, in 2 requests.
`--compare` costs **2 credits** (two version lookups). The `celex` filter and `similarity_top_k=10`
do not change the price. The run prints the figures the API reported for each call; if a response
carries none, it says "not reported" rather than 0. A refused lookup can still cost 1 credit (an
unknown act, for example: a 404 is refunded down to 1 credit); the tool's error does not report
credits, so none are printed for it.
Model usage is billed separately by your model provider.

## Tests

`pytest` runs offline. LexAPI is answered in-process by an `httpx.MockTransport` on an
`http://127.0.0.1:` base URL, the model is a scripted LlamaIndex LLM (`ScriptedLLM` in `offline.py`),
socket connections and name lookups for anything but 127.0.0.1 and httpx's real transports fail for
the whole session, and "today" is fixed. No embedding model is used or needed, and nothing imports
`openai`. The fixtures are synthetic placeholders, and no test asserts legal content. Covered: the
requests sent (as-of body, search `query`, `limit`, `include`, `asOf`), date, question, `--celex` and
`--compare` validation, every Source line in the prompt and no text of a withheld passage or of
another act, the metadata lines the model sees, the prompt's rules, the version lines (as published,
consolidated, not established), compare's same / different / not compared results and date order,
the source list against the passages, the withheld, reason, left-out and credits lines (and "not
reported"), which copy is kept when the search returns the provision too (for each spelling of
`--provision`), that other provisions, languages, acts and versions are never taken for that copy
(and the line saying so when a different version is kept),
the tool's event read from the adapter's dispatcher and its handler removed after each lookup, no
model call when no passage has text, the citation check (the LangChain example's
table, and a test that the two examples' check is the same code), the real `Anthropic` LLM class sending its request to a local stand-in for the Anthropic API (and its
errors), unusable replies and model-provider
errors, 401/402/403/429/500 (credits, quota, payment past due and rate limit, each with its exact
line, for the lookup and for the search) and unreadable answers, sanitised output, the API key never printed, the
model settings, `.env` reading and an unfilled copy of `.env.example`, "today" in UTC whatever the
local time zone, the dashboard link after any 401 and after credits exhausted, the client closed after the run,
the dry run (and its refusal of another act), the
snippet in "The core", the sample outputs and the notice block above, and the output lines shared
with the LangChain example.
Not covered by any test: the live API and a real model.
The tests and `--dry-run` need a checkout of the whole repository: the fixtures are in
`offline/fixtures` at its root, and some tests compare this example with the other Python one.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q
```

## Sample output (illustrative)

Generated with `--dry-run`. The "legal texts" and versions are synthetic placeholders from
`offline/fixtures`, and the answer is scripted, not written by a model. The mock lists the act as
published for dates before 2020 and a placeholder consolidated text (`02016R0679-20990101`) after;
that rule is part of the fixture, not LexAPI's data. The fixture forces one passage to be withheld,
to show what a notice looks like.

```text
$ law-on-a-date --dry-run --as-of 2019-06-01 --provision art_17 "What did Article 17 of the GDPR say about erasure?"
DRY RUN - LexAPI was replaced by a local mock and the model by a scripted reply.
The texts and versions are synthetic placeholders, not the wording or the versions of any act.

Question: What did Article 17 of the GDPR say about erasure?
Act: CELEX 32016R0679
Date asked: 2019-06-01
Provision: art_17
Model: scripted (no model was called)

Version listed for 2019-06-01: 32016R0679, the act as published in the Official Journal.
  Source: CELEX 32016R0679, art_17, version for 2019-06-01: 32016R0679, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679

Answer (scripted for this dry run; passages not printed, see --show-passages):
The provision, in the version listed for the date asked, describes a placeholder request that a placeholder body handles [1]. Another passage describes a placeholder objection to a placeholder activity [2]. LexAPI has no text of the third passage's provision for that date, so it is not described here [3].

Sources:
[1] CELEX 32016R0679, art_17, version for 2019-06-01: 32016R0679, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679
[2] CELEX 32016R0679, art_21, version for 2019-06-01: 32016R0679, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679
[3] CELEX 32016R0679, art_18, text for 2019-06-01 not established, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679 - notice only, no text was given to the model (reason: fetch_timeout)

Withheld for 2019-06-01: 1 passage(s) have no text for that date and arrived as a notice.
A notice gives only LexAPI's reason for having no text for that date; check the source for whether the provision applied then.
Left out: 1 hit(s) not attributed to CELEX 32016R0679.
LexAPI credits used for this run: 9 (version lookup: 4, search: 5). Credits remaining: 491.

Research support, not legal advice. The answer is model-written and may misread the passages; check it against the sources.
```

```text
$ law-on-a-date --dry-run --compare 2018-05-25 2024-01-01
DRY RUN - LexAPI was replaced by a local mock and the model by a scripted reply.
The texts and versions are synthetic placeholders, not the wording or the versions of any act.

Act: CELEX 32016R0679
Dates compared: 2018-05-25 and 2024-01-01

Version listed for 2018-05-25: 32016R0679, the act as published in the Official Journal.
  Source: CELEX 32016R0679, version for 2018-05-25: 32016R0679, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679
Version listed for 2024-01-01: 02016R0679-20990101, a consolidated text (no legal force); the act to cite is CELEX 32016R0679.
  LexAPI's note: [SYNTHETIC FIXTURE] The act as amended, consolidated as of 2099-01-01; not the authentic text. Only the act as published in the Official Journal is authentic (32016R0679).
  Source: CELEX 32016R0679, consolidated text 02016R0679-20990101 (no legal force), version for 2024-01-01: 02016R0679-20990101, https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02016R0679-20990101

Different versions: 32016R0679 is listed for 2018-05-25 and 02016R0679-20990101 for 2024-01-01. This compares the versions LexAPI lists for the dates, not their texts.
LexAPI credits used for this run: 2 (lookup for 2018-05-25: 1, lookup for 2024-01-01: 1). Credits remaining: 498.

Not legal advice. LexAPI provides source texts and citations; it does not assess whether anyone complies with the law.
```
