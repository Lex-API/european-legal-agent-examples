# Offline fixtures

These files are **synthetic**. They follow the response shapes of the LexAPI REST API as
typed by the official SDKs (`@lexapi/client` 0.4.0, `lexapi-client` at the commit the Python examples pin) and described in the
[API reference](https://lex-api.com/docs?ref=github-examples), but none of them was recorded from
the live API, and the legal texts in them are placeholders, not the wording of any act or judgment.
Every placeholder text starts with `[SYNTHETIC FIXTURE`.

- `routes.json` maps a request (method, path, optional body/query substring) to a fixture file.
  Both stub servers (`offline/stub-server.ts` and `gdpr_assistant/offline.py`) read it.
- `llm/` holds the scripted model replies used by `--dry-run` and the tests.
- `legislation-semantic.dora.json`, `legislation-semantic.dora-asof.json` and `llm/dora.json` are
  for the LangChain example. Its own mock transport (`examples/langchain-eu-law-rag/src/dora_rag/offline.py`)
  serves them, not `routes.json`; it copies the date asked into the as-of fixture's `asOf` fields.
- `legislation-as-of.gdpr.*.json`, `legislation-semantic.gdpr-on-a-date.json` and
  `llm/law-on-a-date.json` are for the LlamaIndex example. Its own mock transport
  (`examples/llamaindex-law-on-a-date/src/law_on_a_date/offline.py`) serves them, not `routes.json`;
  it copies the date and provision asked into the answers, and lists the as-published fixture for
  dates before 2020 and the consolidated one (`02016R0679-20990101`, a placeholder number) after.
  That rule belongs to the fixtures, not to LexAPI's data.
