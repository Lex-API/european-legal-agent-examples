# Offline fixtures

These files are **synthetic**. They follow the response shapes of the LexAPI REST API as
typed by the official SDKs (`@lexapi/client` 0.4.0, `lexapi-client` 0.4.0) and described in the
[API reference](https://lex-api.com/docs?ref=github-examples), but none of them was recorded from
the live API, and the legal texts in them are placeholders, not the wording of any act or judgment.
Every placeholder text starts with `[SYNTHETIC FIXTURE`.

- `routes.json` maps a request (method, path, optional body/query substring) to a fixture file.
  Both stub servers (`offline/stub-server.ts` and `gdpr_assistant/offline.py`) read it.
- `llm/` holds the scripted model replies used by `--dry-run` and the tests.
- `legislation-semantic.dora.json`, `legislation-semantic.dora-asof.json` and `llm/dora.json` are
  for the LangChain example. Its own mock transport (`examples/langchain-eu-law-rag/src/dora_rag/offline.py`)
  serves them, not `routes.json`; it copies the date asked into the as-of fixture's `asOf` fields.
