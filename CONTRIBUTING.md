# Contributing

Issues and pull requests are welcome.

- Keep examples small and readable: an explicit loop, no agent framework. Examples built on the
  official LexAPI adapters for LangChain and LlamaIndex are in scope.
- Every example must run with `--dry-run` (no keys) and ship offline tests. Tests must not reach
  the LexAPI service or a model provider; the harness refuses any base URL that is not a
  `http://127.0.0.1:` stub.
- Fixtures are synthetic. Do not commit recorded API responses, legal texts presented as real, or keys.
- Follow the rules in the README's "What these examples are not" section: examples show source
  texts with citations; they do not assess compliance or give legal advice.
- Describe only what the code does. Do not state coverage, speed or accuracy that is not verified.

Before opening a pull request:

```bash
npm ci && npm run lint && npm run typecheck && npm test
(cd examples/gdpr-research-assistant && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q)
(cd examples/langchain-eu-law-rag && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q)
(cd examples/llamaindex-law-on-a-date && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest -q)
```

Each Python example gets its own virtual environment: each pins its own dependencies. The LangChain and LlamaIndex examples need
Python 3.11 or later. The two examples share one citation check; a test in the LlamaIndex example fails when the
copies differ, so change both.

The process-level tests start the TypeScript and GDPR command lines with an explicit environment.
They set `LEXAPI_EXAMPLES_TEST_IGNORE_DOTENV=1`, a test hook (not a feature) that makes the
examples skip reading their own `.env`, so a developer's `.env` cannot change a test's result.
