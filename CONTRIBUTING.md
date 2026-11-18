# Contributing

Issues and pull requests are welcome.

- Keep examples small and readable: an explicit loop, no agent framework.
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
cd examples/gdpr-research-assistant && pip install -e ".[dev]" && ruff check . && ruff format --check . && mypy && pytest -q
```
