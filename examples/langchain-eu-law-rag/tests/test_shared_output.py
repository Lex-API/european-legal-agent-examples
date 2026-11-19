"""The output lines this example shares with examples/llamaindex-law-on-a-date are the same code in both."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2]
OURS = EXAMPLES / "langchain-eu-law-rag" / "src" / "dora_rag"
THEIRS = EXAMPLES / "llamaindex-law-on-a-date" / "src" / "law_on_a_date"
SHARED = {
    "cli.py": (
        *("DEFAULT_ANTHROPIC_MODEL", "USER_AGENT_SUFFIX", "LOCAL_PREFIX", "DASHBOARD_URL", "KEY_HELP"),
        *("FIRST_DATE", "ACT", "DATE_ASKED", "CREDITS_USED", "NOTICE_MEANING", "CLOSING", "ConfigError"),
        *("_DATE", "_EXPORT", "PLACEHOLDERS", "_DASHBOARD_CODES", "error_fields"),
        *("clean", "today", "date_arg", "not_blank", "answer_heading", "load_env_file", "env_key"),
        *("require_keys", "create_client", "one_line", "refusal_line", "model_failure"),
    ),
    "offline.py": ("DRY_RUN_BASE_URL", "DRY_RUN_BANNER", "FixturesNotFoundError", "read_fixture"),
}
SHARED_PIPELINE = ("NO_ANSWER", "HUMAN_PROMPT", "AnswerError", "NoTextError", "ModelCallError")


def _definitions(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            found[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = ast.dump(node)
    return found


def test_the_shared_output_is_the_llamaindex_examples() -> None:
    if not THEIRS.is_dir():
        pytest.skip("needs examples/llamaindex-law-on-a-date from the repository checkout")
    for module, names in SHARED.items():
        ours, theirs = _definitions(OURS / module), _definitions(THEIRS / module)
        for name in names:
            assert ours[name] == theirs[name], f"{module}: {name}"
    ours, theirs = _definitions(OURS / "rag.py"), _definitions(THEIRS / "pipeline.py")
    for name in SHARED_PIPELINE:
        assert ours[name] == theirs[name], f"rag: {name}"
