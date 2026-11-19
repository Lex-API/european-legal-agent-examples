"""The output lines this example shares with examples/langchain-eu-law-rag are the same code in both."""

from __future__ import annotations

import ast
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2]
OURS = EXAMPLES / "llamaindex-law-on-a-date" / "src" / "law_on_a_date"
THEIRS = EXAMPLES / "langchain-eu-law-rag" / "src" / "dora_rag"
SHARED = {
    "cli.py": (
        *("DEFAULT_ANTHROPIC_MODEL", "USER_AGENT_SUFFIX", "LOCAL_PREFIX", "DASHBOARD_URL"),
        *("FIRST_DATE", "DATE_ASKED", "CREDITS_USED", "NOTICE_MEANING", "CLOSING", "_DATE", "ConfigError"),
        *("clean", "today", "date_arg", "answer_heading", "load_env_file", "create_client", "one_line"),
        "model_failure",
    ),
    "offline.py": ("DRY_RUN_BASE_URL",),
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


def test_the_shared_output_is_the_langchain_examples() -> None:
    for module, names in SHARED.items():
        ours, theirs = _definitions(OURS / module), _definitions(THEIRS / module)
        for name in names:
            assert ours[name] == theirs[name], f"{module}: {name}"
    ours, theirs = _definitions(OURS / "pipeline.py"), _definitions(THEIRS / "rag.py")
    for name in SHARED_PIPELINE:
        assert ours[name] == theirs[name], f"pipeline: {name}"
