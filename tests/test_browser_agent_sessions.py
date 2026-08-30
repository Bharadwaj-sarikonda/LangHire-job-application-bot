"""Narrow wiring checks for per-agent OpenRouter sticky-routing sessions."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_node(path: str, name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name
    )


def _call_names(node: ast.AST) -> list[str]:
    names = []
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            names.append(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            names.append(item.func.attr)
    return names


def test_each_browser_agent_run_creates_exactly_one_session_id():
    agent_runs = (
        ("cli/collect_jobs.py", "collect_for_title"),
        ("cli/collect_jobs.py", "fetch_description_for_job"),
        ("cli/apply_jobs.py", "apply_to_job"),
        ("cli/apply_jobs_tailored.py", "fetch_job_description"),
    )

    for path, function_name in agent_runs:
        node = _function_node(path, function_name)
        calls = _call_names(node)
        assert calls.count("uuid4") == 1, f"{path}:{function_name}"
        assert calls.count("Agent") == 1, f"{path}:{function_name}"

        get_llm_calls = [
            item for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "get_llm"
        ]
        session_calls = [
            call for call in get_llm_calls
            if any(keyword.arg == "session_id" for keyword in call.keywords)
        ]
        assert len(session_calls) == 1, f"{path}:{function_name}"


def test_browser_agents_share_the_token_efficient_settings():
    agent_runs = (
        ("cli/collect_jobs.py", "collect_for_title"),
        ("cli/collect_jobs.py", "fetch_description_for_job"),
        ("cli/apply_jobs.py", "apply_to_job"),
        ("cli/apply_jobs_tailored.py", "fetch_job_description"),
    )
    expected = {
        "max_actions_per_step": 5,
        "use_vision": "auto",
        "max_history_items": 10,
        "message_compaction": True,
    }

    for path, function_name in agent_runs:
        node = _function_node(path, function_name)
        agent_call = next(
            item for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "Agent"
        )
        settings = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in agent_call.keywords
            if keyword.arg in expected
        }
        assert settings == expected, f"{path}:{function_name}"


def test_backend_get_llm_overrides_forward_session_id():
    tree = ast.parse((ROOT / "backend/main.py").read_text(encoding="utf-8"))
    overrides = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Lambda)
        and any(arg.arg == "session_id" for arg in node.args.args)
        and isinstance(node.body, ast.Call)
        and isinstance(node.body.func, ast.Name)
        and node.body.func.id == "create_llm"
    ]
    assert len(overrides) == 2
    assert all(
        any(keyword.arg == "session_id" for keyword in override.body.keywords)
        for override in overrides
    )
