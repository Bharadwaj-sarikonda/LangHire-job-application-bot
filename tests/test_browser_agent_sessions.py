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
        expected_settings = {
            **expected,
            "max_actions_per_step": 1
            if (path, function_name) in {
                ("cli/apply_jobs.py", "apply_to_job"),
                ("cli/collect_jobs.py", "collect_for_title"),
                ("cli/collect_jobs.py", "fetch_description_for_job"),
                ("cli/apply_jobs_tailored.py", "fetch_job_description"),
            }
            else 5,
        }
        assert settings == expected_settings, f"{path}:{function_name}"


def test_browser_agents_enable_coordinate_clicking():
    agent_runs = (
        ("cli/collect_jobs.py", "collect_for_title"),
        ("cli/collect_jobs.py", "fetch_description_for_job"),
        ("cli/apply_jobs.py", "apply_to_job"),
        ("cli/apply_jobs_tailored.py", "fetch_job_description"),
    )

    for path, function_name in agent_runs:
        node = _function_node(path, function_name)
        coordinate_clicking_calls = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "set_coordinate_clicking"
        ]
        assert len(coordinate_clicking_calls) == 1, f"{path}:{function_name}"
        assert ast.literal_eval(coordinate_clicking_calls[0].args[0]) is True


def test_parallel_apply_workers_pass_isolated_browser_profiles():
    tree = ast.parse((ROOT / "backend/main.py").read_text(encoding="utf-8"))
    start_apply = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "start_applying"
    )
    source = ast.unparse(start_apply)
    assert "create_worker_profiles(num_workers)" in source
    assert "browser_profile_dir=worker_profile" in source
    assert "remove_worker_profiles(run_dir)" in source


def test_apply_agents_use_the_worker_specific_profile_when_provided():
    apply_source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    tailored_source = (ROOT / "cli/apply_jobs_tailored.py").read_text(encoding="utf-8")

    assert "user_data_dir=str(browser_profile_dir or BROWSER_PROFILE_DIR)" in apply_source
    assert "browser_profile_dir=browser_profile_dir" in tailored_source
    assert "user_data_dir=str(browser_profile_dir or BROWSER_PROFILE_DIR)" in tailored_source


def test_force_stop_targets_temporary_worker_browsers_too():
    source = (ROOT / "backend/main.py").read_text(encoding="utf-8")
    assert 'profile_paths = ("langhire/browser_profile", "langhire/browser_workers")' in source


def test_worker_event_loop_drains_cancelled_tasks_before_closing():
    source = (ROOT / "backend/main.py").read_text(encoding="utf-8")
    assert "asyncio.gather(*pending, return_exceptions=True)" in source


def test_apply_agent_uses_direct_upload_file_instead_of_native_file_chooser():
    source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    assert "use the built-in upload_file action" in source
    assert "NEVER use click, coordinate clicking, evaluate, or keyboard input" in source
    assert "resume_file.is_file()" in source
    assert "For resume/CV uploads, use JavaScript/evaluate" not in source


def test_apply_agent_recovers_from_stale_required_consent_controls():
    source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    assert "CONSENT CHECKBOXES" in source
    assert "that index is stale: abandon it" in source
    assert "Verify a checked state/aria-checked value/checkmark before submitting" in source
    assert "After two distinct failed consent-control methods" in source


def test_apply_agent_requires_exact_secret_tokens_and_email_verification():
    source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    assert "SECRET TOKEN INTEGRITY" in source
    assert "must be EXACTLY the placeholder and nothing else" in source
    assert "one syntactically valid email address only" in source


def test_linkedin_collector_preserves_filters_when_advancing_pages():
    source = (ROOT / "cli/collect_jobs.py").read_text(encoding="utf-8")
    assert "visible Next button or the next numbered page" in source
    assert "Keep the same job title and all current search filters" in source


def test_apply_agent_uses_native_reliability_features():
    node = _function_node("cli/apply_jobs.py", "apply_to_job")
    source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")

    browser_call = next(
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "BrowserSession"
    )
    browser_settings = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in browser_call.keywords
        if keyword.arg == "cross_origin_iframes"
    }
    assert browser_settings == {"cross_origin_iframes": True}

    agent_call = next(
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "Agent"
    )
    agent_settings = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in agent_call.keywords
        if keyword.arg in {"use_vision", "loop_detection_enabled", "loop_detection_window"}
    }
    assert agent_settings == {
        "use_vision": "auto",
        "loop_detection_enabled": True,
        "loop_detection_window": 5,
    }
    assert not any(
        keyword.arg in {"tools", "exclude_tools"} for keyword in agent_call.keywords
    ), "The APPLY agent must retain Browser-Use's native tool set."
    for native_tool in (
        "dropdown_options",
        "select_dropdown",
        "send_keys",
        "evaluate",
        "screenshot",
    ):
        assert native_tool in source
    assert any(keyword.arg == "available_file_paths" for keyword in agent_call.keywords)
    assert "Never blindly retry an interaction" in source
    assert "verify the selected value" in source
    assert "Request a screenshot only when" in source
    assert "If coordinate clicking fails or is unavailable" in source

    assignments = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MAX_STEPS" for target in item.targets)
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Attribute)
    assert assignments[0].value.attr == "MAX_STEPS"

    run_call = next(
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "run"
    )
    max_steps = next(keyword.value for keyword in run_call.keywords if keyword.arg == "max_steps")
    assert isinstance(max_steps, ast.Name) and max_steps.id == "MAX_STEPS"
    assert any(
        keyword.arg == "register_should_stop_callback"
        for keyword in agent_call.keywords
    )


def test_batch_login_is_checked_once_before_worker_profiles_are_created():
    apply_source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    backend_source = (ROOT / "backend/main.py").read_text(encoding="utf-8")

    assert "async def verify_batch_logins" in apply_source
    assert "if not await verify_batch_logins()" in apply_source
    assert apply_source.index("await verify_batch_logins()") < apply_source.index(
        "create_worker_profiles(num_workers)"
    )
    assert "if not await apply_jobs.verify_batch_logins()" in backend_source


def test_batch_login_preflight_finishes_after_verification_without_closing_tabs():
    source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    assert "or attempt to close tabs" in source
    assert "call done(success=true) immediately" in source
    assert "result = await agent.run(max_steps=15)" in source


def test_otp_instructions_select_the_latest_message_inside_a_gmail_thread():
    source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    assert "BOTTOM-most newest message" in source
    assert "Never use the first code shown in a thread" in source


def test_apply_progress_watchdog_fingerprints_browser_use_dom_state():
    source = (ROOT / "cli/apply_jobs.py").read_text(encoding="utf-8")
    assert '"dom_state"' in source


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
