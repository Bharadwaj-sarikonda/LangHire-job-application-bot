from backend.apply.feature_flags import ApplyFeatureFlags


def test_page_orchestrator_is_opt_in_and_full_agent_fallback_is_on_by_default():
    flags = ApplyFeatureFlags.from_env({})
    assert flags.page_orchestrator is False
    assert flags.local_browser_operator is False
    assert flags.full_agent_fallback is True


def test_feature_flags_accept_common_boolean_spellings_and_fail_safe_on_unknown():
    flags = ApplyFeatureFlags.from_env({
        "ENABLE_PAGE_ORCHESTRATOR": "yes",
        "ENABLE_LOCAL_BROWSER_OPERATOR": "1",
        "ENABLE_FULL_AGENT_FALLBACK": "unexpected",
    })
    assert flags.page_orchestrator is True
    assert flags.local_browser_operator is True
    assert flags.full_agent_fallback is True


def test_feature_flags_can_disable_all_new_behavior():
    flags = ApplyFeatureFlags.from_env({
        "ENABLE_PAGE_ORCHESTRATOR": "off",
        "ENABLE_LOCAL_BROWSER_OPERATOR": "true",
        "ENABLE_FULL_AGENT_FALLBACK": "false",
    })
    assert flags.page_orchestrator is False
    assert flags.local_browser_operator is True
    assert flags.full_agent_fallback is False
