"""Regression tests for the narrow Browser Use -> Playwright action adapter."""

from types import SimpleNamespace
from pathlib import Path

from browser_use import Tools
from browser_use.agent.views import ActionResult
from browser_use.tools.registry.views import RegisteredAction

import core.browser_use_playwright as adapter


def test_submission_evidence_requires_live_confirmation():
    assert adapter.submission_evidence(
        "https://ats.example.test/application/confirmation",
        "Thank you for applying. We received your application.",
    )
    assert adapter.submission_evidence(
        "https://ats.example.test/jobs/123",
        "Application submitted successfully.",
    )
    assert adapter.submission_evidence(
        "https://ats.example.test/jobs/123",
        "Review your application, then click Submit Application.",
    ) is None


def test_profile_controls_override_hallucinated_phone_and_resolve_secret_tokens():
    auth = {
        "email": "candidate@example.test",
        "account_email": "login@example.test",
        "phone": "201-492-8580",
    }
    phone = adapter.FormOperation(label="Phone Number", value="4805551234")
    email = adapter.FormOperation(label="Email", value="<secret>email</secret>")

    assert adapter._canonical_value(phone, auth) == "201-492-8580"
    assert adapter._canonical_value(email, auth) == "candidate@example.test"


def test_unknown_question_value_is_preserved():
    operation = adapter.FormOperation(label="Why do you want this role?", value="Because I care")
    assert adapter._canonical_value(operation, {"phone": "201-492-8580"}) == "Because I care"


def test_profile_controls_override_identity_and_home_city_but_not_job_location():
    auth = {"full_name": "Ada Lovelace", "address_city": "San Angelo"}

    assert adapter._canonical_value(
        adapter.FormOperation(label="Name", value="Different Name"), auth
    ) == "Ada Lovelace"
    assert adapter._canonical_value(
        adapter.FormOperation(label="Current City", value="Dallas"), auth
    ) == "San Angelo"
    assert adapter._canonical_value(
        adapter.FormOperation(label="Preferred job location", value="Dallas"), auth
    ) == "Dallas"


def test_profile_address_does_not_override_education_or_employment_locations():
    auth = {"address_city": "San Angelo"}

    assert adapter._canonical_value(
        adapter.FormOperation(label="School City", value="Newark"), auth
    ) == "Newark"
    assert adapter._canonical_value(
        adapter.FormOperation(label="Employer Location City", value="Dallas"), auth
    ) == "Dallas"


def test_semantically_equivalent_dropdown_values_match():
    assert adapter._equivalent("United States of America", "United States")
    assert adapter._equivalent("Yes, authorized to work", "Yes")


class _ValidationControl:
    def __init__(self, info):
        self.info = info
        self.fill_calls = []
        self.select_calls = []

    async def is_visible(self):
        return True

    async def evaluate(self, _script):
        return self.info

    async def fill(self, value):
        self.fill_calls.append(value)
        self.info["value"] = value

    async def select_option(self, *, value):
        self.select_calls.append(value)


class _ValidationControls:
    def __init__(self, controls):
        self.controls = controls

    async def count(self):
        return len(self.controls)

    def nth(self, index):
        return self.controls[index]


class _ValidationPage:
    def __init__(self, controls):
        self.controls = _ValidationControls(controls)

    def locator(self, _selector):
        return self.controls


async def test_submit_validation_corrects_wrong_personal_fact_and_keeps_correct_prefill():
    wrong_city = _ValidationControl({
        "hint": "Current City", "tag": "input", "type": "text", "value": "Dallas",
        "checked": False, "optionLabel": "", "selectedText": "", "options": [],
    })
    correct_email = _ValidationControl({
        "hint": "Email", "tag": "input", "type": "email", "value": "ada@example.test",
        "checked": False, "optionLabel": "", "selectedText": "", "options": [],
    })

    corrected = await adapter._validate_authoritative_fields(
        _ValidationPage([wrong_city, correct_email]),
        {"address_city": "San Angelo", "email": "ada@example.test"},
    )

    assert corrected == 1
    assert wrong_city.fill_calls == ["San Angelo"]
    assert correct_email.fill_calls == []


async def test_submit_validation_keeps_equivalent_dropdown_prefill():
    country = _ValidationControl({
        "hint": "Country", "tag": "select", "type": "select-one", "value": "US",
        "checked": False, "optionLabel": "", "selectedText": "United States of America",
        "options": [{"value": "US", "text": "United States of America"}],
    })

    corrected = await adapter._validate_authoritative_fields(
        _ValidationPage([country]), {"address_country": "United States"}
    )

    assert corrected == 0
    assert country.select_calls == []


def test_frame_url_matching_ignores_ats_tracking_query_parameters():
    hint = "https://careers.example.test/login?in_iframe=1&tracking_id=abc"
    actual = "https://careers.example.test/login"
    assert adapter._same_frame_url(hint, actual) is True


class _Locator:
    def __init__(self):
        self.clicked = False
        self.checked = False
        self.input_type = ""
        self.tag_name = "select"
        self.value = ""
        self.files = []
        self.selected_text = ""
        self.visible = True

    @property
    def first(self):
        return self

    async def count(self):
        return 1

    async def is_visible(self):
        return self.visible

    async def click(self):
        self.clicked = True

    async def check(self):
        self.checked = True

    async def is_checked(self):
        return self.checked

    async def fill(self, value):
        self.value = value

    async def press_sequentially(self, value):
        self.value += value

    async def input_value(self):
        return self.value

    async def set_input_files(self, path):
        self.files = [path]

    async def evaluate(self, script, *_args):
        if "HTMLInputElement" in script:
            return self.input_type
        if "tagName" in script:
            return self.tag_name
        return len(self.files)

    async def get_attribute(self, name):
        if name == "aria-expanded":
            return "false"
        if name == "aria-checked":
            return "true" if self.checked else "false"
        return None

    async def select_option(self, *, label):
        self.selected_text = label
        self.value = label

    def locator(self, _selector):
        return self

    async def text_content(self):
        return self.selected_text


class _AriaCheckboxLocator(_Locator):
    def __init__(self):
        super().__init__()
        self.tag_name = "div"
        self.aria_checked = "false"

    async def is_checked(self):
        raise RuntimeError("not a native input")

    async def get_attribute(self, name):
        if name == "role":
            return "checkbox"
        if name == "aria-checked":
            return self.aria_checked
        return await super().get_attribute(name)

    async def click(self):
        self.aria_checked = "true"


class _Page:
    url = "https://example.test/form"

    def __init__(self, locator):
        self._locator = locator
        self.mouse = _Mouse()

    def locator(self, _xpath):
        return self._locator

    def get_by_label(self, _label, exact=True):
        return self._locator

    def get_by_placeholder(self, _placeholder, exact=True):
        return self._locator

    def get_by_role(self, _role, name=None, exact=True):
        return self._locator

    def get_by_text(self, _text, exact=True):
        return self._locator

    async def wait_for_timeout(self, _milliseconds):
        return None


class _Mouse:
    def __init__(self):
        self.clicks = []

    async def click(self, x, y):
        self.clicks.append((x, y))


class _Session:
    def __init__(self, page):
        self._page = page

    async def page(self, _browser_session):
        return self._page


class _BrowserSession:
    llm_screenshot_size = (1000, 500)
    _original_viewport_size = (2000, 1000)
    cdp_client = None

    async def get_element_by_index(self, _index):
        return SimpleNamespace(xpath="html/body/input[1]")

    async def get_current_page_url(self):
        return _Page.url


def _tools_with_fake_playwright(monkeypatch, locator):
    monkeypatch.setattr(adapter, "_PlaywrightSession", lambda: _Session(_Page(locator)))
    tools = Tools()
    original_models = {name: tools.registry.registry.actions[name].param_model for name in ("click", "input", "upload_file", "select_dropdown", "scroll", "send_keys")}
    adapter.install_playwright_actions(tools)
    return tools, original_models


def test_adapter_preserves_native_browser_use_action_models(monkeypatch):
    tools, original_models = _tools_with_fake_playwright(monkeypatch, _Locator())

    for name, model in original_models.items():
        assert tools.registry.registry.actions[name].param_model is model


def test_adapter_installation_is_idempotent(monkeypatch):
    tools, _ = _tools_with_fake_playwright(monkeypatch, _Locator())
    first_handlers = {
        name: tools.registry.registry.actions[name].function
        for name in ("click", "input", "upload_file", "select_dropdown", "scroll", "send_keys", "playwright_fill_form")
    }

    adapter.install_playwright_actions(tools)

    for name, handler in first_handlers.items():
        assert tools.registry.registry.actions[name].function is handler


async def test_click_uses_playwright_for_browser_use_index(monkeypatch):
    locator = _Locator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["click"]

    result = await action.function(params=action.param_model(index=1), browser_session=_BrowserSession())

    assert locator.clicked is True
    assert "Playwright clicked" in result.extracted_content


async def test_coordinate_click_uses_playwright_with_browser_use_scaling(monkeypatch):
    locator = _Locator()
    page = _Page(locator)
    monkeypatch.setattr(adapter, "_PlaywrightSession", lambda: _Session(page))
    tools = Tools()
    tools.set_coordinate_clicking(True)
    adapter.install_playwright_actions(tools)
    action = tools.registry.registry.actions["click"]

    await action.function(params=action.param_model(coordinate_x=250, coordinate_y=125), browser_session=_BrowserSession())

    assert page.mouse.clicks == [(500, 250)]


async def test_checkbox_click_uses_idempotent_playwright_check(monkeypatch):
    locator = _Locator()
    locator.input_type = "checkbox"
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["click"]

    await action.function(params=action.param_model(index=1), browser_session=_BrowserSession())

    assert locator.checked is True
    assert locator.clicked is False


async def test_aria_checkbox_batch_verification_does_not_falsely_fallback(monkeypatch):
    locator = _AriaCheckboxLocator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["playwright_fill_form"]

    result = await action.function(
        params=action.param_model(
            operations=[{"index": 1, "kind": "checkbox", "checked": True, "label": "Agree"}]
        ),
        browser_session=_BrowserSession(),
    )

    assert locator.aria_checked == "true"
    assert result.metadata["executor"] == "playwright"


async def test_form_batch_uses_authoritative_profile_value(monkeypatch):
    locator = _Locator()
    tools = Tools()
    monkeypatch.setattr(adapter, "_PlaywrightSession", lambda: _Session(_Page(locator)))
    adapter.install_playwright_actions(tools, authoritative_values={"phone": "201-492-8580"})
    action = tools.registry.registry.actions["playwright_fill_form"]

    result = await action.function(
        params=action.param_model(operations=[{
            "index": 1,
            "label": "Phone Number",
            "value": "4805551234",
            "kind": "text",
        }]),
        browser_session=_BrowserSession(),
    )

    assert locator.value == "201-492-8580"
    assert result.metadata["executor"] == "playwright"
    assert result.metadata["authoritative_overrides"] == ["text:Phone Number"]
    assert "successfully completed; do not retry" in result.extracted_content


async def test_single_input_reports_authoritative_override_as_success(monkeypatch):
    class CityBrowserSession(_BrowserSession):
        async def get_element_by_index(self, _index):
            return SimpleNamespace(
                xpath="html/body/input[1]",
                node_value="City",
                attributes={"name": "city"},
            )

    locator = _Locator()
    tools = Tools()
    monkeypatch.setattr(adapter, "_PlaywrightSession", lambda: _Session(_Page(locator)))
    adapter.install_playwright_actions(tools, authoritative_values={"address_city": "San Angelo"})
    action = tools.registry.registry.actions["input"]

    result = await action.function(
        params=action.param_model(index=1, text="Newark", clear=True),
        browser_session=CityBrowserSession(),
    )

    assert locator.value == "San Angelo"
    assert result.error is None
    assert result.metadata["authoritative_override"] is True
    assert "successfully completed; do not retry" in result.extracted_content


async def test_input_uses_playwright_and_verifies_value(monkeypatch):
    locator = _Locator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["input"]

    result = await action.function(
        params=action.param_model(index=1, text="candidate@example.com", clear=True),
        browser_session=_BrowserSession(),
    )

    assert locator.value == "candidate@example.com"
    assert "Playwright filled" in result.extracted_content


async def test_upload_uses_playwright_file_input(monkeypatch):
    locator = _Locator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["upload_file"]

    result = await action.function(
        params=action.param_model(index=1, path="/tmp/resume.pdf"),
        browser_session=_BrowserSession(),
    )

    assert locator.files == ["/tmp/resume.pdf"]
    assert "Playwright uploaded" in result.extracted_content


async def test_select_dropdown_uses_playwright_and_verifies_selection(monkeypatch):
    locator = _Locator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["select_dropdown"]

    result = await action.function(
        params=action.param_model(index=1, text="United States"),
        browser_session=_BrowserSession(),
    )

    assert locator.selected_text == "United States"
    assert "Playwright selected" in result.extracted_content


async def test_batch_form_action_uses_one_verified_playwright_operation(monkeypatch):
    locator = _Locator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["playwright_fill_form"]

    result = await action.function(
        params=action.param_model(
            operations=[
                {"index": 1, "label": "Email", "kind": "text", "value": "candidate@example.com"},
                {"index": 1, "label": "Country", "kind": "select", "value": "United States"},
            ]
        ),
        browser_session=_BrowserSession(),
    )

    assert locator.value == "United States"
    assert result.metadata["executor"] == "playwright"
    assert result.metadata["batch_size"] == 2


async def test_batch_form_action_accepts_browser_use_injected_runtime_arguments(monkeypatch):
    """Exercise the same registry path that previously failed in production."""
    locator = _Locator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)

    result = await tools.registry.execute_action(
        "playwright_fill_form",
        {"operations": [{"index": 1, "kind": "text", "value": "candidate@example.com"}]},
        browser_session=_BrowserSession(),
        page_extraction_llm=object(),
        file_system=object(),
        available_file_paths=[],
        extraction_schema={"type": "object"},
    )

    assert locator.value == "candidate@example.com"
    assert result.metadata["executor"] == "playwright"


async def test_batch_form_action_can_resolve_semantic_field_without_index(monkeypatch):
    locator = _Locator()
    tools, _ = _tools_with_fake_playwright(monkeypatch, locator)
    action = tools.registry.registry.actions["playwright_fill_form"]

    result = await action.function(
        params=action.param_model(
            operations=[{"label": "Email", "kind": "text", "value": "candidate@example.com"}]
        ),
        browser_session=_BrowserSession(),
    )

    assert locator.value == "candidate@example.com"
    assert result.metadata["executor"] == "playwright"


async def test_failed_playwright_action_falls_back_to_native_browser_use(monkeypatch):
    fallback_called = False

    async def native_click(**_kwargs):
        nonlocal fallback_called
        fallback_called = True
        return ActionResult(extracted_content="native fallback")

    locator = _Locator()
    locator.visible = False
    monkeypatch.setattr(adapter, "_PlaywrightSession", lambda: _Session(_Page(locator)))
    tools = Tools()
    original = tools.registry.registry.actions["click"]
    tools.registry.registry.actions["click"] = RegisteredAction(
        name=original.name,
        description=original.description,
        function=native_click,
        param_model=original.param_model,
        terminates_sequence=original.terminates_sequence,
        domains=original.domains,
    )
    adapter.install_playwright_actions(tools)
    action = tools.registry.registry.actions["click"]

    result = await action.function(params=action.param_model(index=1), browser_session=_BrowserSession())

    assert fallback_called is True
    assert "Playwright fallback reason: No visible Playwright locator matched index 1" in result.extracted_content
    assert "Browser Use fallback result: native fallback" in result.extracted_content
    assert result.error is not None
    assert "did not verify the intended target/state change" in result.error


async def test_coordinate_fallback_is_error_when_state_does_not_change(monkeypatch):
    async def native_click(**_kwargs):
        return ActionResult(extracted_content="native coordinate fallback")

    locator = _Locator()
    page = _Page(locator)
    monkeypatch.setattr(adapter, "_PlaywrightSession", lambda: _Session(page))
    tools = Tools()
    original = tools.registry.registry.actions["click"]
    tools.registry.registry.actions["click"] = RegisteredAction(
        name=original.name,
        description=original.description,
        function=native_click,
        param_model=original.param_model,
        terminates_sequence=original.terminates_sequence,
        domains=original.domains,
    )
    adapter.install_playwright_actions(tools)

    result = await tools.registry.registry.actions["click"].function(
        params=SimpleNamespace(index=None, coordinate_x=10, coordinate_y=20),
        browser_session=_BrowserSession(),
    )

    assert result.error is not None
    assert "Browser Use fallback result: native coordinate fallback" in result.error
    assert "did not verify the intended target/state change" in result.error


async def test_batch_fallback_preserves_browser_use_error(monkeypatch):
    async def native_input(**_kwargs):
        return ActionResult(error="native input failed")

    locator = _Locator()
    locator.visible = False
    monkeypatch.setattr(adapter, "_PlaywrightSession", lambda: _Session(_Page(locator)))
    tools = Tools()
    original = tools.registry.registry.actions["input"]
    tools.registry.registry.actions["input"] = RegisteredAction(
        name=original.name,
        description=original.description,
        function=native_input,
        param_model=original.param_model,
        terminates_sequence=original.terminates_sequence,
        domains=original.domains,
    )
    adapter.install_playwright_actions(tools)

    result = await tools.registry.registry.actions["playwright_fill_form"].function(
        params=tools.registry.registry.actions["playwright_fill_form"].param_model(
            operations=[{"index": 1, "kind": "text", "value": "candidate@example.com"}]
        ),
        browser_session=_BrowserSession(),
    )

    assert "Playwright fallback reason: No visible Playwright locator matched index 1" in result.error
    assert "Browser Use fallback error: native input failed" in result.error


def test_application_path_installs_the_native_action_adapter():
    root = Path(__file__).parents[3]
    apply_source = (root / "cli/apply_jobs.py").read_text(encoding="utf-8")
    install_index = apply_source.index(
        "install_playwright_actions(", apply_source.index("async def apply_to_job")
    )
    assert apply_source.index("agent.tools.set_coordinate_clicking(True)") < install_index
    assert "agent.tools" in apply_source[install_index:install_index + 100]

    tailored_source = (root / "cli/apply_jobs_tailored.py").read_text(encoding="utf-8")
    assert "from apply_jobs import apply_to_job as _base_apply" in tailored_source
