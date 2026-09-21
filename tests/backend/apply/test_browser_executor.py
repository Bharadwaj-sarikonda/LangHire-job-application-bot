from collections import deque
from types import SimpleNamespace

import pytest

from backend.apply.browser_executor import BrowserExecutor, BrowserUseActionRunner
from backend.apply.schemas import (
    AnswerStatus,
    BrowserAction,
    BrowserActionType,
    ChoiceOption,
    FieldAnswer,
    FieldControl,
    FormField,
    InteractionKey,
    InteractionMethod,
    LiveElementRef,
    PageButton,
    PageSnapshot,
)


def _ref(snapshot_id="s1", index=1, backend_id=10):
    return LiveElementRef(snapshot_id=snapshot_id, selector_index=index, backend_node_id=backend_id)


def _field(field_id="f1", label="Name", control_type="text", *, value=None, checked=None, options=None, ref=None):
    control_ref = ref or _ref()
    return FormField(
        field_id=field_id,
        label=label,
        control_type=control_type,
        current_value=value,
        controls=[FieldControl(control_ref=control_ref, value=value, checked=checked)],
        options=options or [],
    )


def _snapshot(snapshot_id="s1", *, fields=(), url="https://jobs.test/apply", sig="same", buttons=(), above=0, below=1000, tabs=()):
    return PageSnapshot(
        snapshot_id=snapshot_id,
        url=url,
        title="Application",
        fields=list(fields),
        buttons=list(buttons),
        tabs=list(tabs),
        scroll_position={"above": above, "below": below},
        dom_signature=sig,
        captured_at="2026-09-20T12:00:00Z",
    )


class SequenceObserver:
    def __init__(self, snapshots):
        self.snapshots = deque(snapshots)
        self.last = snapshots[-1] if snapshots else None

    async def observe(self, _session):
        if self.snapshots:
            self.last = self.snapshots.popleft()
        return self.last


class FakeRunner:
    def __init__(self, result=None, on_dispatch=None):
        self.result = result or SimpleNamespace(error=None, extracted_content="done")
        self.on_dispatch = on_dispatch
        self.calls = []

    async def dispatch(self, name, params, session, **kwargs):
        self.calls.append((name, params, session, kwargs))
        if self.on_dispatch:
            return self.on_dispatch(name, params, session)
        return self.result


@pytest.mark.asyncio
async def test_browser_use_runner_executes_registered_action_without_raw_parameter_tracing():
    class Params:
        def __init__(self, **kwargs):
            self.values = kwargs

        def model_dump(self, **kwargs):
            return dict(self.values)

    class Envelope:
        def __init__(self, **kwargs):
            self.input = Params(**kwargs["input"])

    class Registry:
        def __init__(self):
            self.calls = []

        def create_action_model(self, *, include_actions):
            assert include_actions == ["input"]
            return Envelope

        async def execute_action(self, **kwargs):
            self.calls.append(kwargs)
            return "executed"

    registry = Registry()
    tools = SimpleNamespace(registry=registry, act=lambda **_kwargs: pytest.fail("Tools.act would trace raw params"))
    session = object()
    result = await BrowserUseActionRunner(tools=tools, action_timeout=1).dispatch(
        "input", {"index": 4, "text": "private answer", "clear": True}, session,
        sensitive_data={"email": "private answer"},
        available_file_paths=["/tmp/resume.pdf"],
    )
    assert result == "executed"
    assert registry.calls[0]["params"]["text"] == "private answer"
    assert registry.calls[0]["browser_session"] is session
    assert registry.calls[0]["sensitive_data"] == {"email": "private answer"}


@pytest.mark.asyncio
async def test_type_requires_browser_use_result_and_exact_post_value_verification():
    before = _snapshot(fields=[_field()])
    after = _snapshot("s2", fields=[_field(ref=_ref("s2"), value="Jamie Example")], sig="changed")
    runner = FakeRunner()
    executor = BrowserExecutor(runner=runner, observer=SequenceObserver([after]))

    result = await executor.execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="Jamie Example"),
    )

    assert result.verified is True
    assert runner.calls[0][0] == "input"
    assert runner.calls[0][1] == {"index": 1, "text": "Jamie Example", "clear": True}
    assert result.attempts[0].verified is True
    assert "Jamie Example" not in repr(result.attempts)


@pytest.mark.asyncio
async def test_type_action_returning_success_without_value_change_is_not_verified():
    before = _snapshot(fields=[_field()])
    unchanged = _snapshot("s2", fields=[_field(ref=_ref("s2"))])
    executor = BrowserExecutor(runner=FakeRunner(), observer=SequenceObserver([unchanged]))

    result = await executor.execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="Jamie Example"),
    )

    assert result.succeeded is True
    assert result.verified is False
    assert result.error == "Browser action was not verified."


@pytest.mark.asyncio
async def test_textarea_uses_the_same_verified_input_path_as_text_fields():
    before = _snapshot(fields=[_field(control_type="textarea")])
    after = _snapshot("s2", fields=[_field(control_type="textarea", ref=_ref("s2"), value="A thoughtful response")], sig="filled")
    session = object()
    runner = FakeRunner()
    result = await BrowserExecutor(runner, SequenceObserver([after])).execute_answer(
        session, before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="A thoughtful response"),
    )
    assert result.verified is True
    assert runner.calls[0][0] == "input"
    assert runner.calls[0][2] is session


@pytest.mark.asyncio
async def test_stale_index_is_reobserved_remapped_and_retried_once():
    before = _snapshot(fields=[_field(ref=_ref("s1", 1, 10))])
    remapped = _snapshot("s2", fields=[_field(ref=_ref("s2", 7, 17))])
    verified = _snapshot("s3", fields=[_field(ref=_ref("s3", 7, 17), value="4 years")], sig="updated")

    def dispatch(name, params, _session):
        if len(runner.calls) == 1:
            return SimpleNamespace(error=None, extracted_content="Element index 1 not available - page may have changed")
        return SimpleNamespace(error=None, extracted_content="typed")

    runner = FakeRunner(on_dispatch=dispatch)
    observer = SequenceObserver([remapped, remapped, verified])
    executor = BrowserExecutor(runner=runner, observer=observer, max_stale_remaps=1)

    result = await executor.execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="4 years"),
    )

    assert result.verified is True
    assert [call[1]["index"] for call in runner.calls] == [1, 7]
    assert executor.stats["stale_element_events"] == 1
    assert executor.stats["retries"] == 1


@pytest.mark.asyncio
async def test_native_select_uses_known_option_and_verifies_selected_state():
    options = [ChoiceOption(label="Canada", value="CA"), ChoiceOption(label="United States", value="US")]
    before = _snapshot(fields=[_field(control_type="select", options=options)])
    after_field = _field(control_type="select", value="CA", ref=_ref("s2"), options=[
        ChoiceOption(label="Canada", value="CA", selected=True),
        ChoiceOption(label="United States", value="US"),
    ])
    after = _snapshot("s2", fields=[after_field], sig="selected")
    runner = FakeRunner()
    executor = BrowserExecutor(runner=runner, observer=SequenceObserver([after]))

    result = await executor.execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="CA"),
    )

    assert result.verified is True
    assert runner.calls[0][0] == "select_dropdown"
    assert runner.calls[0][1]["text"] == "Canada"


@pytest.mark.asyncio
async def test_radio_select_clicks_only_the_exact_matching_option():
    options = [
        ChoiceOption(label="Yes", value="Y", control_ref=_ref("s1", 1, 10)),
        ChoiceOption(label="No", value="N", control_ref=_ref("s1", 2, 11)),
    ]
    before_field = FormField(
        field_id="f1", label="Authorized?", control_type="radio", options=options,
        controls=[
            FieldControl(control_ref=_ref("s1", 1, 10), value="Y", checked=False),
            FieldControl(control_ref=_ref("s1", 2, 11), value="N", checked=False),
        ],
    )
    before = _snapshot(fields=[before_field])
    after_field = FormField(
        field_id="f1", label="Authorized?", control_type="radio", current_value="Y",
        options=[ChoiceOption(label="Yes", value="Y", selected=True), ChoiceOption(label="No", value="N")],
        controls=[
            FieldControl(control_ref=_ref("s2", 1, 10), value="Y", checked=True),
            FieldControl(control_ref=_ref("s2", 2, 11), value="N", checked=False),
        ],
    )
    after = _snapshot("s2", fields=[after_field], sig="selected")
    runner = FakeRunner()
    executor = BrowserExecutor(runner=runner, observer=SequenceObserver([after]))

    result = await executor.execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="Y"),
    )

    assert result.verified is True
    assert runner.calls[0][0] == "click"
    assert runner.calls[0][1]["index"] == 1


@pytest.mark.asyncio
async def test_checkbox_is_idempotent_and_clicks_only_when_state_differs():
    already_checked = _snapshot(fields=[_field(control_type="checkbox", checked=True)])
    no_call_runner = FakeRunner()
    no_call = await BrowserExecutor(no_call_runner, SequenceObserver([])).execute_answer(
        object(), already_checked,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="Yes"),
    )
    assert no_call.verified is True
    assert no_call_runner.calls == []

    before = _snapshot(fields=[_field(control_type="checkbox", checked=True)])
    after = _snapshot("s2", fields=[_field(control_type="checkbox", ref=_ref("s2"), checked=False)], sig="unchecked")
    runner = FakeRunner()
    result = await BrowserExecutor(runner, SequenceObserver([after])).execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="No"),
    )
    assert result.verified is True
    assert runner.calls[0][0] == "click"


@pytest.mark.asyncio
async def test_date_field_uses_typed_answer_and_verifies_value():
    before = _snapshot(fields=[_field(control_type="date")])
    after = _snapshot("s2", fields=[_field(control_type="date", ref=_ref("s2"), value="1990-04-03")], sig="filled")
    runner = FakeRunner()
    result = await BrowserExecutor(runner, SequenceObserver([after])).execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="1990-04-03"),
    )
    assert result.verified is True
    assert runner.calls[0][0] == "input"


@pytest.mark.asyncio
async def test_upload_requires_approved_file_path_and_visible_file_confirmation(tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"pdf")
    before = _snapshot(fields=[_field(control_type="file")])
    after = _snapshot("s2", fields=[_field(control_type="file", ref=_ref("s2"), value="resume.pdf")], sig="uploaded")
    runner = FakeRunner()
    executor = BrowserExecutor(
        runner,
        SequenceObserver([after]),
        available_file_paths=[str(resume)],
        approved_upload_path=str(resume),
    )

    result = await executor.execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.UNKNOWN, answer=None),
    )

    assert result.verified is True
    assert runner.calls[0][0] == "upload_file"
    assert runner.calls[0][1]["path"] == str(resume.resolve())


@pytest.mark.asyncio
async def test_upload_rejects_unapproved_path_without_dispatch(tmp_path):
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"pdf")
    runner = FakeRunner()
    executor = BrowserExecutor(runner, SequenceObserver([]), approved_upload_path=str(resume))
    before = _snapshot(fields=[_field(control_type="file")])

    result = await executor.execute_action(
        object(), before,
        BrowserAction(action=BrowserActionType.UPLOAD, target_ref="s1:1"),
        field_id="f1",
    )

    assert result.verified is False
    assert runner.calls == []


@pytest.mark.asyncio
async def test_next_button_requires_observed_page_transition():
    button = PageButton(label="Continue", control_ref=_ref("s1", 1, 11))
    before = _snapshot(buttons=[button])
    after = _snapshot("s2", url="https://jobs.test/step2", sig="next", buttons=[PageButton(label="Continue", control_ref=_ref("s2", 2, 12))])
    runner = FakeRunner()
    executor = BrowserExecutor(runner, SequenceObserver([after]))

    result = await executor.execute_action(
        object(), before,
        BrowserAction(action=BrowserActionType.CLICK, target_ref="s1:1"),
    )

    assert result.verified is True
    assert runner.calls[0][1] == {"index": 1}


@pytest.mark.asyncio
async def test_popup_or_new_tab_is_a_verified_navigation_change():
    before = _snapshot(buttons=[PageButton(label="Apply", control_ref=_ref("s1", 1, 11))], tabs=[{"url": "https://jobs.test", "title": "Job", "target_id": "tab-1"}])
    after = _snapshot("s2", tabs=[
        {"url": "https://jobs.test", "title": "Job", "target_id": "tab-1"},
        {"url": "https://ats.test/apply", "title": "Application", "target_id": "tab-2"},
    ])
    result = await BrowserExecutor(FakeRunner(), SequenceObserver([after])).execute_action(
        object(), before,
        BrowserAction(action=BrowserActionType.CLICK, target_ref="s1:1"),
    )
    assert result.verified is True
    assert result.snapshot.tabs[-1]["target_id"] == "tab-2"


@pytest.mark.asyncio
async def test_keyboard_recovery_uses_a_strict_key_and_requires_observed_change():
    button = PageButton(label="Continue", control_ref=_ref("s1", 1, 11))
    before = _snapshot(buttons=[button])
    after = _snapshot("s2", url="https://jobs.test/step2", sig="navigated")
    runner = FakeRunner()
    result = await BrowserExecutor(runner, SequenceObserver([after])).execute_action(
        object(), before,
        BrowserAction(action=BrowserActionType.CLICK, target_ref="s1:1", method=InteractionMethod.KEYBOARD, key=InteractionKey.ENTER),
    )
    assert result.verified is True
    assert runner.calls[0][0] == "send_keys"
    assert runner.calls[0][1] == {"keys": "Enter"}


@pytest.mark.asyncio
async def test_iframe_control_keeps_its_frame_scoped_browser_use_reference():
    frame_ref = LiveElementRef(snapshot_id="s1", selector_index=7, backend_node_id=70, frame_id="cross-origin-frame", target_id="tab-1")
    before = _snapshot(fields=[_field(ref=frame_ref)])
    after = _snapshot("s2", fields=[_field(ref=LiveElementRef(snapshot_id="s2", selector_index=7, backend_node_id=70, frame_id="cross-origin-frame", target_id="tab-1"), value="frame value")], sig="frame-filled")
    runner = FakeRunner()
    result = await BrowserExecutor(runner, SequenceObserver([after])).execute_answer(
        object(), before,
        FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="frame value"),
    )
    assert result.verified is True
    assert result.attempts[0].verified is True
    assert runner.calls[0][1]["index"] == 7


@pytest.mark.asyncio
async def test_scroll_and_wait_are_bounded_and_verified():
    before = _snapshot(above=0, below=2000)
    after = _snapshot("s2", above=700, below=1300)
    runner = FakeRunner()
    executor = BrowserExecutor(runner, SequenceObserver([after]))
    scrolled = await executor.execute_action(
        object(), before,
        BrowserAction(action=BrowserActionType.SCROLL, amount=700, direction="down"),
    )
    assert scrolled.verified is True
    assert runner.calls[0][1]["pages"] == 0.7

    waited = await executor.execute_action(
        object(), before,
        BrowserAction(action=BrowserActionType.WAIT, wait_ms=1),
    )
    assert waited.verified is True
