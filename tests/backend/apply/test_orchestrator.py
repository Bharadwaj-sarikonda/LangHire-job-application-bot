from types import SimpleNamespace

import pytest

from backend.apply.browser_executor import ExecutionResult
from backend.apply.feature_flags import ApplyFeatureFlags
from backend.apply.orchestrator import PageOrchestrator
from backend.apply.schemas import (
    AnswerStatus,
    BrowserActionType,
    FieldAnswer,
    FieldControl,
    FormField,
    LiveElementRef,
    LocalBrowserAction,
    PageAnswerBatch,
    PageButton,
    PageSnapshot,
    RecoveryActionType,
    ValidationMessage,
)


def _field(field_id="f1", label="Full name", current="", control="text", options=None):
    ref = LiveElementRef(snapshot_id="s", selector_index=1, backend_node_id=11, frame_id="frame-1")
    return FormField(
        field_id=field_id,
        label=label,
        control_type=control,
        current_value=current,
        controls=[FieldControl(control_ref=ref, value=current)],
        options=options or [],
    )


def _snapshot(fields=(), button=None, *, status=(), title="Application", url="https://jobs.test/apply", snapshot_id="s"):
    buttons = []
    if button:
        buttons = [PageButton(
            label=button,
            control_ref=LiveElementRef(snapshot_id=snapshot_id, selector_index=8, backend_node_id=88),
        )]
    return PageSnapshot(
        snapshot_id=snapshot_id,
        url=url,
        title=title,
        fields=list(fields),
        buttons=buttons,
        status_messages=list(status),
        dom_signature=f"{snapshot_id}:{title}:{button}:{status}",
        captured_at="2026-09-20T12:00:00Z",
    )


class FakeSession:
    def __init__(self):
        self.started = 0

    async def start(self):
        self.started += 1


class FakeObserver:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.last = self.snapshots[-1]

    async def observe(self, _session):
        if self.snapshots:
            self.last = self.snapshots.pop(0)
        return self.last


class FakePlanner:
    def __init__(self, answers=None):
        self.answer_values = answers or {}
        self.calls = []

    async def plan(self, snapshot, _candidate, _job, field_ids=None):
        ids = set(field_ids or snapshot.field_map())
        self.calls.append(ids)
        answers = []
        for field_id in ids:
            value = self.answer_values.get(field_id, "approved")
            if value is None:
                answers.append(FieldAnswer(field_id=field_id, status=AnswerStatus.UNKNOWN, reason="not known"))
            else:
                answers.append(FieldAnswer(field_id=field_id, status=AnswerStatus.ANSWERED, answer=value))
        return PageAnswerBatch(snapshot_id=snapshot.snapshot_id, answers=answers)


class FakeExecutor:
    def __init__(self, *, answer_results=None, action_results=None, observer=None, failure_snapshot=None):
        self.answer_results = list(answer_results or [True])
        self.action_results = list(action_results or [True])
        self.stats = {"deterministic_actions": 0, "stale_element_events": 0, "validation_failures": 0, "retries": 0}
        self.observer = observer
        self.failure_snapshot = failure_snapshot
        self.answer_calls = []
        self.action_calls = []
        self.runner = SimpleNamespace(dispatch=self._dispatch)

    async def _dispatch(self, *_args, **_kwargs):
        return None

    async def execute_answer(self, _session, snapshot, answer):
        self.answer_calls.append(answer.field_id)
        success = self.answer_results.pop(0) if len(self.answer_results) > 1 else self.answer_results[0]
        self.stats["deterministic_actions"] += 1
        after = (
            await self.observer.observe(_session)
            if success and self.observer
            else (self.failure_snapshot or snapshot)
        )
        return ExecutionResult(success, success, snapshot=after, error=None if success else "unverified")

    async def execute_action(self, _session, snapshot, action, *, deterministic=True, **_kwargs):
        self.action_calls.append(action.action)
        success = self.action_results.pop(0) if len(self.action_results) > 1 else self.action_results[0]
        if deterministic:
            self.stats["deterministic_actions"] += 1
        after = await self.observer.observe(_session) if success and self.observer else snapshot
        return ExecutionResult(success, success, snapshot=after, error=None if success else "unverified")


class FakeOperator:
    def __init__(self, action):
        self.action = action
        self.contexts = []

    async def next_action(self, context):
        self.contexts.append(context)
        if isinstance(self.action, Exception):
            raise self.action
        return self.action


@pytest.mark.asyncio
async def test_page_answers_are_planned_once_then_page_transition_is_observed():
    first = _snapshot([_field()], "Continue")
    after_fill = _snapshot([_field(current="approved")], "Continue", snapshot_id="s2")
    confirmed = _snapshot([], status=["Your application has been submitted"], title="Thank you", snapshot_id="s3")
    observer = FakeObserver([first, after_fill, confirmed])
    planner = FakePlanner()
    executor = FakeExecutor(observer=observer)
    session = FakeSession()

    result = await PageOrchestrator(
        browser_session=session,
        planner=planner,
        executor=executor,
        operator=None,
        candidate_context="candidate facts",
        observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, full_agent_fallback=False),
    ).run()

    assert result.success is True
    assert session.started == 1
    assert planner.calls == [{"f1"}]
    assert executor.answer_calls == ["f1"]
    assert executor.action_calls == [BrowserActionType.CLICK]
    assert result.metrics.page_level_big_llm_calls == 1
    assert result.metrics.completed_fields == 1


@pytest.mark.asyncio
async def test_new_conditional_field_gets_a_new_page_level_answer_call_before_continue():
    first = _snapshot([_field()], "Continue")
    conditional = _snapshot([_field(current="approved"), _field("f2", "Do you need sponsorship?")], "Continue", snapshot_id="s2")
    after_second = _snapshot([_field(current="approved"), _field("f2", "Do you need sponsorship?", "approved")], "Continue", snapshot_id="s3")
    confirmed = _snapshot([], status=["Thank you for applying"], title="Thank you", snapshot_id="s4")
    observer = FakeObserver([first, conditional, after_second, after_second, confirmed])
    planner = FakePlanner()
    executor = FakeExecutor(observer=observer)

    result = await PageOrchestrator(
        browser_session=FakeSession(), planner=planner, executor=executor, operator=None,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, full_agent_fallback=False),
    ).run()

    assert result.success is True
    assert planner.calls == [{"f1"}, {"f2"}]
    assert executor.answer_calls == ["f1", "f2"]
    assert executor.action_calls == [BrowserActionType.CLICK]
    assert result.metrics.page_level_big_llm_calls == 2


@pytest.mark.asyncio
async def test_unknown_answer_hands_off_with_checkpoint_and_same_live_session():
    observer = FakeObserver([_snapshot([_field()])])
    planner = FakePlanner({"f1": None})
    executor = FakeExecutor(observer=observer)
    session = FakeSession()
    checkpoints = []

    async def fallback(checkpoint):
        checkpoints.append(checkpoint)
        return "existing-agent-result"

    result = await PageOrchestrator(
        browser_session=session, planner=planner, executor=executor, operator=None,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, full_agent_fallback=True),
        fallback=fallback,
    ).run()

    assert result.handed_to_fallback is True
    assert result.fallback_result == "existing-agent-result"
    assert session.started == 1
    assert "Unresolved fields" in checkpoints[0]
    assert "Completed fields" in checkpoints[0]
    assert executor.answer_calls == []


@pytest.mark.asyncio
async def test_unknown_answer_gets_one_big_llm_clarification_attempt_before_fallback():
    initial = _snapshot([_field()], "Continue")
    after_fill = _snapshot([_field(current="approved")], "Continue", snapshot_id="s2")
    confirmed = _snapshot([], status=["Thank you for applying"], title="Thank you", snapshot_id="s3")
    observer = FakeObserver([initial, after_fill, confirmed])

    class ResolveOnReask:
        def __init__(self):
            self.calls = []

        async def plan(self, snapshot, _candidate, _job, field_ids=None):
            self.calls.append(set(field_ids or []))
            answer = (
                FieldAnswer(field_id="f1", status=AnswerStatus.UNKNOWN)
                if len(self.calls) == 1
                else FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="approved")
            )
            return PageAnswerBatch(snapshot_id=snapshot.snapshot_id, answers=[answer])

    planner = ResolveOnReask()
    executor = FakeExecutor(observer=observer)
    result = await PageOrchestrator(
        browser_session=FakeSession(), planner=planner, executor=executor, operator=None,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, full_agent_fallback=False),
    ).run()

    assert result.success is True
    assert planner.calls == [{"f1"}, {"f1"}]
    assert executor.answer_calls == ["f1"]
    assert result.metrics.page_level_big_llm_calls == 2


@pytest.mark.asyncio
async def test_local_operator_can_recover_failed_deterministic_type_using_approved_answer():
    first = _snapshot([_field()])
    typed = _snapshot([_field(current="approved")], snapshot_id="s2")
    observer = FakeObserver([first, typed])
    planner = FakePlanner()
    executor = FakeExecutor(answer_results=[False], observer=observer)
    operator = FakeOperator(LocalBrowserAction(
        action="TYPE", target_ref="s:1", answer_ref="f1",
    ))
    # Refs must be scoped to the live post-failure snapshot.
    operator.action = LocalBrowserAction(action="TYPE", target_ref="s:1", answer_ref="f1")
    executor.action_results = [True]

    result = await PageOrchestrator(
        browser_session=FakeSession(), planner=planner, executor=executor, operator=operator,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, local_browser_operator=True, full_agent_fallback=False),
    ).run()

    assert result.completed_field_ids == {"f1"}
    assert result.metrics.local_operator_calls == 1
    assert result.metrics.local_recoveries == 1
    assert operator.contexts[0].approved_answer.answer == "approved"
    assert RecoveryActionType.TYPE in operator.contexts[0].allowed_actions


@pytest.mark.asyncio
async def test_submit_without_confirmation_is_never_clicked_twice():
    form = _snapshot([_field()], "Submit Application")
    after_fill = _snapshot([_field(current="approved")], "Submit Application", snapshot_id="s2")
    no_confirmation = _snapshot([_field(current="approved")], "Submit Application", snapshot_id="s3")
    observer = FakeObserver([form, after_fill, no_confirmation])
    executor = FakeExecutor(observer=observer)
    fallback_checkpoints = []

    async def fallback(checkpoint):
        fallback_checkpoints.append(checkpoint)
        return object()

    result = await PageOrchestrator(
        browser_session=FakeSession(), planner=FakePlanner(), executor=executor, operator=None,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, full_agent_fallback=True), fallback=fallback,
    ).run()

    assert result.handed_to_fallback is True
    assert executor.action_calls == [BrowserActionType.CLICK]
    assert "Submission previously attempted: True" in fallback_checkpoints[0]


@pytest.mark.asyncio
async def test_continue_validation_hands_off_without_reclicking_the_same_control():
    initial = _snapshot([_field()], "Continue")
    after_fill = _snapshot([_field(current="approved")], "Continue", snapshot_id="s2")
    invalid_field = _field(current="approved")
    invalid_field.invalid = True
    validation = _snapshot([invalid_field], "Continue", snapshot_id="s3")
    validation.validation_messages = [ValidationMessage(text="Please review this answer")]
    observer = FakeObserver([initial, after_fill, validation])
    executor = FakeExecutor(observer=observer)
    checkpoints = []

    async def fallback(checkpoint):
        checkpoints.append(checkpoint)
        return object()

    result = await PageOrchestrator(
        browser_session=FakeSession(), planner=FakePlanner(), executor=executor, operator=None,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, full_agent_fallback=True), fallback=fallback,
    ).run()

    assert result.handed_to_fallback is True
    assert result.unresolved_field_ids
    assert executor.action_calls == [BrowserActionType.CLICK]


@pytest.mark.asyncio
async def test_validation_feedback_allows_one_big_llm_replan_before_fallback():
    initial = _snapshot([_field()], "Continue")
    invalid_field = _field()
    invalid_field.invalid = True
    validation = _snapshot([invalid_field], "Continue", snapshot_id="s2")
    validation.validation_messages = [ValidationMessage(text="Enter a valid response")]
    corrected = _snapshot([_field(current="corrected")], "Continue", snapshot_id="s3")
    confirmed = _snapshot([], status=["Your application has been submitted"], title="Thank you", snapshot_id="s4")
    observer = FakeObserver([initial, corrected, confirmed])

    class Replanning:
        def __init__(self):
            self.calls = []

        async def plan(self, snapshot, _candidate, _job, field_ids=None):
            self.calls.append((set(field_ids or []), bool(snapshot.validation_messages)))
            answer = "approved" if len(self.calls) == 1 else "corrected"
            return PageAnswerBatch(snapshot_id=snapshot.snapshot_id, answers=[
                FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer=answer),
            ])

    planner = Replanning()
    executor = FakeExecutor(answer_results=[False, True], observer=observer, failure_snapshot=validation)
    result = await PageOrchestrator(
        browser_session=FakeSession(), planner=planner, executor=executor, operator=None,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, full_agent_fallback=False),
    ).run()

    assert result.success is True
    assert len(planner.calls) == 2
    assert planner.calls[1] == ({"f1"}, True)
    assert executor.answer_calls == ["f1", "f1"]


@pytest.mark.asyncio
async def test_local_operator_failure_hands_off_without_changing_live_session():
    observer = FakeObserver([_snapshot([_field()])])
    executor = FakeExecutor(answer_results=[False], observer=observer)
    operator = FakeOperator(RuntimeError("model provider unavailable"))
    session = FakeSession()
    checkpoints = []

    async def fallback(checkpoint):
        checkpoints.append(checkpoint)
        return "agent-took-over"

    result = await PageOrchestrator(
        browser_session=session, planner=FakePlanner(), executor=executor, operator=operator,
        candidate_context="facts", observer=observer,
        flags=ApplyFeatureFlags(page_orchestrator=True, local_browser_operator=True, full_agent_fallback=True),
        fallback=fallback,
    ).run()

    assert result.handed_to_fallback is True
    assert result.fallback_result == "agent-took-over"
    assert session.started == 1
    assert len(operator.contexts) == 1
    assert executor.answer_calls == ["f1"]
    assert "model provider unavailable" not in checkpoints[0]
