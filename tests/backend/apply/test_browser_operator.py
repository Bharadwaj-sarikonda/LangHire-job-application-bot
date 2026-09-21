from types import SimpleNamespace

import pytest

from backend.apply.browser_operator import (
    BrowserOperatorContext,
    BrowserOperatorError,
    StructuredBrowserOperator,
    as_browser_action,
)
from backend.apply.schemas import (
    AnswerStatus,
    ChoiceOption,
    FieldAnswer,
    FieldControl,
    FormField,
    LiveElementRef,
    LocalBrowserAction,
    PageButton,
    PageSnapshot,
    RecoveryActionType,
)


def _snapshot(field=None):
    button = PageButton(
        label="Continue",
        control_ref=LiveElementRef(snapshot_id="s1", selector_index=3, backend_node_id=30),
    )
    return PageSnapshot(
        snapshot_id="s1",
        url="https://jobs.test/apply",
        title="Apply",
        fields=[field] if field else [],
        buttons=[button],
        validation_messages=[],
        dom_signature="sig",
        captured_at="2026-09-20T12:00:00Z",
    )


def _field(control_type="text", options=None):
    ref = LiveElementRef(snapshot_id="s1", selector_index=1, backend_node_id=10)
    return FormField(
        field_id="f1",
        label="Question",
        control_type=control_type,
        controls=[FieldControl(control_ref=ref, value="")],
        options=options or [],
    )


def _context(field=None, answer=None, allowed=None):
    return BrowserOperatorContext(
        snapshot=_snapshot(field),
        field=field,
        approved_answer=answer,
        previous_attempts=[{"action": "TYPE", "error_code": "action_error"}],
        validation_state=["Choose a valid option"],
        allowed_actions=allowed or [RecoveryActionType.CLICK],
        last_failure="indexed click had no visible effect",
    )


class FakeModel:
    def __init__(self, completion=None, error=None):
        self.completion = completion
        self.error = error
        self.calls = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(completion=self.completion)


@pytest.mark.asyncio
async def test_operator_returns_only_strict_allowed_browser_action():
    model = FakeModel({"action": "CLICK", "target_ref": "s1:3"})
    operator = StructuredBrowserOperator(model)
    context = _context(allowed=[RecoveryActionType.CLICK])

    action = await operator.next_action(context)

    assert action.action == RecoveryActionType.CLICK
    assert as_browser_action(action).target_ref == "s1:3"
    assert operator.calls == 1
    assert model.calls[0][1]["output_format"] is LocalBrowserAction
    prompt = model.calls[0][0][0].content
    assert "approved_answer" in prompt
    assert "validation_state" in prompt
    assert "allowed_actions" in prompt
    assert "indexed click had no visible effect" in prompt


@pytest.mark.asyncio
async def test_operator_rejects_generated_text_extra_fields_and_arbitrary_selectors():
    raw_text = FakeModel({"action": "TYPE", "target_ref": "s1:1", "text": "invented answer"})
    with pytest.raises(BrowserOperatorError, match="schema validation"):
        await StructuredBrowserOperator(raw_text).next_action(
            _context(_field(), FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="approved"), [RecoveryActionType.TYPE])
        )

    selector = FakeModel({"action": "CLICK", "target_ref": "#submit"})
    with pytest.raises(BrowserOperatorError, match="not a live reference"):
        await StructuredBrowserOperator(selector).next_action(_context(allowed=[RecoveryActionType.CLICK]))


@pytest.mark.asyncio
async def test_type_must_reference_the_approved_answer_for_current_field():
    field = _field()
    answer = FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="approved")
    missing_ref = FakeModel({"action": "TYPE", "target_ref": "s1:1"})
    with pytest.raises(BrowserOperatorError, match="approved answer"):
        await StructuredBrowserOperator(missing_ref).next_action(
            _context(field, answer, [RecoveryActionType.TYPE])
        )

    wrong_ref = FakeModel({"action": "TYPE", "target_ref": "s1:1", "answer_ref": "other"})
    with pytest.raises(BrowserOperatorError, match="approved answer"):
        await StructuredBrowserOperator(wrong_ref).next_action(
            _context(field, answer, [RecoveryActionType.TYPE])
        )


@pytest.mark.asyncio
async def test_select_must_match_both_visible_option_and_approved_answer():
    field = _field("select", [ChoiceOption(label="Canada", value="CA"), ChoiceOption(label="United States", value="US")])
    answer = FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="CA")
    accepted = FakeModel({
        "action": "SELECT", "target_ref": "s1:1", "answer_ref": "f1", "option": "Canada",
    })
    result = await StructuredBrowserOperator(accepted).next_action(
        _context(field, answer, [RecoveryActionType.SELECT])
    )
    assert result.option == "Canada"

    incorrect = FakeModel({
        "action": "SELECT", "target_ref": "s1:1", "answer_ref": "f1", "option": "United States",
    })
    with pytest.raises(BrowserOperatorError, match="differs from the approved answer"):
        await StructuredBrowserOperator(incorrect).next_action(
            _context(field, answer, [RecoveryActionType.SELECT])
        )


@pytest.mark.asyncio
async def test_check_requires_checkbox_and_approved_boolean_answer():
    field = _field("checkbox")
    answer = FieldAnswer(field_id="f1", status=AnswerStatus.ANSWERED, answer="Yes")
    model = FakeModel({"action": "CHECK", "target_ref": "s1:1", "answer_ref": "f1"})
    result = await StructuredBrowserOperator(model).next_action(
        _context(field, answer, [RecoveryActionType.CHECK])
    )
    assert result.action == RecoveryActionType.CHECK

    non_checkbox = FakeModel({"action": "CHECK", "target_ref": "s1:1", "answer_ref": "f1"})
    with pytest.raises(BrowserOperatorError, match="only allowed for checkbox"):
        await StructuredBrowserOperator(non_checkbox).next_action(
            _context(_field("text"), answer, [RecoveryActionType.CHECK])
        )


@pytest.mark.asyncio
async def test_operator_rejects_actions_not_allowlisted_and_date_upload_actions():
    model = FakeModel({"action": "WAIT", "wait_ms": 50})
    with pytest.raises(BrowserOperatorError, match="disallowed"):
        await StructuredBrowserOperator(model).next_action(_context(allowed=[RecoveryActionType.CLICK]))

    with pytest.raises(Exception):
        LocalBrowserAction(action="UPLOAD")
    with pytest.raises(Exception):
        LocalBrowserAction(action="DATE")


@pytest.mark.asyncio
async def test_model_errors_are_reported_without_exposing_exception_text():
    model = FakeModel(error=RuntimeError("token=secret-value"))
    with pytest.raises(BrowserOperatorError, match="RuntimeError") as exc:
        await StructuredBrowserOperator(model).next_action(_context(allowed=[RecoveryActionType.FALLBACK]))
    assert "secret-value" not in str(exc.value)

