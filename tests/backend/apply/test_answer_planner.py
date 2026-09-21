import json
from types import SimpleNamespace

import pytest

from backend.apply.answer_planner import AnswerPlanningError, PageAnswerPlanner
from backend.apply.schemas import (
    AnswerStatus,
    ChoiceOption,
    FieldAnswer,
    FieldControl,
    FormField,
    LiveElementRef,
    PageAnswerBatch,
    PageSnapshot,
)


def _field(field_id, label, control_type="text", options=None):
    ref = LiveElementRef(snapshot_id="snapshot-1", selector_index=1, backend_node_id=10)
    return FormField(
        field_id=field_id,
        label=label,
        control_type=control_type,
        options=options or [],
        controls=[FieldControl(control_ref=ref)],
    )


def _snapshot(*fields):
    return PageSnapshot(
        snapshot_id="snapshot-1",
        url="https://jobs.example/apply",
        title="Application",
        fields=list(fields),
        dom_signature="sig",
        captured_at="2026-09-20T12:00:00Z",
    )


class FakeLLM:
    def __init__(self, completion):
        self.completion = completion
        self.calls = []

    async def ainvoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(completion=self.completion)


@pytest.mark.asyncio
async def test_one_call_answers_all_page_fields_with_strict_structured_output():
    snapshot = _snapshot(
        _field("name", "Full name"),
        _field("years", "Years of Python experience"),
    )
    llm = FakeLLM(PageAnswerBatch(
        snapshot_id="snapshot-1",
        answers=[
            FieldAnswer(field_id="name", status=AnswerStatus.ANSWERED, answer="Jamie Example", source="profile"),
            FieldAnswer(field_id="years", status=AnswerStatus.ANSWERED, answer="4", source="profile"),
        ],
    ))

    result = await PageAnswerPlanner(llm).plan(
        snapshot,
        candidate_context="CANDIDATE PROFILE: name=Jamie Example; years_of_python_experience=4",
        job_context={"title": "Engineer"},
    )

    assert len(llm.calls) == 1
    assert llm.calls[0][1]["output_format"] is PageAnswerBatch
    prompt = llm.calls[0][0][0].content
    assert "name" in prompt and "years" in prompt and "Engineer" in prompt
    assert [answer.answer for answer in result.answers] == ["Jamie Example", "4"]


@pytest.mark.asyncio
async def test_choice_answers_must_match_visible_option_and_are_canonicalized():
    snapshot = _snapshot(_field(
        "authorization",
        "Are you authorized to work?",
        "radio",
        [ChoiceOption(label="Yes", value="Y"), ChoiceOption(label="No", value="N")],
    ))
    llm = FakeLLM({
        "snapshot_id": "snapshot-1",
        "answers": [{"field_id": "authorization", "status": "answered", "answer": "yes", "source": "profile"}],
    })

    result = await PageAnswerPlanner(llm).plan(snapshot, "Profile says authorized.")

    assert result.answers[0].answer == "Y"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers, message",
    [
        ([{"field_id": "one", "status": "answered", "answer": "ok"}], "missing"),
        ([
            {"field_id": "one", "status": "answered", "answer": "ok"},
            {"field_id": "one", "status": "answered", "answer": "ok"},
        ], "duplicate"),
        ([
            {"field_id": "one", "status": "answered", "answer": "ok"},
            {"field_id": "other", "status": "answered", "answer": "ok"},
        ], "extra"),
    ],
)
async def test_rejects_missing_duplicate_or_unexpected_field_ids(answers, message):
    snapshot = _snapshot(_field("one", "Question one"), _field("two", "Question two"))
    llm = FakeLLM({"snapshot_id": "snapshot-1", "answers": answers})

    with pytest.raises(AnswerPlanningError, match=message):
        await PageAnswerPlanner(llm).plan(snapshot, "Context")


@pytest.mark.asyncio
async def test_rejects_answer_for_unresolved_field_and_empty_answered_value():
    snapshot = _snapshot(_field("one", "Question one"))
    unresolved = FakeLLM({
        "snapshot_id": "snapshot-1",
        "answers": [{"field_id": "one", "status": "unknown", "answer": "guess"}],
    })
    with pytest.raises(AnswerPlanningError, match="must not include an answer"):
        await PageAnswerPlanner(unresolved).plan(snapshot, "Context")

    empty = FakeLLM({
        "snapshot_id": "snapshot-1",
        "answers": [{"field_id": "one", "status": "answered", "answer": ""}],
    })
    with pytest.raises(AnswerPlanningError, match="empty answer"):
        await PageAnswerPlanner(empty).plan(snapshot, "Context")


@pytest.mark.asyncio
async def test_unknown_is_explicit_and_password_upload_never_reach_answer_model():
    snapshot = _snapshot(
        _field("question", "What is your certification?"),
        _field("password", "Password", "password"),
        _field("resume", "Resume", "file"),
    )
    llm = FakeLLM({
        "snapshot_id": "snapshot-1",
        "answers": [{"field_id": "question", "status": "unknown", "answer": None, "source": "unknown"}],
    })

    result = await PageAnswerPlanner(llm).plan(snapshot, "No certification is documented.")

    assert len(llm.calls) == 1
    prompt = llm.calls[0][0][0].content
    assert '"password"' not in prompt
    assert '"resume"' not in prompt
    assert {answer.field_id: answer.status for answer in result.answers} == {
        "question": AnswerStatus.UNKNOWN,
        "password": AnswerStatus.UNKNOWN,
        "resume": AnswerStatus.UNKNOWN,
    }
    assert result.answers[1].answer is None


@pytest.mark.asyncio
async def test_skips_llm_when_only_non_semantic_controls_are_requested():
    snapshot = _snapshot(_field("password", "Password", "password"))
    llm = FakeLLM(None)

    result = await PageAnswerPlanner(llm).plan(snapshot, "Candidate context")

    assert llm.calls == []
    assert result.answers[0].status == AnswerStatus.UNKNOWN


@pytest.mark.asyncio
async def test_rejects_stale_snapshot_or_unknown_requested_id():
    snapshot = _snapshot(_field("one", "Question one"))
    llm = FakeLLM({"snapshot_id": "old", "answers": [{"field_id": "one", "status": "unknown"}]})
    with pytest.raises(AnswerPlanningError, match="different snapshot_id"):
        await PageAnswerPlanner(llm).plan(snapshot, "Context")

    with pytest.raises(AnswerPlanningError, match="Unknown field IDs"):
        await PageAnswerPlanner(FakeLLM(None)).plan(snapshot, "Context", field_ids={"missing"})


@pytest.mark.asyncio
async def test_answered_choice_with_non_visible_value_is_rejected():
    snapshot = _snapshot(_field(
        "country",
        "Country",
        "select",
        [ChoiceOption(label="Canada", value="CA")],
    ))
    llm = FakeLLM({
        "snapshot_id": "snapshot-1",
        "answers": [{"field_id": "country", "status": "answered", "answer": "USA"}],
    })

    with pytest.raises(AnswerPlanningError, match="not a visible option"):
        await PageAnswerPlanner(llm).plan(snapshot, "Candidate lives in the US.")
