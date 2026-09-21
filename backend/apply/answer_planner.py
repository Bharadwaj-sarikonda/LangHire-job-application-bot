"""One-call, page-scoped semantic answer planning."""

from __future__ import annotations

import json
from typing import Any

from browser_use.llm.messages import UserMessage

from .schemas import AnswerStatus, FieldAnswer, PageAnswerBatch, PageSnapshot


class AnswerPlanningError(ValueError):
    """The answer model returned an incomplete or unsafe page answer batch."""


_NON_SEMANTIC_TYPES = {"password", "file"}
_ALLOWED_SOURCES = {"profile", "saved_qa", "resume", "synthesis", "unknown"}


class PageAnswerPlanner:
    """Ask the configured high-capability model for all unanswered page fields."""

    def __init__(self, llm: Any):
        self.llm = llm

    async def plan(
        self,
        snapshot: PageSnapshot,
        candidate_context: str,
        job_context: dict[str, Any] | None = None,
        field_ids: set[str] | None = None,
    ) -> PageAnswerBatch:
        all_fields = snapshot.field_map()
        requested_ids = set(field_ids) if field_ids is not None else set(all_fields)
        unknown_ids = requested_ids - all_fields.keys()
        if unknown_ids:
            raise AnswerPlanningError(f"Unknown field IDs requested: {sorted(unknown_ids)}")

        # Passwords and file controls need credentials or a controlled upload path,
        # not semantic answer generation. They must never be sent as questions.
        non_semantic = {
            field_id: FieldAnswer(
                field_id=field_id,
                status=AnswerStatus.UNKNOWN,
                reason="This control requires a credential or approved file path.",
                source="unknown",
            )
            for field_id in requested_ids
            if all_fields[field_id].control_type in _NON_SEMANTIC_TYPES
        }
        semantic_ids = requested_ids - non_semantic.keys()
        if not semantic_ids:
            return PageAnswerBatch(snapshot_id=snapshot.snapshot_id, answers=list(non_semantic.values()))

        payload = snapshot.answer_payload(semantic_ids)
        prompt = self._build_prompt(payload, candidate_context, job_context or {})
        response = await self.llm.ainvoke(
            [UserMessage(content=prompt)],
            output_format=PageAnswerBatch,
        )
        answer_batch = self._parse_response(response)
        self._validate_batch(answer_batch, snapshot, semantic_ids)

        # Preserve the snapshot supplied to the model call and put non-semantic
        # controls back into the complete page result.
        answer_batch.snapshot_id = snapshot.snapshot_id
        answer_batch.answers.extend(non_semantic.values())
        self._validate_complete(answer_batch, requested_ids)
        return answer_batch

    @staticmethod
    def _build_prompt(payload: dict[str, Any], candidate_context: str, job_context: dict[str, Any]) -> str:
        return (
            "Answer the application fields on this page in one batch. You determine WHAT each answer is; "
            "a separate browser operator will determine HOW to enter it. Do not describe browser actions.\n\n"
            "Rules:\n"
            "- Return one answer object for every supplied field_id, and no other IDs.\n"
            "- Use only the candidate context and job context below. Never invent or infer missing factual details.\n"
            "- Respect the source priorities and no-guessing instructions in candidate context.\n"
            "- For choice fields, answer with the exact visible option label or value.\n"
            "- If evidence is absent, conflicting, or insufficient, use status=unknown or needs_clarification and answer=null.\n"
            "- If source is supplied, it must be exactly one of: profile, saved_qa, resume, synthesis, unknown.\n"
            "- Include the page snapshot_id exactly as supplied.\n"
            "- Return only the requested structured output.\n\n"
            f"PAGE FIELDS:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            f"JOB CONTEXT:\n{json.dumps(job_context, ensure_ascii=False, default=str)}\n\n"
            f"CANDIDATE CONTEXT:\n{candidate_context}"
        )

    @staticmethod
    def _parse_response(response: Any) -> PageAnswerBatch:
        if isinstance(response, PageAnswerBatch):
            return response
        completion = getattr(response, "completion", response)
        if isinstance(completion, PageAnswerBatch):
            return completion
        if hasattr(completion, "model_dump"):
            completion = completion.model_dump()
        if isinstance(completion, str):
            try:
                completion = json.loads(completion)
            except json.JSONDecodeError as exc:
                raise AnswerPlanningError("Answer model did not return valid JSON.") from exc
        if not isinstance(completion, dict):
            raise AnswerPlanningError("Answer model returned an unsupported structured response.")
        try:
            return PageAnswerBatch.model_validate(completion)
        except Exception as exc:
            raise AnswerPlanningError(f"Answer model returned an invalid page batch: {exc}") from exc

    @staticmethod
    def _validate_batch(batch: PageAnswerBatch, snapshot: PageSnapshot, expected_ids: set[str]) -> None:
        if batch.snapshot_id != snapshot.snapshot_id:
            raise AnswerPlanningError("Answer model returned a different snapshot_id.")
        answer_ids = [answer.field_id for answer in batch.answers]
        if len(answer_ids) != len(set(answer_ids)):
            raise AnswerPlanningError("Answer model returned duplicate field IDs.")
        if set(answer_ids) != expected_ids:
            missing = expected_ids - set(answer_ids)
            extra = set(answer_ids) - expected_ids
            raise AnswerPlanningError(f"Answer field IDs mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

        fields = snapshot.field_map()
        for answer in batch.answers:
            field = fields[answer.field_id]
            if answer.source is not None and answer.source not in _ALLOWED_SOURCES:
                raise AnswerPlanningError(f"Unsupported answer source for {answer.field_id}.")
            if answer.status == AnswerStatus.ANSWERED:
                if not answer.answer or not answer.answer.strip():
                    raise AnswerPlanningError(f"Answered field {answer.field_id} has an empty answer.")
                if field.control_type in _NON_SEMANTIC_TYPES:
                    raise AnswerPlanningError(f"Model must not answer {field.control_type} field {answer.field_id}.")
                if field.options:
                    matched = next(
                        (
                            option for option in field.options
                            if answer.answer.strip().casefold() in {option.label.casefold(), option.value.casefold()}
                        ),
                        None,
                    )
                    if matched is None:
                        raise AnswerPlanningError(f"Answer for choice field {answer.field_id} is not a visible option.")
                    # Canonicalize to the site value while retaining no new text.
                    answer.answer = matched.value
            elif answer.answer not in (None, ""):
                raise AnswerPlanningError(f"Unresolved field {answer.field_id} must not include an answer.")

    @staticmethod
    def _validate_complete(batch: PageAnswerBatch, expected_ids: set[str]) -> None:
        answer_ids = [answer.field_id for answer in batch.answers]
        if len(answer_ids) != len(set(answer_ids)) or set(answer_ids) != expected_ids:
            raise AnswerPlanningError("The final answer batch does not cover the requested page fields exactly once.")
