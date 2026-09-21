"""Model-agnostic local browser recovery interface and strict adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from browser_use.llm.messages import UserMessage

from .browser_executor import BrowserExecutor
from .schemas import (
    AnswerStatus,
    BrowserAction,
    FormField,
    LocalBrowserAction,
    PageSnapshot,
    RecoveryActionType,
)


class BrowserOperatorError(ValueError):
    """Local operator output was invalid, unsafe, or not grounded in the page."""


class BrowserOperatorContextMixin(Protocol):
    snapshot: PageSnapshot
    field: FormField | None
    approved_answer: Any | None
    previous_attempts: list[dict[str, Any]]
    validation_state: list[str]
    allowed_actions: list[RecoveryActionType]
    last_failure: str | None


class BrowserOperator(Protocol):
    """Interface implemented by whichever local browser model is selected later."""

    async def next_action(self, context: "BrowserOperatorContext") -> LocalBrowserAction: ...


_operator_factory: Callable[[], BrowserOperator] | None = None


def register_browser_operator_factory(factory: Callable[[], BrowserOperator] | None) -> None:
    """Install a model-specific factory without coupling the executor to a model."""
    global _operator_factory
    _operator_factory = factory


def create_registered_browser_operator() -> BrowserOperator | None:
    """Return a configured operator, if an integration registered one."""
    return _operator_factory() if _operator_factory is not None else None


class BrowserOperatorContext:
    """Minimal, page-grounded information needed for one recovery decision."""

    def __init__(
        self,
        snapshot: PageSnapshot,
        field: FormField | None,
        approved_answer: Any | None,
        previous_attempts: list[dict[str, Any]],
        validation_state: list[str],
        allowed_actions: list[RecoveryActionType],
        last_failure: str | None = None,
    ):
        self.snapshot = snapshot
        self.field = field
        self.approved_answer = approved_answer
        self.previous_attempts = previous_attempts
        self.validation_state = validation_state
        self.allowed_actions = allowed_actions
        self.last_failure = last_failure

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.model_dump(mode="json"),
            "target_field": self.field.model_dump(mode="json") if self.field else None,
            "approved_answer": (
                {
                    "field_id": self.approved_answer.field_id,
                    "status": self.approved_answer.status.value,
                    "answer": self.approved_answer.answer,
                }
                if self.approved_answer is not None else None
            ),
            "previous_attempts": self.previous_attempts[-8:],
            "validation_state": self.validation_state[-20:],
            "allowed_actions": [action.value for action in self.allowed_actions],
            "last_failure": self.last_failure,
        }


class StructuredBrowserOperator:
    """Adapter for any local chat model that supports structured Pydantic output."""

    def __init__(self, model: Any):
        self.model = model
        self.calls = 0

    async def next_action(self, context: BrowserOperatorContext) -> LocalBrowserAction:
        prompt = (
            "You are a browser interaction recovery operator. You do not answer application questions.\n"
            "Use only the current snapshot and the approved answer supplied below.\n"
            "Return exactly one allowed action in the strict schema.\n"
            "TYPE must reference the approved answer with answer_ref; do not output text or rewrite it.\n"
            "SELECT must match the approved answer and one visible known option.\n"
            "For CLICK, use method=keyboard and a key from Enter, Space, Tab, ArrowDown, ArrowUp, Escape only when keyboard interaction is appropriate.\n"
            "Use only snapshot-scoped target_ref values. Never invent CSS/XPath selectors, JavaScript, "
            "coordinates, answer text, or file paths.\n"
            "If no safe recovery is clear, return FALLBACK.\n\n"
            f"RECOVERY CONTEXT:\n{json.dumps(context.prompt_payload(), ensure_ascii=False, default=str)}"
        )
        self.calls += 1
        try:
            response = await self.model.ainvoke(
                [UserMessage(content=prompt)],
                output_format=LocalBrowserAction,
            )
        except Exception as exc:
            raise BrowserOperatorError(f"Local operator call failed: {type(exc).__name__}") from exc
        action = self._parse(response)
        self._validate(action, context)
        return action

    @staticmethod
    def _parse(response: Any) -> LocalBrowserAction:
        if isinstance(response, LocalBrowserAction):
            return response
        completion = getattr(response, "completion", response)
        if hasattr(completion, "model_dump"):
            completion = completion.model_dump()
        if isinstance(completion, str):
            try:
                completion = json.loads(completion)
            except json.JSONDecodeError as exc:
                raise BrowserOperatorError("Local operator returned invalid JSON.") from exc
        if not isinstance(completion, dict):
            raise BrowserOperatorError("Local operator returned an unsupported response.")
        try:
            return LocalBrowserAction.model_validate(completion)
        except Exception as exc:
            raise BrowserOperatorError(f"Local operator action failed schema validation: {exc}") from exc

    @classmethod
    def _validate(cls, action: LocalBrowserAction, context: BrowserOperatorContext) -> None:
        if action.action not in context.allowed_actions:
            raise BrowserOperatorError("Local operator selected a disallowed action.")

        references = cls._references(context.snapshot)
        if action.action in {RecoveryActionType.CLICK, RecoveryActionType.TYPE, RecoveryActionType.SELECT, RecoveryActionType.CHECK}:
            if not action.target_ref or action.target_ref not in references:
                raise BrowserOperatorError("Action target is not a live reference in the current snapshot.")
        elif action.action == RecoveryActionType.SCROLL:
            if action.target_ref is not None and action.target_ref not in references:
                raise BrowserOperatorError("Scroll target is not a live reference in the current snapshot.")

        if action.action == RecoveryActionType.TYPE:
            cls._validate_answer_ref(action, context)
            assert context.field is not None
            if context.field.control_type in {"password", "file"}:
                raise BrowserOperatorError("Local operator cannot type into credential or file controls.")
        elif action.action == RecoveryActionType.SELECT:
            cls._validate_answer_ref(action, context)
            assert context.field is not None and context.approved_answer is not None
            requested = (action.option or "").strip().casefold()
            approved = (context.approved_answer.answer or "").strip().casefold()
            option = next(
                (
                    item for item in context.field.options
                    if requested in {item.label.casefold(), item.value.casefold()}
                ),
                None,
            )
            if option is None or approved not in {option.label.casefold(), option.value.casefold()}:
                raise BrowserOperatorError("Local operator selected an option that differs from the approved answer.")
        elif action.action == RecoveryActionType.CHECK:
            cls._validate_answer_ref(action, context)
            assert context.field is not None
            if context.field.control_type != "checkbox":
                raise BrowserOperatorError("CHECK is only allowed for checkbox fields.")
            if BrowserExecutor._parse_bool(context.approved_answer.answer if context.approved_answer else None) is None:
                raise BrowserOperatorError("CHECK requires an approved boolean answer.")
        elif action.action == RecoveryActionType.CLICK and context.field is not None:
            if context.field.control_type == "file":
                raise BrowserOperatorError("The local operator cannot open a file chooser.")
            field_references = {
                BrowserExecutor._control_token(control)
                for control in context.field.controls
            }
            field_references.update(
                BrowserExecutor._ref_token(option.control_ref) or ""
                for option in context.field.options
                if option.control_ref is not None
            )
            if action.target_ref not in field_references:
                raise BrowserOperatorError("Field recovery clicks must target the current field or one of its visible options.")

    @staticmethod
    def _validate_answer_ref(action: LocalBrowserAction, context: BrowserOperatorContext) -> None:
        if (
            context.field is None
            or context.approved_answer is None
            or context.approved_answer.status != AnswerStatus.ANSWERED
            or action.answer_ref != context.field.field_id
            or context.approved_answer.field_id != context.field.field_id
        ):
            raise BrowserOperatorError("Action does not reference an approved answer for its target field.")

    @staticmethod
    def _references(snapshot: PageSnapshot) -> set[str]:
        refs: set[str] = set()
        for field in snapshot.fields:
            for control in field.controls:
                refs.add(BrowserExecutor._control_token(control))
            for option in field.options:
                if option.control_ref:
                    refs.add(BrowserExecutor._ref_token(option.control_ref) or "")
        refs.update(BrowserExecutor._ref_token(button.control_ref) or "" for button in snapshot.buttons)
        return refs


def as_browser_action(action: LocalBrowserAction) -> BrowserAction:
    """Convert a validated local action to the executor's broader internal type."""
    return BrowserAction.model_validate(action.model_dump(mode="json"))
