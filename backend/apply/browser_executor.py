"""Deterministic Browser-Use execution with fresh-state verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .page_observer import PageObserver
from .schemas import (
    ActionAttempt,
    AnswerStatus,
    BrowserAction,
    BrowserActionType,
    FieldAnswer,
    FieldControl,
    FormField,
    InteractionKey,
    InteractionMethod,
    LiveElementRef,
    PageButton,
    PageSnapshot,
)


class ActionRunner(Protocol):
    async def dispatch(
        self,
        name: str,
        params: dict[str, Any],
        browser_session: Any,
        *,
        sensitive_data: dict[str, Any] | None = None,
        available_file_paths: list[str] | None = None,
    ) -> Any: ...


class BrowserUseActionRunner:
    """Dispatch through Browser-Use's existing Tools registry on the same session."""

    def __init__(self, tools: Any | None = None, action_timeout: float = 30.0):
        if tools is None:
            from browser_use.tools.service import Tools

            tools = Tools()
        self.tools = tools
        self.action_timeout = action_timeout

    async def dispatch(
        self,
        name: str,
        params: dict[str, Any],
        browser_session: Any,
        *,
        sensitive_data: dict[str, Any] | None = None,
        available_file_paths: list[str] | None = None,
    ) -> Any:
        action_model = self.tools.registry.create_action_model(include_actions=[name])
        action = action_model(**{name: params})
        validated_params = getattr(action, name).model_dump(exclude_unset=True)
        # Call the same registered Browser-Use action directly. Tools.act adds
        # tracing spans whose inputs contain raw action parameters; bypassing
        # that wrapper keeps application answers out of trace/log payloads.
        return await asyncio.wait_for(
            self.tools.registry.execute_action(
                action_name=name,
                params=validated_params,
                browser_session=browser_session,
                sensitive_data=sensitive_data or {"redact_inputs": ""},
                available_file_paths=available_file_paths or [],
            ),
            timeout=self.action_timeout,
        )


@dataclass
class ExecutionResult:
    succeeded: bool
    verified: bool
    stale: bool = False
    unsupported: bool = False
    error: str | None = None
    action_result: Any = None
    snapshot: PageSnapshot | None = None
    attempts: list[ActionAttempt] = field(default_factory=list)


class BrowserExecutor:
    """Execute allowed actions and verify their browser-visible outcome."""

    _ALLOWED_METHODS = {"native", "keyboard"}
    _STALE_MARKERS = (
        "not available - page may have changed",
        "index .* not available",
        "stale element",
        "no node with given id",
        "could not find node",
        "backendnodeid",
        "context was destroyed",
    )

    def __init__(
        self,
        runner: ActionRunner | None = None,
        observer: PageObserver | None = None,
        *,
        max_stale_remaps: int = 1,
        sensitive_data: dict[str, Any] | None = None,
        available_file_paths: list[str] | None = None,
        approved_upload_path: str | None = None,
    ):
        self.runner = runner or BrowserUseActionRunner()
        self.observer = observer or PageObserver()
        self.max_stale_remaps = max(0, max_stale_remaps)
        self.sensitive_data = sensitive_data or {"redact_inputs": ""}
        self.available_file_paths = [str(Path(path).resolve()) for path in (available_file_paths or [])]
        self.approved_upload_path = str(Path(approved_upload_path).resolve()) if approved_upload_path else None
        self.stats: dict[str, int] = {
            "deterministic_actions": 0,
            "stale_element_events": 0,
            "validation_failures": 0,
            "retries": 0,
        }

    async def execute_answer(
        self,
        browser_session: Any,
        snapshot: PageSnapshot,
        answer: FieldAnswer,
    ) -> ExecutionResult:
        initial_field = snapshot.field_map().get(answer.field_id)
        if initial_field is not None and initial_field.control_type == "file":
            expected_name = Path(self.approved_upload_path).name if self.approved_upload_path else None
            if expected_name and initial_field.current_value and Path(initial_field.current_value).name == expected_name:
                return ExecutionResult(True, True, snapshot=snapshot)
            upload = BrowserAction(
                action=BrowserActionType.UPLOAD,
                target_ref=self._control_token(initial_field.controls[0]) if initial_field.controls else None,
            )
            return await self.execute_action(browser_session, snapshot, upload, field_id=answer.field_id)
        if answer.status != AnswerStatus.ANSWERED:
            return ExecutionResult(False, False, error="Field has no approved answer.", snapshot=snapshot)
        if initial_field is not None and self._field_matches_answer(initial_field, answer):
            return ExecutionResult(True, True, snapshot=snapshot)

        attempts: list[ActionAttempt] = []
        current = snapshot
        for remap_count in range(self.max_stale_remaps + 1):
            field = current.field_map().get(answer.field_id)
            if field is None:
                return ExecutionResult(False, False, stale=True, error="Field disappeared during a DOM update.", snapshot=current, attempts=attempts)
            action = self._deterministic_action(field, answer)
            if action is None:
                return ExecutionResult(False, False, unsupported=True, error="No deterministic action for this control.", snapshot=current, attempts=attempts)

            outcome = await self.execute_action(
                browser_session,
                current,
                action,
                field_id=field.field_id,
                answer=answer,
            )
            attempts.extend(outcome.attempts)
            if outcome.verified:
                outcome.attempts = attempts
                return outcome
            if outcome.stale and remap_count < self.max_stale_remaps:
                self.stats["retries"] += 1
                current = await self.observer.observe(browser_session)
                continue
            outcome.attempts = attempts
            return outcome
        return ExecutionResult(False, False, error="Bounded stale-reference recovery exhausted.", snapshot=current, attempts=attempts)

    async def execute_action(
        self,
        browser_session: Any,
        snapshot: PageSnapshot,
        action: BrowserAction,
        *,
        field_id: str | None = None,
        answer: FieldAnswer | None = None,
        approved_answers: dict[str, FieldAnswer] | None = None,
        deterministic: bool = True,
    ) -> ExecutionResult:
        """Execute one strict action. Local-model values are resolved server-side."""
        if action.method is not None and action.method.value not in self._ALLOWED_METHODS:
            return ExecutionResult(False, False, error="Unsupported interaction method.", snapshot=snapshot)
        if action.key is not None and (action.method != InteractionMethod.KEYBOARD or action.action != BrowserActionType.CLICK):
            return ExecutionResult(False, False, error="Keys are only supported for keyboard-assisted clicks.", snapshot=snapshot)
        if action.action in {BrowserActionType.RETRY, BrowserActionType.DONE, BrowserActionType.FALLBACK}:
            return ExecutionResult(
                succeeded=True,
                verified=action.action in {BrowserActionType.DONE, BrowserActionType.FALLBACK},
                snapshot=snapshot,
            )

        resolved_answer = answer
        if action.answer_ref is not None:
            if action.answer_ref != field_id:
                return ExecutionResult(False, False, error="Action referenced a different approved answer.", snapshot=snapshot)
            resolved_answer = resolved_answer or (approved_answers or {}).get(action.answer_ref)
        if action.action in {BrowserActionType.TYPE, BrowserActionType.SELECT, BrowserActionType.CHECK, BrowserActionType.DATE}:
            if resolved_answer is None or resolved_answer.status != AnswerStatus.ANSWERED:
                return ExecutionResult(False, False, error="Action requires an approved answer reference.", snapshot=snapshot)

        field = snapshot.field_map().get(field_id) if field_id else None
        target = self._resolve_target(snapshot, action.target_ref)
        if action.target_ref and target is None:
            return ExecutionResult(False, False, stale=True, error="Target reference is not in this snapshot.", snapshot=snapshot)

        name: str | None = None
        params: dict[str, Any] = {}
        expected_value: str | None = None
        expected_checked: bool | None = None

        if action.action in {BrowserActionType.TYPE, BrowserActionType.DATE}:
            if field is None or resolved_answer is None:
                return ExecutionResult(False, False, error="Typing requires a current field and approved answer.", snapshot=snapshot)
            control = self._choose_control(field, action.target_ref)
            if control is None:
                return ExecutionResult(False, False, stale=True, error="No live control reference for field.", snapshot=snapshot)
            expected_value = resolved_answer.answer
            name = "input"
            params = {"index": control.control_ref.selector_index, "text": expected_value, "clear": True}

        elif action.action == BrowserActionType.SELECT:
            if field is None or resolved_answer is None:
                return ExecutionResult(False, False, error="Select requires a current field and approved answer.", snapshot=snapshot)
            option = self._matching_option(field, action.option or resolved_answer.answer or "")
            if option is None:
                return ExecutionResult(False, False, error="Option is not visible or does not match the approved answer.", snapshot=snapshot)
            if option.selected:
                return ExecutionResult(True, True, snapshot=snapshot)
            if option.control_ref is not None:
                name = "click"
                params = {"index": option.control_ref.selector_index}
                expected_value = option.value
            else:
                control = self._choose_control(field, action.target_ref)
                if control is None:
                    return ExecutionResult(False, False, stale=True, error="Select control reference is stale.", snapshot=snapshot)
                name = "select_dropdown"
                params = {"index": control.control_ref.selector_index, "text": option.label}
                expected_value = option.value

        elif action.action == BrowserActionType.CHECK:
            if field is None or resolved_answer is None:
                return ExecutionResult(False, False, error="Check requires a current field and approved answer.", snapshot=snapshot)
            expected_checked = self._parse_bool(resolved_answer.answer)
            if expected_checked is None:
                return ExecutionResult(False, False, error="Approved answer is not a supported boolean value.", snapshot=snapshot)
            control = self._choose_control(field, action.target_ref)
            if control is None:
                return ExecutionResult(False, False, stale=True, error="Checkbox control reference is stale.", snapshot=snapshot)
            if control.checked is expected_checked:
                return ExecutionResult(True, True, snapshot=snapshot)
            name = "click"
            params = {"index": control.control_ref.selector_index}

        elif action.action == BrowserActionType.CLICK:
            if target is None:
                return ExecutionResult(False, False, error="Click requires a current snapshot-scoped target.", snapshot=snapshot)
            if action.method == InteractionMethod.KEYBOARD:
                name = "send_keys"
                key = action.key or InteractionKey.ENTER
                params = {"keys": key.value}
            else:
                name = "click"
                params = {"index": target.selector_index}
            if field is not None and resolved_answer is not None:
                expected_value = resolved_answer.answer

        elif action.action == BrowserActionType.SCROLL:
            index = target.selector_index if target is not None else 0
            pages = min(2.0, max(0.25, (action.amount or 1000) / 1000.0))
            name = "scroll"
            params = {"down": (action.direction or "down").lower() != "up", "pages": pages, "index": index}

        elif action.action == BrowserActionType.UPLOAD:
            if field is None:
                return ExecutionResult(False, False, error="Upload requires a current file field.", snapshot=snapshot)
            path = self.approved_upload_path
            control = self._choose_control(field, action.target_ref)
            if not path or path not in self.available_file_paths or control is None:
                return ExecutionResult(False, False, error="Upload path or file control is not approved.", snapshot=snapshot)
            name = "upload_file"
            params = {"index": control.control_ref.selector_index, "path": path}
            expected_value = Path(path).name

        elif action.action == BrowserActionType.WAIT:
            delay_ms = min(10000, max(0, action.wait_ms or 250))
            await asyncio.sleep(delay_ms / 1000)
            after = await self.observer.observe(browser_session)
            if deterministic:
                self.stats["deterministic_actions"] += 1
            attempt = self._attempt(action, snapshot, field_id, "wait", True, True, None, "wait completed")
            return ExecutionResult(True, True, snapshot=after, attempts=[attempt])

        else:
            return ExecutionResult(False, False, unsupported=True, error="Unsupported browser action.", snapshot=snapshot)

        if target is not None and target.snapshot_id != snapshot.snapshot_id:
            return ExecutionResult(False, False, stale=True, error="Target reference belongs to an older snapshot.", snapshot=snapshot)

        if deterministic:
            self.stats["deterministic_actions"] += 1
        dispatch_error = None
        action_result = None
        try:
            action_result = await self.runner.dispatch(
                name,
                params,
                browser_session,
                sensitive_data=self.sensitive_data,
                available_file_paths=self.available_file_paths,
            )
            dispatch_error = self._action_error(action_result)
        except Exception as exc:
            dispatch_error = f"{type(exc).__name__}: {exc}"

        after = await self.observer.observe(browser_session)
        stale = self._is_stale(dispatch_error or "")
        if stale:
            self.stats["stale_element_events"] += 1
        verified = self._verify(
            action.action,
            snapshot,
            after,
            field_id,
            expected_value,
            expected_checked,
        )
        if not verified and after.validation_messages != snapshot.validation_messages:
            self.stats["validation_failures"] += 1
        attempt = self._attempt(
            action,
            snapshot,
            field_id,
            name or "unsupported",
            dispatch_error is None,
            verified,
            "stale_dom" if stale else ("action_error" if dispatch_error else None),
            self._observed_change(snapshot, after),
        )
        return ExecutionResult(
            succeeded=dispatch_error is None,
            verified=verified,
            stale=stale,
            error=dispatch_error if dispatch_error else (None if verified else "Browser action was not verified."),
            action_result=action_result,
            snapshot=after,
            attempts=[attempt],
        )

    def _deterministic_action(self, field: FormField, answer: FieldAnswer) -> BrowserAction | None:
        if field.control_type in {"text", "textarea", "contenteditable", "email", "tel", "url", "number", "search", "date", "datetime-local", "month", "time", "week"}:
            kind = BrowserActionType.DATE if field.control_type in {"date", "datetime-local", "month", "time", "week"} else BrowserActionType.TYPE
            return BrowserAction(action=kind, target_ref=self._control_token(field.controls[0]) if field.controls else None, answer_ref=field.field_id)
        if field.control_type == "select" or field.control_type == "combobox":
            option = self._matching_option(field, answer.answer or "")
            if option is None:
                return None
            target = option.control_ref or (field.controls[0].control_ref if field.controls else None)
            return BrowserAction(action=BrowserActionType.SELECT, target_ref=self._ref_token(target), answer_ref=field.field_id, option=option.label)
        if field.control_type == "radio":
            option = self._matching_option(field, answer.answer or "")
            if option is None or option.control_ref is None:
                return None
            return BrowserAction(action=BrowserActionType.SELECT, target_ref=self._ref_token(option.control_ref), answer_ref=field.field_id, option=option.label)
        if field.control_type == "checkbox":
            return BrowserAction(action=BrowserActionType.CHECK, target_ref=self._control_token(field.controls[0]) if field.controls else None, answer_ref=field.field_id)
        if field.control_type == "file":
            return BrowserAction(action=BrowserActionType.UPLOAD, target_ref=self._control_token(field.controls[0]) if field.controls else None)
        return None

    @classmethod
    def _field_matches_answer(cls, field: FormField, answer: FieldAnswer) -> bool:
        expected = (answer.answer or "").strip().casefold()
        if not expected:
            return False
        if field.control_type == "checkbox":
            wanted = cls._parse_bool(answer.answer)
            return wanted is not None and any(control.checked is wanted for control in field.controls)
        if field.current_value and field.current_value.strip().casefold() == expected:
            return True
        return any(
            option.selected
            and expected in {option.value.strip().casefold(), option.label.strip().casefold()}
            for option in field.options
        )

    def _verify(
        self,
        action: BrowserActionType,
        before: PageSnapshot,
        after: PageSnapshot,
        field_id: str | None,
        expected_value: str | None,
        expected_checked: bool | None,
    ) -> bool:
        if action == BrowserActionType.WAIT:
            return True
        if action == BrowserActionType.SCROLL:
            return before.scroll_position != after.scroll_position or before.dom_signature != after.dom_signature
        if field_id:
            field = after.field_map().get(field_id)
            if field is None:
                return False
            if expected_checked is not None:
                return any(control.checked is expected_checked for control in field.controls)
            if expected_value is not None:
                if field.control_type == "file":
                    return bool(field.current_value and Path(field.current_value).name == expected_value)
                values = {field.current_value or ""}
                values.update(option.value for option in field.options if option.selected)
                values.update(option.label for option in field.options if option.selected)
                return any(value.casefold() == expected_value.casefold() for value in values if value)
            return False
        return (
            before.url != after.url
            or before.page_ref != after.page_ref
            or before.dom_signature != after.dom_signature
            or before.tabs != after.tabs
            or before.validation_messages != after.validation_messages
        )

    @staticmethod
    def _resolve_target(snapshot: PageSnapshot, token: str | None) -> LiveElementRef | None:
        if token in (None, "page"):
            return None
        for field in snapshot.fields:
            for control in field.controls:
                if BrowserExecutor._ref_token(control.control_ref) == token:
                    return control.control_ref
            for option in field.options:
                if option.control_ref and BrowserExecutor._ref_token(option.control_ref) == token:
                    return option.control_ref
        for button in snapshot.buttons:
            if BrowserExecutor._ref_token(button.control_ref) == token:
                return button.control_ref
        return None

    @staticmethod
    def _ref_token(ref: LiveElementRef | None) -> str | None:
        if ref is None:
            return None
        return f"{ref.snapshot_id}:{ref.selector_index}"

    @classmethod
    def _control_token(cls, control: FieldControl) -> str:
        return cls._ref_token(control.control_ref) or ""

    @classmethod
    def _choose_control(cls, field: FormField, token: str | None) -> FieldControl | None:
        if token:
            return next((control for control in field.controls if cls._ref_token(control.control_ref) == token), None)
        return field.controls[0] if len(field.controls) == 1 else None

    @staticmethod
    def _matching_option(field: FormField, requested: str) -> Any | None:
        wanted = requested.strip().casefold()
        return next((option for option in field.options if wanted in {option.label.casefold(), option.value.casefold()}), None)

    @staticmethod
    def _parse_bool(value: str | None) -> bool | None:
        if value is None:
            return None
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "y", "1", "checked", "agree", "accept"}:
            return True
        if normalized in {"false", "no", "n", "0", "unchecked", "do not agree", "decline"}:
            return False
        return None

    @staticmethod
    def _action_error(result: Any) -> str | None:
        if result is None:
            return None
        error = getattr(result, "error", None)
        if error:
            return str(error)
        extracted = getattr(result, "extracted_content", None)
        if extracted and BrowserExecutor._is_stale(str(extracted)):
            return str(extracted)
        return None

    @classmethod
    def _is_stale(cls, error: str) -> bool:
        import re

        lowered = error.casefold()
        return any(re.search(marker, lowered) for marker in cls._STALE_MARKERS)

    @staticmethod
    def _observed_change(before: PageSnapshot, after: PageSnapshot) -> str | None:
        if before.url != after.url:
            return "url_changed"
        if before.page_ref != after.page_ref:
            return "page_changed"
        if before.tabs != after.tabs:
            return "tabs_changed"
        if before.dom_signature != after.dom_signature:
            return "dom_changed"
        if before.scroll_position != after.scroll_position:
            return "scroll_changed"
        if before.validation_messages != after.validation_messages:
            return "validation_changed"
        return None

    @staticmethod
    def _attempt(action, snapshot, field_id, method, succeeded, verified, error_code, observed_change) -> ActionAttempt:
        return ActionAttempt(
            action=action.action,
            snapshot_id=snapshot.snapshot_id,
            field_id=field_id,
            method=method,
            succeeded=succeeded,
            verified=verified,
            error_code=error_code,
            observed_change=observed_change,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
