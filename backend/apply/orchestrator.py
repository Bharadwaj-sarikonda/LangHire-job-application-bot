"""Page-scoped apply control flow around an existing Browser-Use session."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

from .answer_planner import PageAnswerPlanner
from .browser_executor import BrowserExecutor, ExecutionResult
from .browser_operator import BrowserOperator, BrowserOperatorContext, as_browser_action
from .feature_flags import ApplyFeatureFlags
from .metrics import OrchestrationMetrics
from .page_observer import PageObserver
from .schemas import (
    AnswerStatus,
    BrowserAction,
    BrowserActionType,
    FieldAnswer,
    FormField,
    PageButton,
    PageSnapshot,
    RecoveryActionType,
)


Fallback = Callable[[str], Awaitable[Any]]


@dataclass
class OrchestrationOutcome:
    success: bool = False
    fallback_result: Any | None = None
    error: str | None = None
    final_snapshot: PageSnapshot | None = None
    answers: dict[str, FieldAnswer] = field(default_factory=dict)
    fields: dict[str, FormField] = field(default_factory=dict)
    completed_field_ids: set[str] = field(default_factory=set)
    unresolved_field_ids: set[str] = field(default_factory=set)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    metrics: OrchestrationMetrics = field(default_factory=OrchestrationMetrics)

    @property
    def handed_to_fallback(self) -> bool:
        return self.fallback_result is not None


class PageOrchestrator:
    """Answer each newly observed form page once, then execute against live DOM."""

    _SUCCESS = re.compile(
        r"(?:application\s+(?:has been\s+)?(?:submitted|received)|"
        r"thank you for applying|successfully applied|submission complete|"
        r"we(?:'|’)ve received your application|your application was sent)",
        re.IGNORECASE,
    )
    _INITIAL_NAV = re.compile(
        r"^(?:easy apply|apply|apply now|start application|begin application|"
        r"continue to application)$",
        re.IGNORECASE,
    )
    _CONTINUE = re.compile(r"^(?:next|continue|review|save and continue|continue application)$", re.IGNORECASE)
    _SUBMIT = re.compile(r"^(?:submit|submit application|finish|complete application)$", re.IGNORECASE)

    def __init__(
        self,
        *,
        browser_session: Any,
        planner: PageAnswerPlanner,
        executor: BrowserExecutor,
        operator: BrowserOperator | None,
        candidate_context: str,
        job_context: dict[str, Any] | None = None,
        flags: ApplyFeatureFlags | None = None,
        fallback: Fallback | None = None,
        observer: PageObserver | None = None,
        watchdog: Any | None = None,
        initial_url: str | None = None,
        max_steps: int = 120,
        max_pages: int = 40,
        max_local_actions: int = 3,
    ):
        self.browser_session = browser_session
        self.planner = planner
        self.executor = executor
        self.operator = operator
        self.candidate_context = candidate_context
        self.job_context = job_context or {}
        self.flags = flags or ApplyFeatureFlags(page_orchestrator=True)
        self.fallback = fallback
        self.observer = observer or executor.observer or PageObserver()
        self.watchdog = watchdog
        self.initial_url = initial_url
        self.max_steps = max_steps
        self.max_pages = max_pages
        self.max_local_actions = max_local_actions
        self._planned: set[str] = set()
        self._reasked: set[str] = set()
        self._completed: set[str] = set()
        self._all_fields: dict[str, FormField] = {}
        self._answers: dict[str, FieldAnswer] = {}
        self._attempts: list[dict[str, Any]] = []
        self._metrics = OrchestrationMetrics()
        self._snapshot: PageSnapshot | None = None
        self._steps = 0
        self._page_generation = 0
        self._submission_attempted = False
        self._set_field_namespace()

    async def run(self) -> OrchestrationOutcome:
        try:
            await self.browser_session.start()
            if self.initial_url:
                await self._navigate_to_initial_url()

            for _ in range(self.max_pages):
                if self._steps >= self.max_steps:
                    return await self._handoff("The page executor reached its bounded action budget.")
                snapshot = await self._observe()
                if self.watchdog is not None and await self.watchdog.should_stop():
                    return await self._handoff(
                        "The existing Browser-Use progress watchdog detected stagnation.",
                        snapshot=snapshot,
                    )
                if self._is_submission_confirmed(snapshot):
                    return self._outcome(success=True, snapshot=snapshot)

                for field in snapshot.fields:
                    self._all_fields[field.field_id] = field
                self._metrics.total_fields = len(self._all_fields)

                newly_seen = {
                    field.field_id for field in snapshot.fields
                    if field.field_id not in self._planned and field.field_id not in self._completed
                }
                if newly_seen:
                    await self._plan_new_fields(snapshot, newly_seen)

                pending = [
                    field for field in snapshot.fields
                    if field.field_id not in self._completed
                ]
                if pending:
                    page_generation_before = self._page_generation
                    unresolved = await self._complete_fields(snapshot, pending)
                    if unresolved:
                        return await self._handoff(
                            "One or more current-page fields could not be safely completed.",
                            unresolved=unresolved,
                            snapshot=self._snapshot or snapshot,
                        )
                    snapshot = self._snapshot or await self._observe()
                    if self._page_generation != page_generation_before:
                        # A field interaction changed the active page/tab. The
                        # new namespace makes repeated questions distinct.
                        continue
                    if any(
                        field.field_id not in self._completed and field.field_id not in self._planned
                        for field in snapshot.fields
                    ):
                        # A conditional control appeared as a result of an
                        # answer. Plan it before considering page navigation.
                        continue

                button = self._next_button(snapshot, has_form=bool(snapshot.fields))
                if button is not None:
                    submit = bool(self._SUBMIT.fullmatch(button.label.strip()))
                    if submit and self._submission_attempted:
                        return await self._handoff(
                            "Submission was already attempted and no confirmation is visible; refusing to submit twice.",
                            snapshot=snapshot,
                        )
                    if submit:
                        self._submission_attempted = True
                    before_click = snapshot
                    clicked = await self._click_button(snapshot, button)
                    if clicked.verified:
                        snapshot = clicked.snapshot or await self._observe()
                        if self._is_submission_confirmed(snapshot):
                            return self._outcome(success=True, snapshot=snapshot)
                        # A submit click without a confirmation must be reviewed
                        # by the full agent, never repeated by this loop.
                        if submit:
                            return await self._handoff(
                                "Submit was clicked, but the page did not confirm receipt.",
                                snapshot=snapshot,
                            )
                        if not self._did_advance_page(before_click, snapshot):
                            unresolved = {
                                field.field_id for field in snapshot.fields
                                if field.invalid
                            }
                            if snapshot.validation_messages and not unresolved:
                                unresolved = {
                                    field.field_id for field in snapshot.fields
                                    if field.field_id in self._completed
                                }
                            for field_id in unresolved:
                                self._completed.discard(field_id)
                            self._metrics.completed_fields = len(self._completed)
                            return await self._handoff(
                                "The continuation control changed the page state without advancing; inspect validation before retrying.",
                                unresolved=unresolved,
                                snapshot=snapshot,
                            )
                        self._snapshot = snapshot
                        self._advance_page_scope()
                        continue
                    if submit:
                        return await self._handoff(
                            "Submit was attempted but could not be verified; it will not be repeated.",
                            snapshot=clicked.snapshot or snapshot,
                        )
                    if self._operator_enabled:
                        recovered = await self._recover_navigation(snapshot, button, clicked)
                        if recovered is not None and recovered.verified:
                            self._snapshot = recovered.snapshot or await self._observe()
                            self._advance_page_scope()
                            continue
                    return await self._handoff(
                        "The next application control did not produce a verifiable page change.",
                        snapshot=clicked.snapshot or snapshot,
                    )

                if self._operator_enabled and not snapshot.fields:
                    navigation = await self._recover_navigation(snapshot, None, None)
                    if navigation is not None and navigation.verified:
                        self._snapshot = navigation.snapshot or await self._observe()
                        self._advance_page_scope()
                        continue

                if self._is_submission_confirmed(snapshot):
                    return self._outcome(success=True, snapshot=snapshot)
                return await self._handoff(
                    "No deterministic continuation control is available on the current page.",
                    snapshot=snapshot,
                )

            return await self._handoff("The page transition budget was exhausted.", snapshot=self._snapshot)
        except Exception as exc:
            # Do not expose raw browser errors: Browser-Use messages can contain
            # field text or secrets supplied to an action.
            return await self._handoff(
                f"Page orchestration failed ({type(exc).__name__}).",
                snapshot=self._snapshot,
            )

    async def _navigate_to_initial_url(self) -> None:
        current = ""
        try:
            current = str((await self.browser_session.get_browser_state_summary(include_screenshot=False)).url or "")
        except Exception:
            pass
        if current == self.initial_url:
            return
        runner = self.executor.runner
        await runner.dispatch("navigate", {"url": self.initial_url, "new_tab": False}, self.browser_session)
        self._steps += 1
        self.executor.stats["deterministic_actions"] += 1
        self._snapshot = await self._observe()

    async def _observe(self) -> PageSnapshot:
        snapshot = await self.observer.observe(self.browser_session)
        self._snapshot = snapshot
        self._metrics.pages_observed += 1
        if self.watchdog is not None:
            try:
                state = await self.browser_session.get_browser_state_summary(
                    include_screenshot=False,
                    cached=True,
                )
                self.watchdog.observe(state, max(self._steps, 1))
            except Exception:
                # Observation remains authoritative; a watchdog-only failure
                # should not interrupt the application or block Agent fallback.
                pass
        for field in snapshot.fields:
            self._all_fields[field.field_id] = field
        self._metrics.total_fields = len(self._all_fields)
        return snapshot

    def _set_field_namespace(self) -> None:
        namespace = f"page{self._page_generation}"
        for observer in {id(self.observer): self.observer, id(self.executor.observer): self.executor.observer}.values():
            setter = getattr(observer, "set_field_namespace", None)
            if callable(setter):
                setter(namespace)

    def _advance_page_scope(self) -> None:
        self._page_generation += 1
        self._set_field_namespace()

    async def _plan_new_fields(self, snapshot: PageSnapshot, field_ids: set[str]) -> None:
        self._planned.update(field_ids)
        semantic = {
            field_id for field_id in field_ids
            if snapshot.field_map()[field_id].control_type not in {"password", "file"}
        }
        if semantic:
            self._metrics.page_level_big_llm_calls += 1
        batch = await self.planner.plan(
            snapshot,
            self.candidate_context,
            self.job_context,
            field_ids=field_ids,
        )
        self._answers.update({answer.field_id: answer for answer in batch.answers})

    async def _complete_fields(self, snapshot: PageSnapshot, fields: list[FormField]) -> set[str]:
        unresolved: set[str] = set()
        current_snapshot = snapshot
        for original in fields:
            if self._steps >= self.max_steps:
                unresolved.add(original.field_id)
                break
            field = current_snapshot.field_map().get(original.field_id)
            if field is None:
                # A conditional transition or re-render changed the page. The
                # next outer pass observes and plans any newly surfaced fields.
                continue
            answer = self._answers.get(field.field_id)
            if answer is None:
                unresolved.add(field.field_id)
                continue
            if answer.status != AnswerStatus.ANSWERED and field.control_type != "file":
                if field.control_type not in {"password", "file"} and field.field_id not in self._reasked:
                    self._reasked.add(field.field_id)
                    self._metrics.page_level_big_llm_calls += 1
                    try:
                        batch = await self.planner.plan(
                            current_snapshot,
                            self.candidate_context,
                            self.job_context,
                            field_ids={field.field_id},
                        )
                        answer = batch.answers[0]
                        self._answers[field.field_id] = answer
                    except Exception:
                        answer = None
                if answer is None or answer.status != AnswerStatus.ANSWERED:
                    unresolved.add(field.field_id)
                    continue

            result = await self._run_field(current_snapshot, field, answer)
            current_snapshot = result.snapshot or await self._observe()
            if result.verified:
                self._completed.add(field.field_id)
                self._metrics.completed_fields = len(self._completed)
                continue

            # A visible validation error makes the prior semantic answer
            # ambiguous. Permit one page-context replan; the local operator may
            # never rewrite or invent the answer itself.
            fresh_field = current_snapshot.field_map().get(field.field_id)
            if (
                field.field_id not in self._reasked
                and (current_snapshot.validation_messages or (fresh_field and fresh_field.invalid))
                and field.control_type not in {"password", "file"}
            ):
                self._reasked.add(field.field_id)
                self._metrics.page_level_big_llm_calls += 1
                try:
                    batch = await self.planner.plan(
                        current_snapshot,
                        self.candidate_context,
                        self.job_context,
                        field_ids={field.field_id},
                    )
                    revised = batch.answers[0]
                    self._answers[field.field_id] = revised
                    if revised.status == AnswerStatus.ANSWERED:
                        retry = await self._run_field(current_snapshot, fresh_field or field, revised)
                        current_snapshot = retry.snapshot or await self._observe()
                        if retry.verified:
                            self._completed.add(field.field_id)
                            self._metrics.completed_fields = len(self._completed)
                            continue
                except Exception:
                    pass

            unresolved.add(field.field_id)
            self._metrics.validation_failures += bool(current_snapshot.validation_messages)
            if result.error:
                self._attempts.append({"field_id": field.field_id, "failure": self._safe_failure(result.error)})
        if (
            current_snapshot.url != snapshot.url
            or current_snapshot.page_ref != snapshot.page_ref
            or current_snapshot.tabs != snapshot.tabs
        ):
            self._advance_page_scope()
        self._snapshot = current_snapshot
        return unresolved

    async def _run_field(self, snapshot: PageSnapshot, field: FormField, answer: FieldAnswer) -> ExecutionResult:
        result = await self.executor.execute_answer(self.browser_session, snapshot, answer)
        self._steps += max(1, len(result.attempts))
        self._sync_executor_metrics()
        if result.verified:
            self._attempts.extend(item.model_dump(mode="json") for item in result.attempts)
            return result

        if not self._operator_enabled or self.operator is None:
            return result

        attempts: list[dict[str, Any]] = [item.model_dump(mode="json") for item in result.attempts]
        failure = self._safe_failure(result.error)
        fresh = result.snapshot or await self._observe()
        for _ in range(self.max_local_actions):
            if self._steps >= self.max_steps:
                break
            live_field = fresh.field_map().get(field.field_id)
            self._metrics.local_operator_calls += 1
            local_action = await self._ask_operator(
                fresh,
                live_field,
                answer,
                attempts,
                failure,
            )
            if local_action is None:
                break
            action = as_browser_action(local_action)
            if action.action == BrowserActionType.FALLBACK:
                break
            if action.action == BrowserActionType.DONE:
                if self._answer_is_present(fresh.field_map().get(field.field_id), answer):
                    self._metrics.local_recoveries += 1
                    return ExecutionResult(True, True, snapshot=fresh, attempts=[])
                failure = "DONE was returned before the approved value was observed."
                fresh = await self._observe()
                attempts.append({"action": "DONE", "verified": False})
                continue
            if action.action == BrowserActionType.RETRY:
                self._metrics.retries += 1
                self.executor.stats["retries"] += 1
                fresh = await self._observe()
                retry = await self.executor.execute_answer(self.browser_session, fresh, answer)
                self._steps += max(1, len(retry.attempts))
                self._sync_executor_metrics()
                attempts.extend(item.model_dump(mode="json") for item in retry.attempts)
                if retry.verified:
                    self._metrics.local_recoveries += 1
                    return retry
                fresh = retry.snapshot or await self._observe()
                failure = self._safe_failure(retry.error)
                continue

            is_value_action = action.action in {
                BrowserActionType.TYPE, BrowserActionType.SELECT, BrowserActionType.CHECK, BrowserActionType.DATE,
            }
            action_result = await self.executor.execute_action(
                self.browser_session,
                fresh,
                action,
                field_id=field.field_id if is_value_action else None,
                answer=answer if is_value_action else None,
                deterministic=False,
            )
            self._steps += max(1, len(action_result.attempts))
            self._sync_executor_metrics()
            attempts.extend(item.model_dump(mode="json") for item in action_result.attempts)
            fresh = await self._observe()  # always refresh refs after a local action
            if action_result.verified:
                # Verify the answer itself, not merely that the click returned.
                if self._answer_is_present(fresh.field_map().get(field.field_id), answer):
                    self._metrics.local_recoveries += 1
                    self._attempts.extend(attempts)
                    return ExecutionResult(True, True, snapshot=fresh, attempts=[])
                # A click may have opened a menu; give the operator the updated
                # snapshot and exact visible option set on the next turn.
                failure = "Interaction changed the page but the approved value is not present yet."
            else:
                failure = self._safe_failure(action_result.error)

        self._attempts.extend(attempts)
        return ExecutionResult(False, False, stale=result.stale, error="Local recovery attempts exhausted.", snapshot=fresh, attempts=[])

    async def _ask_operator(
        self,
        snapshot: PageSnapshot,
        field: FormField | None,
        answer: FieldAnswer,
        attempts: list[dict[str, Any]],
        failure: str,
    ):
        allowed = [
            RecoveryActionType.CLICK,
            RecoveryActionType.SCROLL,
            RecoveryActionType.WAIT,
            RecoveryActionType.RETRY,
            RecoveryActionType.DONE,
            RecoveryActionType.FALLBACK,
        ]
        if field is not None and answer.status == AnswerStatus.ANSWERED:
            if field.control_type in {
                "text", "textarea", "contenteditable", "email", "tel", "url", "number", "search",
                "date", "datetime-local", "month", "time", "week",
            }:
                allowed.append(RecoveryActionType.TYPE)
            if field.control_type in {"select", "combobox", "radio"}:
                allowed.append(RecoveryActionType.SELECT)
            if field.control_type == "checkbox":
                allowed.append(RecoveryActionType.CHECK)
        context = BrowserOperatorContext(
            snapshot=snapshot,
            field=field,
            approved_answer=answer,
            previous_attempts=attempts,
            validation_state=[message.text for message in snapshot.validation_messages],
            allowed_actions=allowed,
            last_failure=failure,
        )
        try:
            return await self.operator.next_action(context)
        except Exception as exc:
            self._attempts.append({"field_id": field.field_id if field else None, "failure": f"operator_{type(exc).__name__}"})
            return None

    async def _click_button(self, snapshot: PageSnapshot, button: PageButton) -> ExecutionResult:
        result = await self.executor.execute_action(
            self.browser_session,
            snapshot,
            BrowserAction(
                action=BrowserActionType.CLICK,
                target_ref=f"{button.control_ref.snapshot_id}:{button.control_ref.selector_index}",
            ),
        )
        self._steps += max(1, len(result.attempts))
        self._sync_executor_metrics()
        return result

    async def _recover_navigation(
        self,
        snapshot: PageSnapshot,
        button: PageButton | None,
        previous: ExecutionResult | None,
    ) -> ExecutionResult | None:
        if not self._operator_enabled or self.operator is None:
            return None
        attempts = [item.model_dump(mode="json") for item in (previous.attempts if previous else [])]
        failure = self._safe_failure(previous.error) if previous else "No safe deterministic navigation control was identified."
        fresh = previous.snapshot if previous and previous.snapshot else snapshot
        for _ in range(self.max_local_actions):
            if self._steps >= self.max_steps:
                return None
            self._metrics.local_operator_calls += 1
            action_model = await self._ask_operator(
                fresh,
                None,
                FieldAnswer(field_id="navigation", status=AnswerStatus.UNKNOWN),
                attempts,
                failure,
            )
            if action_model is None:
                return None
            action = as_browser_action(action_model)
            if action.action == BrowserActionType.FALLBACK:
                return None
            if action.action == BrowserActionType.CLICK:
                target = action.target_ref
                allowed_targets = {
                    f"{candidate.control_ref.snapshot_id}:{candidate.control_ref.selector_index}"
                    for candidate in fresh.buttons
                    if (
                        self._INITIAL_NAV.fullmatch(candidate.label.strip())
                        or self._CONTINUE.fullmatch(candidate.label.strip())
                    )
                    and candidate.label.casefold() not in {"submit", "submit application", "finish", "complete application"}
                    and not (
                        button is not None
                        and candidate.control_ref == button.control_ref
                    )
                }
                if target not in allowed_targets:
                    return None
            if action.action in {BrowserActionType.DONE, BrowserActionType.RETRY}:
                fresh = await self._observe()
                if action.action == BrowserActionType.RETRY:
                    self._metrics.retries += 1
                    continue
                return None
            result = await self.executor.execute_action(
                self.browser_session,
                fresh,
                action,
                deterministic=False,
            )
            self._steps += max(1, len(result.attempts))
            self._sync_executor_metrics()
            attempts.extend(item.model_dump(mode="json") for item in result.attempts)
            fresh = await self._observe()
            if result.verified and (fresh.dom_signature != snapshot.dom_signature or fresh.tabs != snapshot.tabs or fresh.url != snapshot.url):
                self._metrics.local_recoveries += 1
                return result.model_copy(update={"snapshot": fresh, "verified": True})
            failure = self._safe_failure(result.error) if result.error else "Navigation action did not change the page."
        return None

    def _next_button(self, snapshot: PageSnapshot, *, has_form: bool) -> PageButton | None:
        enabled = [button for button in snapshot.buttons if not button.disabled]
        if not has_form:
            return next((button for button in enabled if self._INITIAL_NAV.fullmatch(button.label.strip())), None)
        for button in enabled:
            if self._CONTINUE.fullmatch(button.label.strip()):
                return button
        for button in enabled:
            if self._SUBMIT.fullmatch(button.label.strip()):
                return button
        return None

    def _is_submission_confirmed(self, snapshot: PageSnapshot) -> bool:
        texts = [snapshot.title, *snapshot.status_messages, *(item.text for item in snapshot.validation_messages)]
        if any(self._SUCCESS.search(text or "") for text in texts):
            return True
        path = urlsplit(snapshot.url).path.casefold()
        return any(token in path for token in ("thank-you", "thank_you", "confirmation", "application-submitted"))

    @staticmethod
    def _did_advance_page(before: PageSnapshot, after: PageSnapshot) -> bool:
        if before.url != after.url or before.page_ref != after.page_ref or before.tabs != after.tabs:
            return True
        before_fields = [
            (field.label, field.control_type, field.name, field.required, tuple(option.label for option in field.options))
            for field in before.fields
        ]
        after_fields = [
            (field.label, field.control_type, field.name, field.required, tuple(option.label for option in field.options))
            for field in after.fields
        ]
        if before_fields != after_fields:
            return True
        if [button.label for button in before.buttons] != [button.label for button in after.buttons]:
            return True
        # A step may reuse a URL and identical form structure while replacing
        # field values. Ignore signatures changed only by validation feedback.
        return (
            before.dom_signature != after.dom_signature
            and not before.validation_messages
            and not after.validation_messages
        )

    @staticmethod
    def _answer_is_present(field: FormField | None, answer: FieldAnswer) -> bool:
        if field is None or answer.status != AnswerStatus.ANSWERED:
            return False
        wanted = (answer.answer or "").strip().casefold()
        if field.control_type == "checkbox":
            desired = BrowserExecutor._parse_bool(answer.answer)
            return desired is not None and any(control.checked is desired for control in field.controls)
        values = {field.current_value or ""}
        values.update(option.value for option in field.options if option.selected)
        values.update(option.label for option in field.options if option.selected)
        return bool(wanted and any(value.strip().casefold() == wanted for value in values if value))

    async def _handoff(
        self,
        reason: str,
        *,
        unresolved: set[str] | None = None,
        snapshot: PageSnapshot | None = None,
    ) -> OrchestrationOutcome:
        if unresolved is None:
            current = snapshot or self._snapshot
            unresolved = {
                field.field_id for field in (current.fields if current else [])
                if field.field_id not in self._completed
            }
        if not self.flags.full_agent_fallback or self.fallback is None:
            return self._outcome(success=False, error=reason, snapshot=snapshot, unresolved=unresolved)
        checkpoint = self._build_checkpoint(reason, unresolved, snapshot or self._snapshot)
        self._metrics.fallback_calls += 1
        try:
            result = await self.fallback(checkpoint)
        except Exception as exc:
            return self._outcome(
                success=False,
                error=f"Full Agent fallback failed ({type(exc).__name__}).",
                snapshot=snapshot,
                unresolved=unresolved,
            )
        return self._outcome(
            success=False,
            snapshot=snapshot,
            unresolved=unresolved,
            fallback_result=result,
        )

    def _build_checkpoint(self, reason: str, unresolved: set[str], snapshot: PageSnapshot | None) -> str:
        current = snapshot or self._snapshot
        clean_url = ""
        if current:
            parts = urlsplit(current.url)
            clean_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        completed = [
            {"field_id": field_id, "label": self._all_fields[field_id].label}
            for field_id in sorted(self._completed) if field_id in self._all_fields
        ]
        outstanding = [
            {"field_id": field_id, "label": self._all_fields[field_id].label}
            for field_id in sorted(unresolved) if field_id in self._all_fields
        ]
        page_state = []
        if current:
            page_state = [
                {
                    "field_id": field.field_id,
                    "label": field.label,
                    "control_type": field.control_type,
                    "required": field.required,
                    "options": [option.label for option in field.options],
                    "invalid": field.invalid,
                }
                for field in current.fields
            ]
        failed = [
            {key: value for key, value in attempt.items() if key in {"field_id", "action", "method", "verified", "error_code", "failure"}}
            for attempt in self._attempts[-12:]
        ]
        return (
            "Continue the existing application in the current live browser session. Do not navigate back to the job listing. "
            "The page executor has already completed the listed fields. Inspect each current field before acting. "
            "Never repeat a submission unless the current page explicitly shows the application form is still open and no submission occurred.\n"
            f"Reason for takeover: {reason}\n"
            f"Current page: {clean_url}\n"
            f"Completed fields (do not change): {completed}\n"
            f"Unresolved fields: {outstanding}\n"
            f"Current form summary: {page_state}\n"
            f"Recent failed action metadata: {failed}\n"
            f"Submission previously attempted: {self._submission_attempted}.\n"
            "Continue with the existing application instructions and verify all actions before proceeding."
        )

    def _outcome(
        self,
        *,
        success: bool,
        error: str | None = None,
        snapshot: PageSnapshot | None = None,
        unresolved: set[str] | None = None,
        fallback_result: Any | None = None,
    ) -> OrchestrationOutcome:
        self._sync_executor_metrics()
        return OrchestrationOutcome(
            success=success,
            fallback_result=fallback_result,
            error=error,
            final_snapshot=snapshot,
            answers=dict(self._answers),
            fields=dict(self._all_fields),
            completed_field_ids=set(self._completed),
            unresolved_field_ids=set(unresolved or ()),
            attempts=list(self._attempts),
            metrics=self._metrics,
        )

    @property
    def _operator_enabled(self) -> bool:
        return self.flags.local_browser_operator and self.operator is not None

    def _sync_executor_metrics(self) -> None:
        for name in ("deterministic_actions", "stale_element_events", "validation_failures", "retries"):
            self._metrics_value(name, self.executor.stats.get(name, 0))

    def _metrics_value(self, name: str, value: int) -> None:
        if getattr(self._metrics, name) < value:
            setattr(self._metrics, name, value)

    @staticmethod
    def _safe_failure(error: str | None) -> str:
        if not error:
            return "Browser action did not produce a verifiable state change."
        lowered = error.casefold()
        if "stale" in lowered or "index" in lowered or "node" in lowered:
            return "The page changed and the element reference may be stale."
        if "timeout" in lowered:
            return "The browser action timed out."
        if "validation" in lowered or "invalid" in lowered:
            return "The page reported a validation issue."
        return "The browser action failed or its result could not be verified."
