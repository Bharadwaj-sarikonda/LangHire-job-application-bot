"""Strict, serializable contracts for page observation and execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LiveElementRef(StrictModel):
    """A short-lived Browser-Use selector binding valid only for one snapshot."""

    snapshot_id: str
    selector_index: int
    backend_node_id: int
    target_id: str = ""
    frame_id: str = ""
    session_id: str = ""


class ChoiceOption(StrictModel):
    label: str
    value: str
    selected: bool = False
    control_ref: LiveElementRef | None = None


class FieldControl(StrictModel):
    control_ref: LiveElementRef
    label: str = ""
    value: str | None = None
    checked: bool | None = None


class FormField(StrictModel):
    field_id: str
    label: str
    help_text: str | None = None
    control_type: str
    required: bool = False
    current_value: str | None = None
    options: list[ChoiceOption] = Field(default_factory=list)
    controls: list[FieldControl] = Field(default_factory=list)
    frame_context: str | None = None
    name: str | None = None
    autocomplete: str | None = None
    invalid: bool = False


class PageButton(StrictModel):
    label: str
    control_ref: LiveElementRef
    disabled: bool = False
    role: str = "button"


class ValidationMessage(StrictModel):
    text: str
    field_id: str | None = None
    frame_context: str | None = None


class PageSnapshot(StrictModel):
    """Compact page projection; live refs must not escape their snapshot."""

    snapshot_id: str
    url: str
    title: str
    page_ref: str = ""
    fields: list[FormField] = Field(default_factory=list)
    buttons: list[PageButton] = Field(default_factory=list)
    validation_messages: list[ValidationMessage] = Field(default_factory=list)
    status_messages: list[str] = Field(default_factory=list)
    tabs: list[dict[str, str]] = Field(default_factory=list)
    scroll_position: dict[str, int] = Field(default_factory=dict)
    dom_signature: str
    captured_at: str

    def field_map(self) -> dict[str, FormField]:
        return {field.field_id: field for field in self.fields}

    def answer_payload(self, field_ids: set[str] | None = None) -> dict[str, Any]:
        """Return semantic context for the answer LLM, excluding live DOM refs."""
        selected = [
            field for field in self.fields
            if field_ids is None or field.field_id in field_ids
        ]
        return {
            "page": {"url": self.url, "title": self.title},
            "fields": [
                {
                    "field_id": field.field_id,
                    "label": field.label,
                    "help_text": field.help_text,
                    "control_type": field.control_type,
                    "required": field.required,
                    "current_value": field.current_value,
                    "options": [option.label for option in field.options],
                    "invalid": field.invalid,
                }
                for field in selected
            ],
            "validation_messages": [message.text for message in self.validation_messages],
        }


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    UNKNOWN = "unknown"
    NEEDS_CLARIFICATION = "needs_clarification"


class FieldAnswer(StrictModel):
    field_id: str
    status: AnswerStatus
    answer: str | None = None
    source: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None


class PageAnswerBatch(StrictModel):
    snapshot_id: str
    answers: list[FieldAnswer]


class BrowserActionType(StrEnum):
    CLICK = "CLICK"
    TYPE = "TYPE"
    SELECT = "SELECT"
    CHECK = "CHECK"
    DATE = "DATE"
    SCROLL = "SCROLL"
    WAIT = "WAIT"
    UPLOAD = "UPLOAD"
    RETRY = "RETRY"
    DONE = "DONE"
    FALLBACK = "FALLBACK"


class RecoveryActionType(StrEnum):
    CLICK = "CLICK"
    TYPE = "TYPE"
    SELECT = "SELECT"
    CHECK = "CHECK"
    SCROLL = "SCROLL"
    WAIT = "WAIT"
    RETRY = "RETRY"
    DONE = "DONE"
    FALLBACK = "FALLBACK"


class InteractionMethod(StrEnum):
    NATIVE = "native"
    KEYBOARD = "keyboard"


class InteractionKey(StrEnum):
    ENTER = "Enter"
    SPACE = "Space"
    TAB = "Tab"
    ARROWDOWN = "ArrowDown"
    ARROWUP = "ArrowUp"
    ESCAPE = "Escape"


class BrowserAction(StrictModel):
    """Constrained local-model output. Values are references, never new facts."""

    action: BrowserActionType
    target_ref: str | None = None
    answer_ref: str | None = None
    option: str | None = None
    direction: str | None = None
    amount: int | None = Field(default=None, ge=1, le=2000)
    wait_ms: int | None = Field(default=None, ge=0, le=10000)
    method: InteractionMethod | None = None
    key: InteractionKey | None = None
    reason: str | None = None


class LocalBrowserAction(BrowserAction):
    """The smaller action vocabulary exposed to a local browser model."""

    action: RecoveryActionType


class ActionAttempt(StrictModel):
    action: BrowserActionType | RecoveryActionType
    snapshot_id: str
    field_id: str | None = None
    method: str
    succeeded: bool
    verified: bool
    error_code: str | None = None
    observed_change: str | None = None
    timestamp: str
