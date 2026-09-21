"""Convert Browser-Use's live DOM snapshot into a compact form projection."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from .schemas import (
    ChoiceOption,
    FieldControl,
    FormField,
    LiveElementRef,
    PageButton,
    PageSnapshot,
    ValidationMessage,
)

_FIELD_TAGS = {"input", "textarea", "select"}
_NON_FIELD_INPUT_TYPES = {"button", "submit", "reset", "image", "hidden"}
_BUTTON_TAGS = {"button", "a"}
_MAX_TEXT = 500
_MAX_VALIDATION_MESSAGES = 50


def _attrs(node: Any) -> dict[str, str]:
    value = getattr(node, "attributes", None)
    return value if isinstance(value, dict) else {}


def _tag(node: Any) -> str:
    return str(getattr(node, "tag_name", getattr(node, "node_name", ""))).lower()


def _text(node: Any, limit: int = _MAX_TEXT) -> str:
    if node is None:
        return ""
    getter = getattr(node, "get_all_children_text", None)
    if callable(getter):
        try:
            value = getter()
            if value:
                return _normalize_text(str(value), limit)
        except Exception:
            pass
    value = getattr(node, "node_value", "")
    return _normalize_text(str(value or ""), limit)


def _normalize_text(value: str, limit: int = _MAX_TEXT) -> str:
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _children(node: Any) -> list[Any]:
    result = getattr(node, "children_nodes", None)
    return result if isinstance(result, list) else []


def _visible(node: Any) -> bool:
    attrs = _attrs(node)
    if getattr(node, "is_visible", None) is False:
        return False
    if "hidden" in attrs or attrs.get("aria-hidden", "").lower() == "true":
        return False
    if attrs.get("type", "").lower() == "hidden":
        return False
    styles = getattr(getattr(node, "snapshot_node", None), "computed_styles", None) or {}
    if styles.get("display", "").lower() == "none":
        return False
    if styles.get("visibility", "").lower() == "hidden":
        return False
    try:
        if float(styles.get("opacity", "1")) <= 0:
            return False
    except (ValueError, TypeError):
        pass
    return True


def _node_key(node: Any) -> tuple[str, str, int]:
    return (
        str(getattr(node, "target_id", "") or ""),
        str(getattr(node, "frame_id", "") or ""),
        int(getattr(node, "backend_node_id", 0) or 0),
    )


def _iter_related(nodes: Iterable[Any]) -> list[Any]:
    """Collect nearby DOM nodes for labels/options without returning page HTML."""
    found: dict[int, Any] = {}
    stack = list(nodes)
    while stack and len(found) < 20000:
        node = stack.pop()
        if node is None or id(node) in found:
            continue
        found[id(node)] = node
        parent = getattr(node, "parent_node", None)
        if parent is not None:
            stack.append(parent)
        stack.extend(_children(node))
        stack.extend(getattr(node, "shadow_roots", None) or [])
        content_document = getattr(node, "content_document", None)
        if content_document is not None:
            stack.append(content_document)
    return list(found.values())


def _label(node: Any, related: list[Any], id_map: dict[str, Any]) -> str:
    attrs = _attrs(node)
    direct = attrs.get("aria-label", "").strip()
    if direct:
        return _normalize_text(direct)

    labelled_by = attrs.get("aria-labelledby", "").split()
    labelled = [_text(id_map.get(item), 160) for item in labelled_by]
    labelled = [item for item in labelled if item]
    if labelled:
        return _normalize_text(" ".join(labelled))

    ax_node = getattr(node, "ax_node", None)
    ax_name = str(getattr(ax_node, "name", "") or "").strip()
    if ax_name:
        return _normalize_text(ax_name)

    node_id = attrs.get("id", "")
    if node_id:
        for candidate in related:
            if _tag(candidate) == "label" and _attrs(candidate).get("for") == node_id:
                if _text(candidate):
                    return _text(candidate)

    current = node
    while current is not None:
        tag = _tag(current)
        if tag == "label":
            text = _text(current)
            if text:
                return text
        if tag == "fieldset":
            for child in _children(current):
                if _tag(child) == "legend" and _text(child):
                    return _text(child)
        current = getattr(current, "parent_node", None)

    for key in ("placeholder", "title", "name", "id"):
        value = attrs.get(key, "").strip()
        if value:
            return _normalize_text(value)

    parent = getattr(node, "parent_node", None)
    return _text(parent, 160) or "Unlabelled field"


def _help_text(node: Any, id_map: dict[str, Any]) -> str | None:
    described_by = _attrs(node).get("aria-describedby", "").split()
    values = [_text(id_map.get(item), 200) for item in described_by]
    joined = _normalize_text(" ".join(value for value in values if value))
    return joined or None


def _control_type(node: Any) -> str | None:
    tag = _tag(node)
    attrs = _attrs(node)
    role = attrs.get("role", "").lower()
    input_type = attrs.get("type", "text").lower()
    if role == "combobox":
        return "combobox"
    if tag == "input":
        if input_type in _NON_FIELD_INPUT_TYPES:
            return None
        if input_type in {"radio", "checkbox", "date", "datetime-local", "month", "time", "week", "file"}:
            return input_type
        if input_type == "password":
            return "password"
        return "text" if input_type in {"text", "email", "tel", "url", "number", "search", "password"} else input_type
    if tag == "textarea":
        return "textarea"
    if tag == "select":
        return "select"
    if attrs.get("contenteditable", "").lower() == "true":
        return "contenteditable"
    return {
        "textbox": "text",
        "combobox": "combobox",
        "checkbox": "checkbox",
        "radio": "radio",
        "switch": "checkbox",
    }.get(role)


def _required(attrs: dict[str, str]) -> bool:
    return "required" in attrs or attrs.get("aria-required", "").lower() == "true"


def _value(node: Any, control_type: str) -> str | None:
    attrs = _attrs(node)
    if control_type == "password":
        return None
    snapshot = getattr(node, "snapshot_node", None)
    current = getattr(snapshot, "input_value", None) if snapshot is not None else None
    if current is None:
        current = attrs.get("value")
    if current is None and _tag(node) == "textarea":
        current = _text(node)
    if current is None:
        return None
    return _normalize_text(str(current), 2000)


def _checked(node: Any) -> bool | None:
    snapshot = getattr(node, "snapshot_node", None)
    current = getattr(snapshot, "input_checked", None) if snapshot is not None else None
    if current is not None:
        return bool(current)
    attrs = _attrs(node)
    if "checked" in attrs:
        return True
    if attrs.get("aria-checked", "").lower() in {"true", "false"}:
        return attrs["aria-checked"].lower() == "true"
    return None


def _option_nodes(node: Any) -> list[Any]:
    result: list[Any] = []
    stack = list(_children(node))
    while stack:
        child = stack.pop(0)
        if _tag(child) == "option":
            result.append(child)
        else:
            stack.extend(_children(child))
    return result


def _choice_options(
    node: Any,
    control_type: str,
    snapshot_id: str,
    selector_by_identity: dict[tuple[str, str, int], int],
    related: list[Any],
    id_map: dict[str, Any],
) -> list[ChoiceOption]:
    if control_type == "select":
        options = []
        for child in _option_nodes(node):
            attrs = _attrs(child)
            label = _text(child) or attrs.get("label", "") or attrs.get("value", "")
            if not label:
                continue
            options.append(ChoiceOption(
                label=_normalize_text(label, 200),
                value=attrs.get("value", label),
                selected="selected" in attrs,
            ))
        return options
    if control_type == "combobox":
        attrs = _attrs(node)
        controlled_ids = set((attrs.get("aria-controls") or attrs.get("aria-owns") or "").split())
        roots = [id_map[item] for item in controlled_ids if item in id_map]
        option_nodes: list[Any] = []
        if roots:
            for root in roots:
                option_nodes.extend(
                    item for item in _iter_related([root])
                    if _attrs(item).get("role", "").lower() == "option" and _visible(item)
                )
        elif attrs.get("aria-expanded", "").lower() == "true":
            option_nodes.extend(
                item for item in related
                if _attrs(item).get("role", "").lower() == "option"
                and _visible(item)
                and any(_attrs(parent).get("role", "").lower() == "listbox" for parent in _ancestor_nodes(item))
                )

        option_nodes = sorted(
            {id(item): item for item in option_nodes}.values(),
            key=lambda item: selector_by_identity.get(_node_key(item), 1 << 30),
        )

        options: list[ChoiceOption] = []
        seen: set[tuple[str, str]] = set()
        for option_node in option_nodes:
            option_attrs = _attrs(option_node)
            label = _text(option_node, 200) or option_attrs.get("aria-label", "")
            value = option_attrs.get("value") or option_attrs.get("data-value") or label
            key = (label.casefold(), value.casefold())
            if not label or key in seen:
                continue
            seen.add(key)
            index = selector_by_identity.get(_node_key(option_node))
            ref = PageObserver._static_live_ref(snapshot_id, index, option_node) if index is not None else None
            selected = (
                option_attrs.get("aria-selected", "").lower() == "true"
                or option_attrs.get("aria-checked", "").lower() == "true"
                or "selected" in option_attrs
            )
            options.append(ChoiceOption(label=label, value=value, selected=selected, control_ref=ref))
        return options
    return []


def _ancestor_nodes(node: Any) -> list[Any]:
    ancestors = []
    current = getattr(node, "parent_node", None)
    while current is not None and len(ancestors) < 30:
        ancestors.append(current)
        current = getattr(current, "parent_node", None)
    return ancestors


def _is_button(node: Any) -> bool:
    tag = _tag(node)
    attrs = _attrs(node)
    return (
        tag in _BUTTON_TAGS
        or attrs.get("role", "").lower() in {"button", "link"}
        or (tag == "input" and attrs.get("type", "").lower() in {"submit", "button"})
    )


def _dom_signature(
    url: str,
    title: str,
    fields: list[FormField],
    buttons: list[PageButton],
    validations: list[ValidationMessage],
    statuses: list[str],
) -> str:
    projection = {
        "url": url,
        "title": title,
        "fields": [
            (
                field.field_id,
                field.control_type,
                field.required,
                field.current_value,
                tuple(control.checked for control in field.controls),
                tuple((o.value, o.selected) for o in field.options),
            )
            for field in fields
        ],
        "buttons": [button.label for button in buttons],
        "validation": [message.text for message in validations],
        "status": statuses,
    }
    encoded = json.dumps(projection, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PageObserver:
    """Reads the current Browser-Use state and returns only form-relevant data."""

    def __init__(self, field_namespace: str | None = None):
        self.field_namespace = field_namespace

    def set_field_namespace(self, namespace: str | None) -> None:
        """Scope stable field IDs to a page step while retaining live refs."""
        self.field_namespace = namespace

    async def observe(self, browser_session: Any) -> PageSnapshot:
        state = await browser_session.get_browser_state_summary(
            include_screenshot=False,
            cached=False,
            include_recent_events=True,
        )
        return self.from_browser_state(state, browser_session=browser_session)

    def from_browser_state(self, state: Any, browser_session: Any | None = None) -> PageSnapshot:
        snapshot_id = uuid4().hex
        dom_state = getattr(state, "dom_state", None)
        selector_map = getattr(dom_state, "selector_map", {}) or {}
        indexed_nodes = [(int(index), node) for index, node in selector_map.items()]
        related = _iter_related(node for _, node in indexed_nodes)
        id_map = {
            _attrs(node).get("id"): node
            for node in related
            if _attrs(node).get("id")
        }
        selector_by_identity = {_node_key(node): index for index, node in indexed_nodes}

        visible = [
            (index, node)
            for index, node in indexed_nodes
            if _visible(node)
            or (_tag(node) == "input" and _attrs(node).get("type", "").lower() == "file")
        ]
        raw_fields: list[tuple[int, Any, str, str]] = []
        for index, node in visible:
            field_type = _control_type(node)
            if field_type:
                label = _label(node, related, id_map)
                raw_fields.append((index, node, field_type, label))

        fields = self._build_fields(raw_fields, snapshot_id, related, id_map, selector_by_identity)
        buttons = self._build_buttons(visible, snapshot_id, related, id_map)
        validations = self._build_validation_messages(related, id_map)
        statuses = self._build_status_messages(related)

        url = str(getattr(state, "url", "") or "")
        title = str(getattr(state, "title", "") or "")
        tabs = []
        for tab in getattr(state, "tabs", []) or []:
            tabs.append({
                "url": str(getattr(tab, "url", "") or ""),
                "title": str(getattr(tab, "title", "") or ""),
                "target_id": str(getattr(tab, "target_id", "") or ""),
            })
        page_ref = str(getattr(browser_session, "agent_focus_target_id", "") or "")
        return PageSnapshot(
            snapshot_id=snapshot_id,
            url=url,
            title=title,
            page_ref=page_ref,
            fields=fields,
            buttons=buttons,
            validation_messages=validations,
            status_messages=statuses,
            tabs=tabs,
            scroll_position={
                "above": int(getattr(state, "pixels_above", 0) or 0),
                "below": int(getattr(state, "pixels_below", 0) or 0),
            },
            dom_signature=_dom_signature(url, title, fields, buttons, validations, statuses),
            captured_at=datetime.now(timezone.utc).isoformat(),
        )

    def _live_ref(self, snapshot_id: str, index: int, node: Any) -> LiveElementRef:
        return LiveElementRef(
            snapshot_id=snapshot_id,
            selector_index=index,
            backend_node_id=int(getattr(node, "backend_node_id", 0) or 0),
            target_id=str(getattr(node, "target_id", "") or ""),
            frame_id=str(getattr(node, "frame_id", "") or ""),
            session_id=str(getattr(node, "session_id", "") or ""),
        )

    @staticmethod
    def _static_live_ref(snapshot_id: str, index: int, node: Any) -> LiveElementRef:
        return LiveElementRef(
            snapshot_id=snapshot_id,
            selector_index=index,
            backend_node_id=int(getattr(node, "backend_node_id", 0) or 0),
            target_id=str(getattr(node, "target_id", "") or ""),
            frame_id=str(getattr(node, "frame_id", "") or ""),
            session_id=str(getattr(node, "session_id", "") or ""),
        )

    def _build_fields(self, raw_fields, snapshot_id, related, id_map, selector_by_identity) -> list[FormField]:
        # Radio inputs in one frame/name (or same labelled parent when unnamed) form one field.
        radio_groups: dict[tuple[str, str], list[tuple[int, Any, str, str]]] = {}
        singles = []
        for item in raw_fields:
            index, node, control_type, label = item
            attrs = _attrs(node)
            if control_type == "radio":
                group_key = (
                    str(getattr(node, "frame_id", "") or ""),
                    attrs.get("name") or f"{label}|{getattr(getattr(node, 'parent_node', None), 'backend_node_id', '')}",
                )
                radio_groups.setdefault(group_key, []).append(item)
            else:
                singles.append(item)

        specs: list[tuple[list[tuple[int, Any, str, str]], str]] = [( [item], item[3]) for item in singles]
        for group_items in radio_groups.values():
            labels = [item[3] for item in group_items]
            parent = getattr(group_items[0][1], "parent_node", None)
            group_label = ""
            current = parent
            while current is not None:
                if _tag(current) == "fieldset":
                    legend = next((child for child in _children(current) if _tag(child) == "legend"), None)
                    group_label = _text(legend)
                    if group_label:
                        break
                current = getattr(current, "parent_node", None)
            if not group_label:
                shared = set(labels)
                group_label = next((label for label in labels if label and labels.count(label) > 1), "")
                if not group_label and len(shared) == 1:
                    group_label = labels[0]
            if not group_label:
                group_label = labels[0] if labels else "Radio group"
            specs.append((group_items, group_label))

        fields: list[FormField] = []
        occurrences: dict[str, int] = {}
        for items, label in specs:
            first_index, first_node, control_type, _ = items[0]
            attrs = _attrs(first_node)
            frame = str(getattr(first_node, "frame_id", "") or "") or None
            name = attrs.get("name") or None
            stable = "|".join((frame or "", name or "", attrs.get("id", ""), control_type, label.casefold()))
            occurrence = occurrences.get(stable, 0)
            occurrences[stable] = occurrence + 1
            field_id = "field_" + hashlib.sha256(f"{stable}|{occurrence}".encode()).hexdigest()[:16]
            if self.field_namespace:
                field_id = f"{self.field_namespace}:{field_id}"

            controls = []
            options: list[ChoiceOption] = []
            for index, node, item_type, item_label in items:
                control_ref = self._live_ref(snapshot_id, index, node)
                value = _value(node, item_type)
                checked = _checked(node) if item_type in {"radio", "checkbox"} else None
                controls.append(FieldControl(
                    control_ref=control_ref,
                    label=item_label,
                    value=value,
                    checked=checked,
                ))
                if item_type == "radio":
                    item_attrs = _attrs(node)
                    options.append(ChoiceOption(
                        label=item_label,
                        value=item_attrs.get("value", item_label),
                        selected=bool(checked),
                        control_ref=control_ref,
                    ))

            if control_type == "select":
                options = _choice_options(first_node, control_type, snapshot_id, selector_by_identity, related, id_map)
            elif control_type == "combobox":
                options = _choice_options(first_node, control_type, snapshot_id, selector_by_identity, related, id_map)

            invalid = any(_attrs(node).get("aria-invalid", "").lower() == "true" for _, node, _, _ in items)
            help_text = _help_text(first_node, id_map)
            required = any(_required(_attrs(node)) for _, node, _, _ in items)
            current_value = next((control.value for control in controls if control.checked), None)
            if current_value is None:
                current_value = controls[0].value if controls else None
            if control_type in {"radio", "checkbox"} and current_value is None and any(control.checked for control in controls):
                current_value = next((option.value for option in options if option.selected), None)

            fields.append(FormField(
                field_id=field_id,
                label=label or "Unlabelled field",
                help_text=help_text,
                control_type=control_type,
                required=required,
                current_value=current_value,
                options=options,
                controls=controls,
                frame_context=frame,
                name=name,
                autocomplete=attrs.get("autocomplete") or None,
                invalid=invalid,
            ))
        return fields

    def _build_buttons(self, visible, snapshot_id, related, id_map) -> list[PageButton]:
        buttons = []
        for index, node in visible:
            if not _is_button(node):
                continue
            attrs = _attrs(node)
            label = _label(node, related, id_map) or _text(node)
            if not label or label == "Unlabelled field":
                label = _text(node)
            if not label:
                continue
            buttons.append(PageButton(
                label=label,
                control_ref=self._live_ref(snapshot_id, index, node),
                disabled=("disabled" in attrs or attrs.get("aria-disabled", "").lower() == "true"),
                role=attrs.get("role", "button" if _tag(node) != "a" else "link"),
            ))
        return buttons

    def _build_validation_messages(self, visible, id_map) -> list[ValidationMessage]:
        messages: list[ValidationMessage] = []
        for node in visible:
            if not _visible(node):
                continue
            attrs = _attrs(node)
            role = attrs.get("role", "").lower()
            classes = attrs.get("class", "").lower()
            is_message = role == "alert" or attrs.get("aria-live", "").lower() in {"assertive", "polite"}
            is_message = is_message or ("error" in classes and bool(_text(node)))
            if not is_message:
                continue
            text = _text(node, 300)
            if text and all(item.text != text for item in messages):
                messages.append(ValidationMessage(
                    text=text,
                    frame_context=str(getattr(node, "frame_id", "") or "") or None,
                ))
            if len(messages) >= _MAX_VALIDATION_MESSAGES:
                break
        return messages

    def _build_status_messages(self, nodes) -> list[str]:
        messages: list[str] = []
        for node in nodes:
            attrs = _attrs(node)
            if not _visible(node):
                continue
            if attrs.get("role", "").lower() != "status" and attrs.get("aria-live", "").lower() not in {"polite", "assertive"}:
                continue
            text = _text(node, 300)
            if text and text not in messages:
                messages.append(text)
            if len(messages) >= _MAX_VALIDATION_MESSAGES:
                break
        return messages
