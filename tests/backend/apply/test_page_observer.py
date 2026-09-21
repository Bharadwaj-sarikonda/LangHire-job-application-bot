from types import SimpleNamespace

from backend.apply.page_observer import PageObserver


class Node:
    def __init__(self, tag, attrs=None, text="", *, backend_id=1, frame_id="main", visible=True, value=None, checked=None):
        self.tag_name = tag
        self.node_name = tag.upper()
        self.attributes = attrs or {}
        self.node_value = text
        self.backend_node_id = backend_id
        self.node_id = backend_id
        self.target_id = "tab-1"
        self.frame_id = frame_id
        self.session_id = f"session-{frame_id}"
        self.is_visible = visible
        self.parent_node = None
        self.children_nodes = []
        self.shadow_roots = []
        self.content_document = None
        self.ax_node = SimpleNamespace(name="")
        self.snapshot_node = SimpleNamespace(input_value=value, input_checked=checked, computed_styles={})

    def add(self, *children):
        for child in children:
            child.parent_node = self
            self.children_nodes.append(child)
        return self

    def get_all_children_text(self):
        values = [self.node_value] if self.node_value else []
        for child in self.children_nodes:
            values.append(child.get_all_children_text())
        return " ".join(value for value in values if value)


def _snapshot(nodes, url="https://example.test/app", title="Application"):
    return SimpleNamespace(
        url=url,
        title=title,
        tabs=[SimpleNamespace(url=url, title=title, target_id="tab-1")],
        dom_state=SimpleNamespace(selector_map={i: node for i, node in enumerate(nodes, 1)}),
    )


def test_extracts_label_value_required_help_and_options():
    root = Node("form", backend_id=1)
    label = Node("label", {"for": "city"}, "City", backend_id=2)
    help_text = Node("div", {"id": "city-help"}, "Use your current city", backend_id=3)
    field = Node(
        "input",
        {"id": "city", "name": "city", "type": "text", "required": "", "aria-describedby": "city-help"},
        backend_id=4,
        value="Boston",
    )
    root.add(label, help_text, field)
    country_label = Node("label", {"for": "country"}, "Country", backend_id=5)
    select = Node("select", {"id": "country", "name": "country"}, backend_id=6, value="US")
    select.add(
        Node("option", {"value": "US", "selected": ""}, "United States", backend_id=7),
        Node("option", {"value": "CA"}, "Canada", backend_id=8),
    )
    root.add(country_label, select)

    snapshot = PageObserver().from_browser_state(_snapshot([field, select]))

    assert len(snapshot.fields) == 2
    city = next(field for field in snapshot.fields if field.name == "city")
    assert city.label == "City"
    assert city.current_value == "Boston"
    assert city.required is True
    assert city.help_text == "Use your current city"
    assert city.controls[0].control_ref.snapshot_id == snapshot.snapshot_id
    country = next(field for field in snapshot.fields if field.name == "country")
    assert country.control_type == "select"
    assert [(option.label, option.value, option.selected) for option in country.options] == [
        ("United States", "US", True),
        ("Canada", "CA", False),
    ]


def test_groups_radio_options_by_name_and_uses_legend_as_question():
    form = Node("form", backend_id=1)
    fieldset = Node("fieldset", backend_id=2)
    fieldset.add(Node("legend", text="Are you authorized to work?", backend_id=3))
    yes_label = Node("label", {"for": "auth-yes"}, "Yes", backend_id=4)
    yes = Node("input", {"id": "auth-yes", "name": "authorization", "type": "radio", "value": "yes"}, backend_id=5)
    no_label = Node("label", {"for": "auth-no"}, "No", backend_id=6)
    no = Node("input", {"id": "auth-no", "name": "authorization", "type": "radio", "value": "no"}, backend_id=7, checked=True)
    fieldset.add(yes_label, yes, no_label, no)
    form.add(fieldset)

    snapshot = PageObserver().from_browser_state(_snapshot([yes, no]))

    assert len(snapshot.fields) == 1
    field = snapshot.fields[0]
    assert field.label == "Are you authorized to work?"
    assert field.control_type == "radio"
    assert [(option.label, option.value, option.selected) for option in field.options] == [
        ("Yes", "yes", False),
        ("No", "no", True),
    ]


def test_duplicate_labels_get_distinct_field_ids_and_refs_change_per_snapshot():
    first = Node("input", {"type": "text", "aria-label": "Other", "name": "other_a"}, backend_id=10)
    second = Node("input", {"type": "text", "aria-label": "Other", "name": "other_b"}, backend_id=11)
    observer = PageObserver()
    one = observer.from_browser_state(_snapshot([first, second]))
    two = observer.from_browser_state(_snapshot([first, second]))

    assert len({field.field_id for field in one.fields}) == 2
    assert one.fields[0].field_id == two.fields[0].field_id
    assert one.fields[0].controls[0].control_ref.snapshot_id != two.fields[0].controls[0].control_ref.snapshot_id


def test_field_namespace_disambiguates_the_same_control_on_separate_form_pages():
    field = Node("input", {"type": "text", "name": "answer", "aria-label": "Additional information"}, backend_id=10)
    observer = PageObserver(field_namespace="page0")
    first = observer.from_browser_state(_snapshot([field]))
    observer.set_field_namespace("page1")
    second = observer.from_browser_state(_snapshot([field]))

    assert first.fields[0].field_id != second.fields[0].field_id
    assert first.fields[0].field_id.endswith(second.fields[0].field_id.split(":", 1)[1])


def test_ignores_hidden_and_non_field_controls_and_redacts_password_value():
    hidden = Node("input", {"type": "text", "aria-label": "Hidden"}, backend_id=2, visible=False)
    submit = Node("input", {"type": "submit", "value": "Submit"}, backend_id=3)
    password = Node("input", {"type": "password", "aria-label": "Password"}, backend_id=4, value="secret")
    snapshot = PageObserver().from_browser_state(_snapshot([hidden, submit, password]))

    assert [field.label for field in snapshot.fields] == ["Password"]
    assert snapshot.fields[0].current_value is None


def test_extracts_validation_messages_and_frame_context():
    alert = Node("div", {"role": "alert"}, "Please choose an option", backend_id=2, frame_id="frame-1")
    field = Node("textarea", {"aria-label": "Why interested?", "aria-invalid": "true"}, backend_id=3, frame_id="frame-1")
    snapshot = PageObserver().from_browser_state(_snapshot([alert, field]))

    assert snapshot.validation_messages[0].text == "Please choose an option"
    assert snapshot.validation_messages[0].frame_context == "frame-1"
    assert snapshot.fields[0].invalid is True
    assert snapshot.fields[0].frame_context == "frame-1"


def test_answer_payload_excludes_live_element_references():
    field = Node("input", {"type": "text", "aria-label": "Name"}, backend_id=2)
    snapshot = PageObserver().from_browser_state(_snapshot([field]))

    payload = snapshot.answer_payload()

    assert payload["fields"][0]["label"] == "Name"
    assert "control_ref" not in payload["fields"][0]
    assert "backend_node_id" not in repr(payload)


def test_observes_visible_options_from_an_expanded_react_combobox():
    root = Node("div", backend_id=1)
    combo = Node(
        "input",
        {"role": "combobox", "aria-label": "Country", "aria-expanded": "true", "aria-controls": "country-options", "autocomplete": "country-name"},
        backend_id=2,
        value="Can",
    )
    listbox = Node("div", {"id": "country-options", "role": "listbox"}, backend_id=3)
    canada = Node("div", {"role": "option", "data-value": "CA", "aria-selected": "true"}, "Canada", backend_id=4)
    us = Node("div", {"role": "option", "data-value": "US"}, "United States", backend_id=5)
    listbox.add(canada, us)
    root.add(combo, listbox)

    snapshot = PageObserver().from_browser_state(_snapshot([combo, canada, us]))

    field = snapshot.fields[0]
    assert field.control_type == "combobox"
    assert field.autocomplete == "country-name"
    assert [(option.label, option.value, option.selected) for option in field.options] == [
        ("Canada", "CA", True),
        ("United States", "US", False),
    ]
    assert field.options[0].control_ref.selector_index == 2


def test_hidden_file_input_remains_available_for_browser_use_upload_action():
    file_input = Node("input", {"type": "file", "aria-label": "Resume upload"}, backend_id=9, visible=False)

    snapshot = PageObserver().from_browser_state(_snapshot([file_input]))

    assert len(snapshot.fields) == 1
    assert snapshot.fields[0].control_type == "file"
    assert snapshot.fields[0].label == "Resume upload"
