"""The floating chat assistant, driven through the real Streamlit app (AppTest, offline)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from pawpal_system import Owner, Pet, Priority, Task

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def make_app(tmp_path, monkeypatch, section="home", pets=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ai.llm_client.gemini_available", lambda: (False, "no key"))  # never reach the network
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["owner"] = Owner("Sam", 90, pets=pets if pets is not None else [Pet("Rex", "dog", 5), Pet("Luna", "cat", 3)])
    at.session_state["section"] = section
    return at


def open_chat(at):
    next(b for b in at.button if b.label == "💬 Chat").click()
    return at.run()


def send(at, text):
    at.text_input(key="chat_text").set_value(text)
    next(b for b in at.button if b.label == "Send").click()
    return at.run()


def press(at, label):
    next(b for b in at.button if label in b.label).click()
    return at.run()


def chat_text(at):
    return " ".join(m.value for m in at.markdown)


def test_chat_is_a_bubble_until_opened_and_can_be_closed(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch).run()
    assert not at.exception
    assert any(b.label == "💬 Chat" for b in at.button) and not at.text_input
    at = open_chat(at)
    assert not at.exception and at.text_input(key="chat_text") is not None
    assert any(s.label == "Selected pet" for s in at.selectbox)
    at = press(at, "✕")
    assert not at.text_input and any(b.label == "💬 Chat" for b in at.button)


def test_chat_is_available_on_every_tab(tmp_path, monkeypatch):
    for section in ["home", "pets", "tasks", "advisor", "schedule"]:
        at = make_app(tmp_path, monkeypatch, section=section).run()
        assert any(b.label == "💬 Chat" for b in at.button), section
        assert not at.exception, section


def test_chat_navigates_the_left_hand_tabs(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    at = send(at, "open the tasks")
    assert at.session_state["section"] == "tasks" and any(t.label == "Task title" for t in at.text_input)
    at = send(at, "show Luna's tasks")
    assert at.session_state["pet_filter"] == "Luna"
    at = send(at, "generate the schedule for Rex")
    assert at.session_state["section"] == "schedule" and at.session_state["schedule_pet"] == "Rex"
    assert any("Here is today's plan" in s.value for s in at.success) or any("Add at least one task" in w.value for w in at.warning)
    assert not at.exception


def test_adding_a_task_by_chat_needs_confirmation_then_adds_it_to_the_selected_pet(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    assert next(s for s in at.selectbox if s.label == "Selected pet").value == "Luna"  # newest pet by default
    at = send(at, "add a 20 minute evening walk")
    assert at.session_state["chat_pending"].kind == "add_task"
    assert at.session_state["owner"].pets[1].tasks == []  # nothing yet
    assert any(b.label == "✅ Confirm" for b in at.button)

    at = press(at, "Confirm")
    luna = at.session_state["owner"].pets[1]
    assert [(t.title, t.duration_minutes, t.preferred_time) for t in luna.tasks] == [("Evening walk", 20, "evening")]
    assert at.session_state["owner"].pets[0].tasks == [] and at.session_state["chat_pending"] is None
    assert (tmp_path / "pawpal_test_data.json").exists()  # saved like every other change (to the test file, never the real one)


def test_choosing_another_pet_in_the_chat_changes_who_gets_the_task(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    next(s for s in at.selectbox if s.label == "Selected pet").select("Rex").run()
    at = send(at, "add brush teeth high priority")
    at = press(at, "Confirm")
    rex = at.session_state["owner"].pets[0]
    assert [(t.title, t.priority) for t in rex.tasks] == [("Brush teeth", Priority.HIGH)]
    assert at.session_state["advisor_pet"] == "Rex"  # the choice is shared with the other tabs


def test_cancel_leaves_data_untouched(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    at = send(at, "add a walk for Rex")
    at = press(at, "Cancel")
    assert at.session_state["chat_pending"] is None
    assert all(p.tasks == [] for p in at.session_state["owner"].pets)
    assert "didn't change anything" in chat_text(at)


def test_a_timed_task_from_chat_lands_exactly_in_the_schedule(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    at = send(at, "add nap from 12 pm to 2 pm for Rex")
    at = press(at, "Confirm")
    nap = at.session_state["owner"].pets[0].tasks[0]
    assert (nap.title, nap.start_time, nap.duration_minutes, nap.fixed_time) == ("Nap", "12:00", 120, True)
    at = send(at, "generate the schedule for Rex")
    assert "12:00-14:00" in at.table[0].value["Time"].tolist()


def test_adding_a_pet_by_chat(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch, pets=[]).run())
    assert not any(s.label == "Selected pet" for s in at.selectbox)  # no pets yet
    at = send(at, "add a dog named Mia")
    at = press(at, "Confirm")
    assert [p.name for p in at.session_state["owner"].pets] == ["Mia"]
    at = send(at, "add a walk")
    at = press(at, "Confirm")
    assert [t.title for t in at.session_state["owner"].pets[0].tasks] == ["Walk"]


def test_unsafe_requests_are_refused_in_chat(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    at = send(at, "add 5 units of insulin at 8am")
    assert at.session_state["chat_pending"] is None and "dose" in chat_text(at)
    assert not any(b.label == "✅ Confirm" for b in at.button)


def test_advisor_requests_prefill_the_advisor_and_conversation_survives_tab_changes(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    at = send(at, "suggest a routine for Rex")
    assert at.session_state["section"] == "advisor"
    assert at.text_area(key="advisor_request").value == "suggest a routine for Rex"
    assert next(s for s in at.selectbox if s.label == "Pet").value == "Rex"
    next(b for b in at.sidebar.button if "Tasks" in b.label).click()
    at = at.run()
    assert "suggest a routine for Rex" in chat_text(at) and not at.exception  # history kept across tabs


def test_chat_works_while_the_target_widget_is_on_screen(tmp_path, monkeypatch):
    """Changing a widget's value from chat while that widget is drawn must not raise."""
    at = open_chat(make_app(tmp_path, monkeypatch, section="advisor").run())
    at = send(at, "suggest a routine for Luna")
    assert not at.exception and at.text_area(key="advisor_request").value == "suggest a routine for Luna"
    at = send(at, "switch to Rex")
    assert not at.exception and next(s for s in at.selectbox if s.label == "Pet").value == "Rex"


def test_empty_chat_messages_are_ignored(tmp_path, monkeypatch):
    at = open_chat(make_app(tmp_path, monkeypatch).run())
    at = send(at, "help")
    count = len(at.session_state["chat_messages"])
    at = send(at, "   ")
    assert len(at.session_state["chat_messages"]) == count
