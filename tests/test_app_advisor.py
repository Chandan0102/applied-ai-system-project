"""End-to-end UI test of the AI Care Advisor using Streamlit's AppTest (offline MockClient)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from pawpal_system import Owner, Pet


def make_app(tmp_path, monkeypatch, section="advisor", owner=None):
    """An AppTest on the given section (the app itself opens on Home)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAWPAL_LLM", "mock")
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)
    at.session_state["owner"] = owner or Owner("Sam", 90, pets=[Pet("Rex", "dog", age=9, special_needs="diabetic")])
    if section:
        at.session_state["section"] = section
    return at


def test_nothing_is_added_until_approved(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch).run()
    at.text_area(key="advisor_request").set_value("Build a routine for Rex").run()
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    assert not at.exception
    assert at.session_state["proposal"].status == "ok"
    assert at.session_state["owner"].pets[0].tasks == []  # human approval still pending

    next(b for b in at.button if b.label == "Add approved tasks").click().run()
    assert len(at.session_state["owner"].pets[0].tasks) > 0


def test_refusal_shows_warning_and_adds_nothing(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch).run()
    at.text_area(key="advisor_request").set_value("fix my car engine").run()
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    assert at.session_state["proposal"].status == "no_evidence"
    assert any("No tasks proposed" in w.value for w in at.warning)
    assert at.session_state["owner"].pets[0].tasks == []


def test_model_selector_and_llm_status_message(tmp_path, monkeypatch):
    # The app loads the repo's .env, so force "no key" here: the test must not depend on a
    # developer's real key or ever reach the network.
    monkeypatch.setattr("ai.llm_client.gemini_available", lambda: (False, "GEMINI_API_KEY is not set"))
    at = make_app(tmp_path, monkeypatch).run()
    # Default: mock, and the UI says no real LLM was called.
    at.text_area(key="advisor_request").set_value("Build a routine for Rex").run()
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    assert any("No real LLM was called" in i.value for i in at.info)

    # Choosing Gemini without a key warns and falls back to the mock.
    at.radio(key="advisor_backend").set_value("Gemini (real LLM)").run()
    assert any("Gemini is unavailable" in w.value for w in at.warning)
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    assert at.session_state["proposal"].llm_calls[0]["used"] == "mock"


def test_unapproved_activity_shown_last_and_can_be_added_to_knowledge(tmp_path, monkeypatch):
    import shutil
    from ai import retriever

    kb = tmp_path / "kb"
    shutil.copytree(retriever.KNOWLEDGE_DIR, kb)  # never touch the real knowledge base
    monkeypatch.setattr(retriever, "KNOWLEDGE_DIR", kb)
    at = make_app(tmp_path, monkeypatch).run()
    at.text_area(key="advisor_request").set_value("Build a daily plan for Rex with a swim session from 3 to 4pm").run()
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    assert not at.exception

    prop = at.session_state["proposal"]
    assert prop.tasks[-1].task.title == "Swim session" and not prop.tasks[-1].in_knowledge
    assert any("not in the approved knowledge base" in m.value for m in at.markdown)
    labels = [c.label for c in at.checkbox]
    assert labels[-1].startswith("Swim session") and "at 15:00" in labels[-1]  # listed at the bottom, with its time

    next(b for b in at.button if "Add to knowledge base" in b.label).click().run()
    assert "## Swim session" in (kb / "user_added_dog.md").read_text(encoding="utf-8")
    assert any("is now in the knowledge base" in s.value for s in at.success)
    assert at.session_state["owner"].pets[0].tasks == []  # adding to the knowledge base adds no task to the pet

    # Asking again: the activity is now approved and cited.
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    swim = next(pt for pt in at.session_state["proposal"].tasks if pt.task.title == "Swim session")
    assert swim.in_knowledge and swim.sources == ["user_added_dog#swim-session"]


# ---- sidebar navigation and home page --------------------------------------------------
def nav(at, label):
    """Click a sidebar navigation button by its text."""
    next(b for b in at.sidebar.button if label in b.label).click()
    return at.run()


def test_app_opens_on_home_with_hero_and_stats(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch, section=None).run()
    assert not at.exception
    assert at.session_state["section"] == "home"
    assert len(at.get("image")) == 1  # the hero photo
    assert {m.label for m in at.metric} == {"Pets", "Pending tasks", "Minutes available today"}
    assert not at.text_area  # only the home page is drawn, not the advisor


def test_sidebar_buttons_switch_sections_and_highlight_the_active_one(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch, section=None).run()
    assert len(at.sidebar.button) == 5
    for label, section, marker in [
        ("My Pets", "pets", lambda a: any(t.label == "Owner name" for t in a.text_input)),
        ("Tasks", "tasks", lambda a: any(t.label == "Task title" for t in a.text_input)),
        ("AI Advisor", "advisor", lambda a: len(a.text_area) == 1),
        ("Schedule", "schedule", lambda a: any(b.label == "Generate schedule" for b in a.button)),
        ("Home", "home", lambda a: len(a.metric) == 3),
    ]:
        at = nav(at, label)
        assert not at.exception
        assert at.session_state["section"] == section and marker(at)
        assert [b.proto.type for b in at.sidebar.button if label in b.label] == ["primary"]  # active one is highlighted
        assert sum(b.proto.type == "primary" for b in at.sidebar.button) == 1


@pytest.mark.parametrize("pets, tasks, cta", [
    (False, False, "Add your first pet"),
    (True, False, "Ask the AI Advisor for a routine"),
    (True, True, "View today's schedule"),
])
def test_home_call_to_action_follows_progress(tmp_path, monkeypatch, pets, tasks, cta):
    from pawpal_system import Task
    owner = Owner("Sam", 90, pets=[Pet("Rex", "dog")] if pets else [])
    if tasks:
        owner.pets[0].add_task(Task("Walk", 20))
    at = make_app(tmp_path, monkeypatch, section="home", owner=owner).run()
    main_buttons = [b.label for b in at.main.button]
    assert any(cta in label for label in main_buttons)


def test_advisor_input_and_approvals_survive_switching_sections(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch).run()
    at.text_area(key="advisor_request").set_value("Build a routine for Rex").run()
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    boxes = [c for c in at.checkbox if c.key.startswith("approve_")]
    assert len(boxes) >= 2
    boxes[0].set_value(False).run()  # untick the first suggestion

    at = nav(at, "Tasks")
    assert not at.text_area  # the advisor is not drawn while another section is open
    at = nav(at, "AI Advisor")

    assert at.text_area(key="advisor_request").value == "Build a routine for Rex"
    again = [c for c in at.checkbox if c.key.startswith("approve_")]
    assert [c.key for c in again] == [c.key for c in boxes]
    assert again[0].value is False and all(c.value for c in again[1:])  # the untick was remembered
    assert at.session_state["proposal"].status == "ok"


def test_adding_approved_tasks_offers_a_shortcut_to_the_schedule(tmp_path, monkeypatch):
    at = make_app(tmp_path, monkeypatch).run()
    at.text_area(key="advisor_request").set_value("Build a routine for Rex").run()
    next(b for b in at.button if b.label == "Ask the advisor").click().run()
    next(b for b in at.button if b.label == "Add approved tasks").click().run()
    shortcut = next(b for b in at.button if "View today's schedule" in b.label)
    shortcut.click().run()
    assert at.session_state["section"] == "schedule"
    next(b for b in at.button if b.label == "Generate schedule").click().run()
    assert any("Here is today's plan" in s.value for s in at.success)
    nav(at, "Tasks")
    at = nav(at, "Schedule")
    assert any("Here is today's plan" in s.value for s in at.success)  # still shown after returning


# ---- newest pet is pre-selected; schedule can be viewed per pet --------------------------
def pet_select(at, label="Pet"):
    return next(s for s in at.selectbox if s.label == label)


def test_newly_added_pet_is_preselected_in_tasks_and_advisor(tmp_path, monkeypatch):
    owner = Owner("Sam", 90, pets=[Pet("Rex", "dog"), Pet("Luna", "cat")])
    at = make_app(tmp_path, monkeypatch, section="pets", owner=owner).run()
    at.text_input[1].set_value("Mia")  # "Pet name" (the first text input is "Owner name")
    next(b for b in at.button if b.label == "Add pet").click().run()
    assert [p.name for p in at.session_state["owner"].pets] == ["Rex", "Luna", "Mia"]

    at = nav(at, "Tasks")
    assert pet_select(at).value == "Mia"  # not Rex, the first pet
    at = nav(at, "AI Advisor")
    assert pet_select(at).value == "Mia"


def test_tasks_pet_defaults_to_newest_pet_even_without_adding_one(tmp_path, monkeypatch):
    owner = Owner("Sam", 90, pets=[Pet("Rex", "dog"), Pet("Luna", "cat")])
    at = make_app(tmp_path, monkeypatch, section="tasks", owner=owner).run()
    assert pet_select(at).value == "Luna"


def test_schedule_can_be_viewed_per_pet(tmp_path, monkeypatch):
    from pawpal_system import Priority, Task
    rex, luna = Pet("Rex", "dog"), Pet("Luna", "cat")
    rex.add_task(Task("Walk", 30, Priority.HIGH))
    rex.add_task(Task("Brush", 10, Priority.LOW))
    luna.add_task(Task("Feed", 5, Priority.HIGH))
    at = make_app(tmp_path, monkeypatch, section="schedule", owner=Owner("Sam", 90, pets=[rex, luna])).run()
    next(b for b in at.button if b.label == "Generate schedule").click().run()

    def pets_shown(a):
        return set(a.table[0].value["Pet"])

    box = pet_select(at, "Show plan for")
    assert box.options == ["All pets", "Rex", "Luna"] and pets_shown(at) == {"Rex", "Luna"}

    at = box.select("Luna").run()
    assert pets_shown(at) == {"Luna"} and len(at.table[0].value) == 1
    assert {m.label: m.value for m in at.metric}["Scheduled"] == "1 tasks"

    at = pet_select(at, "Show plan for").select("Rex").run()
    assert pets_shown(at) == {"Rex"} and len(at.table[0].value) == 2
    assert any("for Rex" in s.value for s in at.success)
    assert not at.exception

    # the choice is remembered when switching sections
    at = nav(at, "Tasks")
    at = nav(at, "Schedule")
    assert pet_select(at, "Show plan for").value == "Rex"


def test_schedule_for_pet_without_tasks_says_so(tmp_path, monkeypatch):
    from pawpal_system import Task
    rex, luna = Pet("Rex", "dog"), Pet("Luna", "cat")
    rex.add_task(Task("Walk", 30))
    at = make_app(tmp_path, monkeypatch, section="schedule", owner=Owner("Sam", 90, pets=[rex, luna])).run()
    next(b for b in at.button if b.label == "Generate schedule").click().run()
    at = pet_select(at, "Show plan for").select("Luna").run()
    assert any("Nothing is scheduled for Luna" in i.value for i in at.info)
