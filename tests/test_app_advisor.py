"""End-to-end UI test of the AI Care Advisor using Streamlit's AppTest (offline MockClient)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from pawpal_system import Owner, Pet


def make_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAWPAL_LLM", "mock")
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)
    at.session_state["owner"] = Owner("Sam", 90, pets=[Pet("Rex", "dog", age=9, special_needs="diabetic")])
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
