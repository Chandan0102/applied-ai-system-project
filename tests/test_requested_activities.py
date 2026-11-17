"""Owner-requested activities with times (e.g. a nap 12-2pm), the knowledge-base add flow,
and the scheduler honouring fixed times."""

import shutil
from datetime import time
from pathlib import Path

import pytest

from ai import retriever
from ai.agent import CareAdvisor
from ai.knowledge_store import add_user_guideline, match_activity
from ai.llm_client import MockClient
from ai.logger import AgentLogger
from ai.request_parser import parse_requested_activities
from ai.retriever import Retriever, load_knowledge
from pawpal_system import Owner, Pet, Priority, Scheduler, Task

REAL_KB = Path(retriever.KNOWLEDGE_DIR)


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """A throwaway copy of the knowledge base so tests never modify the real one."""
    dest = tmp_path / "knowledge"
    shutil.copytree(REAL_KB, dest)
    for f in dest.glob("user_added_*.md"):
        f.unlink()
    monkeypatch.setattr(retriever, "KNOWLEDGE_DIR", dest)
    return dest


def advisor():
    return CareAdvisor(MockClient(), AgentLogger(enabled=False))


def acts(text):
    found, _ = parse_requested_activities(text)
    return [(a.title, a.start, a.duration_minutes, a.preferred_time) for a in found]


# ---- parser ----------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("Build a daily plan with afternooon nap from 12 pm to 2Pm", [("Afternooon nap", "12:00", 120, "afternoon")]),
    ("plan with an afternoon nap from 12 to 2", [("Afternoon nap", "12:00", 120, "afternoon")]),
    ("add bath time between 6:30 and 7pm", [("Bath time", "18:30", 30, "evening")]),
    ("routine with a swim from 11 to 2pm", [("Swim", "11:00", 180, "morning")]),
    ("plan for my dog with playtime at 4pm for 30 minutes", [("Playtime", "16:00", 30, "afternoon")]),
    ("plan with a cuddle at 9am for 1 hour", [("Cuddle", "09:00", 60, "morning")]),
    ("build a routine for my cat", []),
])
def test_parser_reads_windows_and_durations(text, expected):
    assert acts(text) == expected


@pytest.mark.parametrize("text", ["daily plan from 12 to 2", "nap from 14:00 to 13:00", "nap from 9am to 9am"])
def test_parser_reports_unusable_windows_instead_of_guessing(text):
    found, notes = parse_requested_activities(text)
    assert found == [] and notes


# ---- knowledge matching and the add-to-knowledge flow ----------------------
def test_match_activity_tolerates_typos_and_rejects_unknowns():
    chunks = load_knowledge()
    assert match_activity("Afternooon nap", "dog", chunks).title == "Afternoon nap"
    assert match_activity("Nap", "cat", chunks).title == "Afternoon nap"
    assert match_activity("Swim session", "dog", chunks) is None
    assert match_activity("Long walk", "dog", chunks) is None  # only 1 of 2 words matches


def test_dog_guidance_does_not_leak_small_pet_tasks():
    ids = {c.chunk_id for c in Retriever().search("daily routine for my dog", species="dog", k=20)}
    assert "small_pets#daily-care-and-feeding" not in ids
    assert "small_pets#daily-care-and-feeding" in {c.chunk_id for c in Retriever().search("feed my hamster", species="hamster")}


def test_add_user_guideline_makes_activity_known_and_is_idempotent(kb):
    assert match_activity("Swim session", "dog", load_knowledge()) is None
    cid = add_user_guideline("Swim session", 60, "dog", preferred_time="afternoon")
    assert cid == "user_added_dog#swim-session"
    assert add_user_guideline("Swim session", 60, "dog") == cid
    assert (kb / "user_added_dog.md").read_text(encoding="utf-8").count("## Swim session") == 1
    assert match_activity("swim session", "dog", load_knowledge()).chunk_id == cid
    assert match_activity("swim session", "cat", load_knowledge()) is None  # filed under dog only


@pytest.mark.parametrize("title, minutes", [("Give 5 units of insulin", 5), ("Insulin shot", 5), ("Feed chocolate", 5), ("x", 5), ("Swim", 500)])
def test_add_user_guideline_refuses_unsafe_or_invalid(kb, title, minutes):
    with pytest.raises(ValueError):
        add_user_guideline(title, minutes, "dog")
    assert not list(kb.glob("user_added_*.md"))


def test_user_added_entry_is_not_pulled_into_unrelated_routines(kb):
    add_user_guideline("Swim session", 60, "dog")
    ids = {c.chunk_id for c in Retriever().search("daily routine for my dog", species="dog", k=30)}
    assert not any(i.startswith("user_added") for i in ids)


# ---- agent -----------------------------------------------------------------
def test_nap_request_is_cited_pinned_and_title_corrected():
    pet = Pet("Rex", "dog", age=5)
    p = advisor().propose("Build a daily plan with afternooon nap from 12 pm to 2Pm", owner=Owner("Sam", 90, pets=[pet]), pet=pet)
    naps = [pt for pt in p.tasks if "nap" in pt.task.title.lower()]
    assert len(naps) == 1  # the model's own nap was replaced, not duplicated
    nap = naps[0]
    assert (nap.task.title, nap.task.start_time, nap.task.duration_minutes) == ("Afternoon nap", "12:00", 120)
    assert nap.task.fixed_time and nap.task.category == "rest"
    assert nap.sources == ["rest_and_sleep#naps-and-quiet-time"] and nap.in_knowledge
    assert any("12:00-14:00" in f for f in nap.flags)


def test_unknown_activity_goes_last_and_is_flagged_not_in_knowledge():
    pet = Pet("Rex", "dog", age=5)
    p = advisor().propose("Build a daily plan with a swim session from 3 to 4pm", owner=Owner("Sam", 90, pets=[pet]), pet=pet)
    assert p.status == "ok"
    assert p.tasks[-1].task.title == "Swim session" and not p.tasks[-1].in_knowledge and p.tasks[-1].sources == []
    assert all(pt.in_knowledge for pt in p.tasks[:-1])
    assert any("Not in the approved knowledge base" in f for f in p.tasks[-1].flags)


def test_user_requested_dose_is_still_blocked():
    p = advisor().propose("daily plan for my dog with 5 units of insulin at 8am for 5 minutes", pet=Pet("Rex", "dog", 5))
    assert not any("units" in pt.task.title.lower() for pt in p.tasks)


def test_user_requested_medication_time_is_flagged_for_vet():
    pet = Pet("Rex", "dog", age=5)
    p = advisor().propose("daily plan with the insulin shot at 8am for 5 minutes", owner=Owner("S", 90, pets=[pet]), pet=pet)
    shot = next(pt for pt in p.tasks if pt.task.title == "Insulin shot")
    assert any("vet" in f.lower() for f in shot.flags) and not shot.in_knowledge


def test_requested_activity_only_request_still_works_without_guidelines():
    pet = Pet("Bean", "dog", age=5)
    p = advisor().propose("swim session from 3 to 4pm", pet=pet)
    assert p.status == "ok" and [pt.task.title for pt in p.tasks] == ["Swim session"]
    assert p.confidence is None  # nothing was generated from guidelines, so no confidence is claimed


def test_after_adding_to_knowledge_the_activity_is_cited_next_time(kb):
    pet = Pet("Rex", "dog", age=5)
    owner = Owner("Sam", 90, pets=[pet])
    first = advisor().propose("plan with a swim session from 3 to 4pm", owner=owner, pet=pet)
    assert not first.tasks[-1].in_knowledge
    add_user_guideline("Swim session", 60, "dog", preferred_time="afternoon")
    second = CareAdvisor(MockClient(), AgentLogger(enabled=False)).propose("plan with a swim session from 3 to 4pm", owner=owner, pet=pet)
    swim = next(pt for pt in second.tasks if pt.task.title == "Swim session")
    assert swim.in_knowledge and swim.sources == ["user_added_dog#swim-session"]


# ---- scheduler -------------------------------------------------------------
def make_owner(minutes=90):
    pet = Pet("Rex", "dog")
    return Owner("Sam", minutes, pets=[pet]), pet


def test_fixed_rest_block_is_placed_exactly_and_uses_no_owner_time():
    owner, pet = make_owner(30)
    pet.add_task(Task("Afternoon nap", 120, Priority.LOW, category="rest", start_time="12:00", fixed_time=True))
    pet.add_task(Task("Walk", 30, Priority.HIGH))
    plan = Scheduler(owner).generate_plan()
    by_title = {i.task.title: i for i in plan.entries}
    assert (by_title["Afternoon nap"].start, by_title["Afternoon nap"].end) == (time(12, 0), time(14, 0))
    assert by_title["Walk"].start == time(8, 0)
    assert plan.total_minutes == 30 and not plan.skipped  # a 120-minute nap fits a 30-minute budget


def test_flexible_tasks_are_pushed_past_a_fixed_block():
    owner, pet = make_owner(minutes=200)
    pet.add_task(Task("Swim", 60, Priority.MEDIUM, start_time="09:00", fixed_time=True))
    pet.add_task(Task("Walk", 30, Priority.HIGH))
    pet.add_task(Task("Groom", 30, Priority.LOW))
    entries = Scheduler(owner, start_time=time(8, 30)).generate_plan().entries
    spans = {i.task.title: (i.start, i.end) for i in entries}
    assert spans["Swim"] == (time(9, 0), time(10, 0))
    assert spans["Walk"] == (time(8, 30), time(9, 0))  # fits in the gap before the fixed block
    assert spans["Groom"][0] == time(10, 0)  # would have overlapped the block, so it moves after it
    assert not any(a != b and spans[a][0] < spans[b][1] and spans[b][0] < spans[a][1] for a in spans for b in spans)


def test_start_time_without_fixed_flag_keeps_original_behaviour():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 30, Priority.HIGH, start_time="18:00"))
    assert Scheduler(owner).generate_plan().entries[0].start == time(8, 0)


def test_fixed_time_needs_a_start_time_and_survives_save_and_load(tmp_path):
    with pytest.raises(ValueError):
        Task("Nap", 30, fixed_time=True)
    owner, pet = make_owner()
    pet.add_task(Task("Nap", 120, category="rest", start_time="12:00", fixed_time=True))
    owner.save_to_json(tmp_path / "d.json")
    assert Owner.load_from_json(tmp_path / "d.json").pets[0].tasks[0].fixed_time is True


def test_species_and_pet_names_are_not_mistaken_for_activities():
    assert acts("daily plan for my dog from 12 to 2") == []
    assert acts("plan for Rex from 12 to 2") == [("Rex", "12:00", 120, "afternoon")]  # without knowing names...
    found, notes = parse_requested_activities("plan for Rex from 12 to 2", frozenset({"rex"}))
    assert found == [] and notes  # ...but the agent passes the pets' names
