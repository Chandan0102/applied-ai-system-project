import os
from pathlib import Path

import streamlit as st

from ai.agent import CareAdvisor, describe_llm_use
from ai.chat_assistant import apply_action, handle_message
from ai.knowledge_store import add_user_guideline
from ai.llm_client import gemini_available, get_client
from pawpal_format import CATEGORY_EMOJI, priority_badge, status_badge, task_label
from pawpal_system import REST_CATEGORY, Owner, Pet, Priority, Scheduler, Task

# PAWPAL_DATA_FILE lets tests (or a second profile) use a different save file; the default is data.json next to this script.
DATA_FILE = Path(os.environ.get("PAWPAL_DATA_FILE") or Path(__file__).parent / "data.json")
HERO_IMAGE = Path(__file__).parent / "assets" / "hero.jpg"

st.set_page_config(
    page_title="PawPal+", page_icon="🐾", layout="wide", initial_sidebar_state="expanded"
)

# --- Application "memory" ---------------------------------------------------
# Streamlit reruns this script on every interaction, so the Owner is kept in
# st.session_state and only created if it isn't there yet. On a fresh start it is
# loaded from data.json, so pets and tasks survive restarts.
if "owner" not in st.session_state:
    try:
        st.session_state.owner = Owner.load_from_json(DATA_FILE)
    except FileNotFoundError:
        st.session_state.owner = Owner(name="Jordan", available_minutes=90)
    except ValueError as err:
        # Keep the unreadable file as a backup instead of overwriting it.
        DATA_FILE.replace(DATA_FILE.with_name(DATA_FILE.name + ".corrupt"))
        st.session_state.owner = Owner(name="Jordan", available_minutes=90)
        st.warning(f"Saved data could not be read and was set aside as data.json.corrupt ({err}).")
owner: Owner = st.session_state.owner


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@st.cache_resource
def get_advisor(backend: str) -> CareAdvisor:
    """One advisor per backend and server process (loads the knowledge base once)."""
    return CareAdvisor(get_client(backend))


def save() -> None:
    """Write the current owner to data.json."""
    try:
        owner.save_to_json(DATA_FILE)
    except OSError as err:
        st.error(f"Could not save your data: {err}")


# --- Navigation -------------------------------------------------------------
# One section is shown at a time. The sidebar buttons only change st.session_state.section
# (in an on_click callback, so the highlighted button is correct on the same run).
SECTIONS = {  # id -> (button label, page title, one-line description)
    "home": ("🏠 Home", "PawPal+", "Plan happier days for every pet."),
    "pets": ("🐕 My Pets", "My Pets", "Tell PawPal+ who you are, how much time you have, and who needs care."),
    "tasks": ("📋 Tasks", "Tasks", "Add care tasks, keep them in order, and fix time clashes."),
    "advisor": ("🤖 AI Advisor", "AI Care Advisor", "Describe what you need; the advisor drafts tasks from vetted guidelines and you approve them."),
    "schedule": ("📅 Schedule", "Today's Schedule", "A plan that fits your available time."),
}
if "section" not in st.session_state or st.session_state.section not in SECTIONS:
    st.session_state.section = "home"


def go(section: str) -> None:
    """Callback: switch the visible section."""
    st.session_state.section = section


# --- Chat assistant (floating, bottom right) ---------------------------------
# All chat logic lives in ai/chat_assistant.py; this file only keeps the conversation and applies results.
# Everything below runs as widget callbacks, i.e. before the page is drawn, so it may set widget state.
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{
        "role": "assistant", "via": "rules",
        "content": "Hi! I can open tabs, switch pets, and add tasks for you. Say **help** to see examples.",
    }]
st.session_state.setdefault("chat_open", False)
st.session_state.setdefault("chat_pending", None)
UI_KEYS = {"section", "advisor_request", "advisor_pet", "schedule_pet", "show_plan", "preferred_pet", "chat_pet", "pet_filter"}


def chat_say(role: str, text: str, via: str = "rules") -> None:
    st.session_state.chat_messages.append({"role": role, "content": text, "via": via})
    del st.session_state.chat_messages[:-40]  # keep the history short


def apply_ui(ui: dict) -> None:
    for key, value in ui.items():
        if key in UI_KEYS:
            st.session_state[key] = value


def chat_llm():
    """The real LLM, only when Gemini is selected in the AI Advisor and available; otherwise None (rules only)."""
    if str(st.session_state.get("advisor_backend", "")).startswith("Gemini") and gemini_available()[0]:
        return getattr(get_advisor("gemini"), "llm", None)
    return None


def chat_send() -> None:
    text = (st.session_state.get("chat_text") or "").strip()
    if not text:
        return
    chat_say("user", text)
    st.session_state.chat_pending = None
    reply = handle_message(text, owner, st.session_state.get("chat_pet"), chat_llm())
    apply_ui(reply.ui)
    chat_say("assistant", reply.text, reply.via)
    st.session_state.chat_pending = reply.action


def chat_confirm() -> None:
    action, st.session_state.chat_pending = st.session_state.chat_pending, None
    if action is None:
        return
    result = apply_action(action, owner)
    save()
    apply_ui(result.ui)
    chat_say("assistant", result.text)


def chat_cancel() -> None:
    st.session_state.chat_pending = None
    chat_say("assistant", "Okay, I didn't change anything.")


def chat_toggle() -> None:
    st.session_state.chat_open = not st.session_state.chat_open


def chat_pet_changed() -> None:
    apply_ui({"preferred_pet": st.session_state.chat_pet, "advisor_pet": st.session_state.chat_pet})


# Streamlit forgets a widget's value when that widget is not drawn for a run. Only one section is drawn
# at a time, so re-assigning these keys keeps typed text, choices and approvals when the user
# switches sections and comes back.
KEPT_KEYS = ("advisor_request", "advisor_pet", "advisor_backend", "pet_filter", "status_filter", "schedule_pet", "chat_pet")
for _key in list(st.session_state.keys()):
    if _key in KEPT_KEYS or _key.startswith("approve_"):
        st.session_state[_key] = st.session_state[_key]

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .stButton > button { justify-content: flex-start; text-align: left;
        padding: 0.55rem 1rem; border-radius: 12px; font-weight: 600; }
    [data-testid="stImage"] img { border-radius: 18px; }
    .st-key-chat_bubble, .st-key-chat_panel { position: fixed; right: 1.25rem; bottom: 1.25rem; left: auto;
        z-index: 999999; }
    .st-key-chat_bubble { width: fit-content !important; }
    .st-key-chat_bubble button { border-radius: 999px; padding: 0.6rem 1.2rem; font-weight: 600;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3); }
    .st-key-chat_panel { width: min(390px, calc(100vw - 2.5rem)) !important; max-height: 82vh; overflow-y: auto;
        padding: 1rem; border-radius: 16px; border: 1px solid rgba(128, 128, 128, 0.4);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)
try:  # an opaque panel background that matches the active theme
    _dark = st.context.theme.type == "dark"
except Exception:
    _dark = False
st.markdown(
    f"<style>.st-key-chat_panel {{ background: {'#1a1c24' if _dark else '#ffffff'}; }}</style>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🐾 PawPal+")
    st.caption("Pet care planning with an AI advisor")
    for _id, (_label, _, _) in SECTIONS.items():
        st.button(
            _label, key=f"nav_{_id}", on_click=go, args=(_id,), width="stretch",
            type="primary" if st.session_state.section == _id else "secondary",
        )
    st.divider()
    _model = "Gemini" if str(st.session_state.get("advisor_backend", "")).startswith("Gemini") else "Mock (offline)"
    st.caption(f"👤 {owner.name}  ·  🐾 {len(owner.pets)} pet(s)  ·  🧠 {_model}")


def page_header(section: str) -> None:
    _, title, subtitle = SECTIONS[section]
    st.title(title)
    st.caption(subtitle)


# --- Home -------------------------------------------------------------------
def render_home() -> None:
    if HERO_IMAGE.exists():
        st.image(str(HERO_IMAGE), width="stretch", caption="Photo: Krista Mangulsone, CC0 (see assets/CREDITS.md)")
    else:  # never leave the landing page empty if the photo is missing
        st.markdown("<h1 style='text-align:center;font-size:5rem'>🐶🐱</h1>", unsafe_allow_html=True)
    st.title("🐾 PawPal+")
    st.subheader("Plan happier days for every pet")
    st.write(
        "Add your pets, let the AI Advisor suggest care tasks grounded in vetted guidelines, "
        "and get a daily plan that fits your time."
    )

    pending = Scheduler(owner).filter_tasks(completed=False)
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.metric("Pets", len(owner.pets))
    with c2, st.container(border=True):
        st.metric("Pending tasks", len(pending))
    with c3, st.container(border=True):
        st.metric("Minutes available today", owner.available_minutes)

    if not owner.pets:
        st.button("🐕 Add your first pet", on_click=go, args=("pets",), type="primary")
    elif not owner.get_all_tasks():
        st.button("🤖 Ask the AI Advisor for a routine", on_click=go, args=("advisor",), type="primary")
    else:
        st.button("📅 View today's schedule", on_click=go, args=("schedule",), type="primary")


# --- My Pets (owner + pets) -------------------------------------------------
def render_pets() -> None:
    page_header("pets")
    st.subheader("Owner")
    col1, col2 = st.columns(2)
    with col1:
        owner.name = st.text_input("Owner name", value=owner.name)
    with col2:
        owner.available_minutes = int(
            st.number_input(
                "Minutes available today", min_value=5, max_value=600,
                value=owner.available_minutes, step=5,
            )
        )

    st.divider()

    st.subheader("Pets")
    with st.form("add_pet_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            pet_name = st.text_input("Pet name")
        with c2:
            species = st.selectbox("Species", ["dog", "cat", "other"])
        if st.form_submit_button("Add pet"):
            name = pet_name.strip()
            if not name:
                st.error("Please enter a pet name.")
            elif any(p.name.lower() == name.lower() for p in owner.pets):
                st.error(f"You already have a pet named {name}.")
            else:
                owner.add_pet(Pet(name=name, species=species))
                st.session_state.preferred_pet = name  # the Tasks form pre-selects the newest pet
                st.session_state.advisor_pet = name  # ...and so does the AI Advisor
                st.session_state.chat_pet = name  # ...and the chat assistant
                st.success(f"Added {name}.")

    if owner.pets:
        st.write("Your pets: " + ", ".join(f"{p.name} ({p.species})" for p in owner.pets))
        st.button("➡️ Next: add tasks", on_click=go, args=("tasks",), type="primary")
    else:
        st.info("No pets yet. Add one above.")


# --- Tasks ------------------------------------------------------------------
def render_tasks() -> None:
    page_header("tasks")
    if not owner.pets:
        st.info("Add a pet first, then you can give it tasks.")
        st.button("🐕 Go to My Pets", on_click=go, args=("pets",))
        return

    pet_names = [p.name for p in owner.pets]
    preferred = st.session_state.get("preferred_pet")
    default_index = pet_names.index(preferred) if preferred in pet_names else len(pet_names) - 1  # newest pet
    with st.form("add_task_form", clear_on_submit=True):
        target = st.selectbox("Pet", pet_names, index=default_index)
        title = st.text_input("Task title", placeholder="Morning walk")
        c1, c2, c3 = st.columns(3)
        with c1:
            duration = st.number_input("Duration (min)", min_value=1, max_value=240, value=20)
        with c2:
            priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)
        with c3:
            when = st.selectbox("Time of day", ["any", "morning", "afternoon", "evening"])
        c4, c5, c6 = st.columns(3)
        with c4:
            category = st.selectbox(
                "Type", list(CATEGORY_EMOJI), format_func=lambda c: f"{CATEGORY_EMOJI[c]} {c}"
            )
        with c5:
            frequency = st.selectbox("Frequency", ["daily", "weekly", "once"])
        with c6:
            fixed_time = st.text_input("Fixed time (HH:MM, optional)", placeholder="18:00")
        if st.form_submit_button("Add task"):
            if not title.strip():
                st.error("Please enter a task title.")
            else:
                pet = next(p for p in owner.pets if p.name == target)
                try:
                    pet.add_task(
                        Task(
                            title=title.strip(),
                            duration_minutes=int(duration),
                            priority=Priority.from_str(priority),
                            category=category,
                            preferred_time=None if when == "any" else when,
                            frequency=frequency,
                            start_time=fixed_time.strip() or None,
                        )
                    )
                    st.session_state.preferred_pet = target  # keep adding for the same pet
                    st.success(f"Added '{title.strip()}' to {target}.")
                except ValueError as err:
                    st.error(str(err))

    scheduler = Scheduler(owner)

    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    # Conflict warnings come first so the owner sees problems before the list.
    conflicts = scheduler.detect_conflicts()
    for warning in conflicts:
        st.warning(
            f"**Time clash.** {warning}\n\n"
            "You can't do these at once. Use *Change a task's time* below to move one of them."
        )

    # Filters: pet and status, applied with Scheduler.filter_tasks().
    f1, f2 = st.columns(2)
    with f1:
        pet_filter = st.selectbox("Show pet", ["All pets"] + [p.name for p in owner.pets], key="pet_filter")
    with f2:
        status_filter = st.radio("Show status", ["Pending", "Completed", "All"], horizontal=True, key="status_filter")

    shown = scheduler.filter_tasks(
        pet_name=None if pet_filter == "All pets" else pet_filter,
        completed={"Pending": False, "Completed": True, "All": None}[status_filter],
    )
    shown = scheduler.sort_by_time(shown)  # chronological, untimed tasks last

    if shown:
        st.caption(f"{len(shown)} task(s), sorted by time (tasks with no fixed time last).")
        st.table(
            [
                {
                    "Time": task.start_time or "-",
                    "Pet": pet.name,
                    "Task": task_label(task),
                    "Minutes": task.duration_minutes,
                    "Priority": priority_badge(task.priority),
                    "Repeats": task.frequency,
                    "Due": task.due_date.isoformat(),
                    "Status": status_badge(task),
                }
                for pet, task in shown
            ]
        )
    elif owner.get_all_tasks():
        st.info("No tasks match these filters.")
    else:
        st.info("No tasks yet. Add one above, or ask the AI Advisor to draft some.")
        st.button("🤖 Ask the AI Advisor", on_click=go, args=("advisor",), key="tasks_to_advisor")

    def label(pet: Pet, task: Task) -> str:
        return f"{pet.name}: {task.title} ({task.start_time or 'no fixed time'}, due {task.due_date})"

    pending = scheduler.sort_by_time(scheduler.filter_tasks(completed=False))
    if pending:
        options = {label(p, t): (p, t) for p, t in pending}

        with st.expander("Mark a task complete"):
            choice = st.selectbox("Task", list(options), key="complete_choice")
            if st.button("Mark complete"):
                pet, task = options[choice]
                nxt = pet.complete_task(task.task_id)
                st.session_state.flash = (
                    f"Completed '{task.title}'. Next one is due {nxt.due_date}."
                    if nxt
                    else f"Completed '{task.title}'."
                )
                save()
                st.rerun()

        with st.expander("Change a task's time"):
            choice = st.selectbox("Task", list(options), key="edit_choice")
            new_time = st.text_input("New fixed time (HH:MM, blank to clear)", key="new_time")
            if st.button("Update time"):
                pet, task = options[choice]
                try:
                    pet.edit_task(task.task_id, start_time=new_time.strip() or None)
                    st.session_state.flash = f"Updated the time for '{task.title}'."
                    save()
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))

    with st.expander("Find a free time slot"):
        st.caption("Looks for the earliest gap between 08:00 and 22:00 around today's fixed-time tasks.")
        need = st.number_input("Minutes needed", min_value=1, max_value=600, value=30, key="slot_minutes")
        if st.button("Find slot"):
            slot = Scheduler(owner).find_next_slot(int(need))
            if slot is None:
                st.warning(f"No free {int(need)}-minute gap left today.")
            else:
                st.success(f"🕒 Next free {int(need)}-minute slot starts at {slot:%H:%M}.")


# --- AI Care Advisor (RAG + agent + guardrails + human approval) -------------
def render_advisor() -> None:
    page_header("advisor")
    st.caption("Nothing is added until you approve it.")
    if not owner.pets:
        st.info("Add a pet first so the advisor can build tasks for it.")
        st.button("🐕 Go to My Pets", on_click=go, args=("pets",), key="advisor_to_pets")
        return

    advisor_pet_name = st.selectbox("Pet", [p.name for p in owner.pets], key="advisor_pet")
    advisor_pet = next(p for p in owner.pets if p.name == advisor_pet_name)
    request_text = st.text_area(
        "What do you need?", key="advisor_request",
        placeholder=f"e.g. Build a daily routine for {advisor_pet.name}",
    )
    can_gemini, why_not = gemini_available()
    backend_label = st.radio(
        "Model", ["Mock (offline)", "Gemini (real LLM)"], horizontal=True, key="advisor_backend",
        help="Gemini needs GEMINI_API_KEY in .env. If a Gemini call fails or returns nothing, the mock is used for that call.",
    )
    use_gemini = backend_label.startswith("Gemini")
    if use_gemini and not can_gemini:
        st.warning(f"Gemini is unavailable ({why_not}); the mock model will be used.")
        use_gemini = False
    if st.button("Ask the advisor"):
        with st.spinner("Retrieving guidelines and verifying tasks..."):
            st.session_state.proposal = get_advisor("gemini" if use_gemini else "mock").propose(
                request_text, owner=owner, pet=advisor_pet, available_minutes=owner.available_minutes
            )
            st.session_state.proposal_pet = advisor_pet.name

    proposal = st.session_state.get("proposal")
    if proposal is None:
        return

    level, llm_msg = describe_llm_use(proposal.llm_calls)
    {"ok": st.success, "warn": st.warning, "info": st.info}[level]("🧠 " + llm_msg)
    for note in proposal.notes:
        st.info("ℹ️ " + note)
    if proposal.status != "ok":
        st.warning(f"No tasks proposed: {proposal.reason}")
    else:
        conf = proposal.confidence
        if conf is not None:
            (st.success if conf.level == "high" else st.warning)(
                f"Confidence {conf.score}/100 ({conf.level}). "
                + ("Review carefully before approving." if conf.level != "high" else "")
            )
        else:
            st.info("No guideline-based tasks were generated, so there is no confidence score; only your own requests are listed.")

        def task_checkbox(pt) -> None:
            t = pt.task
            key = f"approve_{proposal.run_id}_{t.task_id}"
            if key not in st.session_state:  # set the default through session state (also survives section switches)
                st.session_state[key] = not any("already exists" in f for f in pt.flags)
            st.checkbox(
                f"{t.title} — {t.duration_minutes} min, {t.priority.name.lower()}, {t.frequency}"
                + (f", {t.preferred_time}" if t.preferred_time else "")
                + (f", at {t.start_time}" if t.start_time else ""),
                key=key,
            )
            st.caption(
                ("Sources: " + ", ".join(pt.sources) if pt.sources else "No guideline source")
                + ("  |  ⚠️ " + " ".join(pt.flags) if pt.flags else "")
            )

        approved_list = [pt for pt in proposal.tasks if pt.in_knowledge]
        unapproved_list = [pt for pt in proposal.tasks if not pt.in_knowledge]
        for pt in approved_list:
            task_checkbox(pt)

        if unapproved_list:
            st.markdown("**Requested activities not in the approved knowledge base**")
            st.caption(
                "You asked for these, but no care guideline covers them, so they have no source. "
                "You can still add them to the schedule. Use *Add to knowledge base* so future requests "
                "treat them as approved guidance (this is saved on this computer and is not vet-reviewed)."
            )
            species = proposal.species or next(
                (p.species for p in owner.pets if p.name == st.session_state.get("proposal_pet")), ""
            )
            for pt in unapproved_list:
                task_checkbox(pt)
                t = pt.task
                added_key = f"kb_added_{proposal.run_id}_{t.task_id}"
                if st.session_state.get(added_key):
                    st.success(f"'{t.title}' is now in the knowledge base for {species}.")
                elif st.button("📚 Add to knowledge base", key=f"kb_btn_{proposal.run_id}_{t.task_id}"):
                    try:
                        add_user_guideline(
                            t.title, t.duration_minutes, species, t.frequency,
                            t.priority.name.lower(), t.preferred_time,
                        )
                        st.session_state[added_key] = True
                        st.rerun()
                    except ValueError as err:
                        st.error(f"Could not add '{t.title}' to the knowledge base: {err}")

        if proposal.dropped:
            st.info("Blocked by guardrails: " + ", ".join(proposal.dropped))
        if st.button("Add approved tasks"):
            target = next((p for p in owner.pets if p.name == st.session_state.proposal_pet), None)
            added = 0
            for pt in proposal.tasks:
                if target and st.session_state.get(f"approve_{proposal.run_id}_{pt.task.task_id}"):
                    target.add_task(pt.task)
                    added += 1
            st.session_state.proposal = None
            save()
            st.success(f"Added {added} task(s) to {target.name if target else 'your pet'}.")
            st.button("📅 View today's schedule", on_click=go, args=("schedule",), key="advisor_to_schedule")
    with st.expander("How the advisor decided (trace)"):
        for s in proposal.trace:
            st.text(f"{s['step']}: " + str({k: v for k, v in s.items() if k != 'step'})[:400])


# --- Schedule ---------------------------------------------------------------
def render_schedule() -> None:
    page_header("schedule")
    st.caption(
        "The plan fills your available time by priority, starting at 08:00. Tasks with a time you asked for "
        "(like a 12:00 nap) are placed exactly there and other tasks work around them. Pet rest blocks "
        "don't use your time budget."
    )
    if st.button("Generate schedule"):
        st.session_state.show_plan = True  # stays on screen if the user switches section and returns
    if not st.session_state.get("show_plan"):
        return
    if not owner.get_all_tasks():
        st.warning("Add at least one task before generating a schedule.")
        return

    # One plan is built for all pets (they share the time budget); the dropdown only chooses what is shown.
    plan = Scheduler(owner).generate_plan()
    choice = st.selectbox("Show plan for", ["All pets"] + [p.name for p in owner.pets], key="schedule_pet")
    only = None if choice == "All pets" else choice
    entries = [i for i in plan.entries if only is None or i.pet.name == only]
    skipped = [x for x in plan.skipped if only is None or x.pet.name == only]
    used = sum(i.task.duration_minutes for i in entries if i.task.category != REST_CATEGORY)

    m1, m2, m3 = st.columns(3)
    m1.metric("Scheduled", f"{len(entries)} tasks")
    m2.metric("Time used", f"{used} / {owner.available_minutes} min" if only is None else f"{used} min")
    m3.metric("Skipped", f"{len(skipped)} tasks")
    if only is not None:
        st.caption(
            f"All pets together use {plan.total_minutes} / {owner.available_minutes} min; "
            "the time budget is shared between pets."
        )

    if entries:
        st.success("Here is today's plan." if only is None else f"Here is today's plan for {only}.")
        st.table(
            [
                {
                    "Time": f"{i.start:%H:%M}-{i.end:%H:%M}",
                    "Pet": i.pet.name,
                    "Task": task_label(i.task),
                    "Priority": priority_badge(i.task.priority),
                }
                for i in entries
            ]
        )
    elif only is not None:
        st.info(f"Nothing is scheduled for {only} today.")
    else:
        st.info("Nothing to schedule today: every task is done or not due yet.")
    if skipped:
        st.warning("Some tasks did not fit in your available time:")
        for x in skipped:
            st.write(f"- **{x.pet.name}: {x.task.title}** — {x.reason}")
    with st.expander("Why this plan?"):
        if only is None:
            st.text(plan.explain())
        else:
            lines = [f"Included {i.pet.name} - {i.task.title}: {i.reason}" for i in entries]
            lines += [f"Skipped {x.pet.name} - {x.task.title}: {x.reason}" for x in skipped]
            st.text("\n".join(lines) or "No tasks to explain.")


RENDERERS = {
    "home": render_home,
    "pets": render_pets,
    "tasks": render_tasks,
    "advisor": render_advisor,
    "schedule": render_schedule,
}
RENDERERS[st.session_state.section]()


def render_chat() -> None:
    """Floating chat: a bubble at the bottom right that opens a panel on every tab."""
    if not st.session_state.chat_open:
        with st.container(key="chat_bubble"):
            st.button("💬 Chat", key="chat_open_btn", on_click=chat_toggle, help="Open the PawPal assistant")
        return
    with st.container(key="chat_panel"):
        head, close = st.columns([5, 1])
        head.markdown("**💬 PawPal assistant**")
        close.button("✕", key="chat_close_btn", on_click=chat_toggle, help="Close chat")
        names = [p.name for p in owner.pets]
        if names:
            if st.session_state.get("chat_pet") not in names:
                preferred = st.session_state.get("preferred_pet")
                st.session_state.chat_pet = preferred if preferred in names else names[-1]
            st.selectbox(
                "Selected pet", names, key="chat_pet", on_change=chat_pet_changed,
                format_func=lambda n: f"🐾 {n}", help="Tasks you add in the chat go to this pet.",
            )
        st.caption(
            "🧠 Rules, with Gemini for unclear messages" if chat_llm() is not None
            else "🧠 Rules only (offline). Changes need your confirmation."
        )
        # Usual chat order: oldest at the top, newest at the bottom (the box scrolls to the newest message),
        # then the confirm buttons, then the input.
        with st.container(height=260):
            for msg in st.session_state.chat_messages[-12:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("via") == "llm":
                        st.caption("interpreted by Gemini")
        if st.session_state.chat_pending is not None:
            c1, c2 = st.columns(2)
            c1.button("✅ Confirm", key="chat_confirm_btn", on_click=chat_confirm, type="primary", width="stretch")
            c2.button("✖ Cancel", key="chat_cancel_btn", on_click=chat_cancel, width="stretch")
        with st.form("chat_form", clear_on_submit=True, border=False):
            st.text_input("Message", key="chat_text", label_visibility="collapsed",
                          placeholder='e.g. add a 20 minute evening walk')
            st.form_submit_button("Send", on_click=chat_send, width="stretch")


render_chat()

# Persist whatever the user changed during this run.
save()
