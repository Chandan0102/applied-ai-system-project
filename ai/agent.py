"""CareAdvisor: plan -> retrieve -> propose -> verify -> repair -> score.

The LLM only sees retrieved evidence, every task it returns must cite that
evidence, and guardrails verify the result before a human is asked to approve.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ai.confidence import Confidence, score_proposal
from ai.guardrails import VET_NOTE, Issue, is_medication, sanitize_request, validate_tasks
from ai.knowledge_store import match_activity
from ai.llm_client import MockClient
from ai.logger import AgentLogger
from ai.request_parser import parse_requested_activities
from ai.retriever import Chunk, Retriever, load_knowledge, owner_chunks
from pawpal_system import Pet, Priority, Task

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MAX_ROUNDS = 3
MIN_CONFIDENCE_TO_PROPOSE = 25
INTENT_WORDS = {"routine", "plan", "schedule", "care", "daily", "tasks", "task"}
SPECIES_WORDS = ["dog", "cat", "rabbit", "guinea pig", "hamster", "puppy", "kitten"]
CATEGORY_WORDS = {
    "meds": ("insulin", "medic", "pill"),
    "walk": ("walk",),
    "feeding": ("meal", "breakfast", "dinner", "feed", "hay", "water bowl"),
    "grooming": ("brush", "nail", "groom"),
    "enrichment": ("play", "puzzle", "scent", "training"),
    "rest": ("nap", "sleep", "quiet time"),
}


@dataclass
class ProposedTask:
    task: Task
    sources: list[str]
    flags: list[str] = field(default_factory=list)
    origin: str = "guideline"  # "guideline" (cited) or "user" (asked for by the owner, no guideline)
    in_knowledge: bool = True  # False = not in the approved knowledge base; shown separately in the UI


@dataclass
class Proposal:
    status: str  # "ok" | "refused" | "no_evidence"
    reason: str = ""
    tasks: list[ProposedTask] = field(default_factory=list)
    confidence: Optional[Confidence] = None
    issues: list[Issue] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)  # one entry per LLM call (backend used, status)
    notes: list[str] = field(default_factory=list)  # things the owner should know (unreadable times, dropped requests)
    species: Optional[str] = None
    run_id: str = ""


def _category(title: str) -> str:
    low = title.lower()
    for cat, words in CATEGORY_WORDS.items():
        if any(w in low for w in words):
            return cat
    return "general"


def life_stage(species: Optional[str], age: Optional[int], request: str) -> str:
    low = request.lower()
    if "puppy" in low or "kitten" in low:
        return "puppy"
    if age is None:
        return "senior" if re.search(r"\b(senior|elderly|old)\b", low) else "adult"
    if age < 1:
        return "puppy"
    return "senior" if age >= (10 if species == "cat" else 8) else "adult"


def parse_json(raw: str) -> Optional[dict]:
    """Parse the model reply as {"tasks": [...]}; tolerate code fences, reject anything else."""
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and isinstance(data.get("tasks"), list) else None


def describe_llm_use(calls: list[dict]) -> tuple[str, str]:
    """Summarise a proposal's LLM calls as (level, message); level is "ok", "warn" or "info"."""
    if not calls:
        return "info", "No model call was made for this request."
    gemini_ok = [c for c in calls if c.get("used") == "gemini"]
    fell_back = [c for c in calls if c.get("backend") == "gemini" and c.get("used") == "mock"]
    n = len(calls)
    if not fell_back and not gemini_ok:
        return "info", f"Mock model used ({n} call{'s' if n != 1 else ''}). No real LLM was called."
    lines = []
    for c in calls:
        label = f"Round {c['round']}: "
        if c.get("used") == "gemini":
            valid = "valid JSON" if c.get("valid_json") else "reply was not valid JSON (repair loop retried)"
            lines.append(label + f"Gemini answered, non-empty, {valid}.")
        elif c.get("backend") == "gemini":
            why = "returned an empty response" if c.get("status") == "empty" else f"failed ({c.get('detail') or 'unknown error'})"
            lines.append(label + f"Gemini {why}, so the mock model was used instead.")
        else:
            lines.append(label + "mock model used.")
    return ("warn" if fell_back else "ok"), "\n".join(lines)


class CareAdvisor:
    def __init__(self, llm=None, logger: Optional[AgentLogger] = None, few_shot: bool = False) -> None:
        self.llm = llm or MockClient()
        self.logger = logger or AgentLogger()
        self.few_shot = few_shot
        self.knowledge = load_knowledge()
        self.system_prompt = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
        if few_shot:
            self.system_prompt += "\n\n" + (PROMPT_DIR / "fewshot.txt").read_text(encoding="utf-8")

    # ---- planning -------------------------------------------------------
    def _plan(self, request: str, owner, pet: Optional[Pet]) -> dict:
        if pet is None and owner is not None:
            pet = next((p for p in owner.pets if p.name.lower() in request.lower()), None)
        species = pet.species.lower() if pet else None
        age = pet.age if pet else None
        needs = (pet.special_needs or "") if pet else ""
        if species is None:
            for w in SPECIES_WORDS:
                if re.search(rf"\b{w}s?\b", request.lower()):
                    species = {"puppy": "dog", "kitten": "cat"}.get(w, w)
                    break
        if age is None:
            m = re.search(r"(\d+)[- ]?(?:year|yr)", request.lower())
            age = int(m.group(1)) if m else None
        stage = life_stage(species, age, request)
        query = f"{request} {needs} {species or ''}".strip()
        return {"pet": pet, "species": species, "age": age, "stage": stage, "query": query}

    # ---- main entry -----------------------------------------------------
    def propose(self, request: str, owner=None, pet: Optional[Pet] = None,
                available_minutes: Optional[int] = None) -> Proposal:
        proposal = Proposal(status="ok", run_id=uuid.uuid4().hex[:8])

        def step(name: str, **data) -> None:
            proposal.trace.append({"step": name, **data})
            self.logger.log(proposal.run_id, name, **data)

        try:
            return self._run(request, owner, pet, available_minutes, proposal, step)
        except Exception as err:  # last-resort guard: never crash the UI
            proposal.status, proposal.reason, proposal.tasks = "refused", f"internal error: {err!r}", []
            step("error", error=repr(err))
            return proposal

    def _run(self, request, owner, pet, available_minutes, proposal: Proposal, step) -> Proposal:
        clean = sanitize_request(request)
        step("sanitize", original=request, cleaned=clean.text, flags=clean.flags)
        if not clean.text:
            proposal.status = "refused"
            proposal.reason = "The request was empty or contained only disallowed instructions."
            return proposal

        plan = self._plan(clean.text, owner, pet)
        step("plan", species=plan["species"], age=plan["age"], life_stage=plan["stage"], query=plan["query"])

        self.knowledge = load_knowledge()  # re-read each run so owner-added guidelines apply immediately
        proposal.species = plan["species"]
        names = {p.name.lower() for p in (owner.pets if owner else [])} | ({plan["pet"].name.lower()} if plan["pet"] else set())
        activities, parse_notes = parse_requested_activities(clean.text, frozenset(names))
        proposal.notes.extend(parse_notes)
        step("requested_activities", found=[(a.title, a.start, a.duration_minutes) for a in activities], notes=parse_notes)

        # Relevance gate: the request itself (not just the pet's species) must be about pet care.
        on_topic = (
            bool(INTENT_WORDS & set(re.findall(r"[a-z]+", clean.text.lower())))
            or bool(Retriever(self.knowledge).search(clean.text, k=1))
            or bool(activities and plan["species"])
        )
        step("relevance", on_topic=on_topic)
        if not on_topic:
            proposal.status = "no_evidence"
            proposal.reason = "This request does not look like a pet care question, so I won't guess."
            return proposal

        # A request that only names an activity ("swim from 3 to 4pm") is not a request for a whole routine.
        wants_routine = not activities or bool(INTENT_WORDS & set(re.findall(r"[a-z]+", clean.text.lower())))
        step("scope", generate_routine=wants_routine)
        retriever = Retriever(self.knowledge + (owner_chunks(owner) if owner else []))
        chunks = retriever.search(plan["query"], species=plan["species"], stage=plan["stage"]) if wants_routine else []
        knowledge_chunks = [c for c in chunks if c.kind == "knowledge"]
        proposal.retrieved_ids = [c.chunk_id for c in chunks]
        step("retrieve", chunks=[(c.chunk_id, c.score) for c in chunks])
        if not knowledge_chunks and not activities:
            proposal.status = "no_evidence"
            proposal.reason = "No guideline in the knowledge base matches this request, so I won't guess."
            return proposal

        retrieved = {c.chunk_id: c.text for c in chunks}
        existing = {t.title.lower() for t in plan["pet"].tasks} if plan["pet"] else set()
        feedback = ""
        tasks: list[dict] = []
        issues: list[Issue] = []
        needed_repair = False

        for round_no in range(1, MAX_ROUNDS + 1):
            if not knowledge_chunks:
                break  # nothing to ground an LLM draft on; only the owner's own requests are handled
            raw = self.llm.complete(self.system_prompt, self._user_prompt(clean.text, plan, chunks, feedback))
            parsed = parse_json(raw)
            event = dict(getattr(self.llm, "last_event", None) or {"backend": "mock", "status": "ok", "used": "mock", "detail": ""})
            event.update(round=round_no, non_empty=bool(raw.strip()), valid_json=parsed is not None)
            proposal.llm_calls.append(event)
            step("propose", round=round_no, raw=raw[:600], parsed_ok=parsed is not None, llm=event)
            if parsed is None:
                needed_repair = True
                feedback = "FEEDBACK:\n- Your last reply was not valid JSON in the required shape. Return ONLY the JSON."
                tasks, issues = [], []
                continue
            tasks = [t for t in parsed["tasks"] if isinstance(t, dict)]
            issues = validate_tasks(tasks, retrieved, plan["species"], plan["stage"], available_minutes, existing)
            blocking = [i for i in issues if i.severity == "block"]
            step("verify", round=round_no, issues=[(i.code, i.message) for i in issues])
            if not blocking:
                break
            needed_repair = True
            feedback = "FEEDBACK:\n" + "\n".join(f"- {i.message}" for i in blocking)

        blocked_idx = {i.task_index for i in issues if i.severity == "block" and i.task_index is not None}
        proposal.dropped = [str(tasks[i].get("title", "?")) for i in sorted(blocked_idx) if i < len(tasks)]
        proposal.issues = issues
        if proposal.dropped:
            step("drop", dropped=proposal.dropped)

        proposed: list[ProposedTask] = []
        for idx, t in enumerate(tasks):
            if idx in blocked_idx:
                continue
            flags = [i.message for i in issues if i.severity == "warn" and i.task_index == idx]
            if is_medication(t) and not any(VET_NOTE in f for f in flags):
                flags.append(VET_NOTE)
            title = str(t["title"]).strip()
            proposed.append(ProposedTask(
                task=Task(
                    title=title,
                    duration_minutes=t["duration_minutes"],
                    priority=Priority.from_str(str(t["priority"])),
                    category=_category(title),
                    preferred_time=t.get("preferred_time"),
                    frequency=str(t["frequency"]).lower(),
                ),
                sources=[s for s in t.get("sources", []) if s in retrieved],
                flags=flags,
            ))

        conf = None
        if knowledge_chunks:
            conf = score_proposal(
                top_retrieval=max(c.score for c in knowledge_chunks),
                kept=len(proposed),
                proposed=len(tasks),
                warnings=sum(1 for i in issues if i.severity == "warn"),
                needed_repair=needed_repair,
            )
            step("score", score=conf.score, level=conf.level, reasons=conf.reasons)
            if not proposed or conf.score < MIN_CONFIDENCE_TO_PROPOSE:
                if proposed:
                    proposal.notes.append("The guideline-based tasks were too low confidence to show; only your own requests are listed.")
                proposed = []
        proposal.confidence = conf

        extras = self._requested_tasks(activities, plan, existing, proposal, step)
        replaced = {pt.task.title.lower() for pt in extras}
        proposal.tasks = [pt for pt in proposed if pt.task.title.lower() not in replaced] + extras
        proposal.tasks.sort(key=lambda pt: not pt.in_knowledge)  # not-in-knowledge items go last (stable sort)
        if not proposal.tasks:
            proposal.status = "refused"
            proposal.reason = "I could not produce tasks that passed verification with enough confidence."
        return proposal

    def _requested_tasks(self, activities, plan, existing: set, proposal: Proposal, step) -> list[ProposedTask]:
        """Turn owner-requested activities (with times) into tasks.

        If an activity is in the knowledge base it is cited and approved; otherwise it is kept as the
        owner's own request, flagged as not in the knowledge base. Safety rules apply either way.
        """
        out: list[ProposedTask] = []
        for act in activities:
            guide = match_activity(act.title, plan["species"], self.knowledge)
            raw = {
                "title": guide.title if guide else act.title,
                "duration_minutes": act.duration_minutes,
                "priority": guide.priority if guide else "medium",
                "frequency": guide.frequency if guide else "daily",
                "preferred_time": act.preferred_time,
                "sources": [guide.chunk_id] if guide else [],
                "origin": "guideline" if guide else "user",
            }
            evidence = {c.chunk_id: c.text for c in self.knowledge if guide and c.chunk_id == guide.chunk_id}
            issues = validate_tasks([raw], evidence, plan["species"], plan["stage"], None, existing)
            blocking = [i for i in issues if i.severity == "block"]
            step("requested_task", title=raw["title"], in_knowledge=bool(guide), issues=[(i.code, i.message) for i in issues])
            if blocking:
                proposal.dropped.append(raw["title"])
                proposal.notes.append(f"Your request for '{raw['title']}' was not added: " + "; ".join(i.message for i in blocking))
                continue
            end_h, end_m = divmod(int(act.start[:2]) * 60 + int(act.start[3:]) + act.duration_minutes, 60)
            flags = [i.message for i in issues if i.severity == "warn"]
            if is_medication(raw) and not any(VET_NOTE in f for f in flags):
                flags.append(VET_NOTE)
            flags.insert(0, f"Time set from your request: {act.start}-{end_h % 24:02d}:{end_m:02d}.")
            if not guide:
                flags.append("Not in the approved knowledge base, so it has no guideline source.")
            out.append(ProposedTask(
                task=Task(
                    title=raw["title"], duration_minutes=act.duration_minutes,
                    priority=Priority.from_str(raw["priority"]), category=_category(raw["title"]),
                    preferred_time=act.preferred_time, frequency=raw["frequency"], start_time=act.start,
                    fixed_time=True,
                ),
                sources=raw["sources"], flags=flags, origin=raw["origin"], in_knowledge=bool(guide),
            ))
        return out

    @staticmethod
    def _user_prompt(request: str, plan: dict, chunks: list[Chunk], feedback: str) -> str:
        age = plan["age"] if plan["age"] is not None else "unknown"
        pet_line = f"species={plan['species'] or 'unknown'}, age={age}, life_stage={plan['stage']}"
        context = "\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)
        prompt = f"REQUEST: {request}\nPET: {pet_line}\nCONTEXT:\n{context}\n"
        return prompt + (f"\n{feedback}\n" if feedback else "")
