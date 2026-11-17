"""Guardrails: input sanitising and output validation for proposed tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ai.retriever import tokenize

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier)[^.]*",
    r"disregard (all |any |the )?(previous|prior|above|earlier)[^.]*",
    r"(reveal|show|print) (your |the )?(system )?prompt[^.]*",
    r"you are now[^.]*",
    r"forget (your|the) (rules|instructions)[^.]*",
]
DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s*(units?|iu|mg|ml|cc|mcg|tablets?|pills?)\b", re.I)
MED_WORDS = ("insulin", "medic", "pill", "tablet", "dose", "drug", "prescri")
TOXIC = ("chocolate", "grape", "raisin", "onion", "garlic", "xylitol", "macadamia", "alcohol")
VALID_PRIORITIES = {"low", "medium", "high"}
VALID_FREQ = {"once", "daily", "weekly"}
VALID_TOD = {"morning", "afternoon", "evening", None}
MAX_TASK_MINUTES = 120
GROUNDING_MIN_OVERLAP = 0.6
VET_NOTE = "Confirm with your vet before relying on this."


@dataclass
class Issue:
    severity: str  # "block" removes the task, "warn" only flags it
    code: str
    message: str
    task_index: Optional[int] = None


@dataclass
class SanitizedRequest:
    text: str
    flags: list[str] = field(default_factory=list)


def sanitize_request(request: str) -> SanitizedRequest:
    """Strip prompt-injection phrases and normalise whitespace."""
    text = (request or "").strip()
    flags: list[str] = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.I):
            flags.append("prompt_injection")
            text = re.sub(pattern, " ", text, flags=re.I)
    return SanitizedRequest(re.sub(r"\s+", " ", text).strip(), sorted(set(flags)))


def is_medication(task: dict) -> bool:
    title = str(task.get("title", "")).lower()
    return any(w in title for w in MED_WORDS)


def validate_tasks(
    tasks: list[dict],
    retrieved: dict[str, str],
    species: Optional[str],
    life_stage: str,
    available_minutes: Optional[int] = None,
    existing_titles: Optional[set[str]] = None,
) -> list[Issue]:
    """Check each proposed task dict against schema, evidence and safety rules.

    `retrieved` maps chunk id -> chunk text for everything the retriever returned.
    """
    issues: list[Issue] = []
    existing_titles = existing_titles or set()
    for i, t in enumerate(tasks):
        title = str(t.get("title", "")).strip()
        if not title:
            issues.append(Issue("block", "schema", "task has no title", i))
            continue
        low = title.lower()
        mins = t.get("duration_minutes")
        if not isinstance(mins, int) or isinstance(mins, bool) or not (0 < mins <= MAX_TASK_MINUTES):
            issues.append(Issue("block", "schema", f"{title!r}: duration must be 1-{MAX_TASK_MINUTES} minutes", i))
        if str(t.get("priority", "")).lower() not in VALID_PRIORITIES:
            issues.append(Issue("block", "schema", f"{title!r}: invalid priority", i))
        if str(t.get("frequency", "")).lower() not in VALID_FREQ:
            issues.append(Issue("block", "schema", f"{title!r}: invalid frequency", i))
        if t.get("preferred_time") not in VALID_TOD:
            issues.append(Issue("block", "schema", f"{title!r}: invalid preferred_time", i))

        sources = [s for s in (t.get("sources") or []) if isinstance(s, str)]
        valid_sources = [s for s in sources if s in retrieved]
        if t.get("origin") == "user" and not valid_sources:
            pass  # the owner's own request: needs no guideline, but every safety rule below still applies
        elif not valid_sources:
            issues.append(Issue("block", "uncited", f"{title!r} cites no retrieved source", i))
        else:
            title_tokens = set(tokenize(title))
            evidence = set(tokenize(" ".join(retrieved[s] for s in valid_sources)))
            if title_tokens and len(title_tokens & evidence) / len(title_tokens) < GROUNDING_MIN_OVERLAP:
                issues.append(Issue("block", "ungrounded", f"{title!r} is not supported by its cited source", i))

        if DOSE_RE.search(title):
            issues.append(Issue("block", "dose", f"{title!r} contains a dose; doses must come from a vet", i))
        if any(w in low for w in TOXIC) and not any(w in low for w in ("avoid", "never", "check")):
            issues.append(Issue("block", "toxic", f"{title!r} involves a food that can be toxic", i))
        if is_medication(t):
            issues.append(Issue("warn", "medication", f"{title!r} is a medication task. {VET_NOTE}", i))

        for s in valid_sources:
            heading = retrieved[s].splitlines()[0].lower()
            present = {w for w in ("puppies", "adult", "senior") if w in heading}
            wanted = {"puppy": "puppies"}.get(life_stage, life_stage)
            if present and wanted not in present:
                issues.append(Issue("block", "life_stage", f"{title!r} is for a different life stage than {life_stage}", i))
                break
        if low in existing_titles:
            issues.append(Issue("warn", "duplicate", f"{title!r} already exists for this pet", i))

    if available_minutes is not None:
        total = sum(t["duration_minutes"] for t in tasks if isinstance(t.get("duration_minutes"), int))
        if total > available_minutes:
            issues.append(Issue("warn", "budget", f"proposed tasks total {total} min but only {available_minutes} min are available"))
    return issues
