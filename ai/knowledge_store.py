"""Read guideline tasks out of the knowledge base, match owner requests to them, and let the
owner add new (unreviewed) guidelines so future requests can retrieve and cite them."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ai import retriever
from ai.guardrails import is_medication, validate_tasks
from ai.retriever import Chunk, known_species, species_matches, tokenize

TASK_LINE_RE = re.compile(
    r"^- Task:\s*(.+?)\s*\|\s*(\d+)\s*min\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*$", re.M
)
MATCH_THRESHOLD = 0.67  # share of the requested activity's words found in a guideline task title
FUZZY_RATIO = 0.85  # tolerate typos such as "afternooon"
USER_FILE_PREFIX = "user_added_"


@dataclass
class GuidelineTask:
    chunk_id: str
    title: str
    minutes: int
    frequency: str
    priority: str
    preferred_time: str


def guideline_tasks(chunks: list[Chunk], species: Optional[str] = None) -> list[GuidelineTask]:
    """Every `- Task:` line in the knowledge chunks (optionally only those valid for `species`)."""
    out = []
    known = known_species(chunks)
    for c in chunks:
        if c.kind != "knowledge" or not species_matches(c.species, species, known):
            continue
        for m in TASK_LINE_RE.finditer(c.text):
            title, mins, freq, prio, tod = m.groups()
            out.append(GuidelineTask(c.chunk_id, title, int(mins), freq.lower(), prio.lower(), tod.lower()))
    return out


def _same_word(a: str, b: str) -> bool:
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY_RATIO


def match_activity(title: str, species: Optional[str], chunks: list[Chunk]) -> Optional[GuidelineTask]:
    """Find the guideline task an owner-requested activity refers to, or None if it is not in the knowledge base."""
    wanted = tokenize(title)
    if not wanted:
        return None
    best: tuple[float, Optional[GuidelineTask]] = (0.0, None)
    for gt in guideline_tasks(chunks, species):
        have = tokenize(gt.title)
        share = sum(any(_same_word(w, h) for h in have) for w in wanted) / len(wanted)
        if share > best[0]:
            best = (share, gt)
    return best[1] if best[0] >= MATCH_THRESHOLD else None


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ,'()/-]", "", title)).strip()


def add_user_guideline(
    title: str,
    minutes: int,
    species: str,
    frequency: str = "daily",
    priority: str = "medium",
    preferred_time: Optional[str] = None,
    directory: Optional[Path] = None,
) -> str:
    """Append an owner-added task to knowledge/user_added_<species>.md and return its chunk id.

    The entry is only retrieved when a request mentions all of its words, so it does not leak into
    unrelated routines. Raises ValueError if the task fails the safety rules or is a medication task
    (those must come from a vet-reviewed source).
    """
    title = _clean_title(title)
    species = re.sub(r"[^a-z0-9 ]", "", (species or "").lower()).strip()
    if not species:
        raise ValueError("The pet's species is needed to file this guideline.")
    if not (2 <= len(title) <= 60) or not tokenize(title):
        raise ValueError("The activity name must be 2-60 characters.")
    task = {
        "title": title, "duration_minutes": minutes, "priority": priority,
        "frequency": frequency, "preferred_time": preferred_time, "origin": "user",
    }
    if is_medication(task):
        raise ValueError("Medication tasks cannot be added to the knowledge base; they must come from your vet.")
    blocking = [i.message for i in validate_tasks([task], {}, species, "adult") if i.severity == "block"]
    if blocking:
        raise ValueError("; ".join(blocking))

    directory = Path(directory or retriever.KNOWLEDGE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    stem = USER_FILE_PREFIX + re.sub(r"[^a-z0-9]+", "_", species).strip("_")
    path = directory / f"{stem}.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else (
        f"# Owner-Added Guidelines ({species})\nSpecies: {species}\n"
        "Source: Added by the owner in the PawPal+ app (not reviewed by a vet)\n"
    )
    chunk_id = f"{stem}#{retriever._slug(title)}"
    if any(g.chunk_id == chunk_id for g in guideline_tasks(retriever.load_knowledge(directory))):
        return chunk_id  # already there
    words = ", ".join(dict.fromkeys(tokenize(title)))
    entry = (
        f"\n## {title}\nCondition: {words}\n"
        f"Added by the owner. Retrieved only when a request mentions: {words}.\n"
        f"- Task: {title} | {minutes} min | {frequency} | {priority} | {preferred_time or 'morning'}\n"
    )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(existing.rstrip("\n") + "\n" + entry, encoding="utf-8")
    tmp.replace(path)
    return chunk_id
