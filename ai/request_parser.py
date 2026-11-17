"""Extract owner-requested activities with a time window from free text.

Examples understood:
    "... with an afternoon nap from 12 pm to 2pm"   -> Afternoon nap, 12:00, 120 min
    "add bath time between 6:30 and 7pm"             -> Bath time, 18:30..19:00 (end meridiem is inherited)
    "playtime at 4pm for 30 minutes"                 -> Playtime, 16:00, 30 min

Parsing is deterministic (regex), not LLM-based, so a requested time can never be altered by a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_T = r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m?\.?"  # time with a meridiem
_TOPT = r"(\d{1,2})(?::(\d{2}))?\s*(?:([ap])\.?m?\.?)?"  # time, meridiem optional
WINDOW_RE = re.compile(rf"\b(?:from|between)\s+{_TOPT}\s*(?:to|until|till|and|-)\s*{_TOPT}(?![\d:])", re.I)
AT_FOR_RE = re.compile(
    rf"\bat\s+{_TOPT}\s*(?:for|lasting)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\b", re.I
)
CLAUSE_SPLIT = re.compile(r"\b(?:with|plus|add|adding|including|include|and|also|for|then|where|that has)\b|[,.;:!?]", re.I)
ARTICLES = {"a", "an", "the", "my", "some", "his", "her", "their", "our", "in"}
GENERIC = {
    "plan", "routine", "schedule", "daily", "day", "time", "today", "tasks", "task", "care",
    "pet", "pets", "dog", "dogs", "cat", "cats", "puppy", "kitten", "rabbit", "hamster", "guinea", "pig",
}
MAX_TITLE_WORDS = 4


@dataclass
class RequestedActivity:
    title: str
    start: str  # "HH:MM"
    duration_minutes: int
    preferred_time: str  # morning | afternoon | evening


def _to_minutes(hour: int, minute: str | None, meridiem: str | None) -> int | None:
    h, m = hour, int(minute or 0)
    if m > 59:
        return None
    if meridiem:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if meridiem.lower() == "p" else 0)
    elif h > 24 or (h == 24 and m):
        return None
    return h * 60 + m


def _window(g: tuple) -> tuple[int, int] | None:
    """Resolve (h1, m1, ap1, h2, m2, ap2) to (start, end) minutes, inferring missing meridiems."""
    h1, m1, a1, h2, m2, a2 = g
    h1, h2 = int(h1), int(h2)
    end_mer = a2
    start_mer = a1 or a2  # "11 to 2pm" first assumes the same meridiem
    start = _to_minutes(h1, m1, start_mer)
    end = _to_minutes(h2, m2, end_mer)
    if start is None or end is None:
        return None
    if end <= start and not a1 and a2:  # "11 to 2pm" -> 11am to 2pm
        start = _to_minutes(h1, m1, "a")
    if end <= start and not a2 and end // 60 < 12:  # "12 to 2": 24h reading failed, assume pm
        end += 12 * 60
    if start is None or not (0 <= start < end <= 24 * 60):
        return None
    return start, end


def _title_before(text: str, pos: int) -> str:
    clause = CLAUSE_SPLIT.split(text[:pos])[-1]
    words = re.findall(r"[A-Za-z0-9']+", clause)  # digits kept so "5 units of insulin" stays visible to the dose check
    while words and words[0].lower() in ARTICLES:
        words.pop(0)
    words = words[-MAX_TITLE_WORDS:]
    while words and words[0].lower() in ARTICLES:
        words.pop(0)
    return " ".join(words).strip().capitalize()


def _tod(start_minutes: int) -> str:
    return "morning" if start_minutes < 12 * 60 else "afternoon" if start_minutes < 17 * 60 else "evening"


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_requested_activities(text: str, ignore: frozenset = frozenset()) -> tuple[list[RequestedActivity], list[str]]:
    """Return (activities, notes). Notes explain anything that looked like a request but could not be used.

    `ignore` holds extra lower-case words that are not activities (e.g. the pets' names)."""
    generic = GENERIC | set(ignore)
    found: list[RequestedActivity] = []
    notes: list[str] = []

    for m in WINDOW_RE.finditer(text):
        title = _title_before(text, m.start())
        window = _window(m.groups())
        label = title or "(no activity name)"
        if not title or {w.lower() for w in title.split()} <= generic:
            notes.append(f"I saw a time window ({m.group(0).strip()}) but could not tell which activity it is for.")
        elif window is None:
            notes.append(f"I could not read the time window for '{title}' ({m.group(0).strip()}); please use e.g. '12 pm to 2 pm'.")
        else:
            found.append(RequestedActivity(title, _hhmm(window[0]), window[1] - window[0], _tod(window[0])))

    for m in AT_FOR_RE.finditer(text):
        title = _title_before(text, m.start())
        g = m.groups()
        start = _to_minutes(int(g[0]), g[1], g[2])
        amount = float(g[3])
        minutes = round(amount * 60) if g[4].lower().startswith("h") else round(amount)
        if not title or {w.lower() for w in title.split()} <= generic or start is None or minutes <= 0 or start + minutes > 24 * 60:
            notes.append(f"I could not use '{m.group(0).strip()}' as a scheduled activity.")
        else:
            found.append(RequestedActivity(title, _hhmm(start), minutes, _tod(start)))
    return found, notes
