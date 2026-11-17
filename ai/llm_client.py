"""LLM clients: a deterministic offline MockClient and a Gemini wrapper.

Both expose `complete(system_prompt, user_prompt) -> str`. Any API failure makes
GeminiClient return "" so the agent can fall back instead of crashing.
"""

from __future__ import annotations

import json
import os
import re


class MockClient:
    """Offline stand-in that really reads the retrieved CONTEXT and builds tasks from it.

    modes (used by the reliability tests/eval to simulate model misbehaviour):
      good      - follows the rules
      naive     - ignores life stage and invents an uncited dose task, but fixes
                  both once FEEDBACK is supplied
      stubborn  - like naive but never fixes anything (forces the final guardrail drop)
      badjson   - first reply is not JSON, later replies are good
    """

    def __init__(self, mode: str = "good") -> None:
        self.mode = mode
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.mode == "badjson" and self.calls == 1:
            return "Sure! Here is a lovely routine for your pet."
        has_feedback = "FEEDBACK:" in user_prompt
        misbehave = (self.mode == "stubborn") or (self.mode == "naive" and not has_feedback)
        stage = re.search(r"life_stage=(\w+)", user_prompt)
        stage = stage.group(1) if stage else "adult"

        tasks = []
        for chunk_id, body in self._chunks(user_prompt):
            if chunk_id.startswith("owner:"):
                continue
            heading = body.splitlines()[0].lower() if body else ""
            if not misbehave and self._stage_mismatch(heading, stage):
                continue
            for m in re.finditer(r"^- Task:\s*(.+?)\s*\|\s*(\d+)\s*min\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*$", body, re.M):
                title, mins, freq, prio, tod = m.groups()
                tasks.append({
                    "title": title, "duration_minutes": int(mins), "frequency": freq,
                    "priority": prio, "preferred_time": tod, "sources": [chunk_id],
                })
        if misbehave:
            tasks.append({
                "title": "Give 5 units of insulin", "duration_minutes": 5, "frequency": "daily",
                "priority": "high", "preferred_time": "morning", "sources": [],
            })
        return json.dumps({"tasks": tasks})

    @staticmethod
    def _chunks(prompt: str):
        context = prompt.split("CONTEXT:", 1)[-1].split("FEEDBACK:", 1)[0]
        parts = re.split(r"^\[([^\]]+)\]\s*$", context, flags=re.M)
        return list(zip(parts[1::2], (p.strip() for p in parts[2::2])))

    @staticmethod
    def _stage_mismatch(heading: str, stage: str) -> bool:
        stages = {"puppy": "puppies", "adult": "adult", "senior": "senior"}
        words = {"puppies", "adult", "senior"}
        present = {w for w in words if w in heading}
        return bool(present) and stages.get(stage, stage) not in present


class GeminiClient:
    """Minimal Gemini wrapper. Needs GEMINI_API_KEY and the google-genai package.

    complete() never raises: it returns "" on failure and records the outcome in
    `last_event` so callers can tell "empty reply" from "API error".
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or os.getenv("GEMINI_MODEL", "").strip() or self.DEFAULT_MODEL
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY. Set it in .env or use the mock.")
        from google import genai  # imported lazily so offline mode needs no dependency

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.last_event: dict = {}

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.model_name, contents=f"{system_prompt}\n\n{user_prompt}"
            )
            text = (resp.text or "").strip()
        except Exception as err:
            self.last_event = {"backend": "gemini", "status": "error", "detail": f"{type(err).__name__}: {str(err)[:150]}"}
            return ""
        self.last_event = {"backend": "gemini", "status": "ok" if text else "empty", "detail": ""}
        return text


class FallbackClient:
    """Try `primary` (a real LLM); on an error or empty reply, answer with `fallback` (the mock).

    `last_event` records which backend produced the text used:
      {"backend": "gemini"|"mock", "status": "ok"|"empty"|"error", "used": "gemini"|"mock", "detail": str}
    where backend/status describe the primary call and `used` is who supplied the final text.
    """

    def __init__(self, primary, fallback=None) -> None:
        self.primary = primary
        self.fallback = fallback or MockClient()
        self.last_event: dict = {}

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            text = self.primary.complete(system_prompt, user_prompt)
        except Exception as err:  # a primary that raises is treated like an API error
            text = ""
            self.primary.last_event = {"backend": "gemini", "status": "error", "detail": f"{type(err).__name__}: {str(err)[:150]}"}
        event = dict(getattr(self.primary, "last_event", None) or {"backend": "gemini", "status": "ok" if text else "empty", "detail": ""})
        if text and text.strip():
            self.last_event = {**event, "used": event["backend"]}
            return text
        if event.get("status") == "ok":  # blank text without a recorded reason
            event["status"] = "empty"
        self.last_event = {**event, "used": "mock"}
        return self.fallback.complete(system_prompt, user_prompt)


def gemini_available() -> tuple[bool, str]:
    """Whether Gemini can be used right now, with a human-readable reason if not."""
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return False, "GEMINI_API_KEY is not set (add it to .env)"
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False, "the google-genai package is not installed (pip install -r requirements.txt)"
    return True, ""


def get_client(prefer: str | None = None):
    """Return a client. prefer: "mock" or "gemini"; default comes from PAWPAL_LLM (mock if unset).

    "gemini" returns a FallbackClient (Gemini first, mock on error/empty reply). If Gemini
    cannot start at all, the plain MockClient is returned.
    """
    choice = (prefer or os.getenv("PAWPAL_LLM", "mock")).lower()
    if choice == "gemini":
        try:
            return FallbackClient(GeminiClient())
        except Exception:
            return MockClient()
    return MockClient()
