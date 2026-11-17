"""Append-only JSONL logger for every agent step."""

from __future__ import annotations

import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "agent_log.jsonl"


class AgentLogger:
    def __init__(self, path: Path = LOG_PATH, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    def log(self, run_id: str, step: str, **data) -> None:
        if not self.enabled:
            return
        record = {"ts": round(time.time(), 3), "run_id": run_id, "step": step, **data}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass  # logging must never break the app
