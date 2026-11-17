"""Reliability harness: runs the CareAdvisor on scripted cases and writes eval/results.md.

Usage: python -m eval.run_eval            (offline, MockClient)
       PAWPAL_LLM=gemini python -m eval.run_eval   (also runs the real model)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ai.agent import CareAdvisor
from ai.llm_client import GeminiClient, MockClient
from ai.logger import AgentLogger
from pawpal_system import Owner, Pet

HERE = Path(__file__).resolve().parent


def build_owner(spec):
    if not spec:
        return None
    return Owner(spec["name"], spec["available_minutes"],
                 pets=[Pet(p["name"], p["species"], p.get("age"), p.get("special_needs")) for p in spec["pets"]])


def check(case, proposal) -> list[str]:
    """Return a list of failure messages (empty list means the case passed)."""
    exp, fails = case["expect"], []
    titles = [pt.task.title for pt in proposal.tasks]
    joined = " | ".join(titles).lower()
    if proposal.status != exp["status"]:
        fails.append(f"status {proposal.status!r} != {exp['status']!r}")
    for t in exp.get("include", []):
        if t.lower() not in joined:
            fails.append(f"missing task {t!r}")
    for t in exp.get("exclude", []):
        if t.lower() in joined:
            fails.append(f"unexpected task {t!r}")
    if exp.get("flag_contains") and not any(exp["flag_contains"] in f.lower() for pt in proposal.tasks for f in pt.flags):
        fails.append("expected a safety flag")
    rounds = sum(1 for s in proposal.trace if s["step"] == "propose")
    if rounds < exp.get("min_rounds", 0):
        fails.append(f"expected >= {exp['min_rounds']} rounds, got {rounds}")
    if exp.get("dropped_contains") and not any(exp["dropped_contains"] in d.lower() for d in proposal.dropped):
        fails.append("expected a dropped task")
    if exp.get("sanitized") and not any(s.get("flags") for s in proposal.trace if s["step"] == "sanitize"):
        fails.append("injection not flagged")
    if exp.get("owner_source") and not any(i.startswith("owner:") for i in proposal.retrieved_ids):
        fails.append("owner profile was not retrieved")
    for title, start in exp.get("pinned", {}).items():
        pt = next((x for x in proposal.tasks if x.task.title == title), None)
        if pt is None or pt.task.start_time != start or not pt.task.fixed_time:
            fails.append(f"{title!r} not pinned at {start}")
    for title in exp.get("not_in_knowledge", []):
        pts = [x for x in proposal.tasks if x.task.title == title]
        if not pts or pts[0].in_knowledge or proposal.tasks[-1] is not pts[0]:
            fails.append(f"{title!r} should be last and flagged as not in the knowledge base")
    for title in exp.get("cited", []):
        if not any(x.task.title == title and x.sources for x in proposal.tasks):
            fails.append(f"{title!r} should cite a guideline")
    if exp.get("note_contains") and not any(exp["note_contains"] in n for n in proposal.notes):
        fails.append("expected a note for the owner")
    if proposal.status == "ok" and any(not pt.sources and pt.in_knowledge for pt in proposal.tasks):
        fails.append("task without citation")
    return fails


def run(llm_factory=lambda mode: MockClient(mode), label="MockClient") -> list[dict]:
    rows = []
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    for case in cases:
        advisor = CareAdvisor(llm_factory(case.get("mode", "good")), AgentLogger(enabled=False))
        proposal = advisor.propose(case["request"], owner=build_owner(case.get("owner")))
        fails = check(case, proposal)
        rows.append({
            "id": case["id"], "name": case["name"], "status": proposal.status,
            "tasks": len(proposal.tasks), "dropped": len(proposal.dropped),
            "confidence": proposal.confidence.score if proposal.confidence else None,
            "result": "Pass" if not fails else "Fail - " + "; ".join(fails),
        })
    return rows


def to_markdown(rows: list[dict], label: str) -> str:
    passed = sum(r["result"] == "Pass" for r in rows)
    confs = [r["confidence"] for r in rows if r["confidence"] is not None]
    avg = sum(confs) / len(confs) if confs else 0
    lines = [
        f"### Eval results ({label})", "",
        f"**{passed}/{len(rows)} cases passed**; average confidence on proposals: {avg:.0f}/100; "
        f"tasks blocked by guardrails: {sum(r['dropped'] for r in rows)}", "",
        "| ID | Case | Status | Tasks | Dropped | Confidence | Result |",
        "|----|------|--------|-------|---------|------------|--------|",
    ]
    for r in rows:
        conf = "-" if r["confidence"] is None else r["confidence"]
        lines.append(f"| {r['id']} | {r['name']} | {r['status']} | {r['tasks']} | {r['dropped']} | {conf} | {r['result']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = run()
    report = to_markdown(rows, "offline MockClient")
    if "--gemini" in sys.argv:
        try:
            client = GeminiClient()
            report += "\n" + to_markdown(run(lambda mode: client), "Gemini")
        except Exception as err:
            report += f"\nGemini run skipped: {err}\n"
    print(report)
    (HERE / "results.md").write_text(report, encoding="utf-8")
    return 0 if all(r["result"] == "Pass" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
