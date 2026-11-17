"""Confidence score (0-100) for a proposal, with the reasons for each deduction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Confidence:
    score: int
    level: str  # "high" | "medium" | "low"
    reasons: list[str]


def score_proposal(
    top_retrieval: float,
    kept: int,
    proposed: int,
    warnings: int,
    needed_repair: bool,
) -> Confidence:
    reasons: list[str] = []
    retrieval = min(1.0, top_retrieval / 0.6)
    coverage = (kept / proposed) if proposed else 0.0
    agreement = 0.6 if needed_repair else 1.0
    score = 100 * (0.35 * retrieval + 0.35 * coverage + 0.30 * agreement)
    reasons.append(f"retrieval strength {retrieval:.2f}")
    reasons.append(f"{kept}/{proposed} proposed tasks passed verification")
    if needed_repair:
        reasons.append("first draft needed repair (-12)")
    score -= 5 * warnings
    if warnings:
        reasons.append(f"{warnings} warning(s) (-{5 * warnings})")
    score = max(0, min(100, round(score)))
    level = "high" if score >= 70 else "medium" if score >= 40 else "low"
    return Confidence(score, level, reasons)
