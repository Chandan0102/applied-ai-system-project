"""Dependency-free retriever over the markdown knowledge base (plus owner data).

Each `## section` of a knowledge file is one chunk. Scoring is TF-IDF over
lower-cased word tokens. Chunks whose `Species:` header does not include the
requested species are excluded, and anything scoring below MIN_SCORE is dropped
so the agent can say "no evidence" instead of guessing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
MIN_SCORE = 0.15

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "my", "of", "on", "or", "the", "to", "with", "me", "i", "build",
    "make", "create", "plan", "routine", "please", "need", "needs", "year", "old",
    "how", "do", "does", "did", "should", "can", "what", "when", "which", "give",
    "get", "your", "you", "we", "our", "about", "this", "that", "some", "any",
    "has", "have", "had", "was", "were", "will", "would", "help", "good", "best",
}
# Small synonym map so "elderly" finds "senior" and so on.
SYNONYMS = {
    "elderly": "senior", "aged": "senior", "older": "senior",
    "diabetes": "diabetic", "insulin": "diabetic",
    "medicine": "medication", "meds": "medication", "pill": "medication",
    "kitten": "cat", "puppy": "puppies", "dogs": "dog", "cats": "cat",
    "sleep": "nap", "napping": "nap",
}


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens with stopwords removed and synonyms folded."""
    words = re.findall(r"[a-z]+", text.lower())
    return [_stem(SYNONYMS.get(w, w)) for w in words if w not in STOPWORDS and len(w) > 1]


def _stem(word: str) -> str:
    """Very light stemming so feeding/feed and hamsters/hamster match."""
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    species: tuple[str, ...] = ()
    condition: Optional[str] = None  # only retrieved when the query mentions it
    kind: str = "knowledge"  # "knowledge" or "owner"
    score: float = 0.0
    tokens: list[str] = field(default_factory=list, repr=False)


def known_species(chunks: list["Chunk"]) -> set[str]:
    """Species the knowledge base names explicitly (everything except the catch-all "other")."""
    return {s for c in chunks for s in c.species if s != "other"}


def species_matches(chunk_species: tuple, species: Optional[str], known: set) -> bool:
    """Does a chunk apply to this species? "other" covers only species with no dedicated guidance."""
    if not species or not chunk_species:
        return True
    species = species.lower()
    if species in chunk_species:
        return True
    return "other" in chunk_species and species not in known


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def condition_met(condition: Optional[str], query_tokens) -> bool:
    """A chunk's Condition (one or more comma-separated words) must all appear in the query."""
    if not condition:
        return True
    words = re.findall(r"[a-z]+", condition.lower())
    return all(_stem(SYNONYMS.get(w, w)) in query_tokens for w in words)


def load_knowledge(directory: Optional[Path] = None) -> list[Chunk]:
    """Split every *.md file in `directory` (default: knowledge/) into section chunks.

    A `Condition:` line at file level or inside a section limits when that chunk is retrieved.
    """
    chunks: list[Chunk] = []
    for path in sorted(Path(directory or KNOWLEDGE_DIR).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
        species_line = re.search(r"^Species:\s*(.+)$", text, re.M)
        species = tuple(s.strip().lower() for s in species_line.group(1).split(",")) if species_line else ()
        header = re.split(r"^## ", text, maxsplit=1, flags=re.M)[0]
        cond = re.search(r"^Condition:[ \t]*(.+)$", header, re.M)
        source = re.search(r"^Source:\s*(.+)$", text, re.M)
        source_text = source.group(1).strip() if source else path.name
        sections = re.split(r"^## ", text, flags=re.M)
        preamble, bodies = sections[0], sections[1:]
        if not bodies:  # file with no sections: whole file is one chunk
            bodies = [title + "\n" + preamble]
        for body in bodies:
            heading, _, rest = body.partition("\n")
            section_cond = re.search(r"^Condition:[ \t]*(.+)$", rest, re.M)
            if section_cond:
                rest = re.sub(r"^Condition:.*\n?", "", rest, flags=re.M)
            chosen = section_cond or cond
            chunks.append(
                Chunk(
                    chunk_id=f"{path.stem}#{_slug(heading)}",
                    text=f"{title} - {heading.strip()}\n{rest.strip()}",
                    source=f"{path.name} ({source_text})",
                    species=species,
                    condition=chosen.group(1).strip().lower() if chosen else None,
                )
            )
    return chunks


def owner_chunks(owner) -> list[Chunk]:
    """Second retrieval source: facts about the owner's own pets."""
    chunks = []
    for pet in getattr(owner, "pets", []):
        needs = pet.special_needs or "none recorded"
        age = f"{pet.age} years old" if pet.age is not None else "age unknown"
        text = f"Pet profile for {pet.name}: {pet.species}, {age}, special needs: {needs}."
        chunks.append(
            Chunk(
                chunk_id=f"owner:{_slug(pet.name)}",
                text=text,
                source="owner pet profile",
                species=(pet.species.lower(),),
                kind="owner",
            )
        )
    return chunks


class Retriever:
    def __init__(self, chunks: Optional[list[Chunk]] = None) -> None:
        self.chunks = chunks if chunks is not None else load_knowledge()
        for c in self.chunks:
            c.tokens = tokenize(c.text + " " + " ".join(c.species))
        self.known = known_species(self.chunks)
        self._refresh_idf()

    def _refresh_idf(self) -> None:
        n = len(self.chunks) or 1
        df: dict[str, int] = {}
        for c in self.chunks:
            for t in set(c.tokens):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}

    def search(self, query: str, species: Optional[str] = None, k: int = 8,
               stage: Optional[str] = None) -> list[Chunk]:
        """Return up to k chunks scoring >= MIN_SCORE, best first (copies, with score set)."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        results: list[Chunk] = []
        for c in self.chunks:
            if not species_matches(c.species, species, self.known):
                continue  # wrong species: exclude rather than risk unsafe advice
            if stage and c.kind == "knowledge":
                heading = c.text.splitlines()[0].lower()
                present = {w for w in ("puppies", "adult", "senior") if w in heading}
                wanted = {"puppy": "puppies"}.get(stage, stage)
                if present and wanted not in present:
                    continue  # guideline is for a different life stage
                stage_bonus = 1.5 if wanted in present else 1.0
            else:
                stage_bonus = 1.0
            if not condition_met(c.condition, q_tokens):
                continue  # condition-specific guidance (e.g. diabetic) needs an explicit mention
            overlap = [t for t in set(q_tokens) if t in c.tokens]
            if not overlap:
                continue
            weight = sum(self.idf.get(t, 0.0) for t in set(overlap))
            norm = math.sqrt(len(c.tokens)) or 1.0
            score = stage_bonus * weight / norm
            if score >= MIN_SCORE:
                results.append(Chunk(**{**c.__dict__, "score": round(score, 3)}))
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:k]
