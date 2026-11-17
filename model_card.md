# Model Card: PawPal+ Care Advisor

## Intended use
Suggest daily care tasks for a pet owner's pets, grounded in a small curated guideline set, for the owner to review and approve. It is an educational/planning aid, **not veterinary advice**.

## System summary
Retrieval (guideline sections + owner pet profile) -> LLM drafts cited tasks (Gemini, or a deterministic offline `MockClient`) -> rule-based verifier with feedback loop -> confidence score -> human approval. See [README.md](README.md) and [diagrams/system_architecture.mmd](diagrams/system_architecture.mmd).

## Limitations and biases
- **Tiny, hand-written knowledge base.** 10 short files I summarized as educational guidance; they have not been reviewed by a veterinarian. Anything not in them (other species, rare conditions, breed differences) yields no evidence or generic advice.
- **Coverage bias.** Dogs and cats have the most detail; small pets get one file; birds, reptiles and fish have none, so those owners get refusals.
- **Age rules are crude.** Senior = 8+ years (10+ for cats) regardless of breed or size, which is wrong for giant breeds.
- **Keyword retrieval.** TF-IDF with a small synonym map misses paraphrases and can over-retrieve for generic requests; generic requests return many tasks (8-12), which may exceed what an owner wants.
- **Relevance gate is shallow.** A request containing a word like "plan" or "care" passes the gate even if it is off-topic (e.g. "plan an engine repair for my dog" would not be refused).
- **Evaluation is mostly self-consistent.** The 17 eval cases and 202 tests run against a deterministic mock, so they validate the pipeline's logic and guardrails, not the quality of a real LLM. The Gemini path is implemented but I have not measured it. Confidence thresholds (70/40/25) were set by hand, not calibrated.
- **Owner-added knowledge is unreviewed.** The *Add to knowledge base* button saves what the owner typed to `knowledge/user_added_<species>.md`; it is checked by the same dose/toxic/schema rules and medication tasks are refused, but nobody validates the advice, and a wrong entry will be retrieved and cited as if it were guidance in later requests. The UI says it is not vet-reviewed, but the citation itself does not.
- **Time parsing is simple.** Only phrasings like "from 12 to 2", "between 6:30 and 7pm" and "at 4pm for 30 minutes" are understood; ambiguous times such as "12 to 2" are read as afternoon. Activities are limited to 120 minutes by the guardrail, and requested times that clash with other tasks are only flagged when the start times are identical.
- **Fuzzy matching can mislabel.** Typos are tolerated when matching a request to a guideline ("afternooon nap" -> "Afternoon nap"); an unrelated activity that shares most of its words with a guideline could be matched to it.
- **The chat understands a fixed set of phrasings.** It uses rules, not a language model, so unusual wording gets a polite "I'm not sure" (or, with Gemini selected, is passed to Gemini to classify). Chat commands that look like several things can be misread, so every data change shows a Confirm / Cancel card and nothing is applied until it is pressed. Chat-added tasks are the owner's own words: they are not checked against the care guidelines (only against the dose, toxic-food, duration and medication-flag rules).
- **English only**, and the request parser looks for simple patterns ("9-year-old", species words).
- The few-shot prompt option (`few_shot=True`) exists but I did not measure it against the zero-shot baseline, so I make no specialization claim.

## Misuse and prevention
| Risk | Mitigation in the system | Remaining gap |
|---|---|---|
| Someone treats task suggestions as medical advice or doses | Dose patterns blocked; medication tasks always flagged "confirm with your vet"; medication tasks only allowed with retrieved support | A user can ignore the flag; the guideline text itself is unreviewed |
| Prompt injection ("ignore previous instructions...") | Known phrases stripped before the model sees the request; system prompt says the request is data; output is schema-checked and must cite retrieved text | Pattern list is small; novel phrasings could slip through, though uncited output is still rejected |
| Hallucinated or toxic advice | Tasks must cite retrieved chunks and be supported by them; toxic-food terms blocked | Grounding is a token-overlap check, not true entailment |
| Over-trust in confident output | Confidence score and sources are always shown; human approves each task; low confidence shows a warning; very low refuses | The score is heuristic |
| Privacy | Pet data stays in local `data.json`; logs are local and git-ignored; with Gemini enabled the request and retrieved text are sent to Google | Do not put sensitive personal data in requests |

## What surprised me in testing
- Adding the pet's life stage ("adult") to the retrieval query made "How do I fix my car engine" retrieve cat and dog guidance, because the extra word matched documents. A tiny query change produced a confident wrong answer; removing it and filtering by stage afterwards fixed it.
- The Streamlit UI test exposed a flaw the unit tests missed: with a dog selected, an off-topic request still got dog tasks because the species alone matched the knowledge base. That led to the relevance gate.
- My grounding check let "Teleport the dog" through because "dog" appears in a document title; a unit test for exactly this caught it and the threshold moved from 0.5 to 0.6.
- My first answer to "why was the nap ignored?" would have been to just add nap guidance; testing showed the real gap was that the advisor never read times at all, and fixing that exposed a scheduler that ignored `start_time` for placement, so a 12:00 nap would still have appeared at 08:00 in the plan.
- The UI tests were silently saving to my real `data.json`: the app writes next to `app.py`, not the working directory, so a test that only changed the directory still overwrote saved pets. I found it when a new test could not find the file where I expected it. The app now honours `PAWPAL_DATA_FILE` and a shared test fixture points every test at a temporary file.
- Confidence is informative mainly through repair: a draft that needed one feedback round scored 83 vs 95 for a clean one, which matches intuition better than I expected.

## AI collaboration
I used Claude Code to plan and implement this project in one session. This section is drawn from what actually happened in that session; I should confirm and add my own perspective before submitting.

- **Suggestion accepted:** build the AI layer around the existing PawPal+ code rather than rewriting it (retrieval, agent and guardrails live in `ai/` and produce ordinary `Task` objects). It kept all 65 original tests green throughout and made the diff easy to review.
- **Suggestion changed/scoped down:** the first plan included a separate LLM "critic" step and a few-shot vs zero-shot comparison. I did not keep these as written: the critic became a deterministic verifier whose findings are fed back to the LLM (cheaper, testable offline), and the few-shot comparison would have been meaningless against a deterministic mock, so only the option remains and the stretch is not claimed. The first plan also left out required deliverables (Mermaid file, model card sections, parseable eval tables, reasoning traces), and I rejected it until those were added.
- **A suggestion I would verify rather than trust:** an early retrieval design counted any shared word as relevant; testing showed this was too loose, and I tightened it (stopwords, stemming, condition gating, relevance gate) only after seeing the failures above.

## Evaluation (parseable)
See the table in [README.md](README.md#testing-summary) and `eval/results.md` (regenerate with `python -m eval.run_eval`). Summary: 17/17 passed on the offline mock; average confidence 84/100; 104/104 pytest tests passed.

## Human evaluation
Not yet done. Suggested protocol: have two peers rate 10 generated routines for relevance, safety and clarity (Pass/Fail per criterion) and record the results as a table here.
