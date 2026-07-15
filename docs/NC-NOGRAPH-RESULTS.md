# Negative control results: the graph contributes nothing to GAK

**Run 2026-07-13.** Protocol frozen in `NC-NOGRAPH-PROTOCOL.md` **before** the code was
written. Decision rules below were committed in advance and are applied unchanged.

Artifacts: `benchmark/nc_nograph.py` → `results/nc_nograph.json`.
Judge byte-identical to `run_gak.py:92` (gpt-4o, temp 0, json_object), same 88 HF
failure-mode scenarios, same `characteristic_form` rubrics.

## Results

| model | GAK (published, **with** graph) | C1 — no graph, question-**aware** | C2 — no graph, question-**blind** | Δ C1−GAK | Δ C2−GAK |
|---|---|---|---|---|---|
| gpt-4o | 52.3% | **89.8%** (79/88, score 0.949) | 68.2% (60/88, 0.785) | **+37.5** | **+15.9** |
| gpt-4.1 | 73.9% | **89.8%** (79/88, score 0.949) | 69.3% (61/88, 0.789) | **+15.9** | −4.6 |
| claude-sonnet-4-5 | 81.8% | *not run (no Anthropic key)* | — | — | — |

C1's two 79/88 results are **not** a duplicate: the models agree on 78 of 88 scenarios
and pass different sets (5 flip each way). Same count, different scenarios.

## Verdict against the pre-registered rules

Δ = C2 − GAK isolates the graph (C2 reproduces GAK's question-blindness exactly and
deletes only the graph round-trip).

- **gpt-4o: Δ = +15.9 pp → "the graph is actively harmful."** The write/read round-trip
  destroys knowledge the model already had.
- **gpt-4.1: Δ = −4.6 pp → "the graph is a pass-through"** (inside the pre-registered
  ±5 pp band). Stated honestly: this is the one number that mildly favours the graph,
  and it sits just inside the tolerance I declared in advance. It does **not** reach
  the Δ ≤ −5 pp threshold that would have refuted the hypothesis.

**Neither model shows the graph contributing.** H0 (the graph contributes) is rejected.

## The finding that matters

**Simply asking the LLM the question — no graph, no Cypher, no enrichment — beats the
entire GAK pipeline by 15.9 to 37.5 points, and beats even its best published number
(claude, 81.8%) by 8 points.**

GAK is not a graph result. It is the LLM's parametric knowledge of textbook failure
modes, degraded by a round-trip through a database.

## The "model dependence" in paper3 v5 is an artifact

paper3 v5 (drafted, **not uploaded**) reported the 81.8 → 73.9 → 52.3 spread as a
scientific negative result: *"for GAK the enrichment model is a larger lever than the
data layer."* That is wrong.

| pipeline | model spread (gpt-4o vs gpt-4.1) |
|---|---|
| GAK as published (question-blind) | **21.6 pp** |
| C1 (question restored, no graph) | **0.0 pp** — both 79/88 |

The spread is not a property of GAK, of the models, or of the task. It is an artifact of
`run_gak.py:250-255`, where the answer step never receives the question, so all of an
entity's scenarios are graded against one undifferentiated dump. Restore the question and
the models become indistinguishable.

## Root causes (both in `run_gak.py`)

1. **`:250-255` — the answer step never sees the question.** 88 scenarios collapse to 10
   per-entity answers. Costs ~20 pp (C2 → C1) and manufactures the entire model-spread
   artifact.
2. **`:45-52` (`SCHEMA_HINT`) — GAK writes and reads an edge type that does not exist.**
   It instructs the LLM to create `(:Equipment)-[:HAS_FAILURE_MODE]->(:FailureMode)`; no
   ETL loader creates `HAS_FAILURE_MODE`, and it appears in **0** of 239 LLM-generated
   Cypher queries against the live graph. Therefore `baseline_answerable=False` is
   guaranteed by a wrong-edge query, and `enriched_answerable=True` is guaranteed by
   construction. **The published "answerability lift 0 → 100%" is a tautology: it cannot
   fail, for any model, even one emitting garbage.**

## Consequences for paper3

- The GAK 81.8% (Architecture D) does not survive as a graph result.
- The "answerability lift is model-robust and stands" claim in the v5 draft is **vacuous**
  and must be withdrawn.
- The "GAK is model-dependent" negative result in the v5 draft is an **artifact** and must
  be withdrawn.
- **paper3 v5 must not be uploaded as drafted.**

## Caveats

- claude-sonnet-4-5 could not be re-run (no Anthropic key in the workspace). Its 81.8% is
  the published GAK number; its C1/C2 arms are unmeasured. Since C1 = 89.8% for *both* GPT
  models and exceeds claude's GAK score, this does not change the conclusion, but the
  claude C1/C2 cells are genuinely empty.
- The judge is gpt-4o. In C1 it grades gpt-4o's own answers, which risks self-preference —
  but gpt-4.1's answers, graded by the same gpt-4o judge, score **identically** (79/88),
  so self-preference does not explain the result.
