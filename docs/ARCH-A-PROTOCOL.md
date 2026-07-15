# Architecture A: build the baseline we never built

**Frozen 2026-07-13, BEFORE writing any code.**

## Why this exists

Two facts established by code review, both verified against artifacts:

1. **On the 139-scenario snapshot, Architecture A was never run.** Every 139-scenario
   result file in `results/` is ours-over-the-graph (NLQ or deterministic). The 65%
   baseline in the paper is **cited from IBM's KDD paper**, not measured by us.

2. **On the custom-40, what we called Architecture A is not a document baseline.**
   `benchmark/run_baseline.py` loads **no data at all** — no CouchDB, no CSV, no JSON,
   no tools. It sends one LLM call containing the question plus the instruction
   *"Answer based on general industrial knowledge."* Its 85% is GPT-4o answering from
   parametric memory.

So the paper's headline — *"the same GPT-4 model rises from 65% to 82--83%"* — compares
**IBM's number** (their harness, their grader, their undisclosed model config) against
**our number** (our harness, our grader, our model). The model was not held fixed. The
harness was not held fixed. The grader was not held fixed. The entire delta is attributed
to the data layer.

This is the same defect class as paper18's interpolated baseline: a measured treatment
against a borrowed, unmeasured baseline.

## Aggravating factor: our side was tuned on the test set, the baseline was not

| run | pass rate |
|---|---|
| deterministic handlers, 2026-03-11, 5 iterations over 90 min | 68.3 -> 88.5 -> 93.5 -> 97.1 -> **98.6%** |
| NLQ prompt, 2026-03-12, 3 iterations over 3.5 h | 55.4 -> 77.7 -> **82.7%** |

Every iteration was re-scored on the same 139 scenarios. There is **no held-out split
anywhere in the codebase**. The hand-written "Key Conventions" block in the NLQ system
prompt (equipment-ID maps, *"there are NO vibration sensors in this dataset"*, date
formats) is the residue of that tuning. The 65% baseline received zero iterations.

## Hypothesis under test

**H0 (what the paper claims):** the data layer is the lever. Holding the model and the
orchestration fixed and swapping documents -> typed graph lifts pass rate by ~17 points.

**H1:** the +17 is substantially explained by harness, grader, and test-set tuning, and
the true same-harness data-layer effect is materially smaller.

## Method

Run **IBM's own agent** -- their code, their MCP servers, their document data layer --
with a **named** model, and score it with **our** grader. Nothing else changes.

| held fixed | varied |
|---|---|
| scenarios (the 139 / 152) | **the data layer only** |
| model (`gpt-4o`, named and pinned) | |
| grader (our harness pass-scorer, identical to Architecture B) | |

- **Arm A-real:** `openai-agent` (ReAct agent-as-tool) over MCP servers backed by CouchDB
  documents. `_build_run_config` returns `None` for a bare model id, so it uses the direct
  OpenAI API -- no WatsonX needed.
- **Arm B (already measured):** NLQ -> Cypher over the typed graph, same model, same grader.

Both arms are then scored by the *same* grader on the *same* scenarios. The difference is
the data layer and nothing else. That number -- not 65% vs 82% -- is the paper's claim.

## Pre-registered decision rules

Let D = B(gpt-4o) - A-real(gpt-4o), in percentage points, same grader, same scenarios.

- **D >= +12 pp** -> the data-layer claim survives roughly as published. H0 supported.
- **+4 pp <= D < +12 pp** -> the effect is real but materially smaller than the published
  ~17. The paper's magnitude must be restated.
- **-4 pp < D < +4 pp** -> **no data-layer effect.** The published +17 was harness, grader,
  and tuning. H0 rejected.
- **D <= -4 pp** -> the document baseline *beats* the graph on these scenarios.

## What would make me wrong

- If IBM's agent underperforms because we mis-configure it (wrong model, missing data,
  broken MCP server) we would manufacture a *large* D and falsely confirm the paper.
  **Mitigation:** before scoring, sanity-check the agent end-to-end on questions whose
  answers we independently know from the graph (e.g. "What sensors are on Chiller 6?").
  If the agent cannot answer those, the run is invalid and must not be reported.
- Our grader may favour the terse, structured answers our own systems emit over the
  discursive prose an agent emits. **Mitigation:** report the grader's per-scenario
  rationale, and spot-check disagreements by hand.

## Honest limits, stated in advance

- We **cannot** reproduce IBM's published 65% exactly: their agent model is not disclosed
  anywhere in their repo or their shipped trajectories (`react_llm_model_id: 34`), and
  their README says the leaderboard is *"to be revised (WIP with latest models)"*.
- Therefore this run does **not** claim to reproduce IBM. It claims something more useful:
  it builds the **same-harness, same-grader, same-model baseline the paper needed and
  never had.**

## Artifacts

- Driver: `benchmark/run_arch_a.py`
- Output: `results/arch_a_gpt4o.json`
- Every number in any write-up must be generated from that file.
