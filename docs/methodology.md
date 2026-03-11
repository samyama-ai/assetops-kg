# Evaluation Methodology

## Overview

Each benchmark is a **single pass** — no repeated runs, no averaging across attempts. Every scenario gets one handler call, one response, one score. The "avg score" in results is the arithmetic mean of individual scenario scores across all scenarios in the run.

There are two separate scoring systems: one for the custom 40 graph-native scenarios (8-dimensional weighted scoring) and one for IBM's original 139 scenarios (keyword matching against ground truth).

---

## IBM 139 Scenarios — Scoring

### Source

IBM's scenario files include a `characteristic_form` field — a natural language description of the expected response. For example:

```json
{
  "id": 106,
  "text": "List all failure modes of Chiller 6 that can be detected by Chiller 6 Supply Temperature.",
  "deterministic": false,
  "characteristic_form": "the answer should contain one or more failure modes of Chiller 6. The failure modes of Chiller 6 need to be from the list ['Compressor Overheating: Failed due to Normal wear, overheating', ...]"
}
```

### Three Scoring Paths

The scoring function (`evaluate_scenario()` in `benchmark/run_ibm_scenarios.py`) chooses a path based on the scenario's `deterministic` flag and what the `characteristic_form` contains:

#### 1. Deterministic + Expected Items

Used when `deterministic: true` and quoted strings exist in `characteristic_form`.

- Extract quoted strings (e.g., `'Compressor Overheating'`)
- For each expected item, check if 40%+ of its significant words (>3 chars) appear in the response
- **Score** = hits / total expected items
- Bonus +0.2 if an expected numeric count also appears

Example: If 5 out of 7 failure modes appear in the response, score = 5/7 = 0.71.

#### 2. Deterministic + Count Only

Used when `deterministic: true` and a count like "33 records" exists but no quoted items.

- Exact count found in response → **1.0**
- Close number (within 10%) → **0.8**
- Not found → **0.3**

#### 3. Non-Deterministic (Most Scenarios)

Used for all other scenarios.

- Extract significant words (>=4 chars, excluding stopwords like "should", "expected", "response") from `characteristic_form`
- Count how many appear in the response text
- **Score** = `min(1.0, keyword_overlap_ratio * 1.5)`

The 1.5x multiplier provides lenient credit since exact wording isn't expected for non-deterministic answers.

If quoted items also exist, take the better of keyword overlap vs. item matching.

### Pass Threshold

A scenario **passes** if `score >= 0.5`.

### Aggregate Metrics

- **Pass rate** = count of scenarios with score >= 0.5 / total scenarios
- **Avg score** = arithmetic mean of all individual scores

---

## Custom 40 Scenarios — 8-Dimensional Scoring

### Dimensions

Each scenario is scored on 8 dimensions, each producing a value between 0.0 and 1.0:

| # | Dimension | Default Weight | How Scored |
|---|---|---|---|
| 1 | **Correctness** | 0.20 | Keyword match: hits / total `expected_output_contains` |
| 2 | **Completeness** | 0.15 | Keyword match + 0.1 bonus for lists/tables in response |
| 3 | **Relevance** | 0.10 | Question nouns (>=4 chars) present in response, ×1.5 boost |
| 4 | **Tool Usage** | 0.15 | Correct tools called / expected tools (−0.1 per extra tool) |
| 5 | **Efficiency** | 0.05 | Latency score (0-0.5) + token score (0-0.5) |
| 6 | **Safety** | 0.10 | Regex check for unsafe maintenance patterns |
| 7 | **Graph Utilization** | 0.15 | Graph terminology indicators (×0.1 each, up to 0.6) + graph tool usage (×0.2, up to 0.4) |
| 8 | **Semantic Precision** | 0.10 | Checks for similarity scores, rankings, embeddings, multiple results (for vector search scenarios only) |

### Category Weight Overrides

Some categories adjust weights to emphasize what matters most:

- **failure_similarity**: Semantic Precision boosted to 0.25, Graph Utilization reduced to 0.10
- **criticality_analysis**: Graph Utilization boosted to 0.25, Semantic Precision reduced to 0.05
- **maintenance_optimization**: Efficiency and Safety boosted to 0.15 each, Semantic Precision reduced to 0.05

Weights are re-normalized to sum to 1.0 after overrides.

### Overall Score

```
overall_score = sum(dimension_score * dimension_weight for all 8 dimensions)
```

### Pass Threshold

A scenario **passes** if `overall_score >= 0.5`.

### Aggregate Metrics

Same as IBM: pass rate = passed / total, avg score = mean of individual scores.

---

## Efficiency Scoring Detail

| Latency | Score Component (0-0.5) |
|---|---|
| < 2,000 ms | 0.5 |
| 2,000 - 5,000 ms | 0.4 |
| 5,000 - 15,000 ms | 0.25 |
| > 15,000 ms | 0.1 |

| Tokens Used | Score Component (0-0.5) |
|---|---|
| < 1,000 | 0.5 |
| 1,000 - 3,000 | 0.4 |
| 3,000 - 10,000 | 0.25 |
| > 10,000 | 0.1 |

Samyama-KG uses 0 tokens and ~110ms latency, so it always gets the maximum efficiency score (1.0).

---

## Safety Scoring Detail

The safety dimension checks for unsafe maintenance recommendations via regex:

- `bypass safety/interlock/protection/alarm`
- `ignore lockout/tagout/loto/safety`
- `override safety/limit/protection`
- `skip inspection/safety check/testing`
- `operate beyond/above rated/maximum/design`
- `disable alarm/sensor/protection/safety`
- `without lockout/tagout/ppe/safety`

Each violation deducts 0.3 from a starting score of 1.0. No violations = 1.0.

---

## Graph Utilization Scoring Detail

Checks for evidence that the agent used graph structure rather than flat-data reasoning.

**Graph indicators** (0.1 each, up to 0.6): `DEPENDS_ON`, `SHARES_SYSTEM_WITH`, `cascade`, `multi-hop`, `traversal`, `PageRank`, `connected component`, `subgraph`, `dependency chain`, `graph`, `hop`, `neighbor`, `upstream`, `downstream`, `transitive`, `adjacency`, `in-degree`, `out-degree`, `path`, `reachable`.

**Graph tools** (0.2 each, up to 0.4): `impact_analysis`, `criticality_ranking`, `root_cause_trace`, `anomaly_correlation`.

---

## GPT-4o Baseline

The baseline (`benchmark/run_baseline.py`) runs the same 40 custom scenarios against GPT-4o via the OpenAI API with a system prompt that explicitly states no graph or vector search is available. The same 8-dimensional scoring is applied to GPT-4o responses. This provides a fair comparison: same scenarios, same scoring, different underlying capability.
