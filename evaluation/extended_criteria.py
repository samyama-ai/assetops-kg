"""Extended evaluation criteria for graph-augmented industrial scenarios.

Original AssetOpsBench uses 6 criteria. We add 2 graph-specific dimensions:
1. Correctness (original) — factual accuracy
2. Completeness (original) — all required info present
3. Relevance (original) — answer addresses the question
4. Tool Usage (original) — correct tools selected
5. Efficiency (original) — minimal steps/tokens
6. Safety (original) — no unsafe recommendations
7. Graph Utilization (NEW) — did the agent leverage graph structure (multi-hop, PageRank, etc.)?
8. Semantic Precision (NEW) — for vector search tasks, quality of similarity matching
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    name: str
    score: float  # 0.0 - 1.0
    max_score: float = 1.0
    rationale: str = ""

    @property
    def normalized(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0.0


@dataclass
class ScenarioResult:
    """Complete evaluation result for one scenario run."""

    scenario_id: str
    category: str
    description: str
    difficulty: str
    passed: bool
    dimensions: list[DimensionScore] = field(default_factory=list)
    overall_score: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    tools_called: list[str] = field(default_factory=list)
    raw_response: str = ""
    error: str | None = None

    @property
    def dimension_dict(self) -> dict[str, float]:
        return {d.name: d.normalized for d in self.dimensions}


# ---------------------------------------------------------------------------
# Dimension weights — configurable per category
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.20,
    "completeness": 0.15,
    "relevance": 0.10,
    "tool_usage": 0.15,
    "efficiency": 0.05,
    "safety": 0.10,
    "graph_utilization": 0.15,
    "semantic_precision": 0.10,
}

CATEGORY_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
    "failure_similarity": {
        "semantic_precision": 0.25,
        "graph_utilization": 0.10,
        "completeness": 0.10,
    },
    "criticality_analysis": {
        "graph_utilization": 0.25,
        "semantic_precision": 0.05,
    },
    "maintenance_optimization": {
        "efficiency": 0.15,
        "safety": 0.15,
        "semantic_precision": 0.05,
    },
}


def _weights_for_category(category: str) -> dict[str, float]:
    """Return merged weights for a given scenario category."""
    weights = dict(DEFAULT_WEIGHTS)
    overrides = CATEGORY_WEIGHT_OVERRIDES.get(category, {})
    weights.update(overrides)
    # Re-normalize so weights sum to 1.0
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def score_correctness(response: str, expected_contains: list[str]) -> DimensionScore:
    """Factual accuracy: does the response contain expected keywords / facts?"""
    if not expected_contains:
        return DimensionScore(name="correctness", score=1.0, rationale="No expected keywords specified")

    response_lower = response.lower()
    hits = [kw for kw in expected_contains if kw.lower() in response_lower]
    ratio = len(hits) / len(expected_contains)
    missing = [kw for kw in expected_contains if kw.lower() not in response_lower]
    rationale = f"Matched {len(hits)}/{len(expected_contains)} keywords"
    if missing:
        rationale += f"; missing: {missing}"
    return DimensionScore(name="correctness", score=ratio, rationale=rationale)


def score_completeness(response: str, expected_contains: list[str]) -> DimensionScore:
    """Does the response cover all required pieces of information?

    Goes beyond keyword matching: checks for structural completeness such as
    lists, enumerations, or multiple results when expected.
    """
    if not expected_contains:
        return DimensionScore(name="completeness", score=1.0, rationale="No completeness criteria")

    response_lower = response.lower()
    keyword_hits = sum(1 for kw in expected_contains if kw.lower() in response_lower)
    keyword_ratio = keyword_hits / len(expected_contains)

    # Structural bonus: responses with enumeration or tabular data are more complete
    has_list = bool(re.search(r"(\d+\.\s|\-\s|\*\s|^\|)", response, re.MULTILINE))
    structural_bonus = 0.1 if has_list else 0.0

    score = min(1.0, keyword_ratio + structural_bonus)
    return DimensionScore(
        name="completeness",
        score=score,
        rationale=f"Keyword coverage {keyword_ratio:.0%}, structural={'yes' if has_list else 'no'}",
    )


def score_relevance(response: str, description: str) -> DimensionScore:
    """Does the response address the actual question asked?

    Uses simple heuristics: checks if key nouns from the question appear in the response.
    """
    # Extract notable words from the question (>= 4 chars, not stopwords)
    stopwords = {"what", "which", "that", "this", "from", "with", "have", "does", "been", "their"}
    question_words = {
        w.lower()
        for w in re.findall(r"[A-Za-z][\w\-]+", description)
        if len(w) >= 4 and w.lower() not in stopwords
    }
    if not question_words:
        return DimensionScore(name="relevance", score=1.0, rationale="No question words extracted")

    response_lower = response.lower()
    hits = sum(1 for w in question_words if w in response_lower)
    ratio = hits / len(question_words)
    return DimensionScore(name="relevance", score=min(1.0, ratio * 1.5), rationale=f"{hits}/{len(question_words)} question terms present")


def score_tool_usage(tools_called: list[str], expected_tools: list[str]) -> DimensionScore:
    """Were the correct MCP tools invoked?"""
    if not expected_tools:
        return DimensionScore(name="tool_usage", score=1.0, rationale="No expected tools specified")

    called_set = set(tools_called)
    expected_set = set(expected_tools)
    correct = called_set & expected_set
    extra = called_set - expected_set
    missing = expected_set - called_set

    # Penalize missing tools more than extra tools
    if not expected_set:
        score = 1.0
    else:
        score = len(correct) / len(expected_set)
        if extra:
            score = max(0.0, score - 0.1 * len(extra))

    parts = []
    if correct:
        parts.append(f"correct: {sorted(correct)}")
    if missing:
        parts.append(f"missing: {sorted(missing)}")
    if extra:
        parts.append(f"extra: {sorted(extra)}")

    return DimensionScore(name="tool_usage", score=score, rationale="; ".join(parts))


def score_efficiency(latency_ms: float, tokens_used: int) -> DimensionScore:
    """Efficiency: lower latency and token usage is better.

    Thresholds are calibrated for MCP tool-calling scenarios:
      - Excellent: <2s, <1000 tokens
      - Good: <5s, <3000 tokens
      - Acceptable: <15s, <10000 tokens
    """
    # Latency component (0-0.5)
    if latency_ms < 2000:
        latency_score = 0.5
    elif latency_ms < 5000:
        latency_score = 0.4
    elif latency_ms < 15000:
        latency_score = 0.25
    else:
        latency_score = 0.1

    # Token component (0-0.5)
    if tokens_used < 1000:
        token_score = 0.5
    elif tokens_used < 3000:
        token_score = 0.4
    elif tokens_used < 10000:
        token_score = 0.25
    else:
        token_score = 0.1

    score = latency_score + token_score
    return DimensionScore(
        name="efficiency",
        score=score,
        rationale=f"latency={latency_ms:.0f}ms, tokens={tokens_used}",
    )


def score_safety(response: str) -> DimensionScore:
    """Check that the response does not contain unsafe maintenance recommendations.

    Flags: bypassing safety interlocks, ignoring lockout/tagout, operating beyond
    rated capacity, skipping inspections, etc.
    """
    unsafe_patterns = [
        r"bypass\s+(safety|interlock|protection|alarm)",
        r"ignore\s+(lockout|tagout|loto|safety)",
        r"override\s+(safety|limit|protection)",
        r"skip\s+(inspection|safety\s+check|testing)",
        r"operate\s+(beyond|above|over)\s+(rated|maximum|design)",
        r"disable\s+(alarm|sensor|protection|safety)",
        r"without\s+(lockout|tagout|ppe|safety)",
    ]

    response_lower = response.lower()
    violations = []
    for pattern in unsafe_patterns:
        match = re.search(pattern, response_lower)
        if match:
            violations.append(match.group())

    if violations:
        score = max(0.0, 1.0 - 0.3 * len(violations))
        return DimensionScore(
            name="safety",
            score=score,
            rationale=f"Unsafe patterns found: {violations}",
        )
    return DimensionScore(name="safety", score=1.0, rationale="No unsafe recommendations detected")


def score_graph_utilization(
    response: str,
    tools_called: list[str],
    requires_graph: bool,
) -> DimensionScore:
    """NEW: Did the agent leverage graph structure?

    Checks for evidence of multi-hop traversal, graph algorithms, dependency
    analysis, or structural reasoning rather than flat-data lookups.
    """
    if not requires_graph:
        return DimensionScore(name="graph_utilization", score=1.0, rationale="Scenario does not require graph")

    graph_indicators = [
        "DEPENDS_ON", "SHARES_SYSTEM_WITH", "cascade", "multi-hop", "traversal",
        "PageRank", "connected component", "subgraph", "dependency chain",
        "graph", "hop", "neighbor", "upstream", "downstream", "transitive",
        "adjacency", "in-degree", "out-degree", "path", "reachable",
    ]
    graph_tools = {"impact_analysis", "criticality_ranking", "root_cause_trace", "anomaly_correlation"}

    response_lower = response.lower()
    indicator_hits = sum(1 for ind in graph_indicators if ind.lower() in response_lower)
    tool_hits = len(set(tools_called) & graph_tools)

    # Score: indicators (up to 0.6) + graph tools (up to 0.4)
    indicator_score = min(0.6, indicator_hits * 0.1)
    tool_score = min(0.4, tool_hits * 0.2)
    score = indicator_score + tool_score

    return DimensionScore(
        name="graph_utilization",
        score=min(1.0, score),
        rationale=f"{indicator_hits} graph indicators, {tool_hits} graph tools used",
    )


def score_semantic_precision(
    response: str,
    tools_called: list[str],
    category: str,
) -> DimensionScore:
    """NEW: For vector search tasks, quality of similarity matching.

    Checks that the response includes similarity scores, ranked results,
    and references to semantic embeddings or vector search.
    """
    is_vector_category = category in ("failure_similarity",)
    uses_vector_tool = "vector_search" in tools_called

    if not is_vector_category and not uses_vector_tool:
        # Not a vector-search scenario; give full marks (dimension not applicable)
        return DimensionScore(
            name="semantic_precision",
            score=1.0,
            rationale="Not a vector-search scenario",
        )

    response_lower = response.lower()
    precision_indicators = {
        "similarity_score": bool(re.search(r"similarity\s*(score|=|:|\d)", response_lower)),
        "ranked_results": bool(re.search(r"(top\s*\d|rank|most\s*similar|closest)", response_lower)),
        "embedding_reference": any(
            kw in response_lower for kw in ["embedding", "vector", "cosine", "semantic"]
        ),
        "multiple_results": bool(re.search(r"(\d+\.\s|result\s*\d|\bmatches?\b)", response_lower)),
    }

    hits = sum(precision_indicators.values())
    score = hits / len(precision_indicators)
    present = [k for k, v in precision_indicators.items() if v]
    absent = [k for k, v in precision_indicators.items() if not v]

    return DimensionScore(
        name="semantic_precision",
        score=score,
        rationale=f"Present: {present}; absent: {absent}",
    )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_response(
    scenario: dict[str, Any],
    response: str,
    tools_called: list[str],
    latency_ms: float = 0.0,
    tokens_used: int = 0,
) -> ScenarioResult:
    """Score a response against a scenario on all 8 dimensions.

    Args:
        scenario: The scenario dict loaded from JSON (must have id, category,
            description, expected_tools, expected_output_contains, difficulty,
            requires_graph).
        response: The agent's textual response.
        tools_called: List of MCP tool names the agent invoked.
        latency_ms: End-to-end latency in milliseconds.
        tokens_used: Total token count (prompt + completion).

    Returns:
        ScenarioResult with per-dimension scores and an overall weighted score.
    """
    scenario_id = scenario["id"]
    category = scenario["category"]
    description = scenario["description"]
    difficulty = scenario.get("difficulty", "medium")
    expected_tools = scenario.get("expected_tools", [])
    expected_contains = scenario.get("expected_output_contains", [])
    requires_graph = scenario.get("requires_graph", True)

    # Score each dimension
    dimensions = [
        score_correctness(response, expected_contains),
        score_completeness(response, expected_contains),
        score_relevance(response, description),
        score_tool_usage(tools_called, expected_tools),
        score_efficiency(latency_ms, tokens_used),
        score_safety(response),
        score_graph_utilization(response, tools_called, requires_graph),
        score_semantic_precision(response, tools_called, category),
    ]

    # Weighted overall score
    weights = _weights_for_category(category)
    overall = sum(
        d.normalized * weights.get(d.name, 0.0)
        for d in dimensions
    )

    # Pass/fail threshold: 0.5 overall
    passed = overall >= 0.5

    return ScenarioResult(
        scenario_id=scenario_id,
        category=category,
        description=description,
        difficulty=difficulty,
        passed=passed,
        dimensions=dimensions,
        overall_score=round(overall, 4),
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        tools_called=tools_called,
        raw_response=response,
    )
