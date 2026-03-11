"""Run scenarios against Samyama KG MCP tools — measure latency and accuracy.

This is the graph-augmented benchmark that should outperform the GPT-4 baseline.
It uses the MCP server tools (impact_analysis, vector_search, criticality_ranking,
etc.) backed by the Samyama graph database.

Usage:
    python -m benchmark.run_samyama
    python -m benchmark.run_samyama --category criticality_analysis
    python -m benchmark.run_samyama --output samyama_results.json
    python -m benchmark.run_samyama --compare baseline_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.extended_criteria import ScenarioResult, evaluate_response
from evaluation.runner import load_scenarios, format_summary_table, results_to_json


# ---------------------------------------------------------------------------
# MCP client for Samyama tools
# ---------------------------------------------------------------------------

class SamyamaMCPClient:
    """Client for invoking Samyama KG MCP server tools.

    Connects to the MCP server over stdio or SSE and dispatches tool calls.

    TODO: Wire up actual MCP client transport once mcp_server/tools/ are implemented.
    """

    def __init__(self, server_command: str = "python -m mcp_server"):
        self.server_command = server_command
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        # TODO: Initialize MCP client transport
        #   self._client = await mcp.connect_stdio(self.server_command)
        self._connected = True

    async def disconnect(self) -> None:
        """Close the MCP server connection."""
        self._connected = False

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Invoke an MCP tool and return its result dict.

        Supported tools (to be implemented in mcp_server/tools/):
            - impact_analysis: Multi-hop failure cascade analysis
            - anomaly_correlation: Cross-asset anomaly correlation
            - vector_search: Semantic similarity search on failure modes
            - criticality_ranking: PageRank-based asset criticality
            - maintenance_scheduler: Optimization-based scheduling
            - root_cause_trace: Backward traversal for root cause
            - sensor_trend: Temporal sensor data analysis
            - cypher_query: Raw Cypher query execution
        """
        if not self._connected:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        # TODO: Replace with actual tool invocation
        raise NotImplementedError(
            f"MCP tool '{tool_name}' not yet implemented. "
            f"Implement in mcp_server/tools/ and wire up the client."
        )


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------

async def run_samyama_scenario(
    scenario: dict[str, Any],
    client: SamyamaMCPClient,
) -> ScenarioResult:
    """Run a single scenario using Samyama KG MCP tools."""
    scenario_id = scenario["id"]
    category = scenario["category"]
    description = scenario["description"]
    expected_tools = scenario.get("expected_tools", [])

    tools_called: list[str] = []
    response_parts: list[str] = []
    start = time.perf_counter()

    try:
        for tool_name in expected_tools:
            # Build tool-specific parameters
            params = _build_tool_params(tool_name, scenario)
            result = await client.call_tool(tool_name, params)
            tools_called.append(tool_name)
            response_parts.append(json.dumps(result, default=str))
    except NotImplementedError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ScenarioResult(
            scenario_id=scenario_id,
            category=category,
            description=description,
            difficulty=scenario.get("difficulty", "medium"),
            passed=False,
            overall_score=0.0,
            latency_ms=elapsed_ms,
            tools_called=tools_called,
            raw_response="",
            error=str(exc),
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ScenarioResult(
            scenario_id=scenario_id,
            category=category,
            description=description,
            difficulty=scenario.get("difficulty", "medium"),
            passed=False,
            overall_score=0.0,
            latency_ms=elapsed_ms,
            tools_called=tools_called,
            raw_response="",
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    combined_response = "\n".join(response_parts)

    return evaluate_response(
        scenario=scenario,
        response=combined_response,
        tools_called=tools_called,
        latency_ms=elapsed_ms,
        tokens_used=0,
    )


def _build_tool_params(tool_name: str, scenario: dict[str, Any]) -> dict[str, Any]:
    """Build tool-specific parameters from the scenario description.

    Each tool expects different input parameters. This function maps the
    scenario's natural language description to structured tool inputs.
    """
    description = scenario["description"]

    if tool_name == "impact_analysis":
        return {"query": description, "max_hops": 5}
    elif tool_name == "anomaly_correlation":
        return {"query": description, "time_window_hours": 48}
    elif tool_name == "vector_search":
        return {"query": description, "top_k": 5}
    elif tool_name == "criticality_ranking":
        return {"query": description, "algorithm": "pagerank"}
    elif tool_name == "maintenance_scheduler":
        return {"query": description}
    elif tool_name == "root_cause_trace":
        return {"query": description, "max_depth": 5}
    elif tool_name == "sensor_trend":
        return {"query": description, "lookback_days": 30}
    elif tool_name == "cypher_query":
        return {"query": description}
    else:
        return {"query": description}


# ---------------------------------------------------------------------------
# Comparison with baseline
# ---------------------------------------------------------------------------

def compare_with_baseline(
    samyama_results: list[ScenarioResult],
    baseline_path: str,
) -> None:
    """Print a side-by-side comparison of Samyama KG vs baseline results."""
    with open(baseline_path) as f:
        baseline_data = json.load(f)

    baseline_by_id = {r["scenario_id"]: r for r in baseline_data}

    print()
    print("=== COMPARISON: Samyama KG vs Baseline ===")
    header = f"{'ID':<18} {'Category':<28} {'Baseline':>9} {'Samyama':>9} {'Delta':>7}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    total_baseline = 0.0
    total_samyama = 0.0
    count = 0

    for r in sorted(samyama_results, key=lambda x: x.scenario_id):
        baseline = baseline_by_id.get(r.scenario_id)
        if baseline:
            b_score = baseline.get("overall_score", 0.0)
            s_score = r.overall_score
            delta = s_score - b_score
            delta_str = f"{delta:+.3f}"
            print(f"{r.scenario_id:<18} {r.category:<28} {b_score:>9.3f} {s_score:>9.3f} {delta_str:>7}")
            total_baseline += b_score
            total_samyama += s_score
            count += 1
        else:
            print(f"{r.scenario_id:<18} {r.category:<28} {'N/A':>9} {r.overall_score:>9.3f} {'N/A':>7}")

    print(sep)
    if count > 0:
        avg_baseline = total_baseline / count
        avg_samyama = total_samyama / count
        avg_delta = avg_samyama - avg_baseline
        print(
            f"{'AVERAGE':<18} {'':<28} {avg_baseline:>9.3f} {avg_samyama:>9.3f} {avg_delta:>+7.3f}"
        )
        improvement_pct = (avg_delta / avg_baseline * 100) if avg_baseline > 0 else 0
        print(f"\nOverall improvement: {improvement_pct:+.1f}%")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_samyama(
    category: str | None = None,
    output_path: str | None = None,
    compare_path: str | None = None,
) -> list[ScenarioResult]:
    """Run all scenarios against Samyama KG tools and evaluate."""
    scenarios = load_scenarios(category)
    print(f"Samyama KG benchmark: {len(scenarios)} scenarios")

    client = SamyamaMCPClient()
    await client.connect()

    results: list[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {scenario['id']}...", end=" ", flush=True)
        result = await run_samyama_scenario(scenario, client)
        results.append(result)
        status = "PASS" if result.passed else ("ERR" if result.error else "FAIL")
        print(f"{status} ({result.latency_ms:.0f}ms)")

    await client.disconnect()

    print()
    print("=== SAMYAMA KG RESULTS ===")
    print(format_summary_table(results))

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\nResults written to {output_path}")

    if compare_path:
        compare_with_baseline(results, compare_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AssetOps-KG Samyama Benchmark")
    parser.add_argument("--category", type=str, default=None, help="Run only one category")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to baseline results JSON for side-by-side comparison",
    )
    args = parser.parse_args()

    asyncio.run(run_samyama(
        category=args.category,
        output_path=args.output,
        compare_path=args.compare,
    ))


if __name__ == "__main__":
    main()
