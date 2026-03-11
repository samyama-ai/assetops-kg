"""Benchmark runner for AssetOps-KG scenarios.

Loads scenario JSON files, runs each against the MCP server tools,
collects timing and correctness metrics, and prints a summary table.

Usage:
    python -m evaluation.runner                    # run all categories
    python -m evaluation.runner --category failure_similarity
    python -m evaluation.runner --dry-run          # validate scenarios only
    python -m evaluation.runner --output results.json
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


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"

CATEGORY_FILES = [
    "multi_hop_dependency.json",
    "cross_asset_correlation.json",
    "failure_similarity.json",
    "criticality_analysis.json",
    "maintenance_optimization.json",
    "root_cause_analysis.json",
    "temporal_pattern.json",
]


def load_scenarios(category: str | None = None) -> list[dict[str, Any]]:
    """Load scenario definitions from JSON files.

    Args:
        category: If provided, load only scenarios from that category file.
            Otherwise load all categories.

    Returns:
        List of scenario dicts.
    """
    scenarios: list[dict[str, Any]] = []
    files = CATEGORY_FILES
    if category:
        target = f"{category}.json"
        files = [f for f in files if f == target]
        if not files:
            raise FileNotFoundError(
                f"No scenario file for category '{category}'. "
                f"Available: {[f.replace('.json', '') for f in CATEGORY_FILES]}"
            )

    for fname in files:
        fpath = SCENARIOS_DIR / fname
        if not fpath.exists():
            print(f"[WARN] Scenario file not found: {fpath}", file=sys.stderr)
            continue
        with open(fpath) as f:
            data = json.load(f)
        if isinstance(data, list):
            scenarios.extend(data)
        else:
            scenarios.append(data)

    return scenarios


# ---------------------------------------------------------------------------
# MCP tool invocation stub
# ---------------------------------------------------------------------------

async def invoke_mcp_tool(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Invoke a single MCP tool and return its result.

    This is a stub that should be replaced with actual MCP client calls
    once the MCP server tools are implemented.
    """
    # TODO: Replace with actual MCP client invocation, e.g.:
    #   async with mcp_client.connect("stdio://mcp_server") as client:
    #       result = await client.call_tool(tool_name, params)
    #       return result
    raise NotImplementedError(
        f"MCP tool invocation not yet wired. Tool: {tool_name}, params: {params}"
    )


async def run_scenario(scenario: dict[str, Any], dry_run: bool = False) -> ScenarioResult:
    """Execute a single scenario and evaluate the result.

    In dry-run mode, returns a placeholder result without calling tools.
    """
    scenario_id = scenario["id"]
    category = scenario["category"]
    description = scenario["description"]

    if dry_run:
        return ScenarioResult(
            scenario_id=scenario_id,
            category=category,
            description=description,
            difficulty=scenario.get("difficulty", "medium"),
            passed=True,
            overall_score=0.0,
            latency_ms=0.0,
            tokens_used=0,
            tools_called=[],
            raw_response="[dry-run] No execution performed",
        )

    # --- Actual execution path ---
    tools_called: list[str] = []
    response_parts: list[str] = []
    start = time.perf_counter()

    try:
        for tool_name in scenario.get("expected_tools", []):
            result = await invoke_mcp_tool(tool_name, {"query": description})
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


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def format_summary_table(results: list[ScenarioResult]) -> str:
    """Format results as a human-readable summary table."""
    lines: list[str] = []
    header = f"{'ID':<18} {'Category':<28} {'Diff':<8} {'Pass':>5} {'Score':>6} {'Latency':>9} {'Error'}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for r in sorted(results, key=lambda x: x.scenario_id):
        status = "PASS" if r.passed else "FAIL"
        if r.error:
            status = "ERR"
        latency = f"{r.latency_ms:.0f}ms" if r.latency_ms > 0 else "-"
        error_str = (r.error[:40] + "...") if r.error and len(r.error) > 40 else (r.error or "")
        lines.append(
            f"{r.scenario_id:<18} {r.category:<28} {r.difficulty:<8} {status:>5} "
            f"{r.overall_score:>6.2f} {latency:>9} {error_str}"
        )

    lines.append(sep)

    # Aggregates
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    avg_score = sum(r.overall_score for r in results) / total if total > 0 else 0.0

    lines.append(f"Total: {total} | Passed: {passed} | Failed: {failed} | Avg score: {avg_score:.3f}")

    # Per-category breakdown
    categories = sorted(set(r.category for r in results))
    if len(categories) > 1:
        lines.append("")
        lines.append("Per-category breakdown:")
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            cat_pass = sum(1 for r in cat_results if r.passed)
            cat_avg = sum(r.overall_score for r in cat_results) / len(cat_results)
            lines.append(f"  {cat:<30} {cat_pass}/{len(cat_results)} passed, avg={cat_avg:.3f}")

    lines.append(sep)
    return "\n".join(lines)


def results_to_json(results: list[ScenarioResult]) -> list[dict[str, Any]]:
    """Serialize results to JSON-compatible dicts."""
    out = []
    for r in results:
        out.append({
            "scenario_id": r.scenario_id,
            "category": r.category,
            "description": r.description,
            "difficulty": r.difficulty,
            "passed": r.passed,
            "overall_score": r.overall_score,
            "latency_ms": r.latency_ms,
            "tokens_used": r.tokens_used,
            "tools_called": r.tools_called,
            "error": r.error,
            "dimensions": {d.name: {"score": d.score, "rationale": d.rationale} for d in r.dimensions},
        })
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_all(
    category: str | None = None,
    dry_run: bool = False,
    output_path: str | None = None,
) -> list[ScenarioResult]:
    """Load scenarios, run them, print summary, optionally write JSON output."""
    scenarios = load_scenarios(category)
    print(f"Loaded {len(scenarios)} scenarios" + (f" (category: {category})" if category else ""))

    if not scenarios:
        print("[WARN] No scenarios to run.", file=sys.stderr)
        return []

    results: list[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {scenario['id']}...", end=" ", flush=True)
        result = await run_scenario(scenario, dry_run=dry_run)
        results.append(result)
        status = "PASS" if result.passed else ("ERR" if result.error else "FAIL")
        print(status)

    # Print summary
    print()
    print(format_summary_table(results))

    # Write JSON output
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\nResults written to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AssetOps-KG Benchmark Runner")
    parser.add_argument("--category", type=str, default=None, help="Run only scenarios from this category")
    parser.add_argument("--dry-run", action="store_true", help="Validate scenarios without executing tools")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON results")
    args = parser.parse_args()

    asyncio.run(run_all(category=args.category, dry_run=args.dry_run, output_path=args.output))


if __name__ == "__main__":
    main()
