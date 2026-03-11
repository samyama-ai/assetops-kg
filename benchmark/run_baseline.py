"""Run AssetOpsBench baseline: GPT-4 + flat tabular data (no graph).

This establishes the baseline performance that the Samyama KG approach
must beat. Uses the same 40 scenarios but resolves them with:
  - Flat CSV/JSON lookups (no graph traversal)
  - GPT-4 for reasoning (no MCP tools)
  - No vector search (keyword matching only)

Usage:
    python -m benchmark.run_baseline
    python -m benchmark.run_baseline --model gpt-4o --output baseline_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.extended_criteria import ScenarioResult, evaluate_response
from evaluation.runner import load_scenarios, format_summary_table, results_to_json


# ---------------------------------------------------------------------------
# LLM client (OpenAI)
# ---------------------------------------------------------------------------

async def call_llm(
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 2000,
) -> tuple[str, int]:
    """Call OpenAI API and return (response_text, total_tokens).

    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Export it before running the baseline benchmark."
        )

    try:
        import openai
    except ImportError:
        raise ImportError("Install openai package: pip install openai")

    client = openai.AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an industrial maintenance expert. Answer questions about "
                    "equipment failures, dependencies, maintenance scheduling, and asset "
                    "criticality. You have access to flat tabular data about equipment, "
                    "sensors, work orders, and failure modes. You do NOT have access to "
                    "a graph database or vector search."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    text = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0
    return text, tokens


# ---------------------------------------------------------------------------
# Baseline scenario execution
# ---------------------------------------------------------------------------

async def run_baseline_scenario(
    scenario: dict[str, Any],
    model: str = "gpt-4o",
) -> ScenarioResult:
    """Run a single scenario using LLM-only baseline (no graph, no MCP tools)."""
    scenario_id = scenario["id"]
    category = scenario["category"]
    description = scenario["description"]

    prompt = (
        f"Industrial maintenance question:\n\n{description}\n\n"
        "Answer based on general industrial knowledge. "
        "Be specific about equipment types, failure modes, and recommended actions."
    )

    start = time.perf_counter()
    try:
        response_text, tokens_used = await call_llm(prompt, model=model)
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
            tools_called=[],
            raw_response="",
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Baseline uses no tools — tool_usage score will be 0
    return evaluate_response(
        scenario=scenario,
        response=response_text,
        tools_called=[],  # No MCP tools in baseline
        latency_ms=elapsed_ms,
        tokens_used=tokens_used,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_baseline(
    model: str = "gpt-4o",
    category: str | None = None,
    output_path: str | None = None,
    concurrency: int = 3,
) -> list[ScenarioResult]:
    """Run all scenarios against LLM-only baseline."""
    scenarios = load_scenarios(category)
    print(f"Baseline benchmark: {len(scenarios)} scenarios, model={model}")

    semaphore = asyncio.Semaphore(concurrency)

    async def run_with_limit(s: dict[str, Any]) -> ScenarioResult:
        async with semaphore:
            return await run_baseline_scenario(s, model=model)

    results = await asyncio.gather(*[run_with_limit(s) for s in scenarios])
    results = list(results)

    print()
    print("=== BASELINE RESULTS (GPT-4 + flat data, no graph) ===")
    print(format_summary_table(results))

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\nResults written to {output_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AssetOps-KG Baseline Benchmark (GPT-4 + flat data)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--category", type=str, default=None, help="Run only one category")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent API calls")
    args = parser.parse_args()

    asyncio.run(run_baseline(
        model=args.model,
        category=args.category,
        output_path=args.output,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
