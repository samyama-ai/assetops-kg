"""G-05 — GAK's real value is caching, not accuracy: measure the call reduction.

GAK enriches the graph *once per equipment type* (the by-entity cache in
`benchmark/run_gak.py:main`, which groups scenarios by entity and calls
`enrich_entity` once per group). Without that cache the enrichment LLM call would
fire once per scenario. The reduction is therefore a structural ratio of the
committed run's own counts — scenarios / distinct equipment types — not a noisy
measurement. Judge calls (one per scenario) are constant across both regimes and
are excluded from the ratio.

This does NOT re-run GAK; it reads the committed `results/gak_full.json` counts
and writes the caching artifact. The accuracy story is dead (a no-graph control
beats GAK); this is the economics story that replaces it.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "results/gak_full.json"
OUT = HERE / "results/gak_caching.json"


def main() -> None:
    d = json.loads(SRC.read_text())
    s = d["summary"]
    scenarios = s["scenarios"]              # 88 — one enrichment each without the cache
    entities = s["equipment_types"]         # 10 — one enrichment each with the by-entity cache
    reduction = round(scenarios / entities, 2)

    out = {
        "goal": "G-05",
        "metric": "gak_caching_call_reduction_x",
        "enrichment_calls_no_cache": scenarios,
        "enrichment_calls_with_cache": entities,
        "call_reduction_x": reduction,
        "source": "results/gak_full.json",
        "definition": (
            "GAK enriches once per equipment type (by-entity cache in run_gak.py); "
            "without the cache it enriches once per scenario. "
            "reduction = scenarios / distinct equipment types."
        ),
        "excluded": "judge calls (1 per scenario) are constant across both regimes",
        "supersedes_claim": (
            "the earlier '0->100% answerability lift' is circular and dropped; "
            "a no-graph control (89.8%) beats GAK (81.8%) on accuracy — the real "
            "contribution of generation-augmented knowledge is this call reduction, not accuracy"
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"G-05: {scenarios}/{entities} = {reduction}x fewer enrichment calls -> {OUT.name}")


if __name__ == "__main__":
    main()
