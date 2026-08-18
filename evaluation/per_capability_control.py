"""G-03 — isolate the capability classes a no-data LLM genuinely cannot do.

"Structurally impossible with flat documents" is false as a blanket claim: a
gpt-4o with *no data access at all* (`benchmark/run_baseline.py`, results in
`results/baseline_gpt4o_results.json`) already answers 85% of the 40 "graph-native"
scenarios from parametric memory. This groups that no-data control by capability
class, so the structural-advantage claim survives only where the control is low.

Reads the committed baseline artifact; no LLM run. The threshold is <=60%: only
failure_similarity (vector-similarity, 50%) and criticality_analysis
(PageRank-criticality, 60%) genuinely resist a no-data LLM.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "results/baseline_gpt4o_results.json"
OUT = HERE / "results/per_capability_control.json"
THRESHOLD = 60  # a no-data control at or below this => the graph is genuinely required

# Human-readable capability behind each scenario category.
CAPABILITY = {
    "failure_similarity": "vector-similarity",
    "criticality_analysis": "PageRank-criticality",
    "multi_hop_dependency": "multi-hop traversal",
    "cross_asset_correlation": "cross-asset join",
    "root_cause_analysis": "root-cause reasoning",
    "temporal_pattern": "temporal pattern",
    "maintenance_optimization": "optimization",
}


def main() -> None:
    d = json.loads(SRC.read_text())
    rows = d if isinstance(d, list) else d.get("results", d.get("records"))
    agg = collections.defaultdict(lambda: [0, 0])  # category -> [n, passed]
    for r in rows:
        c = r.get("category", "?")
        agg[c][0] += 1
        agg[c][1] += 1 if r.get("passed") else 0

    by_cap = {}
    n_tot = p_tot = 0
    required = []
    for c in sorted(agg):
        n, p = agg[c]
        pct = round(100 * p / n, 1)
        graph_required = pct <= THRESHOLD
        by_cap[c] = {"capability": CAPABILITY.get(c, c), "pass": p, "n": n,
                     "no_data_control_pct": pct, "graph_required": graph_required}
        if graph_required:
            required.append(c)
        n_tot += n
        p_tot += p

    out = {
        "goal": "G-03",
        "control": "no-data gpt-4o parametric baseline (run_baseline.py)",
        "source": "results/baseline_gpt4o_results.json",
        "n_total": n_tot,
        "pass_total": p_tot,
        "no_data_ceiling_pct": round(100 * p_tot / n_tot, 1),
        "threshold_pct": THRESHOLD,
        "graph_required_classes": required,
        "by_capability": by_cap,
        "claim": (
            f"a no-data LLM scores {round(100*p_tot/n_tot,1)}% on the {n_tot} 'graph-native' "
            f"scenarios; the structural graph advantage holds only where the no-data control "
            f"is <= {THRESHOLD}%: " + ", ".join(f"{c} ({by_cap[c]['no_data_control_pct']}%)" for c in required) +
            f". The other {len(by_cap)-len(required)} classes are answered >= 87.5% with no data."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"G-03: ceiling {out['no_data_ceiling_pct']}%, graph-required = {required} -> {OUT.name}")


if __name__ == "__main__":
    main()
