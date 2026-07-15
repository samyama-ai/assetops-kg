#!/usr/bin/env python3
"""Generate every number in the paper3 audit write-up from committed artifacts.

Principle 0: numbers in the write-up are GENERATED here, never transcribed.
Run:  python -m evaluation.paper3_audit_numbers
Emits results/paper3_audit_numbers.json and prints a summary.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"

# scenarios needing a data path absent from IBM's CURRENT upstream (no alert/anomaly
# server), or referencing assets absent from BOTH data layers -> not a documents-vs-graph
# comparison. Detected structurally where possible, else from the agent's own "not found".
def is_contaminated(rec_a: dict) -> str | None:
    q = rec_a["question"].lower()
    resp = (rec_a.get("response") or "").lower()
    if "alert" in q or "anomaly" in q:
        return "no alert/anomaly server in current upstream"
    if any(p in resp for p in ("not listed", "couldn't find", "no asset",
                                "not found in", "isn't listed")):
        return "asset/data absent from the document store"
    return None


def main() -> None:
    out: dict = {}

    # --- Architecture A: the baseline the paper never built --------------------
    a = json.loads((R / "arch_a_gpt4o.json").read_text())
    A = {r["id"]: r for r in a["results"]}
    out["arch_a_valid"] = a["summary"].get("VALID")
    out["arch_a_rate_limited"] = a["summary"].get("rate_limited")
    out["arch_a_full_pct"] = a["summary"]["pass_pct"]
    out["arch_a_context_overflows"] = sum(
        1 for r in a["results"] if "context_length" in (r["error"] or "")
    )

    # --- Architecture B: NLQ->Cypher over the graph, same model, same grader ----
    b = json.loads((R / "repro_nlq_gpt4o_2026-06-29.json").read_text())
    B = {r["id"]: r for r in b}
    ids = sorted(set(A) & set(B))

    na = sum(1 for i in ids if A[i]["passed"])
    nb = sum(1 for i in ids if B[i]["passed"])
    out["n_scenarios"] = len(ids)
    out["arch_a_full"] = round(100 * na / len(ids), 1)
    out["arch_b_full"] = round(100 * nb / len(ids), 1)
    out["delta_full"] = round(out["arch_b_full"] - out["arch_a_full"], 1)

    # --- the data-path-matched (clean) subset ----------------------------------
    contam = {i: is_contaminated(A[i]) for i in ids}
    clean = [i for i in ids if not contam[i]]
    ca = sum(1 for i in clean if A[i]["passed"])
    cb = sum(1 for i in clean if B[i]["passed"])
    out["n_clean"] = len(clean)
    out["n_excluded"] = len(ids) - len(clean)
    out["exclusion_reasons"] = {}
    for i in ids:
        if contam[i]:
            out["exclusion_reasons"][contam[i]] = out["exclusion_reasons"].get(contam[i], 0) + 1
    out["arch_a_clean"] = round(100 * ca / len(clean), 1)
    out["arch_b_clean"] = round(100 * cb / len(clean), 1)
    out["delta_clean"] = round(out["arch_b_clean"] - out["arch_a_clean"], 1)

    # per-type on the clean subset
    types: dict[str, list[int]] = {}
    for i in clean:
        types.setdefault(A[i]["type"], [0, 0, 0])
        t = types[A[i]["type"]]
        t[2] += 1
        t[0] += A[i]["passed"]
        t[1] += B[i]["passed"]
    out["clean_by_type"] = {
        t: {"a_pct": round(100 * v[0] / v[2], 1), "b_pct": round(100 * v[1] / v[2], 1),
            "delta": round(100 * v[1] / v[2] - 100 * v[0] / v[2], 1), "n": v[2]}
        for t, v in sorted(types.items())
    }

    # --- the no-graph control (from the earlier GAK audit) ---------------------
    if (R / "nc_nograph.json").exists():
        nc = json.loads((R / "nc_nograph.json").read_text())
        out["gak_no_graph_c1"] = {m: v["summary"]["C1"]["pass_pct"]
                                  for m, v in nc["arms"].items()}

    # --- the "structurally impossible" custom-40, answered with NO data --------
    bl = json.loads((R / "baseline_gpt4o_results.json").read_text())
    out["custom40_no_data_pct"] = round(100 * sum(1 for r in bl if r["passed"]) / len(bl), 1)

    (R / "paper3_audit_numbers.json").write_text(json.dumps(out, indent=2) + "\n")

    print("wrote results/paper3_audit_numbers.json\n")
    print(f"Architecture A run valid : {out['arch_a_valid']} "
          f"(rate-limited: {out['arch_a_rate_limited']})")
    print(f"context overflows        : {out['arch_a_context_overflows']} "
          f"(genuine document-store limit)\n")
    print(f"FULL {out['n_scenarios']}:  A={out['arch_a_full']}%  B={out['arch_b_full']}%  "
          f"D={out['delta_full']:+.1f}")
    print(f"  excluded {out['n_excluded']}: {out['exclusion_reasons']}")
    print(f"CLEAN {out['n_clean']}: A={out['arch_a_clean']}%  B={out['arch_b_clean']}%  "
          f"D={out['delta_clean']:+.1f}  (paper claims +17)\n")
    print("clean subset, per type:")
    for t, v in out["clean_by_type"].items():
        print(f"  {t:<8} A={v['a_pct']:>5}%  B={v['b_pct']:>5}%  {v['delta']:+.1f}  (n={v['n']})")
    print()
    print(f"custom-40 answered with NO data at all: {out['custom40_no_data_pct']}% "
          f"(\"structurally impossible\" claim)")
    if "gak_no_graph_c1" in out:
        print(f"GAK no-graph control (C1)             : {out['gak_no_graph_c1']}")


if __name__ == "__main__":
    main()
