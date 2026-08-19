"""G-09 — does the docs→graph held-out lift survive strict (format-neutral) grading?

G-08 showed the keyword grader inflates absolute scores via prose-echo leniency.
The core data-layer claim (G-02: docs 66.7% → graph 81.0% = +14.3pp on the
held-out) was measured with that lenient grader. This re-grades the SAME
committed docs and graph responses on the FULL held-out split (all 42, not just
retrieval) with the same format-neutral judge, so the delta is grader-honest.

Reuses the judge prompt from judge_regrade.py — one grader, applied identically.

    OPENAI_API_KEY=... /home/vm-1/projects/venv/bin/python3 -m benchmark.heldout_neutral_regrade \
        --data-dir <staged> --output results/heldout_neutral_regrade.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import openai

from benchmark.judge_regrade import JUDGE

HERE = Path(__file__).resolve().parent.parent


def _rows(p):
    d = json.loads(Path(p).read_text())
    return d if isinstance(d, list) else d.get("results", d.get("scenarios"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="results/heldout_neutral_regrade.json")
    a = ap.parse_args()

    from benchmark.run_ibm_scenarios import load_ibm_scenarios
    scen = {s["id"]: s for s in load_ibm_scenarios(a.data_dir, None)}
    ho = sorted(json.loads((HERE / "scenarios/heldout_split.json").read_text())["ids"])

    docs = {x["id"]: x.get("response", "") for x in _rows(HERE / "results/arch_a_gpt4o.json")}
    graph = {x["id"]: x.get("response", "") for x in _rows(HERE / "results/heldout_nlq_gpt4o.json")}
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def judge(q, cf, r):
        if not r:
            return False
        out = client.chat.completions.create(
            model="gpt-4o", temperature=0,
            messages=[{"role": "user", "content": JUDGE.format(q=q, cf=cf, resp=r[:1500])}])
        return out.choices[0].message.content.strip().upper().startswith("PASS")

    dv, gv = {}, {}
    for i in ho:
        q, cf = scen[i].get("text", ""), scen[i].get("characteristic_form", "")
        dv[i] = judge(q, cf, docs.get(i, ""))
        gv[i] = judge(q, cf, graph.get(i, ""))
        print(f"  id={i}: docs={dv[i]:d} graph={gv[i]:d}")

    n = len(ho)
    d_pct = round(100 * sum(dv.values()) / n, 1)
    g_pct = round(100 * sum(gv.values()) / n, 1)
    out = {
        "goal": "G-09",
        "grader": "format-neutral gpt-4o judge (same as G-08), applied identically",
        "subset": "full held-out split (all 42, not only retrieval)",
        "n": n,
        "documents_neutral_pct": d_pct,
        "graph_neutral_pct": g_pct,
        "heldout_neutral_delta_pp": round(g_pct - d_pct, 1),
        "keyword_grader_reference": {"documents": 66.7, "graph": 81.0, "delta": 14.3},
        "verdict": (
            f"under strict format-neutral grading the data-layer lift is docs {d_pct}% → graph {g_pct}% "
            f"= +{round(g_pct - d_pct, 1)}pp (keyword grader said +14.3). The absolute rates are far lower "
            f"(the keyword grader is lenient), but the docs→graph lift {'survives' if g_pct > d_pct else 'does NOT survive'} "
            f"strict grading."
        ),
    }
    (HERE / a.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nG-09: docs {d_pct}% → graph {g_pct}% = +{round(g_pct - d_pct, 1)}pp (neutral, full held-out, n={n})")


if __name__ == "__main__":
    main()
