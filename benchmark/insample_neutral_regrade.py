"""G-11 — does the in-sample +17.9 / +23.0 lift also compress under strict grading?

G-09 showed the held-out lift compresses only mildly under a format-neutral judge
(+14.3 keyword → +11.9 neutral). This completes the 2×2 (in-sample/held-out ×
keyword/neutral): it re-grades the committed in-sample Arch A (documents) and
Arch B (graph NLQ) responses under the same format-neutral judge, on all 139 and
on the 95 data-path-matched subset, so the headline +17.9/+23.0 gets a
grader-honest counterpart.

    OPENAI_API_KEY=... /home/vm-1/projects/venv/bin/python3 -m benchmark.insample_neutral_regrade \
        --data-dir <staged> --output results/insample_neutral_regrade.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import openai

from benchmark.judge_regrade import JUDGE
from evaluation.paper3_audit_numbers import is_contaminated

HERE = Path(__file__).resolve().parent.parent
ARCH_A = HERE / "results/arch_a_gpt4o.json"
ARCH_B = HERE / "results/repro_nlq_gpt4o_2026-06-29.json"


def _by_id(p):
    d = json.loads(Path(p).read_text())
    r = d if isinstance(d, list) else d.get("results", d.get("scenarios"))
    return {x["id"]: x for x in r}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="results/insample_neutral_regrade.json")
    a = ap.parse_args()

    from benchmark.run_ibm_scenarios import load_ibm_scenarios
    scen = {s["id"]: s for s in load_ibm_scenarios(a.data_dir, None)}
    A, B = _by_id(ARCH_A), _by_id(ARCH_B)
    ids = sorted(set(A) & set(B) & set(scen))
    matched = [i for i in ids if not is_contaminated(A[i])]

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def judge(q, cf, r):
        if not r:
            return False
        out = client.chat.completions.create(
            model="gpt-4o", temperature=0,
            messages=[{"role": "user", "content": JUDGE.format(q=q, cf=cf, resp=r[:1500])}])
        return out.choices[0].message.content.strip().upper().startswith("PASS")

    av, bv = {}, {}
    for i in ids:
        q = scen[i].get("text", "")
        cf = scen[i].get("characteristic_form", "")
        av[i] = judge(q, cf, A[i].get("response", ""))
        bv[i] = judge(q, cf, B[i].get("response", ""))
        print(f"  id={i}: docs={av[i]:d} graph={bv[i]:d}")

    def rates(subset):
        n = len(subset)
        ap_ = round(100 * sum(av[i] for i in subset) / n, 1)
        bp_ = round(100 * sum(bv[i] for i in subset) / n, 1)
        return {"n": n, "documents": ap_, "graph": bp_, "delta_pp": round(bp_ - ap_, 1)}

    out = {
        "goal": "G-11",
        "grader": "format-neutral gpt-4o judge (same as G-08/G-09)",
        "all_139": rates(ids),
        "matched_95": rates(matched),
        "keyword_reference": {"all_139": "+23.0 (58.3→81.3)", "matched": "+17.9 (66.3→84.2)"},
    }
    all_n, m = rates(ids), rates(matched)
    out["verdict"] = (
        f"in-sample under strict format-neutral grading: all-139 docs {all_n['documents']}%→graph "
        f"{all_n['graph']}% = +{all_n['delta_pp']}pp (keyword +23.0); matched docs {m['documents']}%→"
        f"graph {m['graph']}% = +{m['delta_pp']}pp (keyword +17.9). Absolute rates collapse (grader "
        f"leniency) but the data-layer delta {'survives' if m['delta_pp'] > 0 else 'does NOT survive'} strict grading."
    )
    (HERE / a.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nG-11: matched +{m['delta_pp']}pp, all-139 +{all_n['delta_pp']}pp (neutral) "
          f"vs keyword +17.9/+23.0")


if __name__ == "__main__":
    main()
