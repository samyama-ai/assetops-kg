"""G-08 — format-neutral re-grade of docs / graph / SQL on retrieval scenarios.

G-07 showed the naive comparison is confounded by the keyword grader, which
credits the graph tier's tuned prose over SQL's correct-but-tabular output. This
re-grades the SAME committed responses with a single LLM judge (gpt-4o) that is
format-neutral — it credits a correct answer whether it arrives as prose or as
rows — applied identically to all three architectures, on the retrieval-only
subset of the held-out split (generative predict/forecast/recipe scenarios
dropped, since neither backend answers those by querying).

The result isolates the data-model effect from the grading artifact: with format
held neutral and the task held to retrieval, what is the true graph − SQL gap?

    OPENAI_API_KEY=... /home/vm-1/projects/venv/bin/python3 -m benchmark.judge_regrade \
        --data-dir <staged> --output results/sql_clean_regrade.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import openai

HERE = Path(__file__).resolve().parent.parent
GEN_KEYS = ("predict", "forecast", "recipe", "recommend", "probability",
            "generate", "next work order", "next likely")

JUDGE = (
    "You are a strict but FORMAT-NEUTRAL grader for an industrial-maintenance QA "
    "benchmark. You are given the question, a description of the expected answer, and "
    "a candidate response. Credit the response as correct if it CONVEYS the expected "
    "answer, whether it is written as prose or returned as raw table rows / key=value "
    "pairs. Do not penalise tabular or terse formatting. Do not reward fluent prose "
    "that lacks the expected content. Reply with exactly one word: PASS or FAIL.\n\n"
    "QUESTION:\n{q}\n\nEXPECTED ANSWER:\n{cf}\n\nCANDIDATE RESPONSE:\n{resp}\n\nVerdict:"
)


def _rows(p):
    d = json.loads(Path(p).read_text())
    return d if isinstance(d, list) else d.get("results", d.get("scenarios"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="results/sql_clean_regrade.json")
    a = ap.parse_args()

    from benchmark.run_ibm_scenarios import load_ibm_scenarios
    scen = {s["id"]: s for s in load_ibm_scenarios(a.data_dir, None)}
    ho = set(json.loads((HERE / "scenarios/heldout_split.json").read_text())["ids"])

    def generative(i):
        blob = (scen[i].get("characteristic_form", "") + " " + scen[i].get("text", "")).lower()
        return any(k in blob for k in GEN_KEYS)

    retrieval = [i for i in sorted(ho) if i in scen and not generative(i)]

    resp = {
        "documents": {x["id"]: x.get("response", "") for x in _rows(HERE / "results/arch_a_gpt4o.json")},
        "graph": {x["id"]: x.get("response", "") for x in _rows(HERE / "results/heldout_nlq_gpt4o.json")},
        "sql": {x["id"]: x.get("response", "") for x in _rows(HERE / "results/heldout_sql_gpt4o.json")},
    }
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def judge(q, cf, r):
        if not r:
            return False
        out = client.chat.completions.create(
            model="gpt-4o", temperature=0,
            messages=[{"role": "user", "content": JUDGE.format(q=q, cf=cf, resp=r[:1500])}])
        return out.choices[0].message.content.strip().upper().startswith("PASS")

    verdicts = {arch: {} for arch in resp}
    for i in retrieval:
        q = scen[i].get("text", "")
        cf = scen[i].get("characteristic_form", "")
        for arch in resp:
            verdicts[arch][i] = judge(q, cf, resp[arch].get(i, ""))
        print(f"  id={i}: docs={verdicts['documents'][i]:d} graph={verdicts['graph'][i]:d} sql={verdicts['sql'][i]:d}")

    n = len(retrieval)
    rate = {arch: round(100 * sum(verdicts[arch].values()) / n, 1) for arch in resp}
    out = {
        "goal": "G-08",
        "grader": "format-neutral gpt-4o LLM-judge, applied identically to all three arches",
        "subset": "retrieval-only held-out scenarios (generative dropped)",
        "n": n,
        "format_neutral_pass_pct": rate,
        "graph_minus_sql_pp": round(rate["graph"] - rate["sql"], 1),
        "graph_minus_docs_pp": round(rate["graph"] - rate["documents"], 1),
        "sql_minus_docs_pp": round(rate["sql"] - rate["documents"], 1),
        "keyword_grader_reference": {"note": "under the keyword grader (G-07), SQL retrieval was 11.1% — "
                                     "the delta below is how much of that was format artifact"},
        "per_scenario": {str(i): {a: verdicts[a][i] for a in resp} for i in retrieval},
    }
    (HERE / a.output).write_text(json.dumps(out, indent=2) + "\n") if not Path(a.output).is_absolute() \
        else Path(a.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nG-08 format-neutral (retrieval, n={n}): "
          f"docs {rate['documents']}% / graph {rate['graph']}% / SQL {rate['sql']}% "
          f"| graph−SQL = {out['graph_minus_sql_pp']:+}pp")


if __name__ == "__main__":
    main()
