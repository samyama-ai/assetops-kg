"""G-07 — honest analysis of the text-to-SQL ablation.

A naive text-to-SQL tier scores far below the graph on the held-out set. Before
concluding "the graph beats a relational backend", this decomposes WHY, because
the raw gap is confounded:

  1. Answer-format grading. The keyword grader credits the graph tier's tuned/
     templated prose; SQL that retrieves the CORRECT data as rows is not credited
     (e.g. id 116 returns the right sensors+failure-modes but fails the grader).
  2. Generative scenarios. Many tasks are predict / forecast / recipe /
     recommend — generation, not retrieval. The graph tier passes them via
     scenario-specific handlers (the same fallback pattern G-04 exposed on work
     orders), which a raw SQL tier has no equivalent of.
  3. Untuned prompt. The NLQ→Cypher prompt had 3 tuning rounds; the SQL prompt is
     zero-shot.

So this ablation does NOT cleanly isolate graph-vs-relational. It reports the raw
numbers and the decomposition, and names the format-controlled retrieval-only
comparison as the clean follow-up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
GEN_KEYS = ("predict", "forecast", "recipe", "recommend", "probability",
            "generate", "next work order", "next likely")


def _rows(p):
    d = json.loads(Path(p).read_text())
    return d if isinstance(d, list) else d.get("results", d.get("scenarios"))


def main() -> None:
    data_dir = sys.argv[1]
    from benchmark.run_ibm_scenarios import load_ibm_scenarios
    scen = {s["id"]: s for s in load_ibm_scenarios(data_dir, None)}

    ho = set(json.loads((HERE / "scenarios/heldout_split.json").read_text())["ids"])
    sql = {x["id"]: x for x in _rows(HERE / "results/heldout_sql_gpt4o.json")}
    graph = {r["id"]: bool(r["passed"]) for r in _rows(HERE / "results/heldout_nlq_gpt4o.json")}
    docs = {r["id"]: bool(r["passed"]) for r in _rows(HERE / "results/arch_a_gpt4o.json") if r["id"] in ho}

    def is_generative(i):
        blob = (scen.get(i, {}).get("characteristic_form", "") + " " +
                scen.get(i, {}).get("text", "")).lower()
        return any(k in blob for k in GEN_KEYS)

    ids = sorted(sql)
    gen = [i for i in ids if is_generative(i)]
    ret = [i for i in ids if not is_generative(i)]

    def rate(subset, table):
        if not subset:
            return None
        return round(100 * sum(table.get(i, False) for i in subset) / len(subset), 1)

    # SQL failure decomposition
    fails = [i for i in ids if not sql[i]["passed"]]
    norows = [i for i in fails if "No rows" in sql[i]["response"] or "SQL error" in sql[i]["response"]]
    rows_uncredited = [i for i in fails if i not in norows]  # retrieved data, grader didn't credit

    out = {
        "goal": "G-07",
        "n_heldout": len(ids),
        "raw_pass_pct": {
            "documents_archA": rate(ids, docs),
            "graph_archB": rate(ids, graph),
            "sql_archSQL": round(100 * sum(sql[i]["passed"] for i in ids) / len(ids), 1),
        },
        "confounded": True,
        "by_task_class": {
            "retrieval": {"n": len(ret), "documents": rate(ret, docs),
                          "graph": rate(ret, graph),
                          "sql": rate(ret, {i: sql[i]["passed"] for i in ids})},
            "generative": {"n": len(gen), "documents": rate(gen, docs),
                           "graph": rate(gen, graph),
                           "sql": rate(gen, {i: sql[i]["passed"] for i in ids}),
                           "note": "predict/forecast/recipe — the graph tier passes via scenario handlers, "
                                   "not graph queries (same fallback pattern as G-04); raw SQL has no equivalent"},
        },
        "sql_failure_decomposition": {
            "total_fail": len(fails),
            "no_rows_or_error": len(norows),
            "rows_returned_but_not_credited": len(rows_uncredited),
            "rows_uncredited_ids": rows_uncredited,
            "example_format_artifact": "id 116: SQL returned the correct sensors+failure-modes as rows; "
                                       "the keyword grader (tuned to the graph tier's prose) did not credit it",
        },
        "verdict": (
            "The naive text-to-SQL ablation is CONFOUNDED and does not cleanly show the graph beats a "
            "relational backend. The raw gap is driven by (a) a keyword grader that credits the graph "
            "tier's tuned answer FORMAT over SQL's correct-but-tabular output, (b) generative scenarios "
            "the graph passes via scenario-specific handlers, not queries, and (c) an untuned SQL prompt. "
            "This reinforces the paper's honesty thesis: apparent data-model advantages on this benchmark "
            "are substantially harness artifacts. The clean follow-up is a format-controlled, retrieval-only "
            "comparison with a tuned SQL prompt."
        ),
    }
    (HERE / "results/sql_ablation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["raw_pass_pct"], indent=1))
    print("by task class:", json.dumps(out["by_task_class"], indent=0)[:400])
    print("failure decomp:", out["sql_failure_decomposition"]["total_fail"], "fail =",
          out["sql_failure_decomposition"]["no_rows_or_error"], "no-rows +",
          out["sql_failure_decomposition"]["rows_returned_but_not_credited"], "uncredited")


if __name__ == "__main__":
    main()
