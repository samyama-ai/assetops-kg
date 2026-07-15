#!/usr/bin/env python3
"""Generate every GAK number claimed in paper3 v5 from the committed result files.

Principle 0: numbers in the paper are GENERATED from artifacts, never transcribed.
Run:  python -m evaluation.paper3_numbers
Emits results/paper3_numbers.json and prints the LaTeX table body.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The three GAK runs: same 88 scenarios, same graph, same judge; only the
# enrichment model inside the engine's agent runtime differs.
RUNS = [
    ("claude-sonnet-4-5", "gak_full.json"),
    ("gpt-4.1", "repro_gak_gpt41_2026-06-30.json"),
    ("gpt-4o", "repro_gak_2026-06-30.json"),
]


def main() -> None:
    out: dict = {"gak_by_model": {}}

    for model, fname in RUNS:
        s = json.loads((RESULTS / fname).read_text())["summary"]
        out["gak_by_model"][model] = {
            "source_file": fname,
            "scenarios": s["scenarios"],
            "judge_pass": s["scenario_judge_pass"],
            "pass_pct": s["scenario_judge_pass_pct"],
            "avg_judge_score": s["avg_judge_score"],
            "equipment_types": s["equipment_types"],
            "answerability_lift": s["entity_answerability_lift"],
            "answerability_lift_pct": s["entity_answerability_lift_pct"],
        }

    m = out["gak_by_model"]

    # The negative result: spread across enrichment models.
    best, worst = m["claude-sonnet-4-5"], m["gpt-4o"]
    out["gak_model_spread_pp"] = round(best["pass_pct"] - worst["pass_pct"], 1)

    # The structural claim: is the answerability lift model-robust?
    lifts = {k: v["answerability_lift_pct"] for k, v in m.items()}
    out["answerability_lift_model_robust"] = len(set(lifts.values())) == 1
    out["answerability_lift_pct_all"] = sorted(set(lifts.values()))

    # Does any GPT enrichment fall below the 65% document-store baseline?
    BASELINE_PCT = 65.0
    out["baseline_pct"] = BASELINE_PCT
    out["models_below_baseline"] = sorted(
        k for k, v in m.items() if v["pass_pct"] < BASELINE_PCT
    )

    dest = RESULTS / "paper3_numbers.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"wrote {dest}\n")
    print("%-24s %10s %10s %12s" % ("enrichment model", "pass", "rate", "judge score"))
    for model, _ in RUNS:
        v = m[model]
        print(
            "%-24s %6d /%3d %9.1f%% %12.3f"
            % (model, v["judge_pass"], v["scenarios"], v["pass_pct"], v["avg_judge_score"])
        )
    print()
    print(f"spread (best-worst)        : {out['gak_model_spread_pp']} pp")
    print(f"answerability model-robust : {out['answerability_lift_model_robust']} "
          f"(all at {out['answerability_lift_pct_all']}%)")
    print(f"below the {BASELINE_PCT:.0f}% baseline    : {out['models_below_baseline'] or 'none'}")


if __name__ == "__main__":
    main()
