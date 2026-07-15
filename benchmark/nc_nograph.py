#!/usr/bin/env python3
"""Negative control: does the typed graph contribute anything to GAK?

Protocol frozen in docs/NC-NOGRAPH-PROTOCOL.md BEFORE this file was written.

Three arms over the same 88 HF failure-mode scenarios, the same rubrics, and a
judge byte-identical to run_gak.py:92 (gpt-4o, temp 0, json_object):

  GAK  (published)  LLM -> Cypher -> graph -> template dump.  Question-blind.
  C1                LLM answers the question directly.        Question-aware. No graph.
  C2                LLM lists the entity's failure modes in   Question-blind.  No graph.
                    the SAME bulleted shape GAK's template
                    emits; that one string grades all of the
                    entity's scenarios.

C2 is the decisive arm: it reproduces GAK's question-blindness exactly and deletes
only the graph round-trip, so the graph is the sole changed variable.

Usage:
  OPENAI_API_KEY=... python -m benchmark.nc_nograph --models gpt-4o,gpt-4.1
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

WORKERS = int(os.environ.get("NC_WORKERS", "8"))

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCENARIOS = pathlib.Path(
    os.environ.get(
        "GAK_SCENARIOS",
        str(ROOT.parent / "AssetOpsBench/data/hf_failure_mode_sensor_mapping.json"),
    )
)
JUDGE_MODEL = os.environ.get("GAK_JUDGE_MODEL", "gpt-4o")

# GAK's published per-model pass rates, for the Delta computation. Generated, not
# transcribed: read straight out of the committed result files.
GAK_RUNS = {
    "claude-sonnet-4-5": "results/gak_full.json",
    "gpt-4.1": "results/repro_gak_gpt41_2026-06-30.json",
    "gpt-4o": "results/repro_gak_2026-06-30.json",
}


def client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        env = ROOT / ".env"  # symlink -> ~/projects/.env
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit("OPENAI_API_KEY not set and not found in .env")
    # A bare OpenAI() has a 600s timeout. The first run of this control hung for 18
    # minutes on a single request with zero progress. Fail fast and retry instead.
    return OpenAI(api_key=key, timeout=60.0, max_retries=4)


# --- judge: byte-identical to run_gak.py:92 ---------------------------------
def judge(cl: OpenAI, question: str, characteristic_form: str, answer: str) -> dict:
    sys_p = (
        "You are grading an industrial asset-operations answer against a rubric. "
        "The rubric ('characteristic form') states what a correct answer must contain. "
        'Respond with strict JSON: {{"pass": bool, "score": 0.0-1.0, "rationale": "..."}}.'
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"RUBRIC (characteristic form):\n{characteristic_form}\n\n"
        f"ANSWER TO GRADE:\n{answer}\n\n"
        "Grade whether the answer satisfies the rubric. Be strict but fair."
    )
    r = cl.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
    )
    return json.loads(r.choices[0].message.content)


# --- C1: question-aware, no graph -------------------------------------------
def c1_answer(cl: OpenAI, model: str, question: str) -> str:
    r = cl.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": (
                "You are an industrial asset-operations expert. Answer the question "
                "directly and concretely, naming specific failure modes and the sensors "
                "that detect them where relevant."
            )},
            {"role": "user", "content": question},
        ],
    )
    return (r.choices[0].message.content or "").strip()


# --- C2: question-blind, no graph, SAME output shape as GAK's template -------
def c2_answer(cl: OpenAI, model: str, entity: str) -> str:
    """Reproduce GAK's per-entity dump without any graph.

    GAK's template (run_gak.py:130-140) emits exactly:
        Failure modes for {entity}:
        - {name} (severity: {sev}); detected by: {sensor, sensor}
    We hold that format constant so the judge cannot reward formatting instead of
    knowledge -- the graph is then the only thing that differs from GAK.
    """
    r = cl.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": (
                "You are an industrial asset-operations expert."
            )},
            {"role": "user", "content": (
                f"List the standard industrial failure modes of a '{entity}', with each "
                f"mode's severity and the sensors that detect it.\n\n"
                f"Output EXACTLY this format and nothing else:\n"
                f"Failure modes for {entity}:\n"
                f"- <failure mode> (severity: <low|medium|high>); detected by: "
                f"<sensor>, <sensor>"
            )},
        ],
    )
    return (r.choices[0].message.content or "").strip()


def gak_baseline() -> dict[str, float]:
    out = {}
    for model, f in GAK_RUNS.items():
        p = ROOT / f
        if p.exists():
            out[model] = json.loads(p.read_text())["summary"]["scenario_judge_pass_pct"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-4o,gpt-4.1")
    ap.add_argument("--output", default="results/nc_nograph.json")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    scenarios = [json.loads(l) for l in SCENARIOS.read_text().splitlines() if l.strip()]
    by_entity: dict[str, list] = defaultdict(list)
    for s in scenarios:
        by_entity[s["entity"]].append(s)

    cl = client()
    gak = gak_baseline()
    print(f"NC no-graph control: {len(scenarios)} scenarios, {len(by_entity)} entities")
    print(f"  judge={JUDGE_MODEL} (identical to run_gak.py)  models={models}")
    print(f"  GAK reference: {gak}\n")

    results: dict = {"judge_model": JUDGE_MODEL, "gak_reference_pct": gak, "arms": {}}

    for model in models:
        print(f"=== model: {model} ===", flush=True)
        per_arm: dict[str, list] = {"C1": [], "C2": []}

        # C2: one answer per entity (question-blind), reused across its scenarios.
        print("  [C2] generating one entity-level answer per equipment type (no graph)...", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(c2_answer, cl, model, e): e for e in by_entity}
            c2_by_entity = {futs[f]: f.result() for f in as_completed(futs)}
        print(f"  [C2] {len(c2_by_entity)} entity answers done", flush=True)

        def one(s: dict) -> tuple[dict, dict]:
            q, cf, ent = s["text"], s["characteristic_form"], s["entity"]
            a1 = c1_answer(cl, model, q)
            v1 = judge(cl, q, cf, a1)
            v2 = judge(cl, q, cf, c2_by_entity[ent])
            mk = lambda v: {  # noqa: E731
                "id": s["id"], "entity": ent, "question": q,
                "judge_pass": bool(v.get("pass")),
                "judge_score": float(v.get("score", 0.0)),
                "judge_rationale": v.get("rationale", ""),
            }
            return mk(v1), mk(v2)

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(one, s) for s in scenarios]
            for i, f in enumerate(as_completed(futs), 1):
                r1, r2 = f.result()
                per_arm["C1"].append(r1)
                per_arm["C2"].append(r2)
                if i % 10 == 0 or i == len(scenarios):
                    print(f"    {i}/{len(scenarios)} scenarios judged", flush=True)

        summary = {}
        for arm, rows in per_arm.items():
            n = len(rows) or 1
            p = sum(1 for r in rows if r["judge_pass"])
            summary[arm] = {
                "scenarios": len(rows),
                "judge_pass": p,
                "pass_pct": round(100 * p / n, 1),
                "avg_judge_score": round(sum(r["judge_score"] for r in rows) / n, 3),
            }

        g = gak.get(model)
        if g is not None:
            summary["delta_C2_minus_GAK_pp"] = round(summary["C2"]["pass_pct"] - g, 1)
            summary["delta_C1_minus_GAK_pp"] = round(summary["C1"]["pass_pct"] - g, 1)
            summary["gak_pass_pct"] = g

        results["arms"][model] = {"summary": summary, "C1": per_arm["C1"], "C2": per_arm["C2"]}
        print(f"  -> GAK={g}  C1={summary['C1']['pass_pct']}%  C2={summary['C2']['pass_pct']}%\n")

    dest = ROOT / args.output
    dest.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {dest}\n")

    print("%-18s %8s %8s %8s %10s %10s" % ("model", "GAK", "C1", "C2", "C1-GAK", "C2-GAK"))
    for model in models:
        s = results["arms"][model]["summary"]
        print("%-18s %7s%% %7s%% %7s%% %+9s %+9s" % (
            model, s.get("gak_pass_pct", "?"), s["C1"]["pass_pct"], s["C2"]["pass_pct"],
            s.get("delta_C1_minus_GAK_pp", "?"), s.get("delta_C2_minus_GAK_pp", "?"),
        ))


if __name__ == "__main__":
    main()
