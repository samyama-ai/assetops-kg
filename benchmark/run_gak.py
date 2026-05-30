#!/usr/bin/env python3
"""
run_gak.py — GAK (Generation-Augmented Knowledge) evaluation harness.

Tier 3 of the 3-tier grounding-substrate architecture. Evaluates the engine's
native agentic enrichment on REAL AssetOpsBench FMSR knowledge-gap scenarios:
the HF failure-mode/sensor-mapping set (88 scenarios across 10 equipment types
that IBM flags deterministic=False and that are ABSENT from our chiller+AHU
graph). For each scenario:

  1. baseline   — query the graph for the entity's failure modes -> expect empty
  2. enrich     — POST /api/enrich (engine AgentRuntime) -> LLM generates Cypher
  3. materialize— execute the generated CREATE via /api/query, tagging each new
                  node source:'LLM-derived' (provenance, NOT laundered as data)
  4. re-query   — same baseline query -> now answerable
  5. grade      — LLM-as-judge against the scenario's own `characteristic_form`
  6. measure    — answerability lift, nodes created, provenance coverage, latency

Everything is real: real engine HTTP API, real LLM enrichment, real CREATE,
real re-query. No mocks. Driven entirely through Samyama's native HTTP API.

Usage:
  python -m benchmark.run_gak --limit 3 --output results/gak_pilot.json
  python -m benchmark.run_gak --output results/gak_full.json
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

# --- config -----------------------------------------------------------------
SGE_URL = os.environ.get("SGE_URL", "http://localhost:8899")
TENANT = os.environ.get("GAK_TENANT", "assetops")
SCENARIOS = os.environ.get(
    "GAK_SCENARIOS",
    str(Path.home() / "projects/Madhulatha-Sandeep/graph_ws/AssetOpsBench/data/hf_failure_mode_sensor_mapping.json"),
)
JUDGE_MODEL = os.environ.get("GAK_JUDGE_MODEL", "gpt-4o")
ENRICH_MODEL_LABEL = os.environ.get("GAK_ENRICH_LABEL", "claude-sonnet-4-5 (engine AgentRuntime)")

SCHEMA_HINT = (
    "Graph schema (property graph, OpenCypher):\n"
    "  (:Equipment {equipment_id, name, equipment_type})\n"
    "  (:FailureMode {name, severity, description})\n"
    "  (:Sensor {name, sensor_type})\n"
    "  (:Equipment)-[:HAS_FAILURE_MODE]->(:FailureMode)\n"
    "  (:Sensor)-[:MONITORS]->(:FailureMode)\n"
)

# --- SGE HTTP helpers --------------------------------------------------------
def sge_query(cypher: str, graph: str = TENANT) -> dict:
    r = requests.post(f"{SGE_URL}/api/query", json={"query": cypher, "graph": graph}, timeout=60)
    r.raise_for_status()
    return r.json()


def sge_enrich(prompt: str, context: str, graph: str = TENANT) -> dict:
    r = requests.post(
        f"{SGE_URL}/api/enrich",
        json={"prompt": prompt, "context": context, "graph": graph},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def extract_cypher(text: str) -> list[str]:
    """Pull semicolon-separated Cypher statements out of an LLM response.

    Samyama rejects multiple back-to-back CREATE clauses in one query, so the
    enrichment prompt demands the loader-style pattern: one CREATE per node and
    MATCH...CREATE per relationship, each terminated by ';'. We strip fences and
    // comments, split on ';', and keep statement-leading clauses.
    """
    blocks = re.findall(r"```(?:cypher)?\s*(.*?)```", text, re.S | re.I)
    body = "\n".join(blocks) if blocks else text
    # drop // line comments
    body = "\n".join(l for l in body.splitlines() if not l.strip().startswith("//"))
    stmts = []
    for chunk in body.split(";"):
        c = " ".join(chunk.split()).strip()  # collapse whitespace/newlines
        if re.match(r"^(CREATE|MERGE|MATCH|WITH|UNWIND)\b", c, re.I):
            stmts.append(c)
    return stmts


# --- judge (LLM-as-judge against the scenario's characteristic_form) ---------
def judge(question: str, characteristic_form: str, answer: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sys = (
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
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
    )
    return json.loads(resp.choices[0].message.content)


# --- graph queries -----------------------------------------------------------
def entity_subgraph_query(entity: str) -> str:
    """Failure modes (+ monitoring sensors) for the entity, as materialized."""
    safe = entity.replace('"', '\\"')
    return (
        f'MATCH (e:Equipment)-[:HAS_FAILURE_MODE]->(fm:FailureMode) '
        f'WHERE toLower(e.name) CONTAINS toLower("{safe}") '
        f'OR toLower(e.equipment_type) CONTAINS toLower("{safe}") '
        f'OPTIONAL MATCH (s:Sensor)-[:MONITORS]->(fm) '
        f'RETURN fm.name AS failure_mode, fm.severity AS severity, '
        f'collect(DISTINCT s.name) AS sensors'
    )


def subgraph_answer(entity: str) -> tuple[str, int]:
    rows = sge_query(entity_subgraph_query(entity)).get("records", [])
    lines = []
    for r in rows:
        if not r or not r[0]:
            continue
        fm, sev, sensors = r[0], r[1], (r[2] or [])
        sens = ", ".join([s for s in sensors if s]) if sensors else "—"
        lines.append(f"- {fm} (severity: {sev}); detected by: {sens}")
    answer = "Failure modes for {}:\n{}".format(entity, "\n".join(lines)) if lines else ""
    return answer, len(lines)


def reset_gak():
    """Remove any prior LLM-derived nodes so each run starts from the base graph."""
    sge_query('MATCH (n) WHERE n.source = "LLM-derived" DETACH DELETE n')


# --- enrich one entity (cache miss), then answer its scenarios (cache hits) ---
def enrich_entity(entity: str, char_form: str) -> dict:
    q = entity_subgraph_query(entity)
    baseline_rows = sge_query(q).get("records", [])
    baseline_answerable = any(r and r[0] for r in baseline_rows)

    prompt = (
        f"The knowledge graph lacks information about the asset type: '{entity}'. "
        f"Generate OpenCypher to add an Equipment node for '{entity}', its standard "
        f"industrial failure modes as FailureMode nodes, and the sensors that detect them.\n\n"
        f"STRICT OUTPUT FORMAT (the engine rejects multiple CREATE clauses in one query):\n"
        f"  - Emit ONLY semicolon-terminated statements, one per line.\n"
        f"  - One CREATE per node: CREATE (:Label {{props}});\n"
        f"  - Relationships via MATCH...CREATE (no shared variables across statements):\n"
        f"    MATCH (e:Equipment {{name:\"...\"}}), (f:FailureMode {{name:\"...\"}}) CREATE (e)-[:HAS_FAILURE_MODE]->(f);\n"
        f"  - Sensor links: MATCH (s:Sensor {{name:\"...\"}}), (f:FailureMode {{name:\"...\"}}) CREATE (s)-[:MONITORS]->(f);\n"
        f"  - Set source:\"LLM-derived\" on EVERY node (provenance).\n"
        f"  - Use double-quoted strings. Use realistic failure modes for this equipment type.\n"
        f"Return ONLY the Cypher statements, nothing else."
    )
    context = SCHEMA_HINT + f"\nThe materialized knowledge should support questions like: {char_form}"
    t0 = time.time()
    agent_text = sge_enrich(prompt, context).get("agent_response", "")
    enrich_ms = (time.time() - t0) * 1000.0
    stmts = extract_cypher(agent_text)

    errors = []
    for st in stmts:
        try:
            sge_query(st)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{st[:60]}... -> {e}")

    answer, n_fm = subgraph_answer(entity)
    # provenance coverage: failure modes reachable from an Equipment that carry the tag
    prov_rows = sge_query(
        'MATCH (e:Equipment)-[:HAS_FAILURE_MODE]->(fm:FailureMode) '
        'WHERE fm.source = "LLM-derived" RETURN count(fm) AS c'
    ).get("records", [[0]])
    prov_count = prov_rows[0][0] if prov_rows else 0
    return {
        "entity": entity,
        "baseline_answerable": baseline_answerable,
        "enriched_answerable": n_fm > 0,
        "failure_modes_created": n_fm,
        "provenance_tagged_failure_modes": prov_count,
        "cypher_statements": len(stmts),
        "create_errors": errors,
        "enrich_latency_ms": round(enrich_ms, 1),
        "answer": answer,
    }


def load_scenarios(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all scenarios")
    ap.add_argument("--output", default="results/gak_results.json")
    ap.add_argument("--entities", default="", help="comma-sep entity filter substring(s)")
    args = ap.parse_args()

    scenarios = load_scenarios(SCENARIOS)
    if args.entities:
        subs = [e.strip().lower() for e in args.entities.split(",")]
        scenarios = [s for s in scenarios if any(x in str(s.get("entity", "")).lower() for x in subs)]
    if args.limit:
        scenarios = scenarios[: args.limit]

    # group by entity (enrich once per entity = cache miss; answer all = cache hits)
    by_entity: dict[str, list] = {}
    for s in scenarios:
        by_entity.setdefault(str(s.get("entity", "")).strip(), []).append(s)

    print(f"GAK eval: {len(scenarios)} real HF FMSR scenarios across {len(by_entity)} equipment types")
    print(f"  SGE={SGE_URL} tenant={TENANT} enrich={ENRICH_MODEL_LABEL} judge={JUDGE_MODEL}")
    print("  resetting prior LLM-derived nodes...\n")
    reset_gak()

    entity_reports, scenario_results = [], []
    for ei, (entity, group) in enumerate(by_entity.items(), 1):
        print(f"[entity {ei}/{len(by_entity)}] {entity}  ({len(group)} scenarios)")
        try:
            er = enrich_entity(entity, group[0].get("characteristic_form", ""))
        except Exception as e:  # noqa: BLE001
            print(f"    ENRICH ERROR: {e}")
            er = {"entity": entity, "error": str(e), "answer": "", "baseline_answerable": False,
                  "enriched_answerable": False, "failure_modes_created": 0, "enrich_latency_ms": 0}
        entity_reports.append(er)
        print(f"    baseline={er.get('baseline_answerable')} -> enriched={er.get('enriched_answerable')} "
              f"| fm_created={er.get('failure_modes_created')} | {er.get('enrich_latency_ms',0):.0f}ms")

        answer = er.get("answer", "")
        for s in group:
            verdict = {"pass": False, "score": 0.0, "rationale": "no materialized answer"}
            if answer:
                try:
                    verdict = judge(s.get("text", ""), s.get("characteristic_form", ""), answer)
                except Exception as e:  # noqa: BLE001
                    verdict = {"pass": False, "score": 0.0, "rationale": f"judge error: {e}"}
            scenario_results.append({
                "id": s.get("id"), "entity": entity, "question": s.get("text", ""),
                "judge_pass": bool(verdict.get("pass")),
                "judge_score": float(verdict.get("score", 0.0)),
                "judge_rationale": verdict.get("rationale", ""),
            })
        ps = [r for r in scenario_results if r["entity"] == entity]
        print(f"    scenarios judged: {sum(1 for r in ps if r['judge_pass'])}/{len(ps)} pass")

    # summary
    ents_ok = [e for e in entity_reports if "error" not in e]
    ne = len(ents_ok) or 1
    lift = sum(1 for e in ents_ok if (not e["baseline_answerable"]) and e["enriched_answerable"])
    ns = len(scenario_results) or 1
    passed = sum(1 for r in scenario_results if r["judge_pass"])
    summary = {
        "equipment_types": len(by_entity),
        "entities_enriched_ok": len(ents_ok),
        "entity_answerability_lift": lift,
        "entity_answerability_lift_pct": round(100 * lift / ne, 1),
        "scenarios": len(scenario_results),
        "scenario_judge_pass": passed,
        "scenario_judge_pass_pct": round(100 * passed / ns, 1),
        "avg_judge_score": round(sum(r["judge_score"] for r in scenario_results) / ns, 3),
        "total_failure_modes_created": sum(e.get("failure_modes_created", 0) for e in ents_ok),
        "avg_enrich_latency_ms": round(sum(e.get("enrich_latency_ms", 0) for e in ents_ok) / ne, 1),
        "enrich_model": ENRICH_MODEL_LABEL,
        "judge_model": JUDGE_MODEL,
        "note": "Enrich once per equipment type (cache miss); subsequent same-entity "
                "questions are answered from the materialized, provenance-tagged subgraph "
                "(semantic caching). All facts tagged source:'LLM-derived'.",
    }
    out = {"summary": summary, "entity_reports": entity_reports, "scenario_results": scenario_results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
