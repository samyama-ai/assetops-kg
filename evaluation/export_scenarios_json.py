#!/usr/bin/env python3
"""Export the per-scenario drill-down the app's scenario explorer reads.

The audit's headline is a single number (+17.9). This file is how a reviewer gets from
that number to the 139 individual scenarios under it: the question, what Architecture A
answered over IBM's documents, what Architecture B answered over the graph, THE CYPHER B
ACTUALLY GENERATED, what the grader said about each, and -- if the scenario is one of the
44 excluded from the clean comparison -- why.

Principle 0 applies unchanged: every field here is read from a committed result file.
Nothing is transcribed, and nothing is invented for scenarios that lack it.

Separate from audit.json on purpose: audit.json is ~22 KB and every page load needs it;
this is ~1 MB of answer text that only the explorer needs.

Run:  python -m evaluation.export_scenarios_json
Emits: audit-insights/public/data/scenarios.json
"""
from __future__ import annotations

import collections
import json
import pathlib
import re

# The exclusion rule is IMPORTED, not restated -- if the audit's definition of a
# contaminated scenario changes, the explorer changes with it and cannot drift from the
# headline it drills into.
from evaluation.paper3_audit_numbers import is_contaminated

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"

_APP = ROOT.parent / "audit-insights" / "public" / "data" / "scenarios.json"
OUT = _APP if _APP.parent.exists() else ROOT / "_staging" / "scenarios.json"

SCHEMA_VERSION = 2

A_FILE = "arch_a_gpt4o.json"
B_FILE = "repro_nlq_gpt4o_2026-06-29.json"

# Equipment class keywords, taken from the ETL's own controlled vocabulary
# (etl/hf_loader.py `_ENTITY_TO_CLASS`). This axis is DERIVED from question text, so the
# app must show it as such: the keyword list is published in the payload, and anything
# unmatched or ambiguous goes to an explicit bucket rather than being guessed into a
# class. There is no industry/sector field anywhere in this benchmark; we do not invent
# one.
EQUIPMENT_KEYWORDS = {
    "chiller": ["chiller"],
    "ahu": ["ahu", "air handling"],
    "compressor": ["compressor"],
    "pump": ["pump"],
    "motor": ["motor"],
    "turbine": ["turbine"],
    "engine": ["engine"],
    "bearing": ["bearing"],
    "fan": [" fan", "fan "],
    "boiler": ["boiler"],
    "generator": ["generator"],
    "transformer": ["transformer"],
    "cooling_tower": ["cooling tower"],
    "hxu": ["heat exchanger", "hxu"],
    "crac": ["crac"],
    "gearbox": ["gearbox"],
    "rotor": ["rotor"],
}


def load(name):
    return json.loads((R / name).read_text())


def equipment_of(text: str) -> tuple[str, list[str]]:
    """Derive an equipment class from question text. Returns (class, all_matches).

    Honest by construction: no match -> "unclassified"; more than one -> "ambiguous".
    Both are real buckets the UI shows, not silently-dropped rows.
    """
    t = (text or "").lower()
    hits = sorted({k for k, kws in EQUIPMENT_KEYWORDS.items() if any(w in t for w in kws)})
    if not hits:
        return "unclassified", []
    if len(hits) > 1:
        return "ambiguous", hits
    return hits[0], hits


def build() -> dict:
    a_raw = load(A_FILE)
    A = {r["id"]: r for r in a_raw["results"]}
    B = {r["id"]: r for r in load(B_FILE)}

    # gak carries the ONE explicit equipment label in the repo: `entity`, on the 88 fmsr
    # scenarios, joined by integer id (verified 88/88). It does not reach the 139.
    gak_entity = {r["id"]: r["entity"] for r in load("gak_full.json")["scenario_results"]}

    rows = []
    for sid in sorted(A):
        a, b = A[sid], B.get(sid)
        reason = is_contaminated(a)
        eq, eq_hits = equipment_of(a.get("question"))
        nd = (b or {}).get("nlq_details") or {}
        cy = nd.get("cypher_generated") or []
        cy_res = nd.get("cypher_results") or []
        rows.append({
            "id": sid,
            "type": a.get("type"),
            "category": a.get("category"),
            "question": a.get("question"),
            # the 44: why this scenario is not in the clean 95
            "excluded": reason is not None,
            "exclusion_reason": reason,
            # derived axis -- flagged as derived in the payload's meta
            "equipment": eq,
            "equipment_matches": eq_hits,
            "a": {
                "passed": a.get("passed"),
                "score": a.get("score"),
                "latency_ms": a.get("latency_ms"),
                "response": a.get("response"),
                "rationale": a.get("rationale"),
                "error": a.get("error"),
            },
            "b": None if b is None else {
                "passed": b.get("passed"),
                "score": b.get("score"),
                "latency_ms": b.get("latency_ms"),
                "response": b.get("response"),
                "rationale": b.get("rationale"),
                "error": b.get("error"),
                # the trace that makes the graph side auditable: the query it wrote,
                # whether it ran, and how many rows came back.
                "cypher": cy,
                "cypher_results": cy_res,
                "rows_returned": sum(c.get("record_count") or 0 for c in cy_res),
                "tokens_total": nd.get("tokens_total"),
                "retries": nd.get("retries"),
                "model": nd.get("model"),
            },
            # both sides agreed / disagreed -- the fastest way to find the interesting ones
            "flip": (None if b is None else
                     "b_only" if b.get("passed") and not a.get("passed") else
                     "a_only" if a.get("passed") and not b.get("passed") else
                     "both" if a.get("passed") else "neither"),
        })

    eq_counts = collections.Counter(r["equipment"] for r in rows)
    flip_counts = collections.Counter(r["flip"] for r in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_note": ("Generated by evaluation/export_scenarios_json.py from committed "
                           "result files. No number or answer text here is transcribed."),
        "generated_from": f"assetops-kg/results/{A_FILE} + {B_FILE}",
        "n": len(rows),
        "a_label": "Architecture A -- IBM's agent over IBM's documents",
        "b_label": "Architecture B -- the same model over the typed graph",
        "facets": {
            "type": dict(collections.Counter(r["type"] for r in rows).most_common()),
            "category": dict(collections.Counter(r["category"] for r in rows).most_common()),
            "equipment": dict(eq_counts.most_common()),
            "flip": dict(flip_counts.most_common()),
            "exclusion_reason": dict(collections.Counter(
                r["exclusion_reason"] for r in rows if r["exclusion_reason"]).most_common()),
        },
        "facets_meta": {
            "type": {"derived": False, "desc": "IBM's own scenario type."},
            "category": {"derived": False, "desc": "IBM's own question category."},
            "equipment": {
                "derived": True,
                "desc": ("Derived by keyword match over the question text, using the ETL's "
                         "own vocabulary. This benchmark has no equipment field on these 139 "
                         "scenarios and no industry field anywhere, so this axis is inferred "
                         "-- 'unclassified' and 'ambiguous' are real buckets, not dropped rows."),
                "keywords": EQUIPMENT_KEYWORDS,
                "explicit_elsewhere": ("The 88 fmsr scenarios DO carry an explicit `entity` "
                                       "label in gak_full.json; those 88 are not in this 139-set."),
            },
            "flip": {"derived": True,
                     "desc": "Computed from the two graded runs: who passed this scenario."},
            "exclusion_reason": {
                "derived": True,
                "desc": ("From the audit's own is_contaminated() rule, imported rather than "
                         "restated. Note the second rule matches the agent's own 'not found' "
                         "response text, not the data structurally."),
            },
        },
        "gak_entity_note": (f"{len(gak_entity)} fmsr scenarios carry an explicit equipment "
                            f"entity, but on a different scenario set than these {len(rows)}."),
        "scenarios": rows,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)  {data['n']} scenarios")
    print(f"  flips: {data['facets']['flip']}")
    print(f"  equipment: {data['facets']['equipment']}")
    print(f"  excluded: {sum(data['facets']['exclusion_reason'].values())}")


if __name__ == "__main__":
    main()
