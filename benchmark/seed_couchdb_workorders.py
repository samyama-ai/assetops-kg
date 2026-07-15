#!/usr/bin/env python3
"""Give the document baseline the SAME work-order data the graph has.

Why
---
Our knowledge graph was built from AssetOpsBench's 4,248-row work-order file
(`etl/ibm_loader.py:8` -- "Work orders from workorders.csv (~4248 rows)").

IBM's current upstream ships a *2-row* sample at
`src/couchdb/scenarios_data/shared/work_order/workorders.csv`, in a different
(Maximo-style) schema. Seeding CouchDB from it and calling the result
"Architecture A" compares a graph holding 4,248 work orders against a document
store holding 2 -- a 2,000x asymmetry in OUR favour that would manufacture a
large data-layer effect and falsely confirm the paper.

This script loads the same 4,248 rows into CouchDB, mapped into the schema the
current WO MCP server reads.

Fidelity gate
-------------
The benchmark's own rubrics tell us the right answers:
  "work orders for CWC04013 in 2017"            -> 33
  "preventive work orders for CWC04013 in 2017" -> 31
The script asserts both AFTER loading. If they don't hold, the mapping is wrong
and it refuses to leave a bad database behind.

Usage:
    python -m benchmark.seed_couchdb_workorders
"""
from __future__ import annotations

import csv
import io
import subprocess
import sys

import requests

COUCH = "http://localhost:5984"
AUTH = ("admin", "password")
DB = "workorder"
AOB = "/home/vm-1/projects/graph_ws/AssetOpsBench"
SRC_REF = "graph-scenarios:aobench/datalayer/eamlite/db/data/workorders.csv"


def fetch_source() -> list[dict]:
    raw = subprocess.run(
        ["git", "show", SRC_REF], cwd=AOB, capture_output=True, text=True, check=True
    ).stdout
    return list(csv.DictReader(io.StringIO(raw)))


def to_iso(s: str) -> str:
    """'4/6/16 14:00' (M/D/YY H:MM) -> '2016-04-06T14:00:00+00:00'.

    workorders.py builds its date filter as a Mango range on `reportdate` and
    formats bounds with _iso() -> "%Y-%m-%dT%H:%M:%S+00:00". CouchDB compares
    those lexicographically, so a M/D/YY string can never match. Store ISO.
    """
    s = (s or "").strip()
    if not s:
        return ""
    try:
        d, t = (s.split(" ") + ["00:00"])[:2]
        m, day, y = (int(x) for x in d.split("/"))
        hh, mm = (int(x) for x in t.split(":"))
        return f"{2000 + y:04d}-{m:02d}-{day:02d}T{hh:02d}:{mm:02d}:00+00:00"
    except Exception:
        return ""


def to_maximo(r: dict) -> dict:
    """Map the old schema onto the fields the WO MCP server actually queries.

    Only fields the server reads are populated. We do NOT invent values for fields
    the benchmark never asks about (costs, labor, tasks) -- an empty field is
    honest; a fabricated one is not.

    CRITICAL: workorders.py:list_workorders() seeds every Mango selector with
    {"type": "workorder"}. A document without that field is INVISIBLE to every
    query -- the agent answers "no work orders found" and the baseline collapses.
    """
    preventive = (r.get("preventive") or "").strip().upper() == "TRUE"
    finish_iso = to_iso(r.get("actual_finish", ""))
    return {
        "_id": r["wo_id"],
        "type": "workorder",          # <-- required by every selector; without it, nothing matches
        "wonum": r["wo_id"],
        "assetnum": r["equipment_id"],
        "description": r.get("wo_description", ""),
        # The benchmark's WO questions turn on: which asset, which year,
        # preventive vs corrective, and priority.
        "worktype": "PM" if preventive else "CM",
        "preventive": preventive,
        "wopriority": int(r["work_priority"]) if (r.get("work_priority") or "").isdigit() else None,
        "actfinish": finish_iso,
        "reportdate": finish_iso,   # server filters on reportdate; source has only actual_finish
        "siteid": "MAIN",
        "status": "COMP",
        "location": r.get("equipment_name", ""),
        "primary_code": r.get("primary_code", ""),
        "primary_code_description": r.get("primary_code_description", ""),
        "collection": r.get("collection", ""),
        "duration": r.get("duration", ""),
        "actual_labor_hours": r.get("actual_labor_hours", ""),
    }


def main() -> None:
    rows = fetch_source()
    print(f"source work orders: {len(rows)}")

    docs = [to_maximo(r) for r in rows]

    requests.delete(f"{COUCH}/{DB}", auth=AUTH, timeout=30)
    requests.put(f"{COUCH}/{DB}", auth=AUTH, timeout=30).raise_for_status()
    res = requests.post(
        f"{COUCH}/{DB}/_bulk_docs", auth=AUTH, json={"docs": docs}, timeout=180
    )
    res.raise_for_status()
    errs = [d for d in res.json() if d.get("error")]
    n = requests.get(f"{COUCH}/{DB}", auth=AUTH, timeout=30).json()["doc_count"]
    print(f"loaded into CouchDB: {n} docs ({len(errs)} errors)")

    # ---- fidelity gate: the benchmark's own rubrics are the ground truth ----
    def count(pred) -> int:
        return sum(1 for d in docs if pred(d))

    y2017 = lambda d: d["actfinish"].startswith("2017-")  # noqa: E731
    got_all = count(lambda d: d["assetnum"] == "CWC04013" and y2017(d))
    got_pm = count(lambda d: d["assetnum"] == "CWC04013" and y2017(d) and d["preventive"])

    print()
    print("in the documents we wrote:")
    print(f"  CWC04013 / 2017            : {got_all}  (benchmark rubric expects 33)")
    print(f"  CWC04013 / 2017 preventive : {got_pm}  (benchmark rubric expects 31)")

    # Present-in-the-DB is NOT the same as reachable-by-the-server. The first version of
    # this script passed the count check while the WO server still saw ZERO work orders,
    # because every selector is seeded with {"type": "workorder"} and our docs lacked it.
    # So query CouchDB through the server's OWN selector shape.
    sel = {
        "selector": {
            "type": "workorder",
            "assetnum": "CWC04013",
            "reportdate": {"$gte": "2017-01-01T00:00:00+00:00",
                           "$lte": "2017-12-31T23:59:59+00:00"},
        },
        "limit": 1000000,
        "fields": ["wonum", "worktype"],
    }
    via = requests.post(f"{COUCH}/{DB}/_find", auth=AUTH, json=sel, timeout=120).json()
    docs_via = via.get("docs", [])
    via_all = len(docs_via)
    via_pm = sum(1 for d in docs_via if d.get("worktype") == "PM")

    print()
    print("through the WO server's own Mango selector (type+assetnum+reportdate range):")
    print(f"  CWC04013 / 2017            : {via_all}  (expects 33)")
    print(f"  CWC04013 / 2017 preventive : {via_pm}  (expects 31)")

    if (got_all, got_pm) != (33, 31) or (via_all, via_pm) != (33, 31):
        print("\nFIDELITY GATE FAILED -- the data is not reachable through the server's "
              "query path. Refusing to certify this database.", file=sys.stderr)
        sys.exit(1)
    print("\nFIDELITY GATE PASSED: the document store holds the same work-order data the "
          "graph does, AND the WO server can actually retrieve it.")


if __name__ == "__main__":
    main()
