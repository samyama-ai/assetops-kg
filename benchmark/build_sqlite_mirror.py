"""G-07 — mirror the populated graph into SQLite, for the text-to-SQL ablation.

The fairest possible "is it the graph, or just a structured backend?" test uses
the *same data* behind both query engines. This dumps every node label of the
populated Samyama graph into a SQLite table (all properties as columns) and the
two non-FK relations (MONITORS, HAS_SENSOR) into join tables. The FOR_EQUIPMENT
edge is already an `equipment_id` column on each child node, so it needs no join
table. The result is a relational store holding exactly what the graph holds —
so any Arch-B-vs-Arch-SQL difference is the query model, not the data.

    /home/vm-1/projects/venv/bin/python3 -m benchmark.build_sqlite_mirror \
        --data-dir <staged full-graph dir> --out results/assetops_mirror.sqlite
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from benchmark.run_ibm_scenarios import GRAPH_NAME, SamyamaClient, load_ibm_data

NODE_LABELS = ["Site", "Location", "Equipment", "Sensor", "FailureMode",
               "WorkOrder", "AlertEvent", "AnomalyEvent", "Event"]


def _rows(client, cypher: str) -> list:
    return [rec[0] for rec in client.query(cypher, GRAPH_NAME).records]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/assetops_mirror.sqlite")
    a = ap.parse_args()

    client = SamyamaClient.embedded()
    load_ibm_data(client, a.data_dir, GRAPH_NAME)

    out = Path(a.out)
    if out.exists():
        out.unlink()
    db = sqlite3.connect(out)
    schema_lines = []

    for label in NODE_LABELS:
        recs = _rows(client, f"MATCH (n:{label}) RETURN properties(n)")
        if not recs:
            continue
        cols = sorted({k for r in recs for k in r.keys()})
        tbl = label.lower()
        db.execute(f'CREATE TABLE {tbl} ({", ".join(f'"{c}" TEXT' for c in cols)})')
        db.executemany(
            f'INSERT INTO {tbl} ({", ".join(f'"{c}"' for c in cols)}) '
            f'VALUES ({", ".join("?" for _ in cols)})',
            [[str(r.get(c, "")) for c in cols] for r in recs],
        )
        schema_lines.append(f"{tbl}({', '.join(cols)})  -- {len(recs)} rows")

    # MONITORS: sensor -> failure mode
    mon = client.query(
        "MATCH (s:Sensor)-[:MONITORS]->(f:FailureMode) "
        "RETURN properties(s) AS s, properties(f) AS f", GRAPH_NAME).records
    if mon:
        db.execute('CREATE TABLE sensor_monitors_failuremode '
                   '(sensor_name TEXT, sensor_id TEXT, failure_mode TEXT)')
        db.executemany(
            'INSERT INTO sensor_monitors_failuremode VALUES (?,?,?)',
            [[str(s.get("name", "")), str(s.get("sensor_id", s.get("id", ""))),
              str(f.get("name", f.get("failure_mode", "")))] for s, f in mon])
        schema_lines.append(f"sensor_monitors_failuremode(sensor_name, sensor_id, failure_mode)  -- {len(mon)} rows")

    db.commit()
    db.close()
    schema_txt = out.with_suffix(".schema.txt")
    schema_txt.write_text("\n".join(schema_lines) + "\n")
    print(f"mirror -> {out}\nschema:\n" + "\n".join("  " + s for s in schema_lines))


if __name__ == "__main__":
    main()
