"""G-07 — Architecture SQL: text-to-SQL over the relational mirror.

The reviewers' central question: is the graph's advantage over documents really
about the *graph*, or just about having a *structured backend*? This runs the
same scenarios through an LLM→SQL tier over the SQLite mirror (identical data to
the graph, built by build_sqlite_mirror.py), scored by the SAME grader as Arch A
and Arch B. Comparing Arch SQL vs Arch B isolates graph-vs-relational; comparing
Arch SQL vs Arch A (documents) shows how much of the lift is "structure" alone.

    OPENAI_API_KEY=... /home/vm-1/projects/venv/bin/python3 -m benchmark.run_sql \
        --data-dir <staged> --db results/assetops_mirror.sqlite \
        --ids-file scenarios/heldout_split.json --output results/heldout_sql_gpt4o.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import openai

from benchmark.run_ibm_scenarios import evaluate_scenario, load_ibm_scenarios

SQL_SYSTEM = (
    "You translate an industrial-maintenance question into ONE SQLite query over the "
    "schema below, then nothing else. Return only the SQL, no prose, no markdown fence.\n\n"
    "SCHEMA:\n{schema}\n\n"
    "Notes: equipment_id links workorder/alertevent/event/sensor to equipment; "
    "anomalyevent.asset_name is the equipment name (lowercase); "
    "sensor_monitors_failuremode links sensors to failure modes."
)


def extract_sql(text: str) -> str:
    text = re.sub(r"```(?:sql)?", "", text).strip().rstrip(";")
    return text


def answer_via_sql(client, model, schema, question, db) -> tuple[str, str]:
    """Return (response_text, sql). One retry, feeding the error back."""
    messages = [{"role": "system", "content": SQL_SYSTEM.format(schema=schema)},
                {"role": "user", "content": question}]
    sql = ""
    for attempt in range(2):
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
        sql = extract_sql(resp.choices[0].message.content)
        try:
            rows = db.execute(sql).fetchall()
            cols = [d[0] for d in db.execute(sql).description or []]
            if not rows:
                return "No rows returned.", sql
            # Compact textual rendering the keyword grader can read.
            lines = [", ".join(f"{c}={v}" for c, v in zip(cols, r)) for r in rows[:50]]
            return f"{len(rows)} row(s):\n" + "\n".join(lines), sql
        except Exception as e:  # noqa: BLE001
            messages.append({"role": "assistant", "content": sql})
            messages.append({"role": "user",
                             "content": f"That query errored: {e}. Return a corrected SQLite query only."})
    return f"SQL error after retry: {sql}", sql


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--db", default="results/assetops_mirror.sqlite")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--output", default="results/sql_gpt4o.json")
    a = ap.parse_args()

    scenarios = load_ibm_scenarios(a.data_dir, None)
    if a.ids_file:
        keep = set(json.loads(Path(a.ids_file).read_text())["ids"])
        scenarios = [s for s in scenarios if s["id"] in keep]
    schema = Path(a.db).with_suffix(".schema.txt").read_text()
    db = sqlite3.connect(a.db)
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Architecture SQL: {len(scenarios)} scenarios, model={a.model}")
    results = []
    for i, s in enumerate(scenarios, 1):
        t0 = time.time()
        response, sql = answer_via_sql(client, a.model, schema, s["text"], db)
        passed, score, rationale = evaluate_scenario(s, response)
        results.append({"id": s["id"], "type": s.get("type"), "passed": passed,
                        "score": score, "sql": sql, "response": response[:400],
                        "latency_ms": round((time.time() - t0) * 1000)})
        print(f"  [{i}/{len(scenarios)}] id={s['id']} {'PASS' if passed else 'FAIL'} ({score:.2f})")

    npass = sum(r["passed"] for r in results)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nArch SQL: {npass}/{len(results)} = {100*npass/len(results):.1f}% -> {a.output}")


if __name__ == "__main__":
    main()
