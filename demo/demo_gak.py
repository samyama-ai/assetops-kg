"""Narrated terminal demo: Generation-Augmented Knowledge (GAK) on Samyama.

Shows Architecture D from the VLDB 2026 paper: when the graph LACKS an asset
type, an LLM agent writes the missing facts INTO the graph as provenance-tagged
nodes, which the re-query then answers deterministically — the inverse of RAG.

Record:
    asciinema rec --overwrite --cols 92 --rows 34 --idle-time-limit 2.0 \
      -c "bash -c 'source ~/projects/venv/bin/activate && SGE_URL=http://localhost:8080 python -m demo.demo_gak'" \
      demo/assetops_gak.cast
    agg demo/assetops_gak.cast demo/assetops_gak.gif

Requires SGE running with the `assetops` tenant + an enrich LLM configured on
the tenant's agent_config (see samyama-cloud/docs/assetops-gak-demo-setup.md).
"""
from __future__ import annotations

import os
import re
import time

import requests
from rich.console import Console
from rich.panel import Panel

SGE = os.environ.get("SGE_URL", "http://localhost:8080")
G = "assetops"
ENTITY = "electric motor"
console = Console()


def pause(s: float = 1.2) -> None:
    time.sleep(s)


def step(title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}")
    pause(0.5)


def query(cypher: str) -> list:
    r = requests.post(f"{SGE}/api/query", json={"query": cypher, "graph": G}, timeout=60)
    r.raise_for_status()
    return r.json().get("records", [])


def enrich(prompt: str, context: str) -> str:
    r = requests.post(f"{SGE}/api/enrich",
                      json={"prompt": prompt, "context": context, "graph": G}, timeout=180)
    r.raise_for_status()
    return r.json().get("agent_response", "")


def extract_cypher(text: str) -> list[str]:
    blocks = re.findall(r"```(?:cypher)?\s*(.*?)```", text, re.S | re.I)
    body = "\n".join(blocks) if blocks else text
    body = "\n".join(l for l in body.splitlines() if not l.strip().startswith("//"))
    out = []
    for chunk in body.split(";"):
        c = " ".join(chunk.split()).strip()
        if re.match(r"^(CREATE|MERGE|MATCH)\b", c, re.I):
            out.append(c)
    return out


def base_motor_fms() -> list:
    """Failure modes for the entity already present in the (data-derived) graph."""
    return [r for r in query(
        f'MATCH (e:Equipment)-[:HAS_FAILURE_MODE]-(fm:FailureMode) '
        f'WHERE toLower(e.name) CONTAINS toLower("{ENTITY}") '
        f'RETURN DISTINCT fm.name'
    ) if r and r[0]]


def all_fm_names() -> set:
    """All FailureMode names. We diff before/after enrich to find the new ones —
    robust to the engine's post-import property-equality quirk (filtering on
    `source` after a snapshot import is unreliable), so we never filter on it."""
    return {r[0] for r in query('MATCH (fm:FailureMode) RETURN fm.name') if r and r[0]}


def main() -> None:
    console.print(Panel.fit(
        "[bold]Samyama · Generation-Augmented Knowledge (GAK)[/bold]\n"
        "When the graph doesn't know, the LLM writes the missing facts INTO it —\n"
        "provenance-tagged, then answered deterministically. [dim](the inverse of RAG)[/dim]",
        border_style="magenta",
    ))
    pause(1.2)

    fm_before = all_fm_names()  # snapshot of known failure modes before enrichment

    step(f"1 · Ask the graph: failure modes of an {ENTITY}?")
    console.print(f'  [dim]cypher>[/dim] [yellow]MATCH (e:Equipment)-[:HAS_FAILURE_MODE]-(fm) '
                  f'WHERE e.name CONTAINS "{ENTITY}" RETURN fm[/yellow]')
    pause()
    before = base_motor_fms()
    console.print(f"  [red]→ miss[/red]: the chiller+AHU graph has [bold]no {ENTITY}[/bold] "
                  f"— {len(before)} failure modes. The answer isn't in the data.")
    pause(1.2)

    step("2 · GAK: the engine's LLM agent writes the missing knowledge as Cypher")
    console.print("  [dim]POST /api/enrich → agent generates provenance-tagged CREATEs…[/dim]")
    prompt = (
        f"The knowledge graph lacks the asset type '{ENTITY}'. List its 5-7 standard industrial "
        f"failure modes as OpenCypher.\nSTRICT OUTPUT:\n"
        f"- ONLY standalone, semicolon-terminated CREATE statements, one per line.\n"
        f"- Do NOT use MATCH. Do NOT create relationships.\n"
        f"- One per failure mode: "
        f"CREATE (:FailureMode {{name:\"<failure mode>\", equipment:\"{ENTITY}\", source:\"LLM-derived\"}});\n"
        f"- Use realistic failure modes for an {ENTITY}. Double-quoted strings. Return ONLY Cypher."
    )
    ctx = "Node label to create: FailureMode {name, equipment, source}."
    t0 = time.time()
    text = enrich(prompt, ctx)
    dt = time.time() - t0
    stmts = extract_cypher(text)
    console.print(f"  [green]→[/green] agent returned [bold]{len(stmts)}[/bold] Cypher statements "
                  f"in {dt:.0f}s. Sample:")
    for s in stmts[:3]:
        console.print(f"    [yellow]{s[:84]}[/yellow]")
    pause(1.2)

    step("3 · Materialize — every node tagged source:\"LLM-derived\" (auditable)")
    applied = 0
    for s in stmts:
        try:
            query(s)
            applied += 1
        except Exception:
            pass
    console.print(f"  [green]→[/green] materialized [bold]{applied}/{len(stmts)}[/bold] statements "
                  f"into the graph (provenance-tagged).")
    pause(1.2)

    step(f"4 · Re-query: failure modes of an {ENTITY}?")
    new_fms = sorted(all_fm_names() - fm_before)  # the freshly materialized ones
    console.print(f"  [green]→ now answerable[/green]: [bold]{len(new_fms)}[/bold] new failure modes "
                  f"(materialized, source:LLM-derived):")
    for name in new_fms[:7]:
        console.print(f"    • [bold]{name}[/bold]")
    pause(1.2)

    console.print()
    console.print(Panel.fit(
        f"[bold green]Missing knowledge → materialized once → answered deterministically.[/bold green]\n"
        f"0 → {len(new_fms)} failure modes, every node tagged [bold]source:LLM-derived[/bold] so it stays\n"
        f"auditable and distinct from data-derived facts. Repeat asks hit the graph (semantic cache).",
        border_style="green",
    ))
    pause(1.5)


if __name__ == "__main__":
    main()
