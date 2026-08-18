# AssetOps Knowledge Graph

**12,647 nodes. 12,662 edges.** Under a fixed harness (IBM's agent, its MCP servers, CouchDB, gpt-4o, a single grader), swapping *only* the data layer — flat documents → typed graph — lifts task pass rate **66.3% → 84.2% (+17.9pp)** on the 95 data-path-matched scenarios (**+23.0** across all 139). Reported in-sample; full accounting in [`docs/AUDIT-2026-07-13.md`](docs/AUDIT-2026-07-13.md).

> Part of the **Samyama** ecosystem — loaded into and queried via the graph engine at [samyama-ai/samyama-graph](https://github.com/samyama-ai/samyama-graph).
> This repo holds the loader and source-data specifics for the KG.

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue" alt="License"></a>

![AssetOps failure-impact demo](demo/assetops.gif)

## Demo

A narrated walkthrough (load the ISO 14224 + ISA-95 asset graph → most-critical
equipment → DEPENDS_ON failure-impact propagation → triage of priority-1 work
orders raised by high-severity anomalies):

```bash
python -m demo.demo                                                            # run live
asciinema rec --overwrite --cols 92 --rows 32 --idle-time-limit 2.0 \
  -c "bash -c 'source ~/projects/venv/bin/activate && PYTHONUNBUFFERED=1 python -m demo.demo'" \
  demo/assetops.cast                                                           # re-record
agg demo/assetops.cast demo/assetops.gif                                       # convert
```

### Generation-Augmented Knowledge (GAK)

![Generation-Augmented Knowledge demo](demo/assetops_gak.gif)

When the graph *doesn't* have an asset type — here an **electric motor**, absent from the
chiller+AHU graph — the engine's LLM agent writes the missing failure modes **into** the graph
as provenance-tagged nodes (`source:"LLM-derived"`), which the re-query then answers
deterministically. It's the inverse of RAG (write structured facts in, vs. retrieve text out) —
Architecture D from the VLDB 2026 paper.

```bash
python -m demo.demo_gak                                                         # run live (needs SGE + an enrich LLM on the tenant)
agg demo/assetops_gak.cast demo/assetops_gak.gif                                # convert
```

---

On flat document stores IBM report ~65% on AssetOpsBench — but that is their model, harness, and grader (a *cited* figure, not a controlled baseline). Our own document baseline, run under one fixed harness, is 66.3%. We loaded the same data into a knowledge graph and asked:

> *"What equipment is affected if Chiller 6 fails?"*

```cypher
MATCH (e:Equipment {name: 'Chiller-6'})<-[:DEPENDS_ON*1..3]-(downstream:Equipment)
RETURN downstream.name, downstream.criticality_score
ORDER BY downstream.criticality_score DESC
```

| Equipment | Criticality |
|-----------|-------------|
| AHU-3 | 0.92 |
| CRAC-2 | 0.88 |
| AHU-7 | 0.85 |

> The `DEPENDS_ON` topology and `criticality_score` shown above are an analytical layer we add on top of IBM's data (paper §3.2), so this query runs on the *extended* graph; the base graph loaded directly from IBM's sources has **9 node labels and 5 edge types**. Across the 139 IBM scenarios, an instrumented run shows 86 deterministic answers come from a live graph query and 53 from domain-knowledge handlers — see [`docs/information-leakage-analysis.md`](docs/information-leakage-analysis.md).

A separate **deterministic** graph tier answers **137/139** in ~63 ms with zero LLM tokens — but it is tuned in-sample and leans on hardcoded fallbacks (an empty-graph control reproduces most of it), so we report it *with* that control, never as a standalone headline. The defensible, model-held-fixed result is the **+17.9pp** data-layer lift above. Powered by [Samyama Graph](https://github.com/samyama-ai/samyama-graph).

---

## Results

Under one fixed harness — IBM's agent + MCP servers, CouchDB, **gpt-4o**, a single grader — varying **only** the data layer. All figures are in-sample; a held-out split is [tracked as G-02](https://git.samyama.ai/Samyama.ai/enterprise-benchmarks/issues/2).

| Data layer (same harness / model / grader) | Pass rate — 95 matched | Pass rate — all 139 |
|---|---|---|
| Flat documents — Architecture A | 66.3% | 58.3% |
| Typed graph, LLM→Cypher — Architecture B | **84.2%** | **81.3%** |
| **Data-layer lift** | **+17.9pp** | **+23.0pp** |

The 95 "matched" scenarios are those with a data path on both sides; the 44 excluded need an alert/anomaly server or absent-asset data neither side can reach (exclusion is symmetric).

**Read the caveats, not just the lift:**
- IBM's ~65% is a *cited* figure from a different model/harness/grader — **not** a controlled row here, and not directly comparable.
- The deterministic 137/139 (~99%, ~63 ms, 0 tokens) tier is tuned in-sample and mostly hardcoded fallbacks; an empty-graph control reproduces it. Reported only beside that control.
- A **no-data** LLM already scores **85%** on the "graph-native" scenarios — only vector-similarity (50%) and PageRank-criticality (60%) genuinely require the graph.

Full accounting and the retraction of the earlier "65% → 82% → 99% / structurally impossible" framing: [`docs/AUDIT-2026-07-13.md`](docs/AUDIT-2026-07-13.md).

## Schema

**9 node labels** -- Equipment, Sensor, FailureMode, WorkOrder, Location, Site, Event, AnomalyEvent, AlertEvent

**5 edge types** -- CONTAINS_LOCATION, CONTAINS_EQUIPMENT, HAS_SENSOR, FOR_EQUIPMENT, MONITORS

**Data source** -- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) (139 scenarios, 9 asset classes)

## Documentation

New here? Start with the guides:

| Guide | What it covers |
|-------|----------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | prerequisites (Python ≥ 3.10) · install · run the engine (Docker) · load the graph · first query |
| **[docs/QUERYING.md](docs/QUERYING.md)** | ask questions via the **HTTP API** or the **Samyama CLI** |

## Quick Start

**Full walkthrough → [GETTING_STARTED.md](GETTING_STARTED.md).** Needs **Python ≥ 3.10** and **Docker**.

### Build from source (the working path)

The graph is built from IBM's AssetOpsBench dataset — clone it alongside this repo:

```bash
pip install -r requirements.txt
docker run --rm -p 8080:8080 -p 6379:6379 public.ecr.aws/f9f6l5u4/samyama-graph:1.1.0

git clone https://github.com/IBM/AssetOpsBench.git ../AssetOpsBench
curl -X POST http://localhost:8080/api/tenants -H 'Content-Type: application/json' -d '{"id":"assetops","name":"AssetOps KG"}'
python -m etl.loader --data-dir ../AssetOpsBench --url http://localhost:8080 --graph assetops
```

### Snapshot — currently unavailable

The previously-documented `releases/download/kg-snapshots-v5/assetops.sgsnap` **returns 404** (no snapshot
is published yet). Use build-from-source above until one is published; see [GETTING_STARTED.md](GETTING_STARTED.md).

### Benchmarks

```bash
python -m benchmark.run_ibm_scenarios --data-dir ../AssetOpsBench   # deterministic tier (137/139, in-sample; see audit)
python -m benchmark.run_samyama                                      # 100%
```

## Example Queries

```cypher
-- Dependency chain: what breaks if this equipment fails?
MATCH (e:Equipment {name: 'Chiller-6'})<-[:DEPENDS_ON*1..3]-(downstream:Equipment)
RETURN downstream.name, downstream.criticality_score
ORDER BY downstream.criticality_score DESC

-- Failure modes monitored by sensors
MATCH (s:Sensor)<-[:HAS_SENSOR]-(e:Equipment)<-[:MONITORS]-(fm:FailureMode)
RETURN e.name, fm.name, s.type, fm.severity
ORDER BY fm.severity DESC
```

## Links

| | |
|---|---|
| Samyama Graph | [github.com/samyama-ai/samyama-graph](https://github.com/samyama-ai/samyama-graph) |
| The Book | [samyama-ai.github.io/samyama-graph-book](https://samyama-ai.github.io/samyama-graph-book/) |
| IBM AssetOpsBench | [github.com/IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench) |
| Contact | [samyama.dev/contact](https://samyama.dev/contact) |

## License

Apache 2.0
