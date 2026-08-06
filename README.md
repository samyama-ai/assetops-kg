# AssetOps Knowledge Graph

**12,647 nodes. 12,662 edges. IBM AssetOpsBench at 99% accuracy -- deterministic graph queries, zero LLM tokens.**

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

IBM's GPT-4 agents score 65% on their own AssetOpsBench using flat document stores. We loaded the same data into a knowledge graph and asked:

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

**137/139 scenarios passing. 63ms average. Zero tokens.** The bottleneck was the data model, not the LLM. Powered by [Samyama Graph](https://github.com/samyama-ai/samyama-graph).

---

## Results

| Approach | Pass Rate | Avg Latency | Tokens |
|----------|-----------|-------------|--------|
| GPT-4 + flat docs (IBM) | 91/139 (65%) | not reported | not reported |
| GPT-4 + graph NLQ | 114/139 (82%) | ~5,800 ms | ~4,600/scenario |
| **Deterministic (graph)** | **137/139 (99%)** | **63 ms** | **0** |

Same model (GPT-4), same data, +17pp improvement -- proving the gain comes from the data model.

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
python -m benchmark.run_ibm_scenarios --data-dir ../AssetOpsBench   # 99%
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
