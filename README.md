# AssetOps Knowledge Graph

**12,647 nodes. 12,629 edges. IBM AssetOpsBench at 99% accuracy -- deterministic graph queries, zero LLM tokens.**

<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue" alt="License"></a>

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

## Quick Start

### Load from snapshot (recommended)

```bash
# Download (475 KB)
curl -LO https://github.com/samyama-ai/samyama-graph/releases/download/kg-snapshots-v5/assetops.sgsnap

# Start Samyama and import
./target/release/samyama
curl -X POST http://localhost:8080/api/tenants \
  -H 'Content-Type: application/json' \
  -d '{"id":"assetops","name":"AssetOps KG"}'
curl -X POST http://localhost:8080/api/tenants/assetops/snapshot/import \
  -F "file=@assetops.sgsnap"
```

### Build from source and benchmark

```bash
git clone https://github.com/samyama-ai/assetops-kg.git && cd assetops-kg
git clone https://github.com/IBM/AssetOpsBench.git ../AssetOpsBench
pip install -e ".[dev]"
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
