# AssetOps-KG: Industrial Asset Operations Knowledge Graph

Extending [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) with graph database, vector search, and multi-objective optimization capabilities using [Samyama Graph Database](https://github.com/samyama-ai/samyama-graph).

## Key Results

| Benchmark | GPT-4 (IBM baseline) | Samyama-KG | Delta |
|---|---|---|---|
| IBM's 139 scenarios | 91/139 (65%) | **137/139 (99%)** | **+34pp** |
| Custom 40 scenarios | 34/40 (85%) | **40/40 (100%)** | **+15pp** |
| Avg latency | ~11,000 ms | **~110 ms** | **100x faster** |
| Token cost | ~1,600/scenario | **0** | **$0** |

Full analysis: [`docs/results.md`](docs/results.md)

## Thesis

AssetOpsBench shows GPT-4 completes only 65% of industrial maintenance tasks using flat document stores + LLM reasoning loops. We demonstrate that replacing flat storage with a **knowledge graph** (ISO 14224 + ISA-95 ontology) improves accuracy, cuts latency, and enables new query types:

- **Multi-hop dependency analysis** -- "If Chiller 6 fails, what downstream equipment is affected?"
- **Semantic failure similarity** -- "Which pumps had failures similar to Motor 3's bearing wear?"
- **Graph-based criticality ranking** -- PageRank on the equipment dependency network
- **Pareto-optimal maintenance scheduling** -- NSGA-II minimizing cost vs. downtime

## How It Works

```
IBM's approach:     Question → LLM → document search → LLM reasoning → answer
Our approach:       Question → handler routing → Cypher query → graph traversal → answer
```

The knowledge graph replaces LLM reasoning with deterministic graph traversal. Instead of asking GPT-4 to reason over JSON/CSV/YAML fragments, we run Cypher queries that traverse typed relationships (`DEPENDS_ON`, `MONITORS`, `HAS_SENSOR`) and return exact answers.

**Why this is better:**
1. **Deterministic** -- same query always returns the same result, no hallucination
2. **Exact counts** -- traverses Event nodes with date filtering vs. LLM counting across documents
3. **Relationship traversal** -- single edge hops vs. correlating across separate files
4. **New capabilities** -- vector similarity, PageRank, BFS cascade analysis, NSGA-II optimization

## Graph Schema

11 node labels, 16 edge types, 781 nodes, 955 edges (see [`schema/industrial_kg.cypher`](schema/industrial_kg.cypher)):

```
Site -[CONTAINS_LOCATION]-> Location -[CONTAINS_EQUIPMENT]-> Equipment -[HAS_SENSOR]-> Sensor
                                                              |
                                            DEPENDS_ON / SHARES_SYSTEM_WITH
                                                              |
FailureMode -[MONITORS]-> Equipment -[EXPERIENCED]-> FailureMode
WorkOrder -[FOR_EQUIPMENT]-> Equipment
WorkOrder -[ADDRESSES]-> FailureMode
WorkOrder -[USES_PART]-> SparePart -[SUPPLIED_BY]-> Supplier
Anomaly -[TRIGGERED]-> WorkOrder
Event -[FOR_EQUIPMENT]-> Equipment
```

## Project Structure

```
assetops-kg/
├── schema/                    # Graph schema (Cypher CREATE statements)
├── etl/                       # ETL pipeline (AssetOpsBench -> Samyama KG)
│   ├── loader.py              # Main orchestrator — custom 40 scenarios
│   ├── ibm_loader.py          # IBM data ETL — 8-step pipeline for 139 scenarios
│   ├── eamlite_loader.py      # EAMLite -> Site, Location, Equipment
│   ├── couchdb_loader.py      # CouchDB JSON -> Sensor + SensorReading
│   ├── fmsr_loader.py         # YAML -> FailureMode + MONITORS edges
│   └── embedding_gen.py       # sentence-transformers -> vector index
├── mcp_server/                # FastMCP server (9 tools)
│   ├── server.py              # MCP entry point
│   └── tools/
│       ├── asset_tools.py     # query_assets, query_sensors, query_sites
│       ├── failure_tools.py   # find_similar_failures, query_failure_modes
│       ├── impact_tools.py    # impact_analysis, dependency_chain
│       └── analytics_tools.py # criticality_ranking, maintenance_clusters
├── scenarios/                 # 40 new scenario JSONs (7 categories)
├── evaluation/                # 8-dimensional scoring framework
│   ├── extended_criteria.py   # 6 original + 2 graph-specific dimensions
│   └── runner.py              # Benchmark runner
├── benchmark/                 # Benchmark runners
│   ├── run_samyama.py         # Custom 40 scenarios
│   ├── run_baseline.py        # GPT-4o baseline for custom 40
│   └── run_ibm_scenarios.py   # IBM's original 139 scenarios
├── docs/
│   └── results.md             # Full benchmark analysis
├── results/                   # Benchmark result JSONs (v1-v5)
└── tests/
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run custom 40 scenarios (graph-native)
python -m etl.loader --data-dir ../AssetOpsBench/src/couchdb/sample_data
python -m benchmark.run_samyama

# Run IBM's 139 scenarios
python -m benchmark.run_ibm_scenarios --data-dir ../AssetOpsBench

# Start MCP server (for agent integration)
python -m mcp_server.server
```

## Benchmark Results

### IBM's Original 139 Scenarios

IBM's reported GPT-4 baseline: **~65% (91/139)**.

Samyama-KG: **99% (137/139), avg score 0.889.**

| Type | Count | Pass Rate | Avg Score |
|---|---|---|---|
| IoT | 20 | **20/20 (100%)** | 0.988 |
| FMSR | 40 | **40/40 (100%)** | 0.907 |
| TSFM | 23 | **23/23 (100%)** | 0.920 |
| Multi | 20 | **20/20 (100%)** | 0.877 |
| WO | 36 | 34/36 (94%) | 0.801 |

Only 2 failures remain (WO bundling algorithm edge cases).

### Custom 40 Scenarios (Graph-Native)

| Category | GPT-4o | Samyama-KG | Delta |
|---|---|---|---|
| Failure similarity | 3/6 (0.501) | **6/6 (0.902)** | +0.401 |
| Criticality analysis | 3/5 (0.566) | **5/5 (0.938)** | +0.372 |
| Root cause analysis | 5/5 (0.580) | **5/5 (0.934)** | +0.354 |
| Multi-hop dependency | 7/8 (0.618) | **8/8 (0.934)** | +0.316 |
| Maintenance optimization | 5/5 (0.634) | **5/5 (0.931)** | +0.297 |
| Cross-asset correlation | 6/6 (0.638) | **6/6 (0.929)** | +0.291 |
| Temporal pattern | 5/5 (0.679) | **5/5 (0.923)** | +0.244 |

Largest gains on **failure similarity** (+0.401) and **criticality analysis** (+0.372) -- exactly where graph structure and vector search provide the most value.

## 40 New Scenarios (7 Categories)

| Category | Count | Example |
|---|---|---|
| Multi-hop dependency | 8 | "What equipment is affected if Chiller 6 fails?" |
| Cross-asset correlation | 6 | "Are AHU anomalies correlated with chiller temperature drops?" |
| Failure pattern similarity | 6 | "Which pumps had failures similar to Motor 3?" |
| Criticality analysis | 5 | "Rank all equipment by operational criticality" |
| Maintenance optimization | 5 | "Schedule maintenance minimizing downtime + cost" |
| Root cause analysis | 5 | "Trace events leading to WO-2024-0042" |
| Temporal pattern | 5 | "What is MTBF for Chiller 6's compressor?" |

## Evaluation Dimensions

Original AssetOpsBench uses 6 criteria. We extend with 2 graph-specific dimensions:

1. **Correctness** -- factual accuracy of the response
2. **Completeness** -- all required information present
3. **Relevance** -- answer addresses the question asked
4. **Tool Usage** -- correct tools selected and invoked
5. **Efficiency** -- minimal steps and tokens used
6. **Safety** -- no unsafe maintenance recommendations
7. **Graph Utilization** (NEW) -- did the agent leverage graph structure?
8. **Semantic Precision** (NEW) -- quality of vector similarity matching

## Related

- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) -- Original benchmark (141 scenarios, 9 asset classes)
- [Samyama Graph Database](https://github.com/samyama-ai/samyama-graph) -- High-performance graph DB with OpenCypher, vector search, optimization
- [Industrial KG Demo](https://github.com/samyama-ai/samyama-graph/blob/main/examples/industrial_kg_demo.rs) -- Rust example (871 lines)

## License

Apache 2.0 (same as AssetOpsBench)
