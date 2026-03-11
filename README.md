# AssetOps-KG: Industrial Asset Operations Knowledge Graph

Extending [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) with graph database, vector search, and multi-objective optimization capabilities using [Samyama Graph Database](https://github.com/samyama-ai/samyama-graph).

## Thesis

AssetOpsBench shows GPT-4 completes only 65% of industrial maintenance tasks using flat document stores + LLM reasoning loops. We demonstrate that replacing flat storage with a **knowledge graph** (ISO 14224 + ISA-95 ontology) improves accuracy, cuts latency, and enables new query types:

- **Multi-hop dependency analysis** -- "If Chiller 6 fails, what downstream equipment is affected?"
- **Semantic failure similarity** -- "Which pumps had failures similar to Motor 3's bearing wear?"
- **Graph-based criticality ranking** -- PageRank on the equipment dependency network
- **Pareto-optimal maintenance scheduling** -- NSGA-II minimizing cost vs. downtime

## Graph Schema

11 node labels, 16 edge types (see [`schema/industrial_kg.cypher`](schema/industrial_kg.cypher)):

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
```

## Project Structure

```
assetops-kg/
├── schema/                    # Graph schema (Cypher CREATE statements)
├── etl/                       # ETL pipeline (AssetOpsBench -> Samyama KG)
│   ├── loader.py              # Main orchestrator (CLI)
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
├── benchmark/                 # Baseline vs. Samyama comparison
│   ├── run_baseline.py
│   └── run_samyama.py
└── tests/
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Load data (uses AssetOpsBench data if available, otherwise synthetic)
python -m etl.loader --data-dir ../AssetOpsBench/src/couchdb/sample_data

# Start MCP server
python -m mcp_server.server

# Run evaluation
python -m evaluation.runner --scenarios scenarios/
```

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
