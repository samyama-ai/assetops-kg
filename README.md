# AssetOps-KG: Industrial Asset Operations Knowledge Graph

Extending [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) with graph database, vector search, and multi-objective optimization capabilities using [Samyama Graph Database](https://github.com/samyama-ai/samyama-graph).

## Key Results

| Benchmark | GPT-4 (IBM) | NLQ (GPT-4o + graph) | Samyama-KG (deterministic) | Delta vs IBM |
|---|---|---|---|---|
| IBM's 139 scenarios | ~91/139 (65%)* | 115/139 (83%) | **137/139 (99%)** | **+34pp** |
| Avg latency | not reported | 5,874 ms | **63 ms** | -- |
| Avg tokens | not reported | 4,616/scenario | **0** | **$0** |

| Benchmark | GPT-4o (no graph) | Samyama-KG | Delta |
|---|---|---|---|
| Custom 40 scenarios | 34/40 (85%) | **40/40 (100%)** | **+15pp** |
| Avg latency (custom 40) | 11,259 ms | **110 ms** | **103x faster** |

*IBM's reported GPT-4 figure.

**Three tiers of performance emerge:** GPT-4 + flat docs (65%) < GPT-4o + graph via NLQ (83%) < Deterministic handlers + graph (99%). Note: IBM used GPT-4; NLQ used GPT-4o (stronger model), so the +18pp gap is an upper bound on the graph's contribution. GPT-4 NLQ run pending for true same-model comparison.

Full analysis: [`docs/results.md`](docs/results.md) | Scoring methodology: [`docs/methodology.md`](docs/methodology.md) | Reproducing results: [`docs/getting-started.md`](docs/getting-started.md)

## Thesis

AssetOpsBench shows GPT-4 completes only 65% of industrial maintenance tasks using flat document stores + LLM reasoning loops. We demonstrate that replacing flat storage with a **knowledge graph** (ISO 14224 + ISA-95 ontology) improves accuracy, cuts latency, and enables new query types:

- **Multi-hop dependency analysis** -- "If Chiller 6 fails, what downstream equipment is affected?"
- **Semantic failure similarity** -- "Which pumps had failures similar to Motor 3's bearing wear?"
- **Graph-based criticality ranking** -- PageRank on the equipment dependency network
- **Pareto-optimal maintenance scheduling** -- NSGA-II minimizing cost vs. downtime

## How It Works

```
IBM's approach:     Question → LLM → flat document search → LLM reasoning → answer
NLQ approach:       Question → LLM → Cypher generation → graph traversal → LLM synthesis → answer
Handler approach:   Question → deterministic routing → Cypher query → answer (no LLM)
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
│   ├── run_ibm_scenarios.py   # IBM's original 139 scenarios
│   └── run_nlq.py             # NLQ benchmark (GPT-4o generates Cypher)
├── docs/
│   ├── results.md             # Full benchmark analysis
│   ├── methodology.md         # Scoring and evaluation methodology
│   └── getting-started.md     # Setup, reproduction, troubleshooting
├── results/                   # Benchmark result JSONs (v1-v5)
└── tests/
```

## Quick Start

See [`docs/getting-started.md`](docs/getting-started.md) for full setup instructions, prerequisites, and troubleshooting.

```bash
# Clone and install
git clone https://github.com/samyama-ai/assetops-kg.git && cd assetops-kg
git clone https://github.com/IBM/AssetOpsBench.git ../AssetOpsBench
pip install -e ".[dev]"

# Run custom 40 scenarios (100%, avg 0.927)
python -m benchmark.run_samyama --output results/samyama_results.json

# Run IBM's 139 scenarios (99%, avg 0.889)
python -m benchmark.run_ibm_scenarios --data-dir ../AssetOpsBench --output results/ibm_results.json

# Run GPT-4o baseline for comparison (requires OPENAI_API_KEY)
python -m benchmark.run_baseline --output results/baseline_results.json

# Run NLQ benchmark — GPT-4o generates Cypher against the graph (requires OPENAI_API_KEY)
python -m benchmark.run_nlq --output results/nlq_results.json

# Run tests
pytest tests/ -v

# Start MCP server (for agent integration)
python -m mcp_server.server
```

## Benchmark Results

### IBM's Original 139 Scenarios

| Approach | Pass Rate | Avg Score | Avg Latency | Tokens |
|---|---|---|---|---|
| GPT-4 (IBM reported) | ~91/139 (65%) | not reported | not reported | not reported |
| NLQ v3 (GPT-4o + graph) | 115/139 (83%) | 0.789 | 5,874 ms | 4,616/scenario |
| **Deterministic (graph)** | **137/139 (99%)** | **0.889** | **63 ms** | **0** |

#### Per-Type Breakdown (Deterministic vs NLQ)

| Type | Deterministic Pass | Deterministic Avg | NLQ Pass | NLQ Avg |
|---|---|---|---|---|
| IoT (20) | **20/20 (100%)** | 0.988 | 17/20 (85%) | 0.742 |
| FMSR (40) | **40/40 (100%)** | 0.907 | 37/40 (93%) | 0.880 |
| TSFM (23) | **23/23 (100%)** | 0.920 | 21/23 (91%) | 0.936 |
| Multi (20) | **20/20 (100%)** | 0.877 | 8/20 (40%) | 0.605 |
| WO (36) | 34/36 (94%) | 0.801 | 32/36 (89%) | 0.723 |

NLQ with GPT-4o (83%) vs IBM's GPT-4 (65%) = +18pp. This gap reflects both the graph data model and a stronger LLM -- a GPT-4 NLQ run is pending to isolate the graph's contribution. Only 2 deterministic failures remain (WO bundling edge cases).

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

## Evaluation Methodology

Full details: [`docs/methodology.md`](docs/methodology.md)

**Single pass, no repeated runs.** Each scenario gets one handler call, one response, one score. "Avg score" is the arithmetic mean across all scenarios.

**IBM 139 scenarios** are scored by keyword matching against the `characteristic_form` ground truth field. Three paths: strict item matching (deterministic + items), count matching (deterministic + counts), or lenient keyword overlap (non-deterministic, with 1.5x boost). Pass threshold: score >= 0.5.

**Custom 40 scenarios** use 8 weighted dimensions:

| Dimension | Weight | What It Measures |
|---|---|---|
| Correctness | 0.20 | Expected keywords present in response |
| Completeness | 0.15 | Coverage of required information |
| Relevance | 0.10 | Question terms reflected in answer |
| Tool Usage | 0.15 | Correct graph tools invoked |
| Efficiency | 0.05 | Latency and token usage |
| Safety | 0.10 | No unsafe maintenance recommendations |
| Graph Utilization | 0.15 | Evidence of graph traversal, not flat-data reasoning |
| Semantic Precision | 0.10 | Quality of vector similarity matching |

Category-specific weight overrides boost the most relevant dimension (e.g., Semantic Precision → 0.25 for failure similarity scenarios).

## Related

- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench) -- Original benchmark (141 scenarios, 9 asset classes)
- [Samyama Graph Database](https://github.com/samyama-ai/samyama-graph) -- High-performance graph DB with OpenCypher, vector search, optimization
- [Industrial KG Demo](https://github.com/samyama-ai/samyama-graph/blob/main/examples/industrial_kg_demo.rs) -- Rust example (871 lines)

## License

Apache 2.0 (same as AssetOpsBench)
