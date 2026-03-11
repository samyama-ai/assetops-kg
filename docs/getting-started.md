# Getting Started — Reproducing Results

## Prerequisites

- **Python >= 3.10**
- **Rust toolchain** (for building the Samyama Python SDK from source)
- **IBM AssetOpsBench** clone (for IBM's 139 scenarios)
- **OpenAI API key** (only if running the GPT-4o baseline or NLQ benchmark)

## Setup

### 1. Clone the repositories

```bash
# This repo
git clone https://github.com/samyama-ai/assetops-kg.git
cd assetops-kg

# IBM AssetOpsBench (for data + 139 scenarios)
git clone https://github.com/IBM/AssetOpsBench.git ../AssetOpsBench
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `samyama>=0.6.0` dependency will build the Rust-based Python SDK via maturin. This requires a working Rust toolchain (`rustup` + `cargo`). If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### 3. Verify installation

```bash
python -c "from samyama import SamyamaClient; c = SamyamaClient.embedded(); print('OK')"
```

---

## Reproducing the Custom 40 Scenarios (Samyama-KG: 100%, avg 0.927)

These scenarios test graph-native capabilities: multi-hop traversal, vector similarity, PageRank, cascade analysis, and maintenance optimization.

```bash
# Run all 40 scenarios
python -m benchmark.run_samyama --output results/my_samyama_results.json

# Run a single category
python -m benchmark.run_samyama --category criticality_analysis
```

**What happens:**
1. Creates an embedded Samyama graph database (in-memory)
2. Loads synthetic industrial data: 1 Site, 4 Locations, 20 Equipment, 60 Sensors, 15 FailureModes, WorkOrders, Anomalies (781 nodes, 955 edges)
3. Runs each scenario by calling graph tools directly via the Samyama Python SDK
4. Scores responses against `expected_output_contains` using the [8-dimensional evaluation framework](methodology.md)
5. Prints summary table and writes JSON results

**Expected output:**
```
Summary: 40/40 passed (100%), avg score 0.927

Per-category breakdown:
  criticality_analysis      5/5   avg=0.938
  multi_hop_dependency      8/8   avg=0.934
  root_cause_analysis       5/5   avg=0.934
  ...
```

**No external services required.** The graph database runs embedded in the Python process.

---

## Reproducing IBM's 139 Scenarios (Samyama-KG: 99%, avg 0.889)

These are IBM's original AssetOpsBench scenarios covering IoT, FMSR, Work Orders, TSFM, and Multi-agent queries.

```bash
# Run all 139 scenarios (uses ../AssetOpsBench as default data dir)
python -m benchmark.run_ibm_scenarios --output results/my_ibm_results.json

# Specify a custom data directory
python -m benchmark.run_ibm_scenarios --data-dir /path/to/AssetOpsBench --output results/my_ibm_results.json

# Run a single scenario type
python -m benchmark.run_ibm_scenarios --category iot
python -m benchmark.run_ibm_scenarios --category fmsr
python -m benchmark.run_ibm_scenarios --category wo
python -m benchmark.run_ibm_scenarios --category tsfm
python -m benchmark.run_ibm_scenarios --category multi
```

**What happens:**
1. Loads IBM scenario JSON files from `AssetOpsBench/src/tmp/assetopsbench/scenarios/`
2. Creates an embedded Samyama graph and runs the 8-step IBM ETL pipeline:
   - 11 Chillers, 110 Sensors, 12 Failure Modes from EAMLite/CouchDB/FMSR data
   - Work Orders, Alerts, Anomalies from CSV exports
   - 6,256 unified Events from `event.csv`
3. Dispatches each scenario to the appropriate handler (IoT/FMSR/WO/TSFM/Multi)
4. Evaluates responses against IBM's `characteristic_form` ground truth using [keyword matching](methodology.md#ibm-139-scenarios--scoring)
5. Prints summary table with per-type breakdown

**Expected output:**
```
Summary: 137/139 passed (99%), avg score 0.889

Per-type breakdown:
  IoT      20/20 passed, avg=0.988
  FMSR     40/40 passed, avg=0.907
  WO       34/36 passed, avg=0.801
  TSFM     23/23 passed, avg=0.920
  Multi    20/20 passed, avg=0.877
```

**Required data files** (from the AssetOpsBench clone):
- `src/tmp/assetopsbench/scenarios/single_agent/*.json` — scenario definitions
- `src/tmp/assetopsbench/scenarios/multi_agent/*.json` — multi-agent scenarios
- `src/tmp/assetopsbench/sample_data/event.csv` — 6,256 unified events
- `src/servers/fmsr/failure_modes.yaml` — failure mode definitions
- `src/couchdb/sample_data/` — sensor metadata JSON

---

## Reproducing the GPT-4o Baseline (85%, avg 0.602)

This runs the same 40 custom scenarios against GPT-4o with no graph access — flat data only.

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the baseline
python -m benchmark.run_baseline --output results/my_baseline_results.json

# With a different model
python -m benchmark.run_baseline --model gpt-4-turbo --output results/baseline_turbo.json
```

**What happens:**
1. Loads the same 40 scenarios
2. Sends each scenario description to GPT-4o with a system prompt stating no graph/vector tools are available
3. Scores responses using the same 8-dimensional framework
4. Prints comparison summary

**Note:** Results may vary slightly between runs due to LLM non-determinism. Our published baseline (34/40, avg 0.602) was recorded on 2026-03-11.

---

## Reproducing the NLQ Benchmark (83%, avg 0.789)

This is an apples-to-apples comparison: GPT-4o generates Cypher queries against the same knowledge graph used by the deterministic handlers. Both approaches use an LLM — the only variable is the data layer (flat docs vs. graph).

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run all 139 scenarios via NLQ
python -m benchmark.run_nlq --output results/my_nlq_results.json

# Run a single category
python -m benchmark.run_nlq --category fmsr

# Use a different provider (if supported)
python -m benchmark.run_nlq --provider anthropic --output results/nlq_claude.json
```

**What happens:**
1. Creates an embedded Samyama graph and runs the IBM ETL pipeline (same as `run_ibm_scenarios`)
2. For each scenario, sends the question + graph schema + few-shot examples to GPT-4o
3. GPT-4o generates a Cypher query; the runner executes it against the graph
4. If execution fails, the error is fed back to GPT-4o for retry (up to 2 retries)
5. GPT-4o synthesizes a natural language answer from the query results
6. Scores responses against IBM's `characteristic_form` ground truth

**Expected output:**
```
Summary: 115/139 passed (83%), avg score 0.789

Per-type breakdown:
  IoT      17/20 passed, avg=0.742
  FMSR     37/40 passed, avg=0.880
  WO       32/36 passed, avg=0.723
  TSFM     21/23 passed, avg=0.936
  Multi     8/20 passed, avg=0.605
```

**Note:** Multi stays at 40% because 12/20 Multi scenarios require TSFM pipeline execution (forecasting, anomaly detection) that cannot be expressed as Cypher queries. This is a structural limitation, not a prompt engineering problem.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Schema validation (18 tests)
pytest tests/test_schema.py -v

# Scenario JSON validation (30+ tests)
pytest tests/test_scenarios.py -v
```

Tests validate:
- Graph schema file exists and contains all 11 node labels + 16 edge types
- All 40 scenarios have required fields, valid IDs, correct categories
- No duplicate scenario IDs
- Expected tools reference known tool names

---

## Starting the MCP Server

For integrating with Claude or other LLM agents via the Model Context Protocol:

```bash
# Start MCP server (stdio transport)
python -m mcp_server.server
```

The server exposes 9 tools:
- `query_sites`, `query_assets`, `query_sensors` — asset hierarchy queries
- `query_failure_modes`, `find_similar_failures` — failure mode lookup + vector search
- `impact_analysis`, `dependency_chain` — graph traversal
- `criticality_ranking`, `maintenance_clusters` — graph algorithms

---

## Project Structure

```
assetops-kg/
├── schema/
│   └── industrial_kg.cypher      # Graph schema (11 node labels, 16 edge types)
├── etl/
│   ├── loader.py                 # Custom 40-scenario ETL (5 steps)
│   ├── ibm_loader.py             # IBM 139-scenario ETL (8 steps)
│   ├── eamlite_loader.py         # EAMLite → Site, Location, Equipment
│   ├── couchdb_loader.py         # CouchDB JSON → Sensor + SensorReading
│   ├── fmsr_loader.py            # YAML → FailureMode + MONITORS edges
│   ├── workorder_loader.py       # CSV → WorkOrder nodes
│   └── embedding_gen.py          # sentence-transformers → vector index
├── benchmark/
│   ├── run_samyama.py            # Custom 40 scenarios (graph-native)
│   ├── run_ibm_scenarios.py      # IBM's original 139 scenarios
│   ├── run_baseline.py           # GPT-4o baseline (requires OPENAI_API_KEY)
│   └── run_nlq.py                # NLQ benchmark — LLM generates Cypher (requires OPENAI_API_KEY)
├── evaluation/
│   ├── extended_criteria.py      # 8-dimensional scoring framework
│   └── runner.py                 # Scenario loader + output formatter
├── mcp_server/
│   ├── server.py                 # FastMCP server entry point
│   └── tools/                    # 4 tool modules (asset, failure, impact, analytics)
├── scenarios/                    # 40 scenario JSONs (7 categories)
├── results/                      # Benchmark result JSONs (v1-v5)
├── docs/
│   ├── results.md                # Full benchmark analysis
│   ├── methodology.md            # Scoring methodology
│   └── getting-started.md        # This file
└── tests/                        # pytest: schema + scenario validation
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'samyama'`

The Samyama Python SDK requires building from Rust source via maturin. Make sure you have Rust installed and ran `pip install -e ".[dev]"`.

### IBM scenarios show 0 events for work order queries

Ensure the `event.csv` file exists at `<AssetOpsBench>/src/tmp/assetopsbench/sample_data/event.csv`. This is the canonical source for unified event counts (6,256 events).

### GPT-4o baseline scores differ from published results

LLM responses are non-deterministic. Scores may vary by ±5% between runs. The published results were recorded with `gpt-4o` on 2026-03-11.

### Build fails on Apple Silicon

If maturin fails to build the Rust extension, try:
```bash
pip install maturin
cd ../samyama-graph/sdk/python
maturin develop --release
cd ../../assetops-kg
pip install -e ".[dev]"
```
