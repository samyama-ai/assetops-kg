# Reproducibility Guide for IBM

*Prepared for IBM collaboration call — March 26, 2026*

---

## Quick Verification (5 minutes)

### Prerequisites
- Python >= 3.10
- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### Steps

```bash
# 1. Clone both repos
git clone https://github.com/samyama-ai/assetops-kg.git
git clone https://github.com/IBM/AssetOpsBench.git
cd assetops-kg

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Run IBM 139 scenarios (NO API key needed — deterministic)
python -m benchmark.run_ibm_scenarios --data-dir ../AssetOpsBench

# Expected: 137/139 passed (99%), avg score 0.889
```

**Total time**: ~3 min (2 min Rust build, 1 min benchmark). No GPU, no API keys, no cloud services.

---

## Full Evaluation Matrix

| Benchmark | Command | API Key? | Expected Result |
|-----------|---------|----------|-----------------|
| IBM 139 (deterministic) | `python -m benchmark.run_ibm_scenarios` | No | 137/139 (99%) |
| Custom 40 (graph-native) | `python -m benchmark.run_samyama` | No | 40/40 (100%) |
| HF 467 (expanded) | `python -m benchmark.run_hf_expanded` | No | 467/467 (100%) |
| GPT-4o baseline | `python -m benchmark.run_baseline` | Yes (`OPENAI_API_KEY`) | ~34/40 (85% ±3%) |
| NLQ GPT-4 | `python -m benchmark.run_nlq --model gpt-4` | Yes | ~114/139 (82% ±3%) |
| NLQ GPT-4o | `python -m benchmark.run_nlq` | Yes | ~115/139 (83% ±3%) |

**Deterministic benchmarks** (first 3) produce identical results every run.
**LLM benchmarks** (last 3) have ±2-3% variance due to LLM non-determinism.

---

## What Runs Where

```
┌─────────────────────────────────────────────────────┐
│                  Your Machine                       │
│                                                     │
│  assetops-kg/                                       │
│  ├── benchmark/run_ibm_scenarios.py                 │
│  │     ↓ reads scenarios from                       │
│  │   AssetOpsBench/src/tmp/assetopsbench/           │
│  │     ↓ builds graph from IBM's data files         │
│  │   Embedded Samyama Graph (in-process, no server) │
│  │     ↓ executes Cypher queries                    │
│  │   Scenario handlers format answers               │
│  │     ↓ scores against characteristic_form         │
│  └── results/*.json                                 │
│                                                     │
│  [For NLQ only]: OpenAI API ──→ generates Cypher    │
│                  (schema sent, not answers)          │
└─────────────────────────────────────────────────────┘
```

**No external services** for the deterministic benchmark. The graph database runs embedded in the Python process via the Samyama Python SDK (PyO3-based Rust extension).

---

## Data Flow: What Comes From Where

### IBM's Data (Input — Unchanged)

| File | Source | Contents |
|------|--------|----------|
| `AssetOpsBench/src/servers/iot/eamlite/` | EAMLite export | 11 chillers, ISA-95 hierarchy |
| `AssetOpsBench/src/couchdb/sample_data/` | CouchDB JSON | 110 sensors (type, unit, range) |
| `AssetOpsBench/src/servers/fmsr/failure_modes.yaml` | FMSR config | 12 failure modes + sensor mappings |
| `AssetOpsBench/src/tmp/assetopsbench/sample_data/event.csv` | Event log | 6,256 work orders, alerts, anomalies |
| `AssetOpsBench/src/tmp/assetopsbench/scenarios/` | Test scenarios | 139 scenario JSON files |

### Our Transformation (ETL — Graph Construction)

| Step | Input | Output | Lines of Code |
|------|-------|--------|---------------|
| 1. Sites | EAMLite | 1 Site node | 15 |
| 2. Locations | EAMLite | 4 Location nodes + CONTAINS_LOCATION edges | 20 |
| 3. Equipment | EAMLite | 11 Equipment nodes + CONTAINS_EQUIPMENT edges | 30 |
| 4. Sensors | CouchDB JSON | 110 Sensor nodes + HAS_SENSOR edges | 25 |
| 5. Failure modes | FMSR YAML | 12 FailureMode nodes + MONITORS edges | 35 |
| 6. Work orders | event.csv | WorkOrder/Alert/Anomaly nodes | 40 |
| 7. Dependencies | Domain knowledge | DEPENDS_ON/SHARES_SYSTEM_WITH edges | 25 |
| 8. Unified events | event.csv | 6,256 Event nodes | 30 |

**Total ETL**: ~220 lines of Python. The transformation adds structure (edges, labels) but does not modify IBM's data content.

---

## Handling Non-Determinism

### The Three Tiers

| Tier | Determinism | Variance | Use Case |
|------|------------|----------|----------|
| **Deterministic handlers** | 100% reproducible | 0% | Production queries, auditable answers |
| **NLQ (LLM → Cypher)** | Query varies, execution is deterministic | ±2-3% | Exploratory queries, flexible NL interface |
| **IBM baseline (LLM does everything)** | Non-deterministic | ±5-10% | Research comparison |

### How We Mitigate NLQ Variance

1. **Schema grounding**: LLM generates Cypher from a typed schema (14 labels, 21 edge types) — constrained output space
2. **Execution is deterministic**: Different Cypher → same valid answer (if semantically equivalent)
3. **Retry on syntax error**: If Cypher fails to parse, error is fed back for 1 retry
4. **Single-pass reporting**: We report one run, not the best of N

### Same-Model Comparison (Isolating Variables)

| Comparison | Variables | Result |
|-----------|-----------|--------|
| GPT-4 + flat docs (IBM) vs GPT-4 + graph NLQ | Data model only | 65% → 82% (+17pp) |
| GPT-4 NLQ vs GPT-4o NLQ | Model only | 82% → 83% (+1pp) |
| NLQ vs Deterministic | LLM involvement | 83% → 99% (+16pp) |

The +17pp from data model change vs +1pp from model upgrade proves: **the data model is the primary bottleneck**.

---

## For IBM's MCP Team: Integration Path

If IBM wants to reproduce using their own MCP infrastructure:

### Option A: Run Our Benchmark Directly
```bash
git clone https://github.com/samyama-ai/assetops-kg.git
pip install -e ".[dev]"
python -m benchmark.run_ibm_scenarios
```

### Option B: Use Our MCP Server with Your Agent
```bash
# Start our MCP server (exposes 9 graph tools via MCP protocol)
python -m mcp_server.server

# Connect your MCP client to it
# Tools available: query_sites, query_assets, query_sensors,
#   query_failure_modes, find_similar_failures, query_fm_sensor_map,
#   impact_analysis, dependency_chain, criticality_ranking
```

### Option C: Use the Graph Directly
```python
from samyama import SamyamaClient

client = SamyamaClient.embedded()
# Load IBM data
from etl.ibm_loader import load_ibm_data
load_ibm_data(client, data_dir="../AssetOpsBench")

# Query directly
result = client.query("default",
    "MATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor) RETURN e.name, count(s)")
```

---

## Artifacts Summary

| Artifact | URL | Purpose |
|----------|-----|---------|
| Code + benchmarks | https://github.com/samyama-ai/assetops-kg | Full reproducion |
| Results (JSON) | `results/` directory in repo | Raw benchmark outputs |
| Methodology | `docs/methodology.md` | Scoring framework |
| PR #203 | https://github.com/IBM/AssetOpsBench/pull/203 | 40 graph-native scenarios |
| Paper draft | Shared via email | 12-page analysis |
| HuggingFace data | https://huggingface.co/datasets/ibm-research/AssetOpsBench | IBM's 467 scenarios |
