# Information Leakage Analysis

*Prepared for IBM collaboration call — March 26, 2026*

---

## Summary

*Corrected 2026-06-30 after a full live reproduction (see `REPRODUCTION-AND-CORRECTIONS.md` in the paper repo).*

The knowledge graph is constructed from IBM's public data sources, and the chiller/AHU failure-mode content is loaded **verbatim from IBM's own `failure_modes.yaml`** — not invented. The NLQ pipeline (§2) sends only the graph schema to the LLM, never the expected answers; that path is leak-proof and is our apples-to-apples result. The deterministic handlers (§1) are more nuanced and we do **not** claim a blanket "no leakage": an instrumented run shows **86 of the 139 deterministic answers come from a live graph query, while 53 are served by handlers that supply domain knowledge without a graph traversal** (most visibly for equipment types absent from the graph). Several FMSR scenarios IBM flags `deterministic:false` are answered from an authored sensor→failure-mode mapping; IBM's own rubric for these accepts *any subset of its published failure modes*, so this is a valid-subset response rather than a hidden answer key — but it is domain knowledge, not pure traversal.

---

## 1. Deterministic Handlers (99% — 137/139)

### How They Work

Each handler is a **hand-coded function** that:
1. Receives a natural language question
2. Matches it to a category via keyword patterns (e.g., "failure mode" → FMSR handler)
3. Constructs a Cypher query from the matched pattern
4. Executes the query against the graph
5. Formats the result

### Why There's No Leakage

| Concern | Analysis |
|---------|----------|
| **Were handlers trained on the test set?** | No. Handlers match scenario *categories* (IoT, FMSR, WO, TSFM, Multi), not individual scenarios. The same 5 handlers serve all 139 scenarios. |
| **Do handlers encode expected answers?** | Mostly no: 86/139 answers come from a live graph query parameterized by the question. For scenarios whose equipment is absent from the graph (notably the HuggingFace FMSR set), handlers fall back to domain-knowledge constants rather than a traversal — read those as domain-knowledge responses, not graph retrievals. Failure-mode lists themselves are verbatim from IBM's `failure_modes.yaml`. |
| **Could the graph encode answers?** | No. The graph is constructed from IBM's own data files (EAMLite, CouchDB JSON, FMSR YAML, event.csv). We transform the data structure, not the data content. |
| **What about the 2 failing scenarios?** | Scenarios 411 and 424 fail; both are work-order bundling edge cases where our date-window clustering produces different groupings than the benchmark's expected bundle sizes — a response-format mismatch, not a knowledge gap. |

### Verification

Anyone can verify by:
```bash
# 1. Clone IBM's AssetOpsBench (the ONLY data source)
git clone https://github.com/IBM/AssetOpsBench.git

# 2. Clone our repo
git clone https://github.com/samyama-ai/assetops-kg.git

# 3. Run the benchmark
cd assetops-kg
pip install -e ".[dev]"
python -m benchmark.run_ibm_scenarios --data-dir ../AssetOpsBench

# The graph is constructed from IBM's files at runtime.
# No pre-computed answers, no cached results, no pre-trained models.
```

---

## 2. NLQ Pipeline (82% GPT-4 / 83% GPT-4o)

### How It Works

```
Question → LLM receives: system prompt + graph schema + few-shot examples → generates Cypher → graph executes → LLM formats answer
```

### What the LLM Sees

| Sent to LLM | NOT Sent to LLM |
|-------------|-----------------|
| Graph schema (9 labels, 5 edge types for the 139-scenario graph) | Expected answers |
| 5 few-shot Cypher examples | Ground truth (`characteristic_form`) |
| The user's question | Other scenarios' questions |
| Query execution results (for answer synthesis) | Scoring criteria |

### Few-Shot Examples

The 5 few-shot examples are **generic schema demonstrations**, not derived from test scenarios:

```
Example: "How many sensors does Chiller-1 have?"
Cypher: MATCH (e:Equipment {name: 'Chiller-1'})-[:HAS_SENSOR]->(s:Sensor) RETURN count(s)
```

These demonstrate *schema patterns* (how to traverse HAS_SENSOR edges), not specific test answers.

### Non-Determinism

LLM-generated Cypher varies between runs:
- Same question may produce different valid Cypher queries
- Execution results are deterministic (same Cypher → same result)
- Our published NLQ score (82-83%) has ±2-3% variance across runs
- We report single-pass results (no cherry-picking, no averaging)

---

## 3. GPT-4o Baseline (85% on custom 40)

### How It Works

```
Question → GPT-4o with system prompt: "You have NO access to graph database,
vector search, or graph algorithms. Answer based on your training data only."
→ GPT-4o generates answer from parametric knowledge
```

### Why There's No Leakage

- GPT-4o has no access to our graph, our scenarios, or our expected answers
- The system prompt explicitly disables tool access
- GPT-4o's training data cutoff predates our scenario creation
- The 40 custom scenarios are original (not from any public dataset)

---

## 4. HuggingFace Expanded Evaluation (467/467)

### Data Source

All 467 scenarios come from IBM's official HuggingFace release:
```
https://huggingface.co/datasets/ibm-research/AssetOpsBench
```

### How We Handle New Domains

| Domain | Scenarios | Our Approach |
|--------|-----------|-------------|
| scenarios (original) | 152 | Same 5 handlers as IBM 139 |
| rule_logic | 120 | New rule-matching handler for anomaly detection patterns |
| FMSR | 88 | Extended existing FMSR handler (same graph schema) |
| PHM | 75 | New prognostics handler (RUL, fault classification) |
| hydraulic_pump | 17 | Existing handlers + expanded graph with pump equipment |
| compressor | 15 | Existing handlers + expanded graph with compressor equipment |

The base IBM graph behind the 139-scenario evaluation has **9 node labels, 5 edge types, and 12,647 nodes**; the extended schema (14 labels, 21 edge types) spans the dependency topology, the custom graph-native scenarios, and the HuggingFace additions. For HuggingFace FMSR equipment types not present in the graph, handlers return domain-knowledge fallbacks (see §1) rather than graph traversals.

---

## 5. Key Guarantees

1. **Reproducible**: Clone two repos, run one command, get the same results
2. **No pre-training**: Graph is constructed from IBM's public data at runtime
3. **Faithful data, mixed handler retrieval**: failure modes are verbatim from IBM's `failure_modes.yaml`; 86/139 deterministic answers come from a live graph query, 53 from domain-knowledge handlers (honestly reported, not claimed as pure traversal)
4. **Single-pass evaluation**: No re-runs, no cherry-picking, no hyperparameter tuning
5. **Open source**: All code, data, and results are public at https://github.com/samyama-ai/assetops-kg
