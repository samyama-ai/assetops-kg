# Workflow Walkthrough: Before/After Comparison

*Prepared for IBM collaboration call — March 26, 2026*

---

## 1. The Core Shift: Reasoning vs. Retrieval

### IBM's Architecture (Before)

```
User Question
    ↓
LLM (GPT-4) does EVERYTHING:
    1. Parse intent ("which failure modes?")
    2. Select agent (IoT? FMSR? WO?)
    3. Select tool (sites()? sensors()? fm_sr()?)
    4. Formulate arguments
    5. Interpret raw results (JSON/YAML/CSV)
    6. Synthesize answer
    ↓
Answer (65% accuracy)
```

**Problem**: Steps 3-5 are *data operations* — counting, traversing relationships, correlating across documents. LLMs are unreliable at these tasks.

### Knowledge Graph Architecture (After)

```
User Question
    ↓
Handler/NLQ routes to graph query:
    1. Pattern match → Cypher query (deterministic)
       OR
    1. LLM generates Cypher from schema (narrow problem)
    2. Graph engine executes deterministically
    3. Format results
    ↓
Answer (99% deterministic / 82% NLQ)
```

**Key insight**: The LLM's role shrinks from "do everything" to "generate a query from a typed schema" — which it's good at.

---

## 2. Multi-Step Example: Dependency Impact Analysis

### Scenario
> "Pump-CW-1 is scheduled for replacement. Identify all equipment within 2 hops that will be affected, considering dependencies and shared systems."

### IBM Approach (Multi-Agent, Multi-Step)

```
Step 1: LLM selects IoT agent → calls assets() → gets flat JSON list of 11 chillers
Step 2: LLM must REASON about which chillers relate to Pump-CW-1
        → No explicit relationship data in CouchDB → LLM hallucinates connections
Step 3: LLM selects FMSR agent → calls fm_sr() → gets failure modes
Step 4: LLM must CORRELATE failure modes to equipment affected by pump replacement
        → Requires traversing implicit relationships across YAML + JSON documents
Step 5: LLM synthesizes answer
        → Often misses Motor-P1 (depends on Pump-CW-1) because it's not in the same document
```

**Result**: Partial answer, misses downstream dependencies. ~65% accuracy on this type.

### Graph Approach (Single Traversal)

```
Step 1: impact_analysis(equipment="Pump-CW-1", depth=2)
        → Executes Cypher:
          MATCH (e:Equipment {name: "Pump-CW-1"})-[:DEPENDS_ON|SHARES_SYSTEM_WITH*1..2]-(affected)
          RETURN affected.name, type(r), length(path)

Step 2: Returns structured result:
        Hop 1: Chiller-1 (Pump-CW-1 DEPENDS_ON Chiller-1)
                Motor-P1 (Motor-P1 DEPENDS_ON Pump-CW-1)
        Hop 2: AHU-1 (AHU-1 DEPENDS_ON Chiller-1)
                Motor-CH1 (Motor-CH1 DEPENDS_ON Chiller-1)
```

**Result**: Complete, deterministic, 25ms. The graph stores relationships explicitly — no LLM reasoning needed.

### Why This Matters

| Aspect | IBM (LLM reasoning) | Graph (deterministic) |
|--------|---------------------|----------------------|
| Correctness | Misses Motor-P1 | Finds all 4 affected |
| Latency | ~11s (5 LLM calls) | 25ms (1 graph query) |
| Tokens | ~4,600 | 0 |
| Determinism | Varies per run | Identical every time |

---

## 3. Multi-Tool Composition Example

### Scenario
> "Chiller-3 has a vibration anomaly. Find similar historical failures, identify the root failure mode, check what other equipment depends on it, and recommend a maintenance priority."

### Graph Solution: 4 Tools in Sequence

```
Tool 1: find_similar_failures(equipment="Chiller-3", k=5)
        → HNSW vector search on 384-dim failure embeddings
        → Returns: "Bearing Degradation" (0.92 similarity), "Compressor Overheating" (0.78)

Tool 2: cypher_query("MATCH (s:Sensor)-[:MONITORS]->(fm:FailureMode {name: 'Bearing Degradation'})
                       WHERE s.equipment = 'Chiller-3' RETURN s.name, s.type")
        → Returns: vibration_sensor_3 monitors Bearing Degradation

Tool 3: impact_analysis(equipment="Chiller-3", depth=2)
        → Returns: AHU-1, Pump-CW-1 depend on Chiller-3

Tool 4: criticality_ranking()
        → PageRank over dependency graph
        → Chiller-3 rank: 0.087 (3rd most critical of 11 chillers)
```

**Composed answer**: "Chiller-3's vibration anomaly matches historical Bearing Degradation pattern (92% similarity). Sensor vibration_sensor_3 monitors this failure mode. Chiller-3 is the 3rd most critical equipment unit (PageRank 0.087) with 2 downstream dependencies (AHU-1, Pump-CW-1). Recommend immediate inspection with elevated priority."

### What Each Tool Provides

| Tool | Capability | LLM Equivalent |
|------|-----------|---------------|
| `find_similar_failures` | HNSW vector search | None — LLM cannot search embeddings |
| `cypher_query` | Relationship traversal | LLM guesses from flat docs |
| `impact_analysis` | Multi-hop BFS | LLM cannot traverse graphs |
| `criticality_ranking` | PageRank algorithm | LLM cannot compute graph centrality |

---

## 4. MCP Server Architecture

The MCP server exposes 9 tools via the Model Context Protocol, callable by any MCP-compatible agent (Claude, GPT-4, etc.):

```
┌─────────────────────────────────────────┐
│            LLM Agent (Claude/GPT)       │
│  Understands intent, selects tools      │
└──────────────┬──────────────────────────┘
               │ MCP Protocol (stdio/SSE)
┌──────────────▼──────────────────────────┐
│         AssetOps MCP Server             │
│  ┌─────────┐ ┌───────────┐ ┌─────────┐ │
│  │  Asset   │ │  Failure  │ │ Impact  │ │
│  │  Tools   │ │  Tools    │ │  Tools  │ │
│  └────┬─────┘ └────┬──────┘ └────┬────┘ │
│       │            │             │      │
│  ┌────▼────────────▼─────────────▼────┐ │
│  │     Samyama Graph Database         │ │
│  │  781 nodes, 955 edges, HNSW index  │ │
│  │  OpenCypher + Vector Search        │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Tool Categories

| Category | Tools | Graph Operations |
|----------|-------|-----------------|
| **Asset hierarchy** | `query_sites`, `query_assets`, `query_sensors` | Label scans, multi-hop traversals |
| **Failure analysis** | `query_failure_modes`, `find_similar_failures`, `query_fm_sensor_map` | HNSW vector search, edge traversal |
| **Impact analysis** | `impact_analysis`, `dependency_chain` | BFS over DEPENDS_ON/SHARES_SYSTEM_WITH |
| **Analytics** | `criticality_ranking`, `maintenance_clusters` | PageRank, connected components |

---

## 5. Migration to New Assets/Equipment

### Schema-Driven ETL

Adding new equipment types requires only ETL changes — no model retraining, no prompt engineering:

```
Existing (Chillers):           Expanded (HuggingFace 467):
  11 Equipment nodes             → 26 Equipment types
  110 Sensors                    → 1,360 nodes
  12 Failure Modes               → 14 labels, 21 edge types
  781 nodes total                → covers compressors, hydraulic pumps, PHM
```

**Process**:
1. Map new data source to graph schema (labels + edge types)
2. Write ETL loader (typically ~50 lines of Python)
3. Graph queries work immediately — no retraining needed
4. MCP tools are generic (parameterized by equipment name)

**Evidence**: We expanded from 139→467 scenarios across 6 new equipment domains with **zero handler code changes**. All existing tools worked against the expanded graph.
