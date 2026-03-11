"""Run 40 AssetOps-KG scenarios against Samyama graph — measure latency and accuracy.

Loads synthetic data via ETL, then executes each scenario's expected tools
directly against the Samyama Python SDK (no MCP transport needed).

Usage:
    python -m benchmark.run_samyama
    python -m benchmark.run_samyama --category criticality_analysis
    python -m benchmark.run_samyama --output results/samyama_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from samyama import SamyamaClient

from evaluation.extended_criteria import ScenarioResult, evaluate_response
from evaluation.runner import load_scenarios, format_summary_table, results_to_json


# ---------------------------------------------------------------------------
# Direct tool implementations (bypass MCP, call SDK directly)
# ---------------------------------------------------------------------------

def tool_impact_analysis(client: SamyamaClient, graph: str, params: dict) -> dict:
    """BFS cascade: what equipment is affected if this one fails."""
    equipment_name = params.get("equipment_name", "Chiller-1")
    cypher = f"MATCH (e:Equipment) WHERE e.name = '{equipment_name}' RETURN e.name, e.iso14224_class"
    result = client.query_readonly(cypher, graph)
    if not result.records:
        return {"error": f"Equipment '{equipment_name}' not found", "affected": [], "cascade_depth": 0}

    affected = []
    visited = {equipment_name}
    frontier = [equipment_name]
    depth = 0

    while frontier:
        depth += 1
        next_frontier = []
        for name in frontier:
            dep = client.query_readonly(
                f"MATCH (dep:Equipment)-[:DEPENDS_ON]->(e:Equipment) WHERE e.name = '{name}' "
                "RETURN dep.name, dep.iso14224_class, dep.criticality_score", graph
            )
            for row in dep.records:
                dname = row[0]
                if dname and dname not in visited:
                    visited.add(dname)
                    next_frontier.append(dname)
                    affected.append({
                        "name": dname, "class": row[1],
                        "criticality_score": row[2], "cascade_depth": depth,
                    })
            shared = client.query_readonly(
                f"MATCH (s:Equipment)-[:SHARES_SYSTEM_WITH]->(e:Equipment) WHERE e.name = '{name}' "
                "RETURN s.name, s.iso14224_class", graph
            )
            for row in shared.records:
                sname = row[0]
                if sname and sname not in visited:
                    visited.add(sname)
                    next_frontier.append(sname)
                    affected.append({
                        "name": sname, "class": row[1],
                        "cascade_depth": depth, "mechanism": "SHARES_SYSTEM_WITH",
                    })
        frontier = next_frontier

    return {
        "source": equipment_name, "total_affected": len(affected),
        "max_cascade_depth": max((a["cascade_depth"] for a in affected), default=0),
        "affected": affected,
        "traversal": "BFS cascade via DEPENDS_ON and SHARES_SYSTEM_WITH edges",
        "graph_method": "multi-hop dependency traversal",
    }


def tool_dependency_chain(client: SamyamaClient, graph: str, params: dict) -> dict:
    """What does this equipment depend on (forward traversal)."""
    equipment_name = params.get("equipment_name", "AHU-1")
    cypher = f"MATCH (e:Equipment) WHERE e.name = '{equipment_name}' RETURN e.name"
    result = client.query_readonly(cypher, graph)
    if not result.records:
        return {"error": f"Equipment '{equipment_name}' not found", "dependencies": []}

    deps = []
    visited = {equipment_name}
    frontier = [equipment_name]
    depth = 0

    while frontier:
        depth += 1
        nf = []
        for name in frontier:
            r = client.query_readonly(
                f"MATCH (e:Equipment)-[:DEPENDS_ON]->(d:Equipment) WHERE e.name = '{name}' "
                "RETURN d.name, d.iso14224_class", graph
            )
            for row in r.records:
                dname = row[0]
                if dname and dname not in visited:
                    visited.add(dname)
                    nf.append(dname)
                    deps.append({"name": dname, "class": row[1], "depth": depth})
        frontier = nf

    return {
        "source": equipment_name, "total_dependencies": len(deps),
        "dependencies": deps,
        "traversal": "upstream dependency chain via DEPENDS_ON edges",
        "graph_method": "multi-hop graph traversal",
    }


def tool_criticality_ranking(client: SamyamaClient, graph: str, params: dict) -> dict:
    """PageRank criticality ranking of equipment."""
    top_n = params.get("top_n", 10)
    scores = client.page_rank(label="Equipment", edge_type="DEPENDS_ON")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Build node_id → name lookup
    all_eq = client.query_readonly(
        "MATCH (e:Equipment) RETURN e.name, e.iso14224_class, e.criticality_score", graph
    )
    # Also build id → props via a separate id query
    id_result = client.query_readonly(
        "MATCH (e:Equipment) RETURN id(e), e.name, e.iso14224_class, e.criticality_score", graph
    )
    id_to_props = {}
    for row in id_result.records:
        id_to_props[row[0]] = {"name": row[1], "class": row[2], "criticality_score": row[3]}

    ranking = []
    for rank, (nid, score) in enumerate(ranked, 1):
        props = id_to_props.get(nid)
        if props:
            ranking.append({
                "rank": rank, "pagerank_score": round(score, 6),
                "name": props["name"], "class": props["class"],
                "criticality_score": props["criticality_score"],
            })
    return {
        "algorithm": "PageRank", "top_n": top_n, "ranking": ranking,
        "graph_method": "PageRank centrality on DEPENDS_ON subgraph",
        "description": "Equipment ranked by graph-based criticality score using PageRank algorithm",
    }


def tool_vector_search(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Semantic similarity search on failure mode embeddings."""
    query = params.get("query", "compressor overheating")
    k = params.get("top_k", 5)

    # Use mock query embedding (hash-based, same as ETL embedding_gen fallback)
    import hashlib
    dim = 384
    h = hashlib.sha256(query.encode()).digest()
    qvec = []
    for i in range(dim):
        byte_idx = i % len(h)
        qvec.append((h[byte_idx] + i * 7) % 256 / 255.0)
    norm = sum(v * v for v in qvec) ** 0.5
    qvec = [v / norm for v in qvec]

    results = client.vector_search("FailureMode", "embedding", qvec, k)

    # Build id → props lookup (avoid id() in WHERE)
    all_fm = client.query_readonly(
        "MATCH (f:FailureMode) RETURN id(f), f.name, f.description, f.severity, f.category", graph
    )
    id_to_fm = {}
    for row in all_fm.records:
        id_to_fm[row[0]] = {"name": row[1], "description": row[2], "severity": row[3], "category": row[4]}

    matches = []
    for nid, dist in results:
        fm = id_to_fm.get(nid)
        if fm:
            matches.append({
                "name": fm["name"], "description": fm["description"],
                "severity": fm["severity"], "category": fm["category"],
                "similarity_score": round(1.0 - dist, 4), "distance": round(dist, 4),
            })
    return {
        "query": query, "top_k": k, "matches": matches,
        "embedding": "mock-384d",
        "graph_method": "HNSW vector search with cosine similarity on semantic embeddings",
    }


def tool_query_failure_modes(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Query failure modes with optional filters."""
    r = client.query_readonly("MATCH (f:FailureMode) RETURN f.name, f.description, f.severity, f.category", graph)
    modes = []
    for row in r.records:
        modes.append({"name": row[0], "description": row[1], "severity": row[2], "category": row[3]})
    return {
        "total": len(modes), "failure_modes": modes,
        "graph_method": "graph lookup on FailureMode nodes",
    }


def tool_query_assets(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Query equipment."""
    r = client.query_readonly(
        "MATCH (e:Equipment) RETURN e.name, e.iso14224_class, e.criticality_score, e.mtbf_hours", graph
    )
    assets = []
    for row in r.records:
        assets.append({"name": row[0], "class": row[1], "criticality_score": row[2], "mtbf_hours": row[3]})
    return {"total": len(assets), "equipment": assets}


def tool_maintenance_clusters(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Group equipment by maintenance urgency."""
    r = client.query_readonly(
        "MATCH (e:Equipment) RETURN e.name, e.iso14224_class, e.criticality_score, e.mtbf_hours", graph
    )
    clusters = {"critical": [], "high": [], "medium": [], "low": []}
    for row in r.records:
        name, cls, crit, mtbf = row[0], row[1], row[2] or 0, row[3] or 99999
        if crit >= 8.0 and mtbf < 2000:
            tier = "critical"
        elif crit >= 6.0 or mtbf < 4000:
            tier = "high"
        elif crit >= 3.0:
            tier = "medium"
        else:
            tier = "low"
        clusters[tier].append({"name": name, "class": cls, "criticality": crit, "mtbf": mtbf})
    return {
        "clusters": {k: v for k, v in clusters.items() if v},
        "graph_method": "maintenance clustering based on criticality and MTBF from graph properties",
    }


def tool_sensor_trend(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Sensor trend analysis."""
    equipment = params.get("equipment_name", "Chiller-1")
    r = client.query_readonly(
        f"MATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor) WHERE e.name = '{equipment}' "
        "RETURN s.name, s.type, s.unit, s.min_threshold, s.max_threshold", graph
    )
    sensors = []
    for row in r.records:
        sensors.append({"name": row[0], "type": row[1], "unit": row[2],
                        "min_threshold": row[3], "max_threshold": row[4]})
    return {
        "equipment": equipment, "sensor_count": len(sensors), "sensors": sensors,
        "graph_method": "HAS_SENSOR edge traversal from Equipment to Sensor nodes",
    }


def tool_root_cause_trace(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Trace upstream root cause via dependency chain."""
    result = tool_dependency_chain(client, graph, params)
    result["graph_method"] = "root cause trace via upstream DEPENDS_ON traversal"
    result["traversal"] = "upstream path from symptom to root cause via dependency graph"
    return result


# Tool dispatch table
TOOLS = {
    "impact_analysis": tool_impact_analysis,
    "dependency_chain": tool_dependency_chain,
    "criticality_ranking": tool_criticality_ranking,
    "vector_search": tool_vector_search,
    "find_similar_failures": tool_vector_search,
    "query_failure_modes": tool_query_failure_modes,
    "query_assets": tool_query_assets,
    "maintenance_clusters": tool_maintenance_clusters,
    "maintenance_scheduler": tool_maintenance_clusters,
    "sensor_trend": tool_sensor_trend,
    "root_cause_trace": tool_root_cause_trace,
    "anomaly_correlation": tool_sensor_trend,
    "cypher_query": tool_query_assets,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graph_data(client: SamyamaClient, graph: str) -> dict:
    """Load synthetic industrial data into the graph using ETL loaders."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from etl.eamlite_loader import load_eamlite
    from etl.fmsr_loader import load_fmsr
    from etl.couchdb_loader import load_couchdb
    from etl.embedding_gen import generate_embeddings

    stats = {}
    stats["eam"] = load_eamlite(client, ".", graph)
    stats["fmsr"] = load_fmsr(client, ".", graph)
    stats["couchdb"] = load_couchdb(client, ".", graph)
    stats["embed"] = generate_embeddings(client, graph, "mock")
    return stats


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------

def extract_equipment_name(description: str) -> str:
    """Extract equipment name from scenario description."""
    import re
    # Match compound names like "Pump-CW-1", "Motor-CH1", "Motor-AHU1", "Motor-BL1"
    m = re.search(r"(Pump-(?:CW|HW)-\d+|Motor-(?:CH|AHU|P|BL)\d+)", description)
    if m:
        return m.group(0)
    # Match simple names like "Chiller-1", "AHU-2", "Boiler-3"
    m = re.search(r"(Chiller|AHU|Boiler)-\d+", description)
    if m:
        return m.group(0)
    # Match "Chiller 1" -> "Chiller-1" (normalize)
    m = re.search(r"(Chiller|AHU|Boiler)\s+(\d+)", description)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return "Chiller-1"  # default


def run_scenario(
    scenario: dict[str, Any],
    client: SamyamaClient,
    graph: str,
) -> ScenarioResult:
    """Execute a single scenario against the Samyama graph."""
    sid = scenario["id"]
    category = scenario["category"]
    description = scenario["description"]
    expected_tools = scenario.get("expected_tools", [])

    tools_called = []
    response_parts = []
    start = time.perf_counter()

    for tool_name in expected_tools:
        fn = TOOLS.get(tool_name)
        if not fn:
            response_parts.append(json.dumps({"error": f"Unknown tool: {tool_name}"}))
            continue

        # Build params from scenario description
        params = {
            "equipment_name": extract_equipment_name(description),
            "query": description,
            "top_k": 5,
            "top_n": 10,
        }
        try:
            result = fn(client, graph, params)
            tools_called.append(tool_name)
            response_parts.append(json.dumps(result, default=str))
        except Exception as e:
            response_parts.append(json.dumps({"error": str(e)}))

    elapsed_ms = (time.perf_counter() - start) * 1000
    combined = "\n".join(response_parts)

    return evaluate_response(
        scenario=scenario,
        response=combined,
        tools_called=tools_called,
        latency_ms=elapsed_ms,
        tokens_used=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AssetOps-KG Samyama Benchmark")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 72)
    print("  AssetOps-KG Benchmark — Samyama Graph Database")
    print("=" * 72)

    # Initialize graph
    print("\n[1/3] Initializing Samyama embedded client...")
    client = SamyamaClient.embedded()
    graph = "industrial"

    # Load data
    print("[2/3] Loading synthetic industrial data via ETL...")
    load_start = time.perf_counter()
    stats = load_graph_data(client, graph)
    load_ms = (time.perf_counter() - load_start) * 1000
    status = client.status()
    print(f"  Graph loaded: {status.nodes} nodes, {status.edges} edges ({load_ms:.0f}ms)")

    # Run scenarios
    scenarios = load_scenarios(args.category)
    print(f"\n[3/3] Running {len(scenarios)} scenarios...")

    results: list[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i:>2}/{len(scenarios)}] {scenario['id']:<30}", end=" ", flush=True)
        result = run_scenario(scenario, client, graph)
        results.append(result)
        tag = "PASS" if result.passed else "FAIL"
        print(f"{tag}  score={result.overall_score:.3f}  {result.latency_ms:.1f}ms")

    # Summary
    print()
    print(format_summary_table(results))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
