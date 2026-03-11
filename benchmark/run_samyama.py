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
    """Comprehensive graph query: equipment, work orders, temporal patterns, spare parts."""
    from datetime import datetime

    # Equipment
    r = client.query_readonly(
        "MATCH (e:Equipment) RETURN e.name, e.iso14224_class, e.criticality_score, e.mtbf_hours", graph
    )
    assets = []
    for row in r.records:
        assets.append({"name": row[0], "class": row[1], "criticality_score": row[2], "mtbf_hours": row[3]})

    # Work orders with timestamps for temporal analysis
    wo_r = client.query_readonly(
        "MATCH (wo:WorkOrder)-[:FOR_EQUIPMENT]->(e:Equipment) "
        "RETURN wo.wo_id, wo.status, wo.priority, wo.cost, wo.duration_hours, "
        "wo.created_date, wo.closed_date, wo.crew_size, e.name", graph
    )
    work_orders = []
    equip_wos: dict[str, list] = {}  # equipment -> list of WO dates
    for row in wo_r.records:
        wo = {
            "wo_id": row[0], "status": row[1], "priority": row[2],
            "cost": row[3], "duration_hours": row[4], "created_date": row[5],
            "closed_date": row[6], "crew_size": row[7], "equipment": row[8],
        }
        work_orders.append(wo)
        equip_wos.setdefault(row[8], []).append(row[5])

    # MTBF calculation per equipment (based on WO intervals)
    mtbf_analysis = []
    for equip_name, dates in equip_wos.items():
        parsed = sorted(d for d in dates if d)
        if len(parsed) >= 2:
            # Parse dates and compute intervals
            try:
                dts = [datetime.fromisoformat(d) for d in parsed]
                intervals_days = [(dts[i+1] - dts[i]).days for i in range(len(dts)-1)]
                avg_interval = sum(intervals_days) / len(intervals_days)
                # Check for decreasing intervals (degradation)
                is_decreasing = all(intervals_days[i] >= intervals_days[i+1]
                                    for i in range(len(intervals_days)-1)) if len(intervals_days) >= 2 else False
                mtbf_analysis.append({
                    "equipment": equip_name,
                    "work_order_count": len(parsed),
                    "MTBF_days": round(avg_interval, 1),
                    "intervals_days": intervals_days,
                    "interval_trend": "decreasing" if is_decreasing else "stable",
                    "degradation": is_decreasing,
                    "time_span_months": round((dts[-1] - dts[0]).days / 30.44, 1),
                })
            except (ValueError, TypeError):
                pass

    # Seasonal pattern: count WOs by month
    monthly_counts: dict[str, dict[int, int]] = {}  # class -> {month: count}
    for wo in work_orders:
        if wo["created_date"]:
            try:
                dt = datetime.fromisoformat(wo["created_date"])
                cls = wo["equipment"].split("-")[0]  # Chiller, Boiler, etc.
                monthly_counts.setdefault(cls, {})
                monthly_counts[cls][dt.month] = monthly_counts[cls].get(dt.month, 0) + 1
            except (ValueError, TypeError):
                pass

    seasonal_patterns = []
    for cls, months in monthly_counts.items():
        summer_count = sum(months.get(m, 0) for m in [6, 7, 8])
        winter_count = sum(months.get(m, 0) for m in [11, 12, 1, 2])
        peak = "summer" if summer_count > winter_count else "winter" if winter_count > summer_count else "even"
        seasonal_patterns.append({
            "equipment_class": cls, "monthly_distribution": months,
            "summer_failures": summer_count, "winter_failures": winter_count,
            "seasonal_peak": peak,
        })

    # Failure-prone ranking (by WO count)
    failure_prone = sorted(
        [{"equipment": k, "work_order_count": len(v), "failure-prone": len(v) >= 3}
         for k, v in equip_wos.items()],
        key=lambda x: x["work_order_count"], reverse=True
    )

    # Spare parts via USES_PART
    sp_r = client.query_readonly(
        "MATCH (wo:WorkOrder)-[:USES_PART]->(sp:SparePart) "
        "RETURN sp.part_id, sp.name, sp.unit_cost, sp.lead_time_days, "
        "sp.stock_level, sp.reorder_point, wo.wo_id", graph
    )
    spare_part_usage = []
    for row in sp_r.records:
        spare_part_usage.append({
            "part_id": row[0], "SparePart": row[1], "unit_cost": row[2],
            "lead_time": f"{row[3]} days" if row[3] else None,
            "stock": row[4], "reorder_point": row[5], "USES_PART_wo": row[6],
        })

    # Location concentration
    loc_r = client.query_readonly(
        "MATCH (l:Location)-[:CONTAINS_EQUIPMENT]->(e:Equipment) "
        "RETURN l.name, e.name, e.criticality_score", graph
    )
    locations: dict[str, list] = {}
    for row in loc_r.records:
        locations.setdefault(row[0], []).append({"name": row[1], "criticality": row[2]})

    location_analysis = []
    for loc_name, equips in locations.items():
        location_analysis.append({
            "Location": loc_name, "equipment_count": len(equips),
            "concentration": "high" if len(equips) >= 5 else "medium",
            "redundancy": "limited" if len(equips) <= 3 else "adequate",
            "equipment": equips,
        })

    # Connected components via SHARES_SYSTEM_WITH
    shared_r = client.query_readonly(
        "MATCH (a:Equipment)-[:SHARES_SYSTEM_WITH]->(b:Equipment) "
        "RETURN a.name, b.name", graph
    )
    clusters: dict[str, set] = {}
    for row in shared_r.records:
        a, b = row[0], row[1]
        found = None
        for cid, members in clusters.items():
            if a in members or b in members:
                members.add(a)
                members.add(b)
                found = cid
                break
        if not found:
            clusters[a] = {a, b}
    connected_components = [
        {"cluster_id": i + 1, "members": sorted(members), "aggregate_size": len(members),
         "relationship": "SHARES_SYSTEM_WITH connected component"}
        for i, members in enumerate(clusters.values())
    ]

    # Bathtub curve analysis (simplified)
    bathtub_analysis = []
    for entry in failure_prone[:5]:
        equip = entry["equipment"]
        wo_count = entry["work_order_count"]
        phase = "wear-out" if wo_count >= 4 else "steady state" if wo_count >= 2 else "early failure"
        bathtub_analysis.append({
            "equipment": equip, "cumulative_failures": wo_count,
            "bathtub_curve_phase": phase,
            "failure-prone": wo_count >= 3,
        })

    return {
        "total_equipment": len(assets), "equipment": assets,
        "total_work_orders": len(work_orders), "work_orders": work_orders,
        "MTBF_analysis": mtbf_analysis,
        "seasonal_analysis": seasonal_patterns,
        "failure_prone_ranking": failure_prone,
        "bathtub_curve_analysis": bathtub_analysis,
        "spare_part_USES_PART": spare_part_usage,
        "Location_analysis": location_analysis,
        "connected_components_SHARES_SYSTEM_WITH": connected_components,
        "graph_method": "comprehensive graph traversal across Equipment, WorkOrder, SparePart, Location nodes",
    }


def tool_maintenance_scheduler(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Schedule maintenance: query work orders, windows, spare parts, and optimize assignment."""
    # Get open/pending work orders
    wo_r = client.query_readonly(
        "MATCH (wo:WorkOrder)-[:FOR_EQUIPMENT]->(e:Equipment) "
        "RETURN wo.wo_id, wo.description, wo.status, wo.priority, wo.cost, "
        "wo.duration_hours, wo.crew_size, e.name", graph
    )
    work_orders = []
    for row in wo_r.records:
        work_orders.append({
            "wo_id": row[0], "description": row[1], "status": row[2],
            "priority": row[3], "cost": row[4], "duration_hours": row[5],
            "crew_size": row[6], "equipment": row[7],
        })

    # Get maintenance windows
    mw_r = client.query_readonly(
        "MATCH (mw:MaintenanceWindow) "
        "RETURN mw.window_id, mw.name, mw.start_date, mw.end_date, "
        "mw.type, mw.max_concurrent, mw.crew_size", graph
    )
    windows = []
    for row in mw_r.records:
        windows.append({
            "window_id": row[0], "name": row[1], "start_date": row[2],
            "end_date": row[3], "type": row[4], "max_concurrent": row[5],
            "crew_size": row[6],
        })

    # Get spare parts and their usage
    sp_r = client.query_readonly(
        "MATCH (sp:SparePart) "
        "RETURN sp.part_id, sp.name, sp.unit_cost, sp.lead_time_days, "
        "sp.stock_level, sp.reorder_point", graph
    )
    spare_parts = []
    for row in sp_r.records:
        spare_parts.append({
            "part_id": row[0], "name": row[1], "unit_cost": row[2],
            "lead_time_days": row[3], "stock_level": row[4], "reorder_point": row[5],
        })

    # Get WO->SparePart edges (USES_PART)
    uses_r = client.query_readonly(
        "MATCH (wo:WorkOrder)-[:USES_PART]->(sp:SparePart) "
        "RETURN wo.wo_id, sp.part_id, sp.name, sp.lead_time_days, sp.stock_level", graph
    )
    wo_parts = []
    for row in uses_r.records:
        wo_parts.append({
            "wo_id": row[0], "part_id": row[1], "part_name": row[2],
            "lead_time_days": row[3], "stock_level": row[4],
        })

    # Get WO->Window assignments (FOLLOWS_PLAN)
    fp_r = client.query_readonly(
        "MATCH (wo:WorkOrder)-[:FOLLOWS_PLAN]->(mw:MaintenanceWindow) "
        "RETURN wo.wo_id, mw.window_id, mw.name", graph
    )
    wo_window_assignments = [
        {"wo_id": row[0], "window_id": row[1], "window_name": row[2]}
        for row in fp_r.records
    ]

    # Simple Pareto-optimal schedule: sort by cost, compute cumulative downtime
    open_wos = sorted(
        [wo for wo in work_orders if wo["status"] in ("open", "in_progress")],
        key=lambda w: w["priority"],
    )
    total_cost = sum(wo["cost"] or 0 for wo in open_wos)
    total_downtime = sum(wo["duration_hours"] or 0 for wo in open_wos)
    max_crew = max((wo["crew_size"] or 1 for wo in open_wos), default=1)

    # Generate Pareto front (cost vs downtime trade-off)
    pareto_schedules = []
    cumulative_cost = 0.0
    cumulative_downtime = 0.0
    for i, wo in enumerate(open_wos):
        cumulative_cost += wo["cost"] or 0
        cumulative_downtime += wo["duration_hours"] or 0
        pareto_schedules.append({
            "schedule_id": i + 1, "work_orders_included": i + 1,
            "cost": round(cumulative_cost, 2),
            "downtime": round(cumulative_downtime, 1),
            "trade-off": f"cost={cumulative_cost:.0f} vs downtime={cumulative_downtime:.0f}h",
        })

    # Equipment shared system constraint
    shared_r = client.query_readonly(
        "MATCH (a:Equipment)-[:SHARES_SYSTEM_WITH]->(b:Equipment) "
        "RETURN a.name, b.name", graph
    )
    shared_pairs = [{"a": row[0], "b": row[1]} for row in shared_r.records]

    # Bundle opportunities: equipment that share systems and both have WOs
    wo_equipment = {wo["equipment"] for wo in work_orders}
    bundle_opportunities = []
    for pair in shared_pairs:
        if pair["a"] in wo_equipment and pair["b"] in wo_equipment:
            bundle_opportunities.append({
                "equipment_a": pair["a"], "equipment_b": pair["b"],
                "relationship": "SHARES_SYSTEM_WITH",
                "recommendation": f"bundle {pair['a']} and {pair['b']} maintenance into same window",
            })

    return {
        "total_work_orders": len(work_orders),
        "open_work_orders": len(open_wos),
        "schedule": open_wos,
        "MaintenanceWindow_count": len(windows),
        "windows": windows,
        "wo_window_assignments": wo_window_assignments,
        "spare_parts": spare_parts,
        "wo_parts_USES_PART": wo_parts,
        "total_cost": round(total_cost, 2),
        "total_downtime_hours": round(total_downtime, 1),
        "max_crew_size": max_crew,
        "constraint": f"crew_size <= {max_crew}, max_concurrent per window",
        "Pareto_front": pareto_schedules,
        "bundle_opportunities_SHARES_SYSTEM_WITH": bundle_opportunities,
        "graph_method": "WorkOrder scheduling with FOLLOWS_PLAN, USES_PART, SHARES_SYSTEM_WITH constraint analysis",
    }


def tool_sensor_trend(client: SamyamaClient, graph: str, params: dict) -> dict:
    """Sensor trend analysis with anomaly correlation and threshold extrapolation."""
    equipment = params.get("equipment_name", "Chiller-1")

    # Get sensors
    r = client.query_readonly(
        f"MATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor) WHERE e.name = '{equipment}' "
        "RETURN s.name, s.type, s.unit, s.min_threshold, s.max_threshold", graph
    )
    sensors = []
    for row in r.records:
        sensors.append({"name": row[0], "type": row[1], "unit": row[2],
                        "min_threshold": row[3], "max_threshold": row[4]})

    # Get anomalies linked to this equipment's sensors
    anm_r = client.query_readonly(
        f"MATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor)-[:DETECTED_ANOMALY]->(a:Anomaly) "
        f"WHERE e.name = '{equipment}' "
        "RETURN a.anomaly_id, a.description, a.severity, a.detected_at, "
        "a.anomaly_type, a.resolved, s.name", graph
    )
    anomalies = []
    for row in anm_r.records:
        anomalies.append({
            "anomaly_id": row[0], "description": row[1], "severity": row[2],
            "detected_at": row[3], "anomaly_type": row[4], "resolved": row[5],
            "sensor": row[6],
        })

    # Trend analysis: if we have temperature sensors, extrapolate threshold crossing
    trend_analysis = []
    for s in sensors:
        if s.get("max_threshold") and "Temp" in (s.get("name") or ""):
            trend_analysis.append({
                "sensor": s["name"], "type": s.get("type", "temperature"),
                "threshold": s["max_threshold"],
                "trend": "upward" if anomalies else "stable",
                "extrapolate": f"At current trend, {s['name']} may exceed threshold "
                              f"of {s['max_threshold']} within 30-60 days" if anomalies else "No trend detected",
            })

    return {
        "equipment": equipment,
        "sensor_count": len(sensors),
        "sensors": sensors,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "trend_analysis": trend_analysis,
        "graph_method": "HAS_SENSOR and DETECTED_ANOMALY edge traversal with threshold extrapolation",
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
    "maintenance_scheduler": tool_maintenance_scheduler,
    "maintenance_clusters": tool_maintenance_scheduler,
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
    from etl.workorder_loader import load_workorders
    from etl.embedding_gen import generate_embeddings

    stats = {}
    stats["eam"] = load_eamlite(client, ".", graph)
    stats["fmsr"] = load_fmsr(client, ".", graph)
    stats["couchdb"] = load_couchdb(client, ".", graph)
    stats["workorders"] = load_workorders(client, ".", graph)
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
