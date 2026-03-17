"""Handler for 75 PHM (Prognostics and Health Management) scenarios.

Categories:
    - RUL Prediction: Remaining useful life estimation
    - Fault Classification: Identifying fault type from symptoms
    - Engine Health Analysis: Overall health assessment
    - Safety/Policy Evaluation: Compliance and safety checks
    - Cost-Benefit Analysis: Maintenance strategy economics

The KG provides context (asset specs, failure history, maintenance records,
sensor baselines) while predictions combine KG data with domain knowledge
templates.
"""

from __future__ import annotations

import re
import time
from typing import Any

from samyama import SamyamaClient


# ---------------------------------------------------------------------------
# PHM category detection
# ---------------------------------------------------------------------------

_PHM_CATEGORIES = {
    "rul_prediction": [
        "remaining useful life", "rul ", "predict.*life",
        "how long.*last", "time to failure", "end of life",
        "degradation.*predict", "prognos",
    ],
    "fault_classification": [
        "fault classif", "fault diagnos", "fault detect",
        "classify.*fault", "diagnose.*fault", "identify.*fault",
        "what.*fault", "type of fault", "fault type",
        "root cause", "failure classif",
    ],
    "engine_health": [
        "health.*analy", "health.*assess", "health.*monitor",
        "health index", "health score", "health status",
        "condition.*assess", "condition.*monitor",
        "engine health", "overall health", "asset health",
    ],
    "safety_policy": [
        "safety.*evaluat", "safety.*assess", "safety.*policy",
        "compliance", "regulation", "standard",
        "maintenance policy", "maintenance standard",
        "inspection.*policy", "risk.*assess", "risk.*evaluat",
    ],
    "cost_benefit": [
        "cost.*benefit", "cost.*analy", "economic.*analy",
        "maintenance.*cost", "total cost", "lifecycle cost",
        "return on investment", "roi ", "payback",
        "preventive vs corrective", "maintenance strategy",
        "cost effective", "budget",
    ],
}


def _classify_phm_category(text: str) -> str:
    """Determine which PHM sub-category a scenario belongs to."""
    text_lower = text.lower()
    best_cat = "engine_health"  # default
    best_score = 0
    for cat, patterns in _PHM_CATEGORIES.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


# ---------------------------------------------------------------------------
# Equipment extraction
# ---------------------------------------------------------------------------

_EQUIPMENT_RE = re.compile(
    r"((?:Chiller|AHU|Boiler|Pump|Motor|Turbine|Compressor|Generator"
    r"|Fan|Conveyor|Gearbox|Engine|Bearing)(?:[- ]\w+)?)",
    re.IGNORECASE,
)


def _extract_equipment(text: str) -> str | None:
    """Extract equipment identifier from scenario text."""
    m = _EQUIPMENT_RE.search(text)
    if m:
        return m.group(1).strip()
    # Try generic asset patterns
    m2 = re.search(r"asset\s+(\S+(?:\s+\S+)?)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# KG query helpers
# ---------------------------------------------------------------------------

def _query_failure_history(
    client: SamyamaClient, graph: str, equipment_name: str,
) -> list[dict[str, Any]]:
    """Query historical failure events and work orders for equipment."""
    cypher = (
        "MATCH (wo:WorkOrder)-[:FOR_EQUIPMENT]->(e:Equipment) "
        f"WHERE e.name CONTAINS '{equipment_name}' "
        "RETURN wo.wo_id, wo.description, wo.status, wo.primary_code, "
        "wo.actual_finish, wo.preventive, wo.wo_type, wo.cost, wo.duration_hours"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    history = []
    for row in result.records:
        history.append({
            "wo_id": row[0], "description": row[1], "status": row[2],
            "primary_code": row[3], "actual_finish": row[4],
            "preventive": row[5], "wo_type": row[6],
            "cost": row[7], "duration_hours": row[8],
        })
    return history


def _query_maintenance_records(
    client: SamyamaClient, graph: str, equipment_name: str,
) -> list[dict[str, Any]]:
    """Query maintenance records linked to equipment."""
    cypher = (
        "MATCH (e:Equipment)-[:HAS_MAINTENANCE]->(m:MaintenanceRecord) "
        f"WHERE e.name CONTAINS '{equipment_name}' "
        "RETURN m.record_id, m.type, m.description, m.date, "
        "m.cost, m.duration_hours, m.outcome"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    records = []
    for row in result.records:
        records.append({
            "record_id": row[0], "type": row[1], "description": row[2],
            "date": row[3], "cost": row[4], "duration_hours": row[5],
            "outcome": row[6],
        })
    return records


def _query_asset_profile(
    client: SamyamaClient, graph: str, equipment_name: str,
) -> dict[str, Any]:
    """Query asset specifications, MTBF, criticality, and sensor baseline."""
    # Equipment properties
    cypher = (
        "MATCH (e:Equipment) "
        f"WHERE e.name CONTAINS '{equipment_name}' "
        "RETURN e.name, e.iso14224_class, e.criticality_score, "
        "e.mtbf_hours, e.equipment_id, e.type"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return {"name": equipment_name, "found": False}

    if not result.records:
        return {"name": equipment_name, "found": False}

    row = result.records[0]
    profile: dict[str, Any] = {
        "name": row[0], "iso14224_class": row[1],
        "criticality_score": row[2], "mtbf_hours": row[3],
        "equipment_id": row[4], "type": row[5], "found": True,
    }

    # Sensor count
    try:
        sr = client.query_readonly(
            f"MATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor) "
            f"WHERE e.name CONTAINS '{equipment_name}' "
            f"RETURN count(s)",
            graph,
        )
        profile["sensor_count"] = sr.records[0][0] if sr.records else 0
    except Exception:
        profile["sensor_count"] = 0

    return profile


def _query_failure_modes(
    client: SamyamaClient, graph: str, equipment_name: str,
) -> list[dict[str, Any]]:
    """Query failure modes and their signatures from KG."""
    cypher = (
        "MATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor)-[:MONITORS]->(fm:FailureMode) "
        f"WHERE e.name CONTAINS '{equipment_name}' "
        "RETURN fm.name, fm.description, fm.severity, fm.category, "
        "collect(s.sensor_type)"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    modes = []
    for row in result.records:
        modes.append({
            "name": row[0], "description": row[1], "severity": row[2],
            "category": row[3], "sensor_signatures": row[4] if row[4] else [],
        })

    # Fallback: query FailureMode nodes by asset_type
    if not modes:
        try:
            fm_r = client.query_readonly(
                "MATCH (fm:FailureMode) "
                f"WHERE fm.asset_type CONTAINS '{equipment_name}' "
                "RETURN fm.name, fm.description, fm.severity, fm.category",
                graph,
            )
            for row in fm_r.records:
                modes.append({
                    "name": row[0], "description": row[1],
                    "severity": row[2], "category": row[3],
                    "sensor_signatures": [],
                })
        except Exception:
            pass

    return modes


def _query_maintenance_policies(
    client: SamyamaClient, graph: str, equipment_name: str,
) -> list[dict[str, Any]]:
    """Query maintenance policies and compliance rules from KG."""
    cypher = (
        "MATCH (e:Equipment)-[:SUBJECT_TO]->(p:MaintenancePolicy) "
        f"WHERE e.name CONTAINS '{equipment_name}' "
        "RETURN p.policy_id, p.name, p.description, p.frequency, "
        "p.compliance_standard, p.last_audit"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    policies = []
    for row in result.records:
        policies.append({
            "policy_id": row[0], "name": row[1], "description": row[2],
            "frequency": row[3], "compliance_standard": row[4],
            "last_audit": row[5],
        })
    return policies


def _query_similar_equipment_history(
    client: SamyamaClient, graph: str, equipment_name: str,
) -> list[dict[str, Any]]:
    """Query failure history of similar equipment (same class) for comparison."""
    # Get equipment class
    try:
        cr = client.query_readonly(
            f"MATCH (e:Equipment) WHERE e.name CONTAINS '{equipment_name}' "
            "RETURN e.iso14224_class",
            graph,
        )
        if not cr.records or not cr.records[0][0]:
            return []
        eq_class = cr.records[0][0]
    except Exception:
        return []

    # Find similar equipment and their work order counts
    cypher = (
        "MATCH (e:Equipment) "
        f"WHERE e.iso14224_class = '{eq_class}' "
        f"AND NOT e.name CONTAINS '{equipment_name}' "
        "RETURN e.name, e.mtbf_hours, e.criticality_score"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    similar = []
    for row in result.records:
        similar.append({
            "name": row[0], "mtbf_hours": row[1],
            "criticality_score": row[2],
        })
    return similar


# ---------------------------------------------------------------------------
# Sub-handlers by PHM category
# ---------------------------------------------------------------------------

def _handle_rul_prediction(
    client: SamyamaClient, graph: str, equipment_name: str,
    profile: dict, failure_history: list, maintenance_records: list,
    failure_modes: list, similar_equipment: list,
) -> dict[str, Any]:
    """Build RUL prediction response from KG context."""
    tools = ["query_asset_profile", "query_failure_history",
             "query_similar_equipment"]

    # Compute MTBF from failure history
    dates = sorted(
        wo.get("actual_finish") for wo in failure_history
        if wo.get("actual_finish")
    )
    mtbf_estimate = None
    intervals = []
    if len(dates) >= 2:
        from datetime import datetime
        try:
            dts = [datetime.fromisoformat(d) for d in dates]
            intervals = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
            mtbf_estimate = sum(intervals) / len(intervals) if intervals else None
        except (ValueError, TypeError):
            pass

    # Compare with similar equipment
    sim_mtbf_values = [
        s["mtbf_hours"] for s in similar_equipment
        if s.get("mtbf_hours")
    ]
    fleet_avg_mtbf = (
        sum(sim_mtbf_values) / len(sim_mtbf_values) if sim_mtbf_values else None
    )

    # Degradation trend
    trend = "stable"
    if len(intervals) >= 3:
        if all(intervals[i] >= intervals[i + 1] for i in range(len(intervals) - 1)):
            trend = "accelerating_degradation"
        elif intervals[-1] < intervals[0] * 0.5:
            trend = "degrading"

    response_lines = [
        f"Remaining Useful Life (RUL) prediction for {equipment_name}:",
        "",
        "Asset Health Profile:",
        f"  Equipment class: {profile.get('iso14224_class', 'N/A')}",
        f"  Criticality score: {profile.get('criticality_score', 'N/A')}",
        f"  Catalog MTBF: {profile.get('mtbf_hours', 'N/A')} hours",
        f"  Sensor count: {profile.get('sensor_count', 0)}",
        "",
        "Failure Pattern Analysis:",
        f"  Historical failures: {len(failure_history)}",
        f"  Observed MTBF: {mtbf_estimate:.0f} days" if mtbf_estimate else
        "  Observed MTBF: insufficient data",
        f"  Degradation trend: {trend}",
        f"  Failure intervals (days): {intervals}" if intervals else
        "  Failure intervals: no data",
    ]

    if failure_modes:
        response_lines.append("")
        response_lines.append("Known failure modes:")
        for fm in failure_modes[:5]:
            response_lines.append(
                f"  - {fm.get('name', 'Unknown')} "
                f"(severity: {fm.get('severity', 'N/A')})"
            )

    if similar_equipment:
        response_lines.append("")
        response_lines.append(
            f"Fleet comparison ({len(similar_equipment)} similar assets):"
        )
        if fleet_avg_mtbf:
            response_lines.append(
                f"  Fleet average MTBF: {fleet_avg_mtbf:.0f} hours"
            )
        for se in similar_equipment[:3]:
            response_lines.append(
                f"  - {se['name']}: MTBF={se.get('mtbf_hours', 'N/A')}h, "
                f"criticality={se.get('criticality_score', 'N/A')}"
            )

    return {
        "response": "\n".join(response_lines),
        "tools_used": tools,
        "category": "rul_prediction",
        "mtbf_estimate_days": mtbf_estimate,
        "degradation_trend": trend,
        "failure_count": len(failure_history),
        "fleet_avg_mtbf_hours": fleet_avg_mtbf,
        "similar_equipment_count": len(similar_equipment),
    }


def _handle_fault_classification(
    client: SamyamaClient, graph: str, equipment_name: str,
    profile: dict, failure_history: list, failure_modes: list,
) -> dict[str, Any]:
    """Build fault classification response from KG context."""
    tools = ["query_asset_profile", "query_failure_modes", "query_failure_history"]

    # Build fault signature table
    fault_signatures = []
    for fm in failure_modes:
        sigs = fm.get("sensor_signatures", [])
        fault_signatures.append({
            "fault_type": fm.get("name"),
            "description": fm.get("description"),
            "severity": fm.get("severity"),
            "sensor_indicators": sigs,
        })

    # Historical fault distribution
    from collections import Counter
    code_counts = Counter(
        wo.get("primary_code") for wo in failure_history
        if wo.get("primary_code")
    )

    response_lines = [
        f"Fault classification analysis for {equipment_name}:",
        "",
        f"Known fault types: {len(failure_modes)}",
        "",
    ]

    if fault_signatures:
        response_lines.append("Fault Signatures:")
        for i, fs in enumerate(fault_signatures, 1):
            response_lines.append(f"  {i}. {fs['fault_type']}")
            response_lines.append(f"     {fs['description']}")
            if fs["sensor_indicators"]:
                response_lines.append(
                    f"     Sensor indicators: "
                    f"{', '.join(str(s) for s in fs['sensor_indicators'])}"
                )
            response_lines.append(f"     Severity: {fs['severity']}")

    if code_counts:
        response_lines.append("")
        response_lines.append("Historical fault distribution:")
        for code, count in code_counts.most_common():
            response_lines.append(f"  - {code}: {count} occurrences")

    return {
        "response": "\n".join(response_lines),
        "tools_used": tools,
        "category": "fault_classification",
        "fault_signatures": fault_signatures,
        "historical_distribution": dict(code_counts),
    }


def _handle_engine_health(
    client: SamyamaClient, graph: str, equipment_name: str,
    profile: dict, failure_history: list, maintenance_records: list,
) -> dict[str, Any]:
    """Build engine/asset health assessment from KG context."""
    tools = ["query_asset_profile", "query_failure_history",
             "query_maintenance_records"]

    # Compute health index (simplified)
    criticality = profile.get("criticality_score") or 0.5
    failure_count = len(failure_history)

    # More failures = lower health
    failure_penalty = min(failure_count * 0.1, 0.5)
    health_index = max(0.0, min(1.0, 1.0 - failure_penalty))

    # Maintenance ratio (preventive vs corrective)
    preventive_count = sum(
        1 for wo in failure_history
        if str(wo.get("preventive", "")).lower() in ("true", "1", "yes")
    )
    corrective_count = failure_count - preventive_count
    maint_ratio = (
        preventive_count / failure_count if failure_count > 0 else 0.0
    )

    # Health status
    if health_index >= 0.8:
        health_status = "Good"
    elif health_index >= 0.5:
        health_status = "Fair"
    else:
        health_status = "Poor"

    response_lines = [
        f"Health assessment for {equipment_name}:",
        "",
        f"Health index: {health_index:.2f} ({health_status})",
        f"Criticality score: {criticality}",
        f"Equipment class: {profile.get('iso14224_class', 'N/A')}",
        "",
        "Maintenance History Summary:",
        f"  Total work orders: {failure_count}",
        f"  Preventive: {preventive_count}",
        f"  Corrective: {corrective_count}",
        f"  Preventive/total ratio: {maint_ratio:.2f}",
        f"  Maintenance records: {len(maintenance_records)}",
    ]

    # Cost summary
    total_cost = sum(
        wo.get("cost") or 0 for wo in failure_history
        if wo.get("cost")
    )
    total_downtime = sum(
        wo.get("duration_hours") or 0 for wo in failure_history
        if wo.get("duration_hours")
    )
    if total_cost or total_downtime:
        response_lines.append("")
        response_lines.append("Cost Impact:")
        response_lines.append(f"  Total maintenance cost: ${total_cost:,.2f}")
        response_lines.append(
            f"  Total downtime: {total_downtime:.1f} hours"
        )

    return {
        "response": "\n".join(response_lines),
        "tools_used": tools,
        "category": "engine_health",
        "health_index": round(health_index, 3),
        "health_status": health_status,
        "failure_count": failure_count,
        "preventive_ratio": round(maint_ratio, 3),
        "total_cost": total_cost,
        "total_downtime_hours": total_downtime,
    }


def _handle_safety_policy(
    client: SamyamaClient, graph: str, equipment_name: str,
    profile: dict, failure_history: list, policies: list,
) -> dict[str, Any]:
    """Build safety/policy evaluation from KG context."""
    tools = ["query_asset_profile", "query_maintenance_policies",
             "query_failure_history"]

    response_lines = [
        f"Safety and policy evaluation for {equipment_name}:",
        "",
        f"Equipment class: {profile.get('iso14224_class', 'N/A')}",
        f"Criticality: {profile.get('criticality_score', 'N/A')}",
    ]

    if policies:
        response_lines.append("")
        response_lines.append(f"Applicable maintenance policies ({len(policies)}):")
        for p in policies:
            response_lines.append(f"  - {p.get('name', 'N/A')}")
            if p.get("compliance_standard"):
                response_lines.append(
                    f"    Standard: {p['compliance_standard']}"
                )
            if p.get("frequency"):
                response_lines.append(
                    f"    Required frequency: {p['frequency']}"
                )
    else:
        response_lines.append("")
        response_lines.append("No specific maintenance policies found in KG.")
        response_lines.append("Recommended standards:")
        response_lines.append("  - ISO 14224: Equipment reliability data collection")
        response_lines.append("  - ISO 55000: Asset management systems")
        response_lines.append("  - OSHA machinery safety requirements")

    # Compliance assessment from failure history
    overdue_count = sum(
        1 for wo in failure_history
        if str(wo.get("status", "")).lower() in ("overdue", "past_due")
    )
    if overdue_count:
        response_lines.append("")
        response_lines.append(
            f"COMPLIANCE WARNING: {overdue_count} overdue maintenance items"
        )

    # Risk assessment
    criticality = profile.get("criticality_score") or 0.5
    risk_level = "High" if criticality >= 0.8 else "Medium" if criticality >= 0.5 else "Low"
    response_lines.append("")
    response_lines.append(f"Risk assessment: {risk_level}")
    response_lines.append(
        f"Based on criticality score of {criticality} and "
        f"{len(failure_history)} historical failures"
    )

    return {
        "response": "\n".join(response_lines),
        "tools_used": tools,
        "category": "safety_policy",
        "policies": policies,
        "policy_count": len(policies),
        "overdue_maintenance": overdue_count,
        "risk_level": risk_level,
    }


def _handle_cost_benefit(
    client: SamyamaClient, graph: str, equipment_name: str,
    profile: dict, failure_history: list, maintenance_records: list,
) -> dict[str, Any]:
    """Build cost-benefit analysis from KG context."""
    tools = ["query_asset_profile", "query_failure_history",
             "query_maintenance_records"]

    # Separate preventive vs corrective
    preventive_wos = [
        wo for wo in failure_history
        if str(wo.get("preventive", "")).lower() in ("true", "1", "yes")
    ]
    corrective_wos = [
        wo for wo in failure_history
        if str(wo.get("preventive", "")).lower() not in ("true", "1", "yes")
    ]

    prev_cost = sum(wo.get("cost") or 0 for wo in preventive_wos)
    corr_cost = sum(wo.get("cost") or 0 for wo in corrective_wos)
    total_cost = prev_cost + corr_cost

    prev_downtime = sum(wo.get("duration_hours") or 0 for wo in preventive_wos)
    corr_downtime = sum(wo.get("duration_hours") or 0 for wo in corrective_wos)
    total_downtime = prev_downtime + corr_downtime

    # Cost ratio
    cost_ratio = prev_cost / corr_cost if corr_cost > 0 else 0.0

    # Recommendation
    if cost_ratio < 0.3 and len(corrective_wos) > len(preventive_wos):
        recommendation = (
            "Increase preventive maintenance frequency. Current corrective "
            "costs significantly exceed preventive, indicating reactive strategy."
        )
    elif cost_ratio > 2.0:
        recommendation = (
            "Review preventive maintenance scope. Preventive costs are high "
            "relative to corrective, which may indicate over-maintenance."
        )
    else:
        recommendation = (
            "Maintenance strategy appears balanced. Continue monitoring "
            "cost trends for optimization opportunities."
        )

    response_lines = [
        f"Cost-benefit analysis for {equipment_name}:",
        "",
        "Maintenance Cost Breakdown:",
        f"  Preventive maintenance: ${prev_cost:,.2f} "
        f"({len(preventive_wos)} work orders, {prev_downtime:.1f}h downtime)",
        f"  Corrective maintenance: ${corr_cost:,.2f} "
        f"({len(corrective_wos)} work orders, {corr_downtime:.1f}h downtime)",
        f"  Total: ${total_cost:,.2f} ({total_downtime:.1f}h total downtime)",
        "",
        f"Preventive/corrective cost ratio: {cost_ratio:.2f}",
        "",
        f"Recommendation: {recommendation}",
    ]

    return {
        "response": "\n".join(response_lines),
        "tools_used": tools,
        "category": "cost_benefit",
        "preventive_cost": prev_cost,
        "corrective_cost": corr_cost,
        "total_cost": total_cost,
        "preventive_downtime_hours": prev_downtime,
        "corrective_downtime_hours": corr_downtime,
        "cost_ratio": round(cost_ratio, 3),
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

def handle_phm(
    client: SamyamaClient,
    scenario: dict[str, Any],
    tenant: str = "default",
) -> dict[str, Any]:
    """Handle PHM (Prognostics and Health Management) scenarios.

    Classifies the scenario into a PHM sub-category (RUL, fault
    classification, health analysis, safety/policy, cost-benefit),
    queries the KG for relevant asset context, and produces a
    structured response.

    Parameters
    ----------
    client : SamyamaClient
        Samyama Python SDK client (embedded or remote).
    scenario : dict
        Scenario dict with at least ``text`` (or ``description``) and ``id``.
    tenant : str
        Graph / tenant name (default ``"default"``).

    Returns
    -------
    dict
        ``{"response": ..., "tools_used": [...], "latency_ms": ...}``
    """
    text = scenario.get("text") or scenario.get("description", "")
    sid = scenario.get("id", "?")
    graph = tenant
    tools_used: list[str] = []
    start = time.perf_counter()

    try:
        # Classify PHM sub-category
        phm_category = _classify_phm_category(text)
        equipment_name = _extract_equipment(text)

        if not equipment_name:
            # Try a broader extraction
            equipment_name = "Equipment"

        # Query common KG context
        profile = _query_asset_profile(client, graph, equipment_name)
        tools_used.append("query_asset_profile")

        failure_history = _query_failure_history(client, graph, equipment_name)
        tools_used.append("query_failure_history")

        # Dispatch to sub-handler
        if phm_category == "rul_prediction":
            maintenance_records = _query_maintenance_records(
                client, graph, equipment_name,
            )
            tools_used.append("query_maintenance_records")

            failure_modes = _query_failure_modes(client, graph, equipment_name)
            tools_used.append("query_failure_modes")

            similar = _query_similar_equipment_history(
                client, graph, equipment_name,
            )
            tools_used.append("query_similar_equipment")

            sub_result = _handle_rul_prediction(
                client, graph, equipment_name, profile,
                failure_history, maintenance_records, failure_modes, similar,
            )

        elif phm_category == "fault_classification":
            failure_modes = _query_failure_modes(client, graph, equipment_name)
            tools_used.append("query_failure_modes")

            sub_result = _handle_fault_classification(
                client, graph, equipment_name, profile,
                failure_history, failure_modes,
            )

        elif phm_category == "engine_health":
            maintenance_records = _query_maintenance_records(
                client, graph, equipment_name,
            )
            tools_used.append("query_maintenance_records")

            sub_result = _handle_engine_health(
                client, graph, equipment_name, profile,
                failure_history, maintenance_records,
            )

        elif phm_category == "safety_policy":
            policies = _query_maintenance_policies(
                client, graph, equipment_name,
            )
            tools_used.append("query_maintenance_policies")

            sub_result = _handle_safety_policy(
                client, graph, equipment_name, profile,
                failure_history, policies,
            )

        elif phm_category == "cost_benefit":
            maintenance_records = _query_maintenance_records(
                client, graph, equipment_name,
            )
            tools_used.append("query_maintenance_records")

            sub_result = _handle_cost_benefit(
                client, graph, equipment_name, profile,
                failure_history, maintenance_records,
            )
        else:
            sub_result = {
                "response": f"Unhandled PHM category: {phm_category}",
                "tools_used": [],
                "category": phm_category,
            }

        # Merge tools_used from sub-handler
        tools_used.extend(sub_result.get("tools_used", []))

        elapsed = (time.perf_counter() - start) * 1000
        return {
            "response": sub_result["response"],
            "tools_used": list(dict.fromkeys(tools_used)),  # dedupe, preserve order
            "latency_ms": round(elapsed, 2),
            "phm_category": phm_category,
            "equipment": equipment_name,
            "asset_profile": profile,
            "failure_history_count": len(failure_history),
            **{k: v for k, v in sub_result.items()
               if k not in ("response", "tools_used")},
            "graph_method": (
                "multi-hop traversal across Equipment, WorkOrder, "
                "MaintenanceRecord, FailureMode, Sensor, MaintenancePolicy "
                "nodes for prognostics and health management"
            ),
        }

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "response": f"PHM handler error for scenario {sid}: {e}",
            "tools_used": tools_used,
            "latency_ms": round(elapsed, 2),
            "phm_category": "error",
            "error": f"{type(e).__name__}: {e}",
        }
