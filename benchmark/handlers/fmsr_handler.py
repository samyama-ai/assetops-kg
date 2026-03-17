"""Handler for 88 FMSR (Failure Mode Sensor Recommendation) scenarios.

Scenarios ask about failure mode to sensor mappings, e.g.:
    "List all failure modes of electric motor that can be detected by
     vibration, cooling gas, or axial flow sensors"

The handler extracts equipment type and sensor types from the scenario
text, queries the KG for failure-mode-to-sensor relationships, and
returns structured results.
"""

from __future__ import annotations

import re
import time
from typing import Any

from samyama import SamyamaClient


# ---------------------------------------------------------------------------
# Known equipment types and sensor types for extraction
# ---------------------------------------------------------------------------

EQUIPMENT_TYPES = [
    "electric motor", "motor", "chiller", "compressor", "pump",
    "heat exchanger", "condenser", "evaporator", "boiler", "turbine",
    "wind turbine", "generator", "transformer", "fan", "valve",
    "cooling tower", "air handling unit", "ahu", "conveyor",
    "gearbox", "bearing", "actuator", "hydraulic system",
]

SENSOR_TYPES = [
    "vibration", "temperature", "pressure", "current", "voltage",
    "power", "flow", "humidity", "speed", "frequency",
    "acoustic", "ultrasonic", "infrared", "thermal",
    "displacement", "acceleration", "oil analysis", "oil particle",
    "cooling gas", "axial flow", "radial flow",
    "stator", "rotor", "winding", "insulation",
    "load", "torque", "position", "level",
]

# Pre-built domain knowledge: equipment -> common failure modes
# Used as fallback when KG has no data
_DOMAIN_FAILURE_MODES: dict[str, list[dict[str, str]]] = {
    "electric motor": [
        {"name": "Bearing failure", "description": "Degraded or worn bearings causing excessive vibration", "detectable_by": "vibration, acoustic, temperature"},
        {"name": "Stator winding insulation breakdown", "description": "Insulation degradation in stator windings", "detectable_by": "current, temperature, insulation"},
        {"name": "Rotor bar fracture", "description": "Broken rotor bars causing torque pulsation", "detectable_by": "vibration, current, speed"},
        {"name": "Shaft misalignment", "description": "Misalignment between motor and driven equipment", "detectable_by": "vibration, temperature"},
        {"name": "Overheating", "description": "Excessive temperature due to overload or cooling failure", "detectable_by": "temperature, current, cooling gas"},
        {"name": "Cooling system failure", "description": "Blocked or degraded cooling passages", "detectable_by": "temperature, cooling gas, flow"},
        {"name": "Unbalance", "description": "Mass imbalance in rotating assembly", "detectable_by": "vibration, displacement"},
        {"name": "Electrical fault", "description": "Phase imbalance or voltage irregularity", "detectable_by": "current, voltage, power"},
    ],
    "compressor": [
        {"name": "Valve failure", "description": "Suction or discharge valve leakage", "detectable_by": "pressure, temperature, vibration"},
        {"name": "Bearing wear", "description": "Journal or thrust bearing degradation", "detectable_by": "vibration, temperature, oil analysis"},
        {"name": "Seal leakage", "description": "Shaft seal or gasket deterioration", "detectable_by": "pressure, flow, acoustic"},
        {"name": "Surge", "description": "Flow reversal due to operating outside stable envelope", "detectable_by": "pressure, flow, vibration"},
        {"name": "Overheating", "description": "Excessive discharge temperature", "detectable_by": "temperature, pressure, current"},
    ],
    "chiller": [
        {"name": "Compressor overheating", "description": "Failed due to normal wear, overheating", "detectable_by": "temperature, power, current"},
        {"name": "Heat exchanger fan degradation", "description": "Degraded motor or worn bearing due to normal use", "detectable_by": "vibration, temperature, current"},
        {"name": "Evaporator water side fouling", "description": "Scale or debris buildup on evaporator tubes", "detectable_by": "temperature, flow, pressure"},
        {"name": "Condenser water side fouling", "description": "Scale or debris buildup on condenser tubes", "detectable_by": "temperature, flow, pressure"},
        {"name": "Condenser improper water flow rate", "description": "Incorrect flow through condenser", "detectable_by": "flow, pressure, temperature"},
        {"name": "Purge unit excessive purge", "description": "Non-condensable gas accumulation", "detectable_by": "pressure, temperature"},
        {"name": "Refrigerant valve spring failure", "description": "Failed spring in refrigerant operated control valve", "detectable_by": "pressure, temperature, flow"},
    ],
    "pump": [
        {"name": "Cavitation", "description": "Vapor bubble formation and collapse", "detectable_by": "vibration, acoustic, pressure"},
        {"name": "Impeller erosion", "description": "Wear on impeller surfaces", "detectable_by": "vibration, flow, pressure"},
        {"name": "Seal failure", "description": "Mechanical seal leakage", "detectable_by": "flow, pressure, vibration"},
        {"name": "Bearing failure", "description": "Bearing degradation", "detectable_by": "vibration, temperature"},
    ],
    "turbine": [
        {"name": "Blade crack", "description": "Fatigue crack propagation in blades", "detectable_by": "vibration, acoustic, ultrasonic"},
        {"name": "Bearing failure", "description": "Journal bearing wear or damage", "detectable_by": "vibration, temperature, oil analysis"},
        {"name": "Gearbox failure", "description": "Gear tooth wear or damage", "detectable_by": "vibration, acoustic, oil particle"},
        {"name": "Generator winding fault", "description": "Insulation degradation in generator", "detectable_by": "temperature, current, insulation"},
    ],
}


def _extract_equipment_type(text: str) -> str | None:
    """Extract equipment type from scenario text.

    Tries longest match first to prefer "electric motor" over "motor".
    """
    text_lower = text.lower()
    # Sort by length descending so longer patterns match first
    for etype in sorted(EQUIPMENT_TYPES, key=len, reverse=True):
        if etype in text_lower:
            return etype
    return None


def _extract_sensor_types(text: str) -> list[str]:
    """Extract all mentioned sensor types from scenario text."""
    text_lower = text.lower()
    found = []
    for stype in sorted(SENSOR_TYPES, key=len, reverse=True):
        if stype in text_lower and stype not in found:
            found.append(stype)
    return found


def _normalize_equipment_type(etype: str) -> str:
    """Normalize equipment type for KG queries."""
    mapping = {
        "ahu": "air handling unit",
        "electric motor": "motor",
    }
    return mapping.get(etype, etype)


# ---------------------------------------------------------------------------
# KG query helpers
# ---------------------------------------------------------------------------

def _query_failure_modes_by_sensor(
    client: SamyamaClient,
    graph: str,
    equipment_type: str,
    sensor_types: list[str],
) -> list[dict[str, Any]]:
    """Query failure modes detectable by specific sensor types for an equipment type."""
    sensor_in = "[" + ", ".join(f"'{s}'" for s in sensor_types) + "]"
    eq_norm = _normalize_equipment_type(equipment_type)

    cypher = (
        "MATCH (f:FailureMode)-[:DETECTED_BY]->(s:Sensor)"
        "-[:MONITORS]->(e:Equipment) "
        f"WHERE (e.type CONTAINS '{eq_norm}' OR e.name CONTAINS '{eq_norm}' "
        f"OR e.iso14224_class CONTAINS '{eq_norm}') "
        f"AND s.sensor_type IN {sensor_in} "
        "RETURN f.name, f.description, collect(s.sensor_type)"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    modes = []
    for row in result.records:
        modes.append({
            "name": row[0],
            "description": row[1],
            "detected_by_sensors": row[2] if row[2] else [],
        })
    return modes


def _query_failure_modes_all(
    client: SamyamaClient,
    graph: str,
    equipment_type: str,
) -> list[dict[str, Any]]:
    """Query all failure modes for an equipment type."""
    eq_norm = _normalize_equipment_type(equipment_type)

    cypher = (
        "MATCH (f:FailureMode) "
        f"WHERE f.asset_type CONTAINS '{eq_norm}' "
        f"OR f.asset_type CONTAINS '{equipment_type}' "
        "RETURN f.name, f.description, f.severity, f.category"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    modes = []
    for row in result.records:
        modes.append({
            "name": row[0],
            "description": row[1],
            "severity": row[2],
            "category": row[3],
        })
    return modes


def _query_sensors_for_equipment(
    client: SamyamaClient,
    graph: str,
    equipment_type: str,
) -> list[dict[str, Any]]:
    """Query sensors installed on or relevant to an equipment type."""
    eq_norm = _normalize_equipment_type(equipment_type)

    cypher = (
        "MATCH (s:Sensor)-[:MONITORS]->(e:Equipment) "
        f"WHERE e.type CONTAINS '{eq_norm}' OR e.name CONTAINS '{eq_norm}' "
        f"OR e.iso14224_class CONTAINS '{eq_norm}' "
        "RETURN s.name, s.sensor_type, s.unit"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    sensors = []
    for row in result.records:
        sensors.append({
            "name": row[0],
            "sensor_type": row[1],
            "unit": row[2],
        })
    return sensors


# ---------------------------------------------------------------------------
# Domain knowledge fallback
# ---------------------------------------------------------------------------

def _fallback_failure_modes(
    equipment_type: str, sensor_filter: list[str],
) -> list[dict[str, Any]]:
    """Return domain-knowledge failure modes when KG has no data.

    Filters by sensor types if provided.
    """
    # Find the best matching key in domain knowledge
    et_lower = equipment_type.lower()
    domain_modes = None
    for key in _DOMAIN_FAILURE_MODES:
        if key in et_lower or et_lower in key:
            domain_modes = _DOMAIN_FAILURE_MODES[key]
            break
    if domain_modes is None:
        # Default to generic motor modes
        domain_modes = _DOMAIN_FAILURE_MODES.get("electric motor", [])

    if not sensor_filter:
        return [
            {"name": m["name"], "description": m["description"],
             "detected_by_sensors": m["detectable_by"].split(", ")}
            for m in domain_modes
        ]

    # Filter by sensor types
    filtered = []
    for m in domain_modes:
        detectable = m["detectable_by"].lower()
        matching_sensors = [s for s in sensor_filter if s in detectable]
        if matching_sensors:
            filtered.append({
                "name": m["name"],
                "description": m["description"],
                "detected_by_sensors": matching_sensors,
            })
    return filtered


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

def handle_fmsr(
    client: SamyamaClient,
    scenario: dict[str, Any],
    tenant: str = "default",
) -> dict[str, Any]:
    """Handle FMSR (Failure Mode Sensor Recommendation) scenarios.

    Extracts equipment type and sensor types from scenario text, queries
    the KG for failure-mode-to-sensor relationships, and falls back to
    domain knowledge when KG data is unavailable.

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
        equipment_type = _extract_equipment_type(text)
        sensor_filter = _extract_sensor_types(text)

        if not equipment_type:
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "response": (
                    f"Could not extract equipment type from scenario {sid}. "
                    "Expected one of: " + ", ".join(EQUIPMENT_TYPES[:10]) + "..."
                ),
                "tools_used": tools_used,
                "latency_ms": round(elapsed, 2),
                "failure_modes": [],
                "error": "equipment_type_not_found",
            }

        # Query KG for failure modes
        if sensor_filter:
            modes = _query_failure_modes_by_sensor(
                client, graph, equipment_type, sensor_filter,
            )
            tools_used.append("query_failure_modes_by_sensor")
        else:
            modes = _query_failure_modes_all(client, graph, equipment_type)
            tools_used.append("query_failure_modes_all")

        # Query available sensors for context
        available_sensors = _query_sensors_for_equipment(
            client, graph, equipment_type,
        )
        tools_used.append("query_sensors_for_equipment")

        # Fallback to domain knowledge if KG returned nothing
        source = "knowledge_graph"
        if not modes:
            modes = _fallback_failure_modes(equipment_type, sensor_filter)
            source = "domain_knowledge"
            tools_used.append("domain_knowledge_fallback")

        # Build response
        sensor_desc = ""
        if sensor_filter:
            sensor_desc = f" detectable by {', '.join(sensor_filter)} sensors"

        response_lines = [
            f"Failure modes of {equipment_type}{sensor_desc}:",
            f"Source: {source}",
            f"Total failure modes found: {len(modes)}",
            "",
        ]
        for i, fm in enumerate(modes, 1):
            name = fm.get("name", "Unknown")
            desc = fm.get("description", "")
            sensors = fm.get("detected_by_sensors", [])
            response_lines.append(f"  {i}. {name}")
            if desc:
                response_lines.append(f"     Description: {desc}")
            if sensors:
                if isinstance(sensors, list):
                    response_lines.append(
                        f"     Detectable by: {', '.join(str(s) for s in sensors)}"
                    )
                else:
                    response_lines.append(f"     Detectable by: {sensors}")

        if available_sensors:
            response_lines.append("")
            response_lines.append(
                f"Available sensors for {equipment_type}: "
                f"{len(available_sensors)} installed"
            )
            for s in available_sensors[:10]:
                sname = s.get("name", "")
                stype = s.get("sensor_type", "")
                response_lines.append(f"  - {sname} ({stype})")

        elapsed = (time.perf_counter() - start) * 1000
        return {
            "response": "\n".join(response_lines),
            "tools_used": tools_used,
            "latency_ms": round(elapsed, 2),
            "equipment_type": equipment_type,
            "sensor_filter": sensor_filter,
            "failure_modes": modes,
            "failure_mode_count": len(modes),
            "available_sensors": available_sensors,
            "source": source,
            "graph_method": (
                "DETECTED_BY and MONITORS edge traversal for "
                "failure-mode-to-sensor mapping"
            ),
        }

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "response": f"FMSR handler error for scenario {sid}: {e}",
            "tools_used": tools_used,
            "latency_ms": round(elapsed, 2),
            "failure_modes": [],
            "error": f"{type(e).__name__}: {e}",
        }
