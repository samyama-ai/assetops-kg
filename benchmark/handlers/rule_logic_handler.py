"""Handler for 120 monitoring-rule / Analysis & Inference scenarios.

Scenarios ask about anomaly detection rules for equipment, e.g.:
    "Please find anomalies for asset AHU HUR00118 between
     2021-04-11T23:00:00.000Z and 2021-04-12T05:15:00.000Z"

The handler extracts the equipment name and optional time range from the
scenario text, queries the KG for monitoring rules and sensor readings,
applies rules to readings, and returns structured anomaly results.
"""

from __future__ import annotations

import re
import time
from typing import Any

from samyama import SamyamaClient


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

_EQUIP_RE = re.compile(r"asset\s+(\S+(?:\s+\S+)?)", re.IGNORECASE)
_ISO_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
)
_BETWEEN_RE = re.compile(
    r"between\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)"
    r"\s+and\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)",
    re.IGNORECASE,
)
_SENSOR_TYPE_RE = re.compile(
    r"(temperature|pressure|vibration|humidity|power|flow|current|voltage"
    r"|speed|level|frequency)",
    re.IGNORECASE,
)


def _extract_equipment_name(text: str) -> str | None:
    """Extract equipment name from scenario text.

    Matches patterns like:
        "asset AHU HUR00118"
        "asset Chiller 6"
        "asset CWC04013"
    """
    m = _EQUIP_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: look for known equipment patterns
    for pat in [
        r"(AHU\s+\S+)",
        r"(Chiller\s+\d+)",
        r"(CWC\d+)",
        r"(CQPA\s+AHU\s+\w+)",
    ]:
        m2 = re.search(pat, text, re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
    return None


def _extract_time_range(text: str) -> tuple[str | None, str | None]:
    """Extract start and end timestamps from scenario text."""
    m = _BETWEEN_RE.search(text)
    if m:
        return m.group(1), m.group(2)
    # Try to find any two ISO timestamps
    timestamps = _ISO_TS_RE.findall(text)
    if len(timestamps) >= 2:
        return timestamps[0], timestamps[1]
    if len(timestamps) == 1:
        return timestamps[0], None
    return None, None


def _extract_sensor_type(text: str) -> str | None:
    """Extract sensor type keyword from scenario text."""
    m = _SENSOR_TYPE_RE.search(text)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------------------
# KG query helpers
# ---------------------------------------------------------------------------

def _query_monitoring_rules(
    client: SamyamaClient,
    graph: str,
    equipment_name: str,
) -> list[dict[str, Any]]:
    """Query monitoring rules associated with an equipment node."""
    cypher = (
        "MATCH (e:Equipment)-[:HAS_RULE]->(r:MonitoringRule) "
        f"WHERE e.name CONTAINS '{equipment_name}' "
        "RETURN r.rule_id, r.name, r.description, r.condition, "
        "r.threshold, r.severity, r.sensor_type"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    rules = []
    for row in result.records:
        rules.append({
            "rule_id": row[0],
            "name": row[1],
            "description": row[2],
            "condition": row[3],
            "threshold": row[4],
            "severity": row[5],
            "sensor_type": row[6],
        })
    return rules


def _query_sensor_readings(
    client: SamyamaClient,
    graph: str,
    equipment_name: str,
    start_ts: str | None,
    end_ts: str | None,
) -> list[dict[str, Any]]:
    """Query sensor readings for equipment within an optional time range."""
    where_parts = [f"e.name CONTAINS '{equipment_name}'"]
    if start_ts:
        where_parts.append(f"sr.timestamp >= '{start_ts}'")
    if end_ts:
        where_parts.append(f"sr.timestamp <= '{end_ts}'")
    where_clause = " AND ".join(where_parts)

    cypher = (
        "MATCH (e:Equipment)<-[:MONITORS]-(s:Sensor)"
        "-[:PRODUCED_READING]->(sr:SensorReading) "
        f"WHERE {where_clause} "
        "RETURN s.name, s.sensor_type, sr.value, sr.timestamp, sr.unit"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    readings = []
    for row in result.records:
        readings.append({
            "sensor_name": row[0],
            "sensor_type": row[1],
            "value": row[2],
            "timestamp": row[3],
            "unit": row[4],
        })
    return readings


def _query_anomaly_events(
    client: SamyamaClient,
    graph: str,
    equipment_name: str,
    start_ts: str | None,
    end_ts: str | None,
) -> list[dict[str, Any]]:
    """Query existing anomaly events linked to an equipment node."""
    where_parts = [f"e.name CONTAINS '{equipment_name}'"]
    if start_ts:
        where_parts.append(f"a.detected_at >= '{start_ts}'")
    if end_ts:
        where_parts.append(f"a.detected_at <= '{end_ts}'")
    where_clause = " AND ".join(where_parts)

    cypher = (
        "MATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor)"
        "-[:DETECTED_ANOMALY]->(a:Anomaly) "
        f"WHERE {where_clause} "
        "RETURN a.anomaly_id, a.description, a.severity, a.detected_at, "
        "a.anomaly_type, a.resolved, s.name, s.sensor_type"
    )
    try:
        result = client.query_readonly(cypher, graph)
    except Exception:
        return []

    anomalies = []
    for row in result.records:
        anomalies.append({
            "anomaly_id": row[0],
            "description": row[1],
            "severity": row[2],
            "detected_at": row[3],
            "anomaly_type": row[4],
            "resolved": row[5],
            "sensor_name": row[6],
            "sensor_type": row[7],
        })
    return anomalies


# ---------------------------------------------------------------------------
# Rule application logic
# ---------------------------------------------------------------------------

def _apply_rules_to_readings(
    rules: list[dict], readings: list[dict],
) -> list[dict[str, Any]]:
    """Apply monitoring rules to sensor readings and detect anomalies.

    For each rule, check if any reading violates the threshold condition.
    Returns a list of detected anomaly dicts.
    """
    detected: list[dict[str, Any]] = []

    for rule in rules:
        threshold = rule.get("threshold")
        condition = str(rule.get("condition") or "exceeds").lower()
        rule_sensor = str(rule.get("sensor_type") or "").lower()

        # Match readings to this rule by sensor type
        matched_readings = [
            r for r in readings
            if rule_sensor and rule_sensor in str(r.get("sensor_type") or "").lower()
        ]
        if not matched_readings and readings:
            # If no sensor type match, apply to all readings as fallback
            matched_readings = readings

        for reading in matched_readings:
            val = reading.get("value")
            if val is None or threshold is None:
                continue
            try:
                val_f = float(val)
                thr_f = float(threshold)
            except (ValueError, TypeError):
                continue

            violated = False
            if "exceed" in condition or "above" in condition or ">" in condition:
                violated = val_f > thr_f
            elif "below" in condition or "<" in condition:
                violated = val_f < thr_f
            elif "equal" in condition or "=" in condition:
                violated = abs(val_f - thr_f) < 1e-9
            else:
                # Default: exceeds
                violated = val_f > thr_f

            if violated:
                detected.append({
                    "rule_id": rule.get("rule_id"),
                    "rule_name": rule.get("name"),
                    "severity": rule.get("severity", "medium"),
                    "sensor_name": reading.get("sensor_name"),
                    "sensor_type": reading.get("sensor_type"),
                    "reading_value": val_f,
                    "threshold": thr_f,
                    "condition": condition,
                    "timestamp": reading.get("timestamp"),
                    "description": (
                        f"Rule '{rule.get('name')}' violated: "
                        f"value {val_f} {condition} threshold {thr_f}"
                    ),
                })

    return detected


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

def handle_rule_logic(
    client: SamyamaClient,
    scenario: dict[str, Any],
    tenant: str = "default",
) -> dict[str, Any]:
    """Handle monitoring rule / Analysis & Inference scenarios.

    Extracts equipment name and time range from scenario text, queries KG
    for monitoring rules and sensor readings, applies rules to detect
    anomalies, and returns structured results.

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
        equipment_name = _extract_equipment_name(text)
        start_ts, end_ts = _extract_time_range(text)
        sensor_type = _extract_sensor_type(text)

        if not equipment_name:
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "response": f"Could not extract equipment name from scenario {sid}.",
                "tools_used": tools_used,
                "latency_ms": round(elapsed, 2),
                "anomalies": [],
                "error": "equipment_name_not_found",
            }

        # 1. Query monitoring rules for the equipment
        rules = _query_monitoring_rules(client, graph, equipment_name)
        tools_used.append("query_monitoring_rules")

        # 2. Query sensor readings in the time range
        readings = _query_sensor_readings(
            client, graph, equipment_name, start_ts, end_ts,
        )
        tools_used.append("query_sensor_readings")

        # 3. Query pre-existing anomaly events
        existing_anomalies = _query_anomaly_events(
            client, graph, equipment_name, start_ts, end_ts,
        )
        tools_used.append("query_anomaly_events")

        # 4. Apply rules to readings to detect new anomalies
        detected_anomalies = _apply_rules_to_readings(rules, readings)
        tools_used.append("apply_monitoring_rules")

        # Merge existing and newly detected anomalies
        all_anomalies = existing_anomalies + detected_anomalies

        # Build human-readable response
        time_range_str = ""
        if start_ts and end_ts:
            time_range_str = f" between {start_ts} and {end_ts}"
        elif start_ts:
            time_range_str = f" from {start_ts}"

        response_lines = [
            f"Anomaly analysis for {equipment_name}{time_range_str}:",
            f"Monitoring rules evaluated: {len(rules)}",
            f"Sensor readings analyzed: {len(readings)}",
            f"Pre-existing anomalies: {len(existing_anomalies)}",
            f"Rule-detected anomalies: {len(detected_anomalies)}",
            f"Total anomalies found: {len(all_anomalies)}",
        ]

        if all_anomalies:
            response_lines.append("")
            response_lines.append("Anomalies:")
            for i, anom in enumerate(all_anomalies[:20], 1):
                desc = anom.get("description", "Unknown anomaly")
                sev = anom.get("severity", "medium")
                ts = anom.get("detected_at") or anom.get("timestamp", "")
                sensor = anom.get("sensor_name", "")
                response_lines.append(
                    f"  {i}. [{sev}] {desc}"
                    + (f" (sensor: {sensor})" if sensor else "")
                    + (f" at {ts}" if ts else "")
                )
        elif not rules and not readings:
            response_lines.append(
                f"No monitoring rules or sensor readings found for {equipment_name}. "
                "Ensure equipment data and rules are loaded in the knowledge graph."
            )
        else:
            response_lines.append(
                "No anomalies detected within the specified parameters."
            )

        elapsed = (time.perf_counter() - start) * 1000
        return {
            "response": "\n".join(response_lines),
            "tools_used": tools_used,
            "latency_ms": round(elapsed, 2),
            "equipment": equipment_name,
            "time_range": {"start": start_ts, "end": end_ts},
            "rules_count": len(rules),
            "readings_count": len(readings),
            "anomalies": all_anomalies,
            "anomaly_count": len(all_anomalies),
            "graph_method": (
                "HAS_RULE, MONITORS, PRODUCED_READING, DETECTED_ANOMALY "
                "edge traversal with threshold-based rule evaluation"
            ),
        }

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "response": f"Rule logic handler error for scenario {sid}: {e}",
            "tools_used": tools_used,
            "latency_ms": round(elapsed, 2),
            "anomalies": [],
            "error": f"{type(e).__name__}: {e}",
        }
