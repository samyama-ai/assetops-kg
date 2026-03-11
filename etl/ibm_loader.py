"""IBM AssetOpsBench loader — loads the full IBM dataset into a Samyama graph.

Loads:
  1. Site / Location / Equipment hierarchy (11 chillers)
  2. Sensors (10 per chiller = 110 total)
  3. Failure modes (7 Chiller + 5 AHU = 12 total)
  4. MONITORS edges (Chiller 6 sensors -> failure modes)
  5. Work orders from workorders.csv (~4248 rows)
  6. Alert events from alert_events.csv (~1465 rows)
  7. Anomaly events from anomaly_events.csv (~541 rows)
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from samyama import SamyamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EQUIPMENT_MAP: Dict[str, str] = {
    "CWC04006": "Chiller 6",
    "CWC04007": "Chiller 7",
    "CWC04009": "Chiller 9",
    "CWC04010": "Chiller 10",
    "CWC04012": "Chiller 12",
    "CWC04013": "Chiller 13",
    "CWC04014": "Chiller 14",
    "CWC04701": "Chiller 1",
    "CWC04702": "Chiller 2",
    "CWC04703": "Chiller 3",
    "CWC04704": "Chiller 4",
}

SENSOR_SUFFIXES: List[str] = [
    "Chiller % Loaded",
    "Chiller Efficiency",
    "Condenser Water Flow",
    "Condenser Water Return To Tower Temperature",
    "Liquid Refrigerant Evaporator Temperature",
    "Power Input",
    "Return Temperature",
    "Schedule",
    "Supply Temperature",
    "Tonnage",
]

CHILLER_FAILURE_MODES: List[Dict[str, str]] = [
    {"name": "Compressor Overheating: Failed due to Normal wear, overheating",
     "severity": "critical"},
    {"name": "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
     "severity": "high"},
    {"name": "Evaporator Water side fouling",
     "severity": "high"},
    {"name": "Condenser Water side fouling",
     "severity": "high"},
    {"name": "Condenser Improper water side flow rate",
     "severity": "medium"},
    {"name": "Purge Unit Excessive purge",
     "severity": "medium"},
    {"name": "Refrigerant Operated Control Valve Failed spring",
     "severity": "high"},
]

AHU_FAILURE_MODES: List[Dict[str, str]] = [
    {"name": "Pressure Regulators Diaphragm failure",
     "severity": "high"},
    {"name": "Steam Heating Coils Air side fouling",
     "severity": "medium"},
    {"name": "Belts or sheaves Wear",
     "severity": "medium"},
    {"name": "Improper switch position",
     "severity": "low"},
    {"name": "Solenoid Valves Bound due to hardened grease",
     "severity": "medium"},
]

# Sensor suffix -> list of failure mode names that the sensor monitors.
# Only applied to Chiller 6 sensors.
SENSOR_FM_MAP: Dict[str, List[str]] = {
    "Supply Temperature": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Evaporator Water side fouling",
    ],
    "Return Temperature": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Evaporator Water side fouling",
    ],
    "Condenser Water Flow": [
        "Condenser Water side fouling",
        "Condenser Improper water side flow rate",
    ],
    "Condenser Water Return To Tower Temperature": [
        "Condenser Water side fouling",
        "Condenser Improper water side flow rate",
    ],
    "Power Input": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
    ],
    "Chiller Efficiency": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
        "Evaporator Water side fouling",
        "Condenser Water side fouling",
        "Condenser Improper water side flow rate",
        "Purge Unit Excessive purge",
        "Refrigerant Operated Control Valve Failed spring",
    ],
    "Chiller % Loaded": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
        "Evaporator Water side fouling",
        "Condenser Water side fouling",
        "Condenser Improper water side flow rate",
        "Purge Unit Excessive purge",
        "Refrigerant Operated Control Valve Failed spring",
    ],
    "Tonnage": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Evaporator Water side fouling",
    ],
    "Liquid Refrigerant Evaporator Temperature": [
        "Evaporator Water side fouling",
        "Refrigerant Operated Control Valve Failed spring",
    ],
    # Schedule has no direct failure-mode mapping
}

WO_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(val: Any) -> str:
    """Escape a value for safe embedding in a Cypher string literal.

    Strings are wrapped in double-quotes with internal quotes escaped.
    Numbers and booleans are returned as-is.
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val)
    # Escape backslashes first, then double-quotes
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _props(d: Dict[str, Any]) -> str:
    """Build a Cypher property map literal: {key1: val1, key2: val2}."""
    parts = [f"{k}: {_esc(v)}" for k, v in d.items()]
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Section loaders
# ---------------------------------------------------------------------------

def _load_hierarchy(client: SamyamaClient, graph: str) -> Dict[str, int]:
    """Create 1 Site, 1 Location, and 11 Equipment nodes with edges."""
    # Site
    client.query(
        'CREATE (s:Site {name: "MAIN"})',
        graph,
    )
    # Location
    client.query(
        'CREATE (l:Location {name: "Main Building"})',
        graph,
    )
    # Site -> Location
    client.query(
        'MATCH (s:Site {name: "MAIN"}), (l:Location {name: "Main Building"}) '
        "CREATE (s)-[:CONTAINS_LOCATION]->(l)",
        graph,
    )

    equip_count = 0
    for eq_id, eq_name in EQUIPMENT_MAP.items():
        props = {
            "equipment_id": eq_id,
            "name": eq_name,
            "asset_type": "Chiller",
        }
        client.query(f"CREATE (e:Equipment {_props(props)})", graph)

        # Location -> Equipment
        client.query(
            f'MATCH (l:Location {{name: "Main Building"}}), '
            f'(e:Equipment {{equipment_id: "{eq_id}"}}) '
            f"CREATE (l)-[:CONTAINS_EQUIPMENT]->(e)",
            graph,
        )
        equip_count += 1

    logger.info("Hierarchy: 1 site, 1 location, %d equipment", equip_count)
    return {"sites": 1, "locations": 1, "equipment": equip_count}


def _load_sensors(client: SamyamaClient, graph: str) -> Dict[str, int]:
    """Create 10 sensors per chiller (110 total) with HAS_SENSOR edges."""
    sensor_count = 0
    for eq_id, eq_name in EQUIPMENT_MAP.items():
        for suffix in SENSOR_SUFFIXES:
            sensor_name = f"{eq_name} {suffix}"
            props = {
                "name": sensor_name,
                "sensor_type": suffix,
                "equipment_id": eq_id,
            }
            client.query(f"CREATE (s:Sensor {_props(props)})", graph)

            # Equipment -> Sensor
            client.query(
                f'MATCH (e:Equipment {{equipment_id: "{eq_id}"}}), '
                f"(s:Sensor {{name: {_esc(sensor_name)}}}) "
                f"CREATE (e)-[:HAS_SENSOR]->(s)",
                graph,
            )
            sensor_count += 1

    logger.info("Sensors: %d created", sensor_count)
    return {"sensors": sensor_count}


def _load_failure_modes(client: SamyamaClient, graph: str) -> Dict[str, int]:
    """Create 12 FailureMode nodes (7 Chiller + 5 AHU)."""
    fm_count = 0
    for fm in CHILLER_FAILURE_MODES:
        props = {
            "name": fm["name"],
            "description": fm["name"],
            "asset_type": "Chiller",
            "severity": fm["severity"],
        }
        client.query(f"CREATE (f:FailureMode {_props(props)})", graph)
        fm_count += 1

    for fm in AHU_FAILURE_MODES:
        props = {
            "name": fm["name"],
            "description": fm["name"],
            "asset_type": "AHU",
            "severity": fm["severity"],
        }
        client.query(f"CREATE (f:FailureMode {_props(props)})", graph)
        fm_count += 1

    logger.info("Failure modes: %d created", fm_count)
    return {"failure_modes": fm_count}


def _load_monitors_edges(client: SamyamaClient, graph: str) -> Dict[str, int]:
    """Create MONITORS edges from Chiller 6 sensors to relevant failure modes."""
    monitors_count = 0
    chiller6_id = "CWC04006"
    chiller6_name = EQUIPMENT_MAP[chiller6_id]

    for suffix, fm_names in SENSOR_FM_MAP.items():
        sensor_name = f"{chiller6_name} {suffix}"
        for fm_name in fm_names:
            client.query(
                f"MATCH (s:Sensor {{name: {_esc(sensor_name)}}}), "
                f"(f:FailureMode {{name: {_esc(fm_name)}}}) "
                f"CREATE (s)-[:MONITORS]->(f)",
                graph,
            )
            monitors_count += 1

    logger.info("MONITORS edges: %d created", monitors_count)
    return {"monitors_edges": monitors_count}


def _load_work_orders(
    client: SamyamaClient, data_dir: str, graph: str
) -> Dict[str, int]:
    """Load work orders from workorders.csv, batch-creating nodes + edges."""
    csv_path = os.path.join(
        data_dir,
        "aobench", "datalayer", "eamlite", "db", "data", "workorders.csv",
    )
    if not os.path.isfile(csv_path):
        logger.warning("workorders.csv not found at %s — skipping", csv_path)
        return {"work_orders": 0, "wo_edges": 0}

    wo_count = 0
    wo_edge_count = 0
    batch: List[Dict[str, Any]] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            batch.append(row)
            if len(batch) >= WO_BATCH_SIZE:
                created, edges = _flush_wo_batch(client, batch, graph)
                wo_count += created
                wo_edge_count += edges
                batch = []

        # Remainder
        if batch:
            created, edges = _flush_wo_batch(client, batch, graph)
            wo_count += created
            wo_edge_count += edges

    logger.info("Work orders: %d created, %d edges", wo_count, wo_edge_count)
    return {"work_orders": wo_count, "wo_edges": wo_edge_count}


def _flush_wo_batch(
    client: SamyamaClient,
    batch: List[Dict[str, Any]],
    graph: str,
) -> tuple[int, int]:
    """Create a batch of WorkOrder nodes and their FOR_EQUIPMENT edges."""
    created = 0
    edges = 0
    for row in batch:
        props = {
            "wo_id": row.get("wo_id", ""),
            "wo_description": row.get("wo_description", ""),
            "collection": row.get("collection", ""),
            "primary_code": row.get("primary_code", ""),
            "primary_code_description": row.get("primary_code_description", ""),
            "secondary_code": row.get("secondary_code", ""),
            "secondary_code_description": row.get("secondary_code_description", ""),
            "equipment_id": row.get("equipment_id", ""),
            "equipment_name": row.get("equipment_name", ""),
            "preventive": row.get("preventive", ""),
            "work_priority": row.get("work_priority", ""),
            "actual_finish": row.get("actual_finish", ""),
            "duration": row.get("duration", ""),
            "actual_labor_hours": row.get("actual_labor_hours", ""),
        }

        client.query(f"CREATE (w:WorkOrder {_props(props)})", graph)
        created += 1

        eq_id = row.get("equipment_id", "").strip()
        if eq_id and eq_id in EQUIPMENT_MAP:
            client.query(
                f'MATCH (w:WorkOrder {{wo_id: {_esc(props["wo_id"])}}}), '
                f'(e:Equipment {{equipment_id: "{eq_id}"}}) '
                f"CREATE (w)-[:FOR_EQUIPMENT]->(e)",
                graph,
            )
            edges += 1

    return created, edges


def _load_alert_events(
    client: SamyamaClient, data_dir: str, graph: str
) -> Dict[str, int]:
    """Load alert events from alert_events.csv."""
    csv_path = os.path.join(
        data_dir,
        "src", "tmp", "assetopsbench", "sample_data", "alert_events.csv",
    )
    if not os.path.isfile(csv_path):
        logger.warning("alert_events.csv not found at %s — skipping", csv_path)
        return {"alert_events": 0, "alert_edges": 0}

    alert_count = 0
    alert_edge_count = 0

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            eq_id = row.get("equipment_id", "").strip()
            eq_name = row.get("equipment_name", "").strip()
            rule_id = row.get("rule_id", "").strip()
            start_time = row.get("start_time", "").strip()
            end_time = row.get("end_time", "").strip()

            # Use a composite key to allow multiple alerts
            alert_key = f"{eq_id}_{rule_id}_{start_time}"
            props = {
                "alert_key": alert_key,
                "equipment_id": eq_id,
                "equipment_name": eq_name,
                "rule_id": rule_id,
                "start_time": start_time,
                "end_time": end_time,
            }

            client.query(f"CREATE (a:AlertEvent {_props(props)})", graph)
            alert_count += 1

            if eq_id and eq_id in EQUIPMENT_MAP:
                client.query(
                    f"MATCH (a:AlertEvent {{alert_key: {_esc(alert_key)}}}), "
                    f'(e:Equipment {{equipment_id: "{eq_id}"}}) '
                    f"CREATE (a)-[:FOR_EQUIPMENT]->(e)",
                    graph,
                )
                alert_edge_count += 1

    logger.info("Alert events: %d created, %d edges", alert_count, alert_edge_count)
    return {"alert_events": alert_count, "alert_edges": alert_edge_count}


def _load_unified_events(
    client: SamyamaClient, data_dir: str, graph: str
) -> Dict[str, int]:
    """Load unified events from event.csv (work orders + alerts + anomalies)."""
    csv_path = os.path.join(
        data_dir,
        "src", "tmp", "assetopsbench", "sample_data", "event.csv",
    )
    if not os.path.isfile(csv_path):
        logger.warning("event.csv not found at %s — skipping", csv_path)
        return {"unified_events": 0, "unified_event_edges": 0}

    event_count = 0
    edge_count = 0

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            event_id = row.get("event_id", "").strip()
            event_group = row.get("event_group", "").strip()
            event_category = row.get("event_category", "").strip()
            event_type = row.get("event_type", "").strip()
            description = row.get("description", "").strip()
            eq_id = row.get("equipment_id", "").strip()
            eq_name = row.get("equipment_name", "").strip()
            event_time = row.get("event_time", "").strip()

            props: Dict[str, Any] = {
                "event_id": event_id,
                "event_group": event_group,
                "event_category": event_category,
                "event_type": event_type,
                "description": description,
                "equipment_id": eq_id,
                "equipment_name": eq_name,
                "event_time": event_time,
            }

            client.query(f"CREATE (ev:Event {_props(props)})", graph)
            event_count += 1

            if eq_id and eq_id in EQUIPMENT_MAP:
                client.query(
                    f"MATCH (ev:Event {{event_id: {_esc(event_id)}}}), "
                    f'(e:Equipment {{equipment_id: "{eq_id}"}}) '
                    f"CREATE (ev)-[:FOR_EQUIPMENT]->(e)",
                    graph,
                )
                edge_count += 1

    logger.info("Unified events: %d created, %d edges", event_count, edge_count)
    return {"unified_events": event_count, "unified_event_edges": edge_count}


def _load_anomaly_events(
    client: SamyamaClient, data_dir: str, graph: str
) -> Dict[str, int]:
    """Load anomaly events from anomaly_events.csv."""
    csv_path = os.path.join(
        data_dir,
        "src", "tmp", "assetopsbench", "sample_data", "anomaly_events.csv",
    )
    if not os.path.isfile(csv_path):
        logger.warning("anomaly_events.csv not found at %s — skipping", csv_path)
        return {"anomaly_events": 0, "anomaly_edges": 0}

    # Build a lowercase equipment name -> equipment_id lookup
    name_to_id: Dict[str, str] = {}
    for eq_id, eq_name in EQUIPMENT_MAP.items():
        name_to_id[eq_name.lower()] = eq_id

    anomaly_count = 0
    anomaly_edge_count = 0

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            timestamp = row.get("timestamp", "").strip()
            kpi = row.get("KPI", "").strip()
            asset_name = row.get("asset_name", "").strip()
            value = row.get("value", "").strip()
            upper_bound = row.get("upper_bound", "").strip()
            lower_bound = row.get("lower_bound", "").strip()
            anomaly_score = row.get("anomaly_score", "").strip()

            anomaly_key = f"anomaly_{idx}"
            props: Dict[str, Any] = {
                "anomaly_key": anomaly_key,
                "timestamp": timestamp,
                "kpi": kpi,
                "asset_name": asset_name,
                "anomaly_score": anomaly_score,
            }

            # Store numeric values where possible
            for field, raw in [("value", value), ("upper_bound", upper_bound),
                               ("lower_bound", lower_bound)]:
                props[field] = raw

            client.query(f"CREATE (an:AnomalyEvent {_props(props)})", graph)
            anomaly_count += 1

            # Match asset_name case-insensitively to equipment
            eq_id = name_to_id.get(asset_name.lower())
            if eq_id:
                client.query(
                    f"MATCH (an:AnomalyEvent {{anomaly_key: {_esc(anomaly_key)}}}), "
                    f'(e:Equipment {{equipment_id: "{eq_id}"}}) '
                    f"CREATE (an)-[:FOR_EQUIPMENT]->(e)",
                    graph,
                )
                anomaly_edge_count += 1

    logger.info(
        "Anomaly events: %d created, %d edges", anomaly_count, anomaly_edge_count,
    )
    return {"anomaly_events": anomaly_count, "anomaly_edges": anomaly_edge_count}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_ibm_data(
    client: SamyamaClient, data_dir: str, graph: str = "ibm"
) -> dict:
    """Load the full IBM AssetOpsBench dataset into a Samyama graph.

    Args:
        client:   An initialised SamyamaClient (embedded or remote).
        data_dir: Root path of the AssetOpsBench repo
                  (e.g. ~/projects/Madhulatha-Sandeep/AssetOpsBench).
        graph:    Target graph name (default "ibm").

    Returns:
        A stats dict with counts of every node and edge type created.
    """
    data_dir = str(Path(data_dir).expanduser().resolve())
    logger.info("Loading IBM AssetOpsBench data from %s into graph '%s'", data_dir, graph)

    stats: Dict[str, int] = {}

    # 1. Site / Location / Equipment hierarchy
    logger.info("Step 1/7: Loading asset hierarchy...")
    hierarchy = _load_hierarchy(client, graph)
    stats.update(hierarchy)

    # 2. Sensors (10 per chiller = 110)
    logger.info("Step 2/7: Loading sensors...")
    sensors = _load_sensors(client, graph)
    stats.update(sensors)

    # 3. Failure modes (12 total)
    logger.info("Step 3/7: Loading failure modes...")
    fm = _load_failure_modes(client, graph)
    stats.update(fm)

    # 4. MONITORS edges (Chiller 6 sensors -> failure modes)
    logger.info("Step 4/7: Loading MONITORS edges...")
    monitors = _load_monitors_edges(client, graph)
    stats.update(monitors)

    # 5. Work orders from CSV
    logger.info("Step 5/7: Loading work orders...")
    wo = _load_work_orders(client, data_dir, graph)
    stats.update(wo)

    # 6. Alert events from CSV
    logger.info("Step 6/7: Loading alert events...")
    alerts = _load_alert_events(client, data_dir, graph)
    stats.update(alerts)

    # 7. Anomaly events from CSV
    logger.info("Step 7/8: Loading anomaly events...")
    anomalies = _load_anomaly_events(client, data_dir, graph)
    stats.update(anomalies)

    # 8. Unified events from event.csv (6,256 events with ISO dates)
    logger.info("Step 8/8: Loading unified events...")
    events = _load_unified_events(client, data_dir, graph)
    stats.update(events)

    logger.info("IBM data load complete. Stats: %s", stats)
    return stats
