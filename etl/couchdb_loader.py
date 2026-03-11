"""CouchDB sensor data loader.

Reads chiller*_sensordata_couchdb.json files from the AssetOpsBench data
directory and creates Sensor + SensorReading nodes linked to the Equipment
hierarchy already present in the graph.

If no JSON files are found, creates synthetic sensor data (3 sensors per
equipment, 10 readings each) so the rest of the pipeline can proceed.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from samyama import SamyamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensor definitions per equipment class (used for both real and synthetic)
# ---------------------------------------------------------------------------

SENSOR_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "chiller": [
        {"suffix": "CondWaterRetTemp",   "type": "temperature", "unit": "F",    "min": 40,  "max": 120, "sampling_rate_hz": 0.067},
        {"suffix": "Efficiency",         "type": "efficiency",  "unit": "kW/t", "min": 0.1, "max": 1.5, "sampling_rate_hz": 0.067},
        {"suffix": "Tonnage",            "type": "load",        "unit": "tons", "min": 0,   "max": 1000, "sampling_rate_hz": 0.067},
        {"suffix": "SupplyTemp",         "type": "temperature", "unit": "F",    "min": 38,  "max": 55,  "sampling_rate_hz": 0.067},
        {"suffix": "ReturnTemp",         "type": "temperature", "unit": "F",    "min": 50,  "max": 65,  "sampling_rate_hz": 0.067},
        {"suffix": "CondWaterFlow",      "type": "flow",        "unit": "GPM",  "min": 500, "max": 10000, "sampling_rate_hz": 0.067},
        {"suffix": "PowerInput",         "type": "power",       "unit": "kW",   "min": 50,  "max": 800, "sampling_rate_hz": 0.067},
        {"suffix": "PercentLoaded",      "type": "load",        "unit": "%",    "min": 0,   "max": 110, "sampling_rate_hz": 0.067},
        {"suffix": "EvapRefrigTemp",     "type": "temperature", "unit": "F",    "min": 10,  "max": 50,  "sampling_rate_hz": 0.067},
        {"suffix": "SetpointTemp",       "type": "temperature", "unit": "F",    "min": 40,  "max": 50,  "sampling_rate_hz": 0.067},
    ],
    "ahu": [
        {"suffix": "SupplyAirTemp",  "type": "temperature", "unit": "F",   "min": 50, "max": 80,   "sampling_rate_hz": 0.067},
        {"suffix": "ReturnAirTemp",  "type": "temperature", "unit": "F",   "min": 65, "max": 85,   "sampling_rate_hz": 0.067},
        {"suffix": "FanSpeed",       "type": "speed",       "unit": "RPM", "min": 0,  "max": 1800, "sampling_rate_hz": 0.067},
    ],
    "pump": [
        {"suffix": "DischargePressure", "type": "pressure",    "unit": "PSI",  "min": 20,  "max": 120,  "sampling_rate_hz": 0.1},
        {"suffix": "FlowRate",          "type": "flow",        "unit": "GPM",  "min": 100, "max": 5000, "sampling_rate_hz": 0.1},
        {"suffix": "Vibration",         "type": "vibration",   "unit": "mm/s", "min": 0,   "max": 15,   "sampling_rate_hz": 1.0},
    ],
    "motor": [
        {"suffix": "Current",   "type": "current",     "unit": "A",    "min": 5,  "max": 200, "sampling_rate_hz": 0.1},
        {"suffix": "Vibration", "type": "vibration",   "unit": "mm/s", "min": 0,  "max": 10,  "sampling_rate_hz": 1.0},
        {"suffix": "BearingTemp", "type": "temperature", "unit": "F",  "min": 100, "max": 220, "sampling_rate_hz": 0.067},
    ],
    "boiler": [
        {"suffix": "SteamPressure",   "type": "pressure",    "unit": "PSI",  "min": 5,  "max": 150, "sampling_rate_hz": 0.067},
        {"suffix": "WaterTemp",       "type": "temperature", "unit": "F",    "min": 140, "max": 250, "sampling_rate_hz": 0.067},
        {"suffix": "ExhaustGasTemp",  "type": "temperature", "unit": "F",    "min": 300, "max": 600, "sampling_rate_hz": 0.067},
    ],
}


def _escape(val: Any) -> str:
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _props_string(props: dict[str, Any]) -> str:
    parts = [f"{k}: {_escape(v)}" for k, v in props.items()]
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Real CouchDB JSON loader
# ---------------------------------------------------------------------------

def _load_couchdb_json(
    client: SamyamaClient, data_dir: str, graph: str
) -> Dict[str, int] | None:
    """Try to load chiller*_sensordata_couchdb.json files.

    The JSON structure is an array of objects, each with:
      - asset_id: str
      - timestamp: str (ISO 8601)
      - <sensor_name>: float  (remaining keys are sensor readings)

    Returns stats dict if files found, None otherwise.
    """
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob("chiller*_sensordata_couchdb.json"))

    # Also check under src/couchdb/sample_data/ (AssetOpsBench layout)
    if not json_files:
        alt_path = data_path / "src" / "couchdb" / "sample_data"
        json_files = sorted(alt_path.glob("chiller*_sensordata_couchdb.json"))
    if not json_files:
        alt_path = data_path / "couchdb" / "sample_data"
        json_files = sorted(alt_path.glob("chiller*_sensordata_couchdb.json"))

    if not json_files:
        return None

    sensors_created: set[str] = set()
    readings_count = 0
    sample_interval = 100  # keep every 100th reading

    for json_file in json_files:
        logger.info("Loading CouchDB JSON: %s", json_file)
        with open(json_file) as f:
            data = json.load(f)

        # Handle both array format and {docs: [...]} format
        if isinstance(data, dict) and "docs" in data:
            records = data["docs"]
        elif isinstance(data, list):
            records = data
        else:
            logger.warning("Unexpected JSON structure in %s, skipping", json_file)
            continue

        for idx, record in enumerate(records):
            asset_id = record.get("asset_id", "")
            timestamp = record.get("timestamp", "")

            # Extract sensor names (everything except asset_id and timestamp)
            sensor_keys = [
                k for k in record.keys()
                if k not in ("asset_id", "timestamp", "_id", "_rev")
            ]

            # Create Sensor nodes (once per sensor name)
            for sensor_key in sensor_keys:
                if sensor_key not in sensors_created:
                    # Derive sensor type from key name
                    key_lower = sensor_key.lower()
                    if "temp" in key_lower:
                        stype = "temperature"
                    elif "flow" in key_lower:
                        stype = "flow"
                    elif "pressure" in key_lower:
                        stype = "pressure"
                    elif "efficiency" in key_lower:
                        stype = "efficiency"
                    elif "power" in key_lower:
                        stype = "power"
                    elif "vibration" in key_lower:
                        stype = "vibration"
                    else:
                        stype = "gauge"

                    # Clean sensor name for Cypher (replace special chars)
                    clean_name = sensor_key.replace('"', '\\"')
                    props = {
                        "name": clean_name,
                        "type": stype,
                        "unit": "raw",
                        "min_threshold": 0.0,
                        "max_threshold": 999.0,
                        "sampling_rate_hz": 0.067,
                    }
                    client.query(f"CREATE (s:Sensor {_props_string(props)})", graph)

                    # Link to equipment — match by asset_id prefix
                    # CouchDB data uses "Chiller 6", our equipment uses "Chiller-*"
                    # Try to match by class
                    equip_class = asset_id.split()[0].lower() if asset_id else ""
                    if equip_class:
                        match_cypher = (
                            f'MATCH (e:Equipment), (s:Sensor {{name: "{clean_name}"}}) '
                            f'WHERE e.iso14224_class = "{equip_class}" '
                            f"CREATE (e)-[:HAS_SENSOR]->(s)"
                        )
                        try:
                            client.query(match_cypher, graph)
                        except Exception:
                            logger.debug("Could not link sensor %s to %s equipment", clean_name, equip_class)

                    sensors_created.add(sensor_key)

            # Create SensorReading nodes (sampled)
            if idx % sample_interval == 0:
                for sensor_key in sensor_keys:
                    value = record.get(sensor_key)
                    if value is None:
                        continue
                    clean_name = sensor_key.replace('"', '\\"')
                    reading_props = {
                        "timestamp": timestamp,
                        "value": round(float(value), 4),
                        "unit": "raw",
                        "quality_flag": "good",
                    }
                    client.query(
                        f"CREATE (r:SensorReading {_props_string(reading_props)})",
                        graph,
                    )

                    # Link reading to sensor
                    link_cypher = (
                        f'MATCH (s:Sensor {{name: "{clean_name}"}}), '
                        f'(r:SensorReading {{timestamp: "{timestamp}", value: {round(float(value), 4)}}}) '
                        f"CREATE (s)-[:PRODUCED_READING]->(r)"
                    )
                    try:
                        client.query(link_cypher, graph)
                    except Exception:
                        logger.debug("Could not link reading to sensor %s", clean_name)

                    readings_count += 1

    return {
        "sensors": len(sensors_created),
        "readings": readings_count,
    }


# ---------------------------------------------------------------------------
# Synthetic sensor data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_sensors(
    client: SamyamaClient, graph: str
) -> Dict[str, int]:
    """Generate synthetic sensor data: 3 sensors per equipment, 10 readings each."""
    sensors_count = 0
    readings_count = 0

    # Fetch all equipment names and their classes
    result = client.query_readonly(
        "MATCH (e:Equipment) RETURN e.name, e.iso14224_class", graph
    )

    equipment_list: list[tuple[str, str]] = []
    for record in result.records:
        name = record[0] if record[0] is not None else ""
        eq_class = record[1] if record[1] is not None else ""
        if name and eq_class:
            equipment_list.append((str(name), str(eq_class)))

    if not equipment_list:
        logger.warning("No equipment found in graph; cannot create sensors")
        return {"sensors": 0, "readings": 0}

    base_time = datetime(2024, 1, 1, 0, 0, 0)
    random.seed(42)  # reproducible

    for equip_name, equip_class in equipment_list:
        templates = SENSOR_TEMPLATES.get(equip_class, SENSOR_TEMPLATES["motor"])
        # Take up to 3 sensors per equipment
        chosen = templates[:3]

        for tmpl in chosen:
            sensor_name = f"{equip_name}-{tmpl['suffix']}"
            props = {
                "name": sensor_name,
                "type": tmpl["type"],
                "unit": tmpl["unit"],
                "min_threshold": tmpl["min"],
                "max_threshold": tmpl["max"],
                "sampling_rate_hz": tmpl["sampling_rate_hz"],
            }
            client.query(f"CREATE (s:Sensor {_props_string(props)})", graph)
            sensors_count += 1

            # Link to equipment
            link_cypher = (
                f'MATCH (e:Equipment {{name: "{equip_name}"}}), '
                f'(s:Sensor {{name: "{sensor_name}"}}) '
                f"CREATE (e)-[:HAS_SENSOR]->(s)"
            )
            client.query(link_cypher, graph)

            # Create 10 readings per sensor
            for i in range(10):
                ts = base_time + timedelta(minutes=15 * i)
                value = round(
                    random.uniform(float(tmpl["min"]), float(tmpl["max"])), 4
                )
                reading_props = {
                    "timestamp": ts.isoformat(),
                    "value": value,
                    "unit": tmpl["unit"],
                    "quality_flag": "good",
                }
                client.query(
                    f"CREATE (r:SensorReading {_props_string(reading_props)})",
                    graph,
                )

                # Link reading to sensor
                link_cypher = (
                    f'MATCH (s:Sensor {{name: "{sensor_name}"}}), '
                    f'(r:SensorReading {{timestamp: "{ts.isoformat()}", value: {value}}}) '
                    f"CREATE (s)-[:PRODUCED_READING]->(r)"
                )
                try:
                    client.query(link_cypher, graph)
                except Exception:
                    logger.debug("Could not link reading for sensor %s", sensor_name)

                readings_count += 1

    return {"sensors": sensors_count, "readings": readings_count}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_couchdb(
    client: SamyamaClient, data_dir: str, graph: str = "industrial"
) -> Dict[str, int]:
    """Load sensor data into the knowledge graph.

    Tries to read CouchDB JSON exports from data_dir.  Falls back to
    synthetic sensor generation if no files are found.

    Returns dict with counts: {sensors, readings}.
    """
    stats = _load_couchdb_json(client, data_dir, graph)
    if stats is not None:
        logger.info("Loaded real CouchDB data: %s", stats)
        return stats

    logger.info("No CouchDB JSON files found; generating synthetic sensor data")
    return _generate_synthetic_sensors(client, graph)
