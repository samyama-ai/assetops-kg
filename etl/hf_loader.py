"""HuggingFace AssetOpsBench scenario loader.

Reads the six HF JSON files (467 rows total) and populates the
Industrial KG with:
  - Equipment nodes for new equipment types
  - FailureMode nodes extracted from FMSR characteristic_form
  - Sensor nodes extracted from FMSR text / characteristic_form
  - MonitoringRule nodes from rule_logic scenarios
  - PHMScenario nodes from prognostics_and_health_management scenarios
  - HFScenario nodes for every row (umbrella reference)
  - Relationship edges: MONITORS, EXPERIENCED, HAS_RULE,
    HAS_PHM_SCENARIO, TARGETS_EQUIPMENT, INVOLVES_FAILURE_MODE,
    TESTS_RULE

All operations are idempotent — duplicate checks via MATCH before CREATE.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from samyama import SamyamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cypher helpers (same pattern as fmsr_loader / eamlite_loader)
# ---------------------------------------------------------------------------


def _escape(val: Any) -> str:
    """Escape a value for embedding in a Cypher property literal."""
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return '""'
    return str(val)


def _props(props: dict[str, Any]) -> str:
    """Build a Cypher property-map string: {key: value, ...}."""
    parts = [f"{k}: {_escape(v)}" for k, v in props.items() if v is not None]
    return "{" + ", ".join(parts) + "}"


def _esc(text: str) -> str:
    """Escape a string for use inside a Cypher quoted literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Entity normalisation
# ---------------------------------------------------------------------------

# Map raw entity strings from the JSON files to canonical iso14224_class
# values used in the KG Equipment nodes.
_ENTITY_TO_CLASS: dict[str, str] = {
    # FMSR config entities
    "electric motor": "electric_motor",
    "pump": "pump",
    "power transformer": "power_transformer",
    "industrial gas turbine": "industrial_gas_turbine",
    "compressor": "compressor",
    "reciprocating internal combustion engine": "reciprocating_engine",
    "fan": "fan",
    "aero gas turbine": "aero_gas_turbine",
    "steam turbine": "steam_turbine",
    "electric generator": "electric_generator",
    # Rule-logic entities
    "AHU": "ahu",
    "ahu": "ahu",
    "CRAC": "crac",
    "crac": "crac",
    "Pump": "pump",
    "Boiler": "boiler",
    "boiler": "boiler",
    "HXU": "hxu",
    "hxu": "hxu",
    "Cooling Tower": "cooling_tower",
    "cooling tower": "cooling_tower",
    "Chiller": "chiller",
    "chiller": "chiller",
    # PHM entities
    "turbofan engine": "turbofan_engine",
    "induction motor": "induction_motor",
    "induction motor ": "induction_motor",  # trailing space in data
    "bearing": "bearing",
    "bearings": "bearing",
    "rotor": "rotor",
    "gearbox": "gearbox",
    "gearboxes": "gearbox",
    "turbine": "turbine",
    "engine": "turbofan_engine",
    "engines": "turbofan_engine",
    "motor": "motor",
    # Multiagent entities
    "hydrolic_pump": "hydraulic_pump",
    "hydraulic_pump": "hydraulic_pump",
    # Generic / base-scenario entities (from hf_scenarios.json)
    "Equipment": "equipment",
    "Site": "site",
    "WindTurbine": "wind_turbine",
    "wind_turbine": "wind_turbine",
}

# Human-readable display names for each iso14224_class
_CLASS_DISPLAY: dict[str, str] = {
    "electric_motor": "Electric Motor",
    "pump": "Pump",
    "power_transformer": "Power Transformer",
    "industrial_gas_turbine": "Industrial Gas Turbine",
    "compressor": "Compressor",
    "reciprocating_engine": "Reciprocating Internal Combustion Engine",
    "fan": "Fan",
    "aero_gas_turbine": "Aero Gas Turbine",
    "steam_turbine": "Steam Turbine",
    "electric_generator": "Electric Generator",
    "crac": "CRAC",
    "hxu": "HXU",
    "cooling_tower": "Cooling Tower",
    "turbofan_engine": "Turbofan Engine",
    "induction_motor": "Induction Motor",
    "bearing": "Bearing",
    "rotor": "Rotor",
    "gearbox": "Gearbox",
    "turbine": "Turbine",
    "hydraulic_pump": "Hydraulic Pump",
    "ahu": "AHU",
    "chiller": "Chiller",
    "boiler": "Boiler",
    "motor": "Motor",
    "equipment": "Equipment",
    "site": "Site",
    "wind_turbine": "Wind Turbine",
}

# Sensor type mapping: canonical FMSR sensor names from ISO 14224
_FMSR_SENSOR_TYPES: dict[str, dict[str, str]] = {
    "electric_motor": {
        "vibration": "Vibration Sensor",
        "cooling_gas": "Cooling Gas Sensor",
        "axial_flux": "Axial Flux Sensor",
        "power": "Power Sensor",
        "current": "Current Sensor",
        "temperature": "Temperature Sensor",
    },
    "steam_turbine": {
        "vibration": "Vibration Sensor",
        "oil_debris": "Oil Debris Sensor",
        "length_measurement": "Length Measurement Sensor",
        "temperature": "Temperature Sensor",
    },
    "aero_gas_turbine": {
        "vibration": "Vibration Sensor",
        "speed": "Speed Sensor",
        "fuel_pressure": "Fuel Pressure Sensor",
        "oil_debris": "Oil Debris Sensor",
        "compressor_pressure": "Compressor Pressure Sensor",
        "air_flow": "Air Flow Sensor",
        "temperature": "Temperature Sensor",
        "oil_pressure": "Oil Pressure Sensor",
    },
    "industrial_gas_turbine": {
        "speed": "Speed Sensor",
        "air_flow": "Air Flow Sensor",
        "exhaust_temperature": "Exhaust Temperature Sensor",
        "fuel_pressure": "Fuel Pressure Sensor",
        "compressor_efficiency": "Compressor Efficiency Sensor",
        "vibration": "Vibration Sensor",
        "compressor_pressure": "Compressor Pressure Sensor",
        "output_power": "Output Power Sensor",
    },
    "pump": {
        "vibration": "Vibration Sensor",
        "flow_rate": "Flow Rate Sensor",
        "pressure": "Pressure Sensor",
        "temperature": "Temperature Sensor",
        "acoustic": "Acoustic Sensor",
    },
    "power_transformer": {
        "dissolved_gas": "Dissolved Gas Analyzer",
        "temperature": "Temperature Sensor",
        "ultrasound": "Ultrasound Sensor",
        "frequency_response": "Frequency Response Analyzer",
        "oil_level": "Oil Level Sensor",
        "partial_discharge": "Partial Discharge Sensor",
    },
    "compressor": {
        "vibration": "Vibration Sensor",
        "pressure": "Pressure Sensor",
        "temperature": "Temperature Sensor",
        "speed": "Speed Sensor",
        "oil_debris": "Oil Debris Sensor",
        "acoustic": "Acoustic Sensor",
        "oil_level": "Oil Level Sensor",
    },
    "reciprocating_engine": {
        "cylinder_pressure": "Cylinder Pressure Sensor",
        "fuel_flow": "Fuel Flow Sensor",
        "vibration": "Vibration Sensor",
        "oil_debris": "Oil Debris Sensor",
        "exhaust_temperature": "Exhaust Temperature Sensor",
        "oil_pressure": "Oil Pressure Sensor",
        "temperature": "Temperature Sensor",
    },
    "fan": {
        "vibration": "Vibration Sensor",
        "temperature": "Temperature Sensor",
        "noise": "Noise Sensor",
        "current": "Current Sensor",
    },
    "electric_generator": {
        "vibration": "Vibration Sensor",
        "temperature": "Temperature Sensor",
        "current": "Current Sensor",
        "partial_discharge": "Partial Discharge Sensor",
    },
}

# Compressor multiagent sensor set
_COMPRESSOR_SENSORS = [
    ("pressure_1", "Pressure Sensor 1", "bar"),
    ("pressure_2", "Pressure Sensor 2", "bar"),
    ("temperature_1", "Temperature Sensor 1", "C"),
    ("temperature_2", "Temperature Sensor 2", "C"),
    ("vibration", "Vibration Sensor", "mm/s"),
    ("motor_current", "Motor Current Sensor", "A"),
    ("valve_position", "Valve Position Sensor", "%"),
]

# Hydraulic pump multiagent sensor set
_HYDRAULIC_PUMP_SENSORS = [
    ("PS1", "Pressure Sensor PS1", "bar"),
    ("PS2", "Pressure Sensor PS2", "bar"),
    ("PS3", "Pressure Sensor PS3", "bar"),
    ("PS4", "Pressure Sensor PS4", "bar"),
    ("PS5", "Pressure Sensor PS5", "bar"),
    ("PS6", "Pressure Sensor PS6", "bar"),
    ("FS1", "Flow Sensor FS1", "l/min"),
    ("FS2", "Flow Sensor FS2", "l/min"),
    ("TS1", "Temperature Sensor TS1", "C"),
    ("TS2", "Temperature Sensor TS2", "C"),
    ("TS3", "Temperature Sensor TS3", "C"),
    ("TS4", "Temperature Sensor TS4", "C"),
    ("VS1", "Vibration Sensor VS1", "mm/s"),
    ("EPS1", "Motor Power Sensor EPS1", "W"),
    ("SE", "Efficiency Factor SE", "%"),
    ("CE", "Cooling Efficiency CE", "%"),
    ("CP", "Cooling Power CP", "kW"),
]


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_failure_modes(characteristic_form: str) -> list[str]:
    """Extract failure mode names from bracket-delimited lists in
    the characteristic_form field.

    Returns de-duplicated, cleaned failure mode strings.
    """
    matches = re.findall(r"\[([^\]]+)\]", characteristic_form)
    modes: set[str] = set()
    for match_str in matches:
        # Items may contain commas inside names (e.g. "through fault e.g. lightning strike")
        # so we split carefully
        items = re.split(r",\s*(?=[A-Za-z])", match_str)
        for item in items:
            cleaned = item.strip().strip("'\"").strip()
            # Skip items that look like sensor names (contain "Sensor")
            if cleaned and len(cleaned) > 3 and "Sensor" not in cleaned:
                modes.add(cleaned)
    return sorted(modes)


def _extract_sensors_from_text(text: str) -> list[str]:
    """Extract sensor type mentions from a FMSR question text."""
    # Common sensor keywords found in the FMSR texts
    sensor_patterns = [
        r"vibration",
        r"cooling gas",
        r"axial flux",
        r"power",
        r"current",
        r"temperature",
        r"oil debris",
        r"speed",
        r"fuel (?:pressure|flow)",
        r"compressor (?:pressure|efficiency)",
        r"air flow",
        r"output power",
        r"flow rate",
        r"pressure",
        r"ultrasound",
        r"dissolved gas(?: analysis)?",
        r"frequency response(?: analysis)?",
        r"cylinder pressure",
        r"exhaust temperature",
        r"oil (?:pressure|level|analysis)",
        r"partial discharge",
        r"acoustic",
        r"noise",
        r"voltage",
        r"length measurement",
    ]
    found: list[str] = []
    text_lower = text.lower()
    for pattern in sensor_patterns:
        if re.search(pattern, text_lower):
            found.append(pattern.replace(r"(?:", "").replace(")", "").replace(r"\s+", " "))
    return found


def _normalise_entity(raw: str) -> str | None:
    """Convert a raw entity string to its canonical iso14224_class.

    Returns None for empty or unrecognised entities.
    """
    raw = raw.strip()
    if not raw:
        return None
    key = raw.lower() if raw.lower() in _ENTITY_TO_CLASS else raw
    return _ENTITY_TO_CLASS.get(key)


def _make_fm_name(description: str) -> str:
    """Create a short kebab-case id from a failure mode description."""
    words = re.split(r"[\s/:,]+", description)
    meaningful = [w.lower() for w in words[:5] if len(w) > 1]
    return "-".join(meaningful[:4]).replace('"', "").replace("'", "")


# ---------------------------------------------------------------------------
# Idempotent node/edge creation wrappers
# ---------------------------------------------------------------------------


def _ensure_equipment(
    client: SamyamaClient,
    graph: str,
    iso_class: str,
    created_equip: set[str],
) -> str:
    """Create an Equipment node for the given class if not already created.

    Returns the equipment name used as the unique key.
    """
    if iso_class in created_equip:
        return _CLASS_DISPLAY.get(iso_class, iso_class)

    display = _CLASS_DISPLAY.get(iso_class, iso_class.replace("_", " ").title())
    name = f"HF-{display}"

    # Check if equipment of this class already exists (from eamlite or prior run)
    result = client.query_readonly(
        f'MATCH (e:Equipment {{iso14224_class: "{_esc(iso_class)}"}}) RETURN count(e)',
        graph,
    )
    existing = 0
    if result.records:
        val = result.records[0][0]
        existing = int(val) if val is not None else 0

    if existing == 0:
        props = {
            "name": name,
            "iso14224_class": iso_class,
            "isa95_level": 1,
            "status": "reference",
            "criticality_score": 5,
        }
        client.query(f"CREATE (e:Equipment {_props(props)})", graph)
        logger.info("Created Equipment node: %s (%s)", name, iso_class)

    created_equip.add(iso_class)
    return name


def _ensure_failure_mode(
    client: SamyamaClient,
    graph: str,
    description: str,
    equip_class: str,
    created_fm: set[str],
) -> str | None:
    """Create a FailureMode node if it does not exist yet.

    Returns the failure mode short name, or None if skipped.
    """
    short_name = _make_fm_name(description)
    if not short_name or len(short_name) < 3:
        return None

    key = f"{equip_class}:{short_name}"
    if key in created_fm:
        return short_name

    # Check for duplicate
    result = client.query_readonly(
        f'MATCH (fm:FailureMode {{name: "{_esc(short_name)}"}}) RETURN count(fm)',
        graph,
    )
    existing = 0
    if result.records:
        val = result.records[0][0]
        existing = int(val) if val is not None else 0

    if existing == 0:
        fm_props = {
            "name": short_name,
            "description": description,
            "severity": "medium",
            "iso14224_mechanism": "unknown",
            "category": equip_class,
        }
        client.query(f"CREATE (fm:FailureMode {_props(fm_props)})", graph)

    created_fm.add(key)
    return short_name


def _ensure_sensor(
    client: SamyamaClient,
    graph: str,
    sensor_name: str,
    sensor_type: str,
    unit: str,
    created_sensors: set[str],
) -> None:
    """Create a Sensor node if it does not exist yet."""
    if sensor_name in created_sensors:
        return

    result = client.query_readonly(
        f'MATCH (s:Sensor {{name: "{_esc(sensor_name)}"}}) RETURN count(s)',
        graph,
    )
    existing = 0
    if result.records:
        val = result.records[0][0]
        existing = int(val) if val is not None else 0

    if existing == 0:
        s_props = {
            "name": sensor_name,
            "type": sensor_type,
            "unit": unit,
        }
        client.query(f"CREATE (s:Sensor {_props(s_props)})", graph)

    created_sensors.add(sensor_name)


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read a newline-delimited JSON file into a list of dicts."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_fmsr_scenarios(
    client: SamyamaClient,
    graph: str,
    rows: list[dict],
    created_equip: set[str],
    created_fm: set[str],
    created_sensors: set[str],
    stats: dict[str, int],
) -> None:
    """Process hf_failure_mode_sensor_mapping.json rows.

    Creates Equipment, FailureMode, Sensor nodes and
    EXPERIENCED, MONITORS edges.
    """
    print("  Loading FMSR failure-mode-sensor mapping (88 rows)...")

    for i, row in enumerate(rows, 1):
        entity_raw = row.get("entity", "")
        iso_class = _normalise_entity(entity_raw)
        if iso_class is None:
            continue

        equip_name = _ensure_equipment(client, graph, iso_class, created_equip)

        # Extract failure modes from characteristic_form
        cf = row.get("characteristic_form", "")
        failure_modes = _extract_failure_modes(cf)

        for fm_desc in failure_modes:
            fm_name = _ensure_failure_mode(client, graph, fm_desc, iso_class, created_fm)
            if fm_name is None:
                continue

            # EXPERIENCED edge: Equipment -> FailureMode
            try:
                client.query(
                    f'MATCH (e:Equipment {{iso14224_class: "{_esc(iso_class)}"}}), '
                    f'(fm:FailureMode {{name: "{_esc(fm_name)}"}}) '
                    f"MERGE (e)-[:EXPERIENCED]->(fm)",
                    graph,
                )
                stats["experienced_edges"] += 1
            except Exception:
                # MERGE may not be supported; fall back to CREATE with guard
                try:
                    client.query(
                        f'MATCH (e:Equipment {{iso14224_class: "{_esc(iso_class)}"}}), '
                        f'(fm:FailureMode {{name: "{_esc(fm_name)}"}}) '
                        f"CREATE (e)-[:EXPERIENCED]->(fm)",
                        graph,
                    )
                    stats["experienced_edges"] += 1
                except Exception as exc:
                    logger.debug("EXPERIENCED edge skipped: %s", exc)

        # Create sensors for this equipment type from the canonical mapping
        sensor_map = _FMSR_SENSOR_TYPES.get(iso_class, {})
        for sensor_key, sensor_display in sensor_map.items():
            s_name = f"{equip_name}-{sensor_display}"
            _ensure_sensor(client, graph, s_name, sensor_display, "", created_sensors)
            stats["sensors"] += 1

            # MONITORS edge: Sensor -> Equipment (sensor monitors equipment)
            try:
                client.query(
                    f'MATCH (s:Sensor {{name: "{_esc(s_name)}"}}), '
                    f'(e:Equipment {{iso14224_class: "{_esc(iso_class)}"}}) '
                    f"CREATE (s)-[:MONITORS]->(e)",
                    graph,
                )
                stats["monitors_edges"] += 1
            except Exception as exc:
                logger.debug("MONITORS edge skipped: %s", exc)

        if i % 20 == 0:
            print(f"    ... processed {i}/{len(rows)} FMSR rows")

    stats["failure_modes"] = len(created_fm)
    print(f"    FMSR complete: {len(created_fm)} failure modes, "
          f"{stats['monitors_edges']} MONITORS edges")


def _load_rule_logic(
    client: SamyamaClient,
    graph: str,
    rows: list[dict],
    created_equip: set[str],
    stats: dict[str, int],
) -> None:
    """Process hf_rule_logic.json rows.

    Creates MonitoringRule nodes and HAS_RULE edges.
    """
    print("  Loading rule-logic monitoring rules (120 rows)...")

    for i, row in enumerate(rows, 1):
        rule_id = str(row.get("id", f"rule_{i}"))
        entity_raw = row.get("entity", "")
        iso_class = _normalise_entity(entity_raw)

        # Truncate the rule description for the node (full text may be huge)
        text = row.get("text", "")
        description = text[:500] if len(text) > 500 else text

        rule_props = {
            "rule_id": rule_id,
            "description": description,
            "entity": entity_raw,
            "category": row.get("category", ""),
            "deterministic": row.get("deterministic", True),
            "group": row.get("group", ""),
        }

        # Check if this rule already exists
        result = client.query_readonly(
            f'MATCH (r:MonitoringRule {{rule_id: "{_esc(rule_id)}"}}) RETURN count(r)',
            graph,
        )
        existing = 0
        if result.records:
            val = result.records[0][0]
            existing = int(val) if val is not None else 0

        if existing == 0:
            client.query(
                f"CREATE (r:MonitoringRule {_props(rule_props)})", graph
            )
        stats["monitoring_rules"] += 1

        # HAS_RULE edge: Equipment -> MonitoringRule
        if iso_class:
            _ensure_equipment(client, graph, iso_class, created_equip)
            try:
                client.query(
                    f'MATCH (e:Equipment {{iso14224_class: "{_esc(iso_class)}"}}), '
                    f'(r:MonitoringRule {{rule_id: "{_esc(rule_id)}"}}) '
                    f"CREATE (e)-[:HAS_RULE]->(r)",
                    graph,
                )
                stats["has_rule_edges"] += 1
            except Exception as exc:
                logger.debug("HAS_RULE edge skipped: %s", exc)

        if i % 30 == 0:
            print(f"    ... processed {i}/{len(rows)} rule-logic rows")

    print(f"    Rule-logic complete: {stats['monitoring_rules']} rules, "
          f"{stats['has_rule_edges']} HAS_RULE edges")


def _load_phm_scenarios(
    client: SamyamaClient,
    graph: str,
    rows: list[dict],
    created_equip: set[str],
    stats: dict[str, int],
) -> None:
    """Process hf_prognostics_and_health_management.json rows.

    Creates PHMScenario nodes and HAS_PHM_SCENARIO edges.
    """
    print("  Loading PHM scenarios (75 rows)...")

    for i, row in enumerate(rows, 1):
        scenario_id = str(row.get("id", f"phm_{i}"))
        entity_raw = row.get("entity", "")
        iso_class = _normalise_entity(entity_raw)

        text = row.get("text", "")
        description = text[:500] if len(text) > 500 else text

        phm_props = {
            "scenario_id": scenario_id,
            "task_type": row.get("category", ""),
            "entity": entity_raw,
            "description": description,
            "deterministic": row.get("deterministic", True),
            "group": row.get("group", ""),
        }

        # Check for existing
        result = client.query_readonly(
            f'MATCH (p:PHMScenario {{scenario_id: "{_esc(scenario_id)}"}}) RETURN count(p)',
            graph,
        )
        existing = 0
        if result.records:
            val = result.records[0][0]
            existing = int(val) if val is not None else 0

        if existing == 0:
            client.query(
                f"CREATE (p:PHMScenario {_props(phm_props)})", graph
            )
        stats["phm_scenarios"] += 1

        # HAS_PHM_SCENARIO edge
        if iso_class:
            _ensure_equipment(client, graph, iso_class, created_equip)
            try:
                client.query(
                    f'MATCH (e:Equipment {{iso14224_class: "{_esc(iso_class)}"}}), '
                    f'(p:PHMScenario {{scenario_id: "{_esc(scenario_id)}"}}) '
                    f"CREATE (e)-[:HAS_PHM_SCENARIO]->(p)",
                    graph,
                )
                stats["has_phm_edges"] += 1
            except Exception as exc:
                logger.debug("HAS_PHM_SCENARIO edge skipped: %s", exc)

        if i % 20 == 0:
            print(f"    ... processed {i}/{len(rows)} PHM rows")

    print(f"    PHM complete: {stats['phm_scenarios']} scenarios, "
          f"{stats['has_phm_edges']} HAS_PHM_SCENARIO edges")


def _load_multiagent_compressor(
    client: SamyamaClient,
    graph: str,
    rows: list[dict],
    created_equip: set[str],
    created_sensors: set[str],
    stats: dict[str, int],
) -> None:
    """Process hf_compressor.json rows.

    Creates compressor Equipment with its sensor set.
    """
    print("  Loading compressor multiagent scenarios (15 rows)...")

    iso_class = "compressor"
    _ensure_equipment(client, graph, iso_class, created_equip)

    # Create the compressor sensor set
    equip_name = f"HF-{_CLASS_DISPLAY[iso_class]}"
    for sensor_key, sensor_display, unit in _COMPRESSOR_SENSORS:
        s_name = f"{equip_name}-{sensor_display}"
        _ensure_sensor(client, graph, s_name, sensor_display, unit, created_sensors)

        # HAS_SENSOR edge: Equipment -> Sensor
        try:
            client.query(
                f'MATCH (e:Equipment {{name: "{_esc(equip_name)}"}}), '
                f'(s:Sensor {{name: "{_esc(s_name)}"}}) '
                f"CREATE (e)-[:HAS_SENSOR]->(s)",
                graph,
            )
        except Exception as exc:
            logger.debug("HAS_SENSOR edge skipped: %s", exc)

    stats["multiagent_compressor"] = len(rows)
    print(f"    Compressor multiagent: {len(rows)} scenarios, "
          f"{len(_COMPRESSOR_SENSORS)} sensors")


def _load_multiagent_hydraulic_pump(
    client: SamyamaClient,
    graph: str,
    rows: list[dict],
    created_equip: set[str],
    created_sensors: set[str],
    stats: dict[str, int],
) -> None:
    """Process hf_hydrolic_pump.json rows.

    Creates hydraulic pump Equipment with its sensor set.
    """
    print("  Loading hydraulic pump multiagent scenarios (17 rows)...")

    iso_class = "hydraulic_pump"
    _ensure_equipment(client, graph, iso_class, created_equip)

    equip_name = f"HF-{_CLASS_DISPLAY[iso_class]}"
    for sensor_key, sensor_display, unit in _HYDRAULIC_PUMP_SENSORS:
        s_name = f"{equip_name}-{sensor_display}"
        _ensure_sensor(client, graph, s_name, sensor_display, unit, created_sensors)

        try:
            client.query(
                f'MATCH (e:Equipment {{name: "{_esc(equip_name)}"}}), '
                f'(s:Sensor {{name: "{_esc(s_name)}"}}) '
                f"CREATE (e)-[:HAS_SENSOR]->(s)",
                graph,
            )
        except Exception as exc:
            logger.debug("HAS_SENSOR edge skipped: %s", exc)

    stats["multiagent_hydraulic_pump"] = len(rows)
    print(f"    Hydraulic pump multiagent: {len(rows)} scenarios, "
          f"{len(_HYDRAULIC_PUMP_SENSORS)} sensors")


def _load_hf_scenario_nodes(
    client: SamyamaClient,
    graph: str,
    all_rows: list[tuple[str, dict]],
    created_equip: set[str],
    stats: dict[str, int],
) -> None:
    """Create an HFScenario umbrella node for every row across all files.

    Also creates TARGETS_EQUIPMENT edges where entity is known.
    """
    print("  Creating HFScenario umbrella nodes (467 total)...")

    for i, (source_file, row) in enumerate(all_rows, 1):
        scenario_id = f"hf_{source_file}_{row.get('id', i)}"
        entity_raw = row.get("entity", "")
        iso_class = _normalise_entity(entity_raw)

        text = row.get("text", "")
        description = text[:300] if len(text) > 300 else text

        hf_props = {
            "scenario_id": scenario_id,
            "type": row.get("type", ""),
            "entity": entity_raw,
            "category": row.get("category", ""),
            "group": row.get("group", ""),
            "deterministic": row.get("deterministic", False),
            "text": description,
        }

        # Check for existing
        result = client.query_readonly(
            f'MATCH (h:HFScenario {{scenario_id: "{_esc(scenario_id)}"}}) RETURN count(h)',
            graph,
        )
        existing = 0
        if result.records:
            val = result.records[0][0]
            existing = int(val) if val is not None else 0

        if existing == 0:
            client.query(
                f"CREATE (h:HFScenario {_props(hf_props)})", graph
            )
        stats["hf_scenarios"] += 1

        # TARGETS_EQUIPMENT edge
        if iso_class:
            _ensure_equipment(client, graph, iso_class, created_equip)
            try:
                client.query(
                    f'MATCH (h:HFScenario {{scenario_id: "{_esc(scenario_id)}"}}), '
                    f'(e:Equipment {{iso14224_class: "{_esc(iso_class)}"}}) '
                    f"CREATE (h)-[:TARGETS_EQUIPMENT]->(e)",
                    graph,
                )
                stats["targets_equip_edges"] += 1
            except Exception as exc:
                logger.debug("TARGETS_EQUIPMENT edge skipped: %s", exc)

        if i % 100 == 0:
            print(f"    ... processed {i}/{len(all_rows)} HFScenario nodes")

    print(f"    HFScenario nodes: {stats['hf_scenarios']}, "
          f"TARGETS_EQUIPMENT edges: {stats['targets_equip_edges']}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_hf_scenarios(
    client: SamyamaClient,
    data_dir: str = "data",
    tenant: str = "default",
    graph: str = "industrial",
) -> dict[str, int]:
    """Load expanded AssetOpsBench HF data into the KG.

    Reads the six HF JSON files from *data_dir* and creates:
      - Equipment nodes for new equipment types
      - FailureMode nodes from FMSR characteristic_form
      - Sensor nodes for each equipment type
      - MonitoringRule nodes from rule_logic
      - PHMScenario nodes from PHM data
      - HFScenario umbrella nodes for all 467 rows
      - Corresponding relationship edges

    Args:
        client: SamyamaClient instance (embedded or remote).
        data_dir: Path to the directory containing hf_*.json files.
        tenant: Tenant id (unused currently, reserved for multi-tenancy).
        graph: Graph name in Samyama.

    Returns:
        Dict with creation counts for all node/edge types.
    """
    data_path = Path(data_dir)

    # Expected file names
    files = {
        "fmsr": "hf_failure_mode_sensor_mapping.json",
        "rule_logic": "hf_rule_logic.json",
        "phm": "hf_prognostics_and_health_management.json",
        "compressor": "hf_compressor.json",
        "hydraulic_pump": "hf_hydrolic_pump.json",  # note: filename has typo
    }

    # Validate that files exist
    for key, fname in files.items():
        fpath = data_path / fname
        if not fpath.exists():
            logger.warning("HF file not found: %s — skipping %s", fpath, key)

    # Shared tracking sets (for idempotency)
    created_equip: set[str] = set()
    created_fm: set[str] = set()
    created_sensors: set[str] = set()

    stats: dict[str, int] = {
        "equipment": 0,
        "failure_modes": 0,
        "sensors": 0,
        "monitors_edges": 0,
        "experienced_edges": 0,
        "monitoring_rules": 0,
        "has_rule_edges": 0,
        "phm_scenarios": 0,
        "has_phm_edges": 0,
        "hf_scenarios": 0,
        "targets_equip_edges": 0,
        "multiagent_compressor": 0,
        "multiagent_hydraulic_pump": 0,
    }

    print("[bold]HF Loader:[/bold] Loading AssetOpsBench HuggingFace scenarios...")

    # Collect all rows for the umbrella HFScenario pass
    all_rows: list[tuple[str, dict]] = []

    # 1. FMSR — failure modes and sensor mapping
    fmsr_path = data_path / files["fmsr"]
    if fmsr_path.exists():
        fmsr_rows = _read_jsonl(fmsr_path)
        _load_fmsr_scenarios(
            client, graph, fmsr_rows,
            created_equip, created_fm, created_sensors, stats,
        )
        all_rows.extend(("fmsr", r) for r in fmsr_rows)
    else:
        print(f"  WARNING: {fmsr_path} not found, skipping FMSR")

    # 2. Rule logic — monitoring rules
    rule_path = data_path / files["rule_logic"]
    if rule_path.exists():
        rule_rows = _read_jsonl(rule_path)
        _load_rule_logic(client, graph, rule_rows, created_equip, stats)
        all_rows.extend(("rule_logic", r) for r in rule_rows)
    else:
        print(f"  WARNING: {rule_path} not found, skipping rule_logic")

    # 3. PHM — prognostics and health management
    phm_path = data_path / files["phm"]
    if phm_path.exists():
        phm_rows = _read_jsonl(phm_path)
        _load_phm_scenarios(client, graph, phm_rows, created_equip, stats)
        all_rows.extend(("phm", r) for r in phm_rows)
    else:
        print(f"  WARNING: {phm_path} not found, skipping PHM")

    # 4. Compressor multiagent
    comp_path = data_path / files["compressor"]
    if comp_path.exists():
        comp_rows = _read_jsonl(comp_path)
        _load_multiagent_compressor(
            client, graph, comp_rows,
            created_equip, created_sensors, stats,
        )
        all_rows.extend(("compressor", r) for r in comp_rows)
    else:
        print(f"  WARNING: {comp_path} not found, skipping compressor")

    # 5. Hydraulic pump multiagent
    hp_path = data_path / files["hydraulic_pump"]
    if hp_path.exists():
        hp_rows = _read_jsonl(hp_path)
        _load_multiagent_hydraulic_pump(
            client, graph, hp_rows,
            created_equip, created_sensors, stats,
        )
        all_rows.extend(("hydraulic_pump", r) for r in hp_rows)
    else:
        print(f"  WARNING: {hp_path} not found, skipping hydraulic pump")

    # Also load the main hf_scenarios.json if present
    scenarios_path = data_path / "hf_scenarios.json"
    if scenarios_path.exists():
        scenario_rows = _read_jsonl(scenarios_path)
        all_rows.extend(("scenarios", r) for r in scenario_rows)

    # 6. Create HFScenario umbrella nodes for ALL rows
    _load_hf_scenario_nodes(client, graph, all_rows, created_equip, stats)

    # Final equipment count
    stats["equipment"] = len(created_equip)

    # Summary
    total_nodes = (
        stats["equipment"]
        + stats["failure_modes"]
        + stats["sensors"]
        + stats["monitoring_rules"]
        + stats["phm_scenarios"]
        + stats["hf_scenarios"]
    )
    total_edges = (
        stats["monitors_edges"]
        + stats["experienced_edges"]
        + stats["has_rule_edges"]
        + stats["has_phm_edges"]
        + stats["targets_equip_edges"]
    )

    print(f"\n  HF Load Complete:")
    print(f"    Equipment types:     {stats['equipment']}")
    print(f"    Failure modes:       {stats['failure_modes']}")
    print(f"    Sensors:             {stats['sensors']}")
    print(f"    Monitoring rules:    {stats['monitoring_rules']}")
    print(f"    PHM scenarios:       {stats['phm_scenarios']}")
    print(f"    HF scenario nodes:   {stats['hf_scenarios']}")
    print(f"    Total new nodes:     ~{total_nodes}")
    print(f"    Total new edges:     ~{total_edges}")

    return stats


# ---------------------------------------------------------------------------
# CLI convenience — run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/user/projects/Madhulatha-Sandeep/graph_ws/AssetOpsBench/data"
    )
    graph_name = sys.argv[2] if len(sys.argv) > 2 else "industrial"

    print(f"Data dir: {data_dir}")
    print(f"Graph:    {graph_name}")

    client = SamyamaClient.embedded()
    result = load_hf_scenarios(client, data_dir=data_dir, graph=graph_name)
    print(f"\nFinal stats: {json.dumps(result, indent=2)}")
