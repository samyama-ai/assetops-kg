"""FMSR (Failure Mode and Sensor Reasoning) loader.

Loads failure mode definitions from the AssetOpsBench failure_modes.yaml
and creates FailureMode nodes with MONITORS edges to matching equipment.

Falls back to a hardcoded copy of the AssetOpsBench failure modes when
the YAML file is not found at the expected paths.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

from samyama import SamyamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded failure modes (from AssetOpsBench src/servers/fmsr/failure_modes.yaml)
# ---------------------------------------------------------------------------

_BUILTIN_FAILURE_MODES: dict[str, list[str]] = {
    "chiller": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
        "Evaporator Water side fouling",
        "Condenser Water side fouling",
        "Condenser Improper water side flow rate",
        "Purge Unit Excessive purge",
        "Refrigerant Operated Control Valve Failed spring",
    ],
    "ahu": [
        "Pressure Regulators Diaphragm failure",
        "Steam Heating Coils Air side fouling",
        "Belts or sheaves Wear",
        "Improper switch position",
        "Solenoid Valves Bound due to hardened grease",
    ],
    "pump": [
        "Impeller erosion or cavitation damage",
        "Mechanical seal leakage",
        "Bearing failure due to misalignment",
        "Coupling wear or fatigue fracture",
    ],
    "motor": [
        "Stator winding insulation breakdown",
        "Bearing overheating due to lubrication failure",
        "Rotor bar cracking",
        "Shaft misalignment causing vibration",
    ],
    "boiler": [
        "Tube corrosion and wall thinning",
        "Scale buildup reducing heat transfer",
        "Burner flame instability",
        "Safety valve malfunction",
        "Refractory lining degradation",
    ],
}

# Map from failure mode text to structured attributes
_SEVERITY_MAP = {
    "Compressor Overheating": ("high", "overheating", "thermal"),
    "Heat Exchangers": ("medium", "wear", "mechanical"),
    "Evaporator Water side fouling": ("medium", "fouling", "chemical"),
    "Condenser Water side fouling": ("medium", "fouling", "chemical"),
    "Condenser Improper water side flow rate": ("medium", "flow_degradation", "hydraulic"),
    "Purge Unit Excessive purge": ("low", "leakage", "refrigerant"),
    "Refrigerant Operated Control Valve": ("medium", "spring_failure", "mechanical"),
    "Pressure Regulators Diaphragm": ("medium", "diaphragm_rupture", "mechanical"),
    "Steam Heating Coils Air side fouling": ("medium", "fouling", "thermal"),
    "Belts or sheaves Wear": ("low", "wear", "mechanical"),
    "Improper switch position": ("low", "misconfiguration", "electrical"),
    "Solenoid Valves Bound": ("medium", "seizure", "mechanical"),
    "Impeller erosion": ("high", "erosion", "hydraulic"),
    "Mechanical seal leakage": ("high", "leakage", "mechanical"),
    "Bearing failure": ("high", "bearing_failure", "mechanical"),
    "Coupling wear": ("medium", "fatigue", "mechanical"),
    "Stator winding": ("high", "insulation_breakdown", "electrical"),
    "Bearing overheating": ("high", "overheating", "thermal"),
    "Rotor bar cracking": ("high", "cracking", "mechanical"),
    "Shaft misalignment": ("medium", "misalignment", "mechanical"),
    "Tube corrosion": ("high", "corrosion", "chemical"),
    "Scale buildup": ("medium", "fouling", "chemical"),
    "Burner flame instability": ("high", "instability", "combustion"),
    "Safety valve malfunction": ("critical", "valve_failure", "mechanical"),
    "Refractory lining": ("medium", "degradation", "thermal"),
}


def _classify_failure_mode(text: str) -> tuple[str, str, str]:
    """Return (severity, iso14224_mechanism, category) for a failure mode text."""
    for prefix, attrs in _SEVERITY_MAP.items():
        if text.startswith(prefix):
            return attrs
    return ("medium", "unknown", "general")


def _escape(val: Any) -> str:
    if isinstance(val, str):
        # Escape inner double quotes and backslashes
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(val)


def _props_string(props: dict[str, Any]) -> str:
    parts = [f"{k}: {_escape(v)}" for k, v in props.items()]
    return "{" + ", ".join(parts) + "}"


def _load_yaml_failure_modes(data_dir: str) -> dict[str, list[str]] | None:
    """Try to find and load failure_modes.yaml from various locations."""
    search_paths = [
        Path(data_dir) / "failure_modes.yaml",
        Path(data_dir) / "src" / "servers" / "fmsr" / "failure_modes.yaml",
        Path(data_dir) / "servers" / "fmsr" / "failure_modes.yaml",
        Path(data_dir) / "fmsr" / "failure_modes.yaml",
    ]

    for path in search_paths:
        if path.exists():
            logger.info("Found failure_modes.yaml at %s", path)
            try:
                import yaml
                with open(path) as f:
                    return yaml.safe_load(f)
            except ImportError:
                logger.warning("PyYAML not installed; reading YAML manually")
                return _parse_simple_yaml(path)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", path, e)

    return None


def _parse_simple_yaml(path: Path) -> dict[str, list[str]]:
    """Minimal YAML parser for the simple failure_modes.yaml format.

    The file has the structure:
        equipment_type:
          - "failure mode description"
    """
    result: dict[str, list[str]] = {}
    current_key: str | None = None

    with open(path) as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.startswith("#"):
                continue

            # Top-level key (no leading whitespace, ends with colon)
            if not line[0].isspace() and stripped.endswith(":"):
                current_key = stripped[:-1].strip()
                result[current_key] = []
            # List item under current key
            elif current_key is not None and stripped.lstrip().startswith("- "):
                item = stripped.lstrip()[2:].strip()
                # Remove surrounding quotes if present
                if item.startswith('"') and item.endswith('"'):
                    item = item[1:-1]
                result[current_key].append(item)

    return result


def _make_short_name(description: str) -> str:
    """Create a short identifier from a failure mode description."""
    # Take first few meaningful words
    words = re.split(r"[\s:,]+", description)
    meaningful = [w for w in words[:4] if len(w) > 2]
    return "-".join(meaningful[:3]).replace('"', "")


def load_fmsr(
    client: SamyamaClient, data_dir: str, graph: str = "industrial"
) -> Dict[str, int]:
    """Load failure modes into the knowledge graph.

    Attempts to read failure_modes.yaml from data_dir.  Falls back to
    the built-in copy of AssetOpsBench failure modes.

    Returns dict with counts: {failure_modes, monitors_edges}.
    """
    fm_data = _load_yaml_failure_modes(data_dir)
    if fm_data is None:
        logger.info("No YAML found; using built-in failure modes")
        fm_data = _BUILTIN_FAILURE_MODES

    failure_modes_count = 0
    monitors_edges_count = 0

    for equip_class, modes in fm_data.items():
        equip_class_lower = equip_class.strip().lower()

        for description in modes:
            short_name = _make_short_name(description)
            severity, mechanism, category = _classify_failure_mode(description)

            props = {
                "name": short_name,
                "description": description,
                "severity": severity,
                "iso14224_mechanism": mechanism,
                "category": category,
            }

            # Create FailureMode node
            client.query(
                f"CREATE (fm:FailureMode {_props_string(props)})", graph
            )
            failure_modes_count += 1

            # Create MONITORS edge to each equipment of matching class
            # FailureMode -[:MONITORS]-> Equipment
            escaped_name = short_name.replace("\\", "\\\\").replace('"', '\\"')
            match_cypher = (
                f'MATCH (fm:FailureMode {{name: "{escaped_name}"}}), '
                f'(e:Equipment {{iso14224_class: "{equip_class_lower}"}}) '
                f"CREATE (fm)-[:MONITORS]->(e)"
            )
            try:
                # Count how many equipment nodes match
                count_result = client.query_readonly(
                    f'MATCH (e:Equipment {{iso14224_class: "{equip_class_lower}"}}) '
                    f"RETURN count(e)",
                    graph,
                )
                equip_count = 0
                if count_result.records:
                    val = count_result.records[0][0]
                    equip_count = int(val) if val is not None else 0

                client.query(match_cypher, graph)
                monitors_edges_count += equip_count
            except Exception as e:
                logger.debug(
                    "Could not create MONITORS edge for %s -> %s: %s",
                    short_name, equip_class_lower, e,
                )

    logger.info(
        "FMSR load complete: %d failure modes, %d MONITORS edges",
        failure_modes_count, monitors_edges_count,
    )

    return {
        "failure_modes": failure_modes_count,
        "monitors_edges": monitors_edges_count,
    }
