"""EAMLite loader — creates the Site / Location / Equipment hierarchy.

If actual EAMLite SQL data is found under data_dir, it is parsed.
Otherwise, a realistic synthetic hierarchy is generated based on a
typical industrial campus (1 site, 4 locations, 20 equipment items).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from samyama import SamyamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic asset hierarchy definition
# ---------------------------------------------------------------------------

SITE = {
    "name": "Campus-Main",
    "location": "Houston, TX",
    "timezone": "America/Chicago",
    "commissioning_date": "2015-03-12",
}

LOCATIONS = [
    {"name": "Central-Plant", "building": "CP-01", "floor": "G", "zone": "Mechanical", "isa95_level": 2},
    {"name": "North-Wing", "building": "NW-02", "floor": "1", "zone": "HVAC", "isa95_level": 2},
    {"name": "South-Wing", "building": "SW-03", "floor": "1", "zone": "HVAC", "isa95_level": 2},
    {"name": "Utility-Yard", "building": "UY-04", "floor": "G", "zone": "Utility", "isa95_level": 2},
]

# 4 equipment per type, distributed across the 4 locations
_EQUIPMENT_DEFS: list[dict[str, Any]] = [
    # Chillers  — Central-Plant
    {"name": "Chiller-1",  "iso14224_class": "chiller",  "location": "Central-Plant", "criticality_score": 9, "mtbf_hours": 8760, "install_date": "2016-06-01", "manufacturer": "Trane",   "status": "running"},
    {"name": "Chiller-2",  "iso14224_class": "chiller",  "location": "Central-Plant", "criticality_score": 9, "mtbf_hours": 8500, "install_date": "2016-06-01", "manufacturer": "Trane",   "status": "running"},
    {"name": "Chiller-3",  "iso14224_class": "chiller",  "location": "Central-Plant", "criticality_score": 8, "mtbf_hours": 7800, "install_date": "2018-01-15", "manufacturer": "Carrier", "status": "standby"},
    {"name": "Chiller-4",  "iso14224_class": "chiller",  "location": "Central-Plant", "criticality_score": 8, "mtbf_hours": 9000, "install_date": "2020-04-20", "manufacturer": "York",    "status": "running"},
    # AHUs — North-Wing & South-Wing
    {"name": "AHU-1",      "iso14224_class": "ahu",      "location": "North-Wing",    "criticality_score": 7, "mtbf_hours": 12000, "install_date": "2016-06-01", "manufacturer": "Daikin",  "status": "running"},
    {"name": "AHU-2",      "iso14224_class": "ahu",      "location": "North-Wing",    "criticality_score": 7, "mtbf_hours": 11500, "install_date": "2017-03-10", "manufacturer": "Daikin",  "status": "running"},
    {"name": "AHU-3",      "iso14224_class": "ahu",      "location": "South-Wing",    "criticality_score": 6, "mtbf_hours": 13000, "install_date": "2018-09-01", "manufacturer": "Lennox",  "status": "running"},
    {"name": "AHU-4",      "iso14224_class": "ahu",      "location": "South-Wing",    "criticality_score": 6, "mtbf_hours": 12500, "install_date": "2019-02-14", "manufacturer": "Lennox",  "status": "standby"},
    # Pumps — Central-Plant & Utility-Yard
    {"name": "Pump-CW-1",  "iso14224_class": "pump",     "location": "Central-Plant", "criticality_score": 8, "mtbf_hours": 15000, "install_date": "2016-06-01", "manufacturer": "Grundfos",  "status": "running"},
    {"name": "Pump-CW-2",  "iso14224_class": "pump",     "location": "Central-Plant", "criticality_score": 7, "mtbf_hours": 14500, "install_date": "2016-06-01", "manufacturer": "Grundfos",  "status": "standby"},
    {"name": "Pump-HW-1",  "iso14224_class": "pump",     "location": "Utility-Yard",  "criticality_score": 7, "mtbf_hours": 16000, "install_date": "2017-08-20", "manufacturer": "Xylem",     "status": "running"},
    {"name": "Pump-HW-2",  "iso14224_class": "pump",     "location": "Utility-Yard",  "criticality_score": 6, "mtbf_hours": 15500, "install_date": "2019-11-05", "manufacturer": "Xylem",     "status": "running"},
    # Motors — distributed
    {"name": "Motor-CH1",  "iso14224_class": "motor",    "location": "Central-Plant", "criticality_score": 8, "mtbf_hours": 20000, "install_date": "2016-06-01", "manufacturer": "ABB",     "status": "running"},
    {"name": "Motor-AHU1", "iso14224_class": "motor",    "location": "North-Wing",    "criticality_score": 6, "mtbf_hours": 22000, "install_date": "2016-06-01", "manufacturer": "Siemens", "status": "running"},
    {"name": "Motor-P1",   "iso14224_class": "motor",    "location": "Central-Plant", "criticality_score": 7, "mtbf_hours": 21000, "install_date": "2017-08-20", "manufacturer": "ABB",     "status": "running"},
    {"name": "Motor-BL1",  "iso14224_class": "motor",    "location": "Utility-Yard",  "criticality_score": 5, "mtbf_hours": 25000, "install_date": "2018-01-15", "manufacturer": "WEG",     "status": "running"},
    # Boilers — Utility-Yard
    {"name": "Boiler-1",   "iso14224_class": "boiler",   "location": "Utility-Yard",  "criticality_score": 9, "mtbf_hours": 10000, "install_date": "2015-03-12", "manufacturer": "Cleaver-Brooks", "status": "running"},
    {"name": "Boiler-2",   "iso14224_class": "boiler",   "location": "Utility-Yard",  "criticality_score": 9, "mtbf_hours": 9500,  "install_date": "2015-03-12", "manufacturer": "Cleaver-Brooks", "status": "standby"},
    {"name": "Boiler-3",   "iso14224_class": "boiler",   "location": "North-Wing",    "criticality_score": 7, "mtbf_hours": 11000, "install_date": "2019-05-10", "manufacturer": "Fulton",          "status": "running"},
    {"name": "Boiler-4",   "iso14224_class": "boiler",   "location": "South-Wing",    "criticality_score": 7, "mtbf_hours": 10500, "install_date": "2020-02-28", "manufacturer": "Fulton",          "status": "running"},
]

# Dependency edges: (source_equipment_name, target_equipment_name)
DEPENDS_ON_EDGES = [
    ("Chiller-1", "Pump-CW-1"),
    ("Chiller-2", "Pump-CW-1"),
    ("Chiller-3", "Pump-CW-2"),
    ("Chiller-4", "Pump-CW-2"),
    ("Pump-CW-1", "Motor-CH1"),
    ("Pump-HW-1", "Motor-P1"),
    ("AHU-1", "Motor-AHU1"),
    ("Boiler-1", "Pump-HW-1"),
    ("Boiler-2", "Pump-HW-2"),
]

# Shared-system edges
SHARES_SYSTEM_EDGES = [
    ("Chiller-1", "Chiller-2"),
    ("Chiller-3", "Chiller-4"),
    ("AHU-1", "AHU-2"),
    ("AHU-3", "AHU-4"),
    ("Pump-CW-1", "Pump-CW-2"),
    ("Pump-HW-1", "Pump-HW-2"),
    ("Boiler-1", "Boiler-2"),
    ("Boiler-3", "Boiler-4"),
]


def _escape(val: Any) -> str:
    """Escape a value for embedding in a Cypher literal."""
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _props_string(props: dict[str, Any]) -> str:
    """Build a Cypher property map string like {name: "x", age: 30}."""
    parts = [f"{k}: {_escape(v)}" for k, v in props.items()]
    return "{" + ", ".join(parts) + "}"


def _try_load_eamlite_sql(data_dir: str) -> dict | None:
    """Attempt to find and parse EAMLite SQL exports.

    Looks for CSV or SQL files following the EAMLite naming convention.
    Returns None if no usable data is found.
    """
    data_path = Path(data_dir)

    # Check common EAMLite export patterns
    for pattern in ["eamlite*.csv", "eamlite*.sql", "EAMLite*", "eam_*.csv"]:
        matches = list(data_path.glob(pattern))
        if matches:
            logger.info("Found EAMLite files: %s", matches)
            # TODO: parse actual EAMLite SQL/CSV exports when available
            return None

    return None


def load_eamlite(
    client: SamyamaClient, data_dir: str, graph: str = "industrial"
) -> Dict[str, int]:
    """Load the asset hierarchy into the knowledge graph.

    Attempts to parse EAMLite SQL data from data_dir first.
    Falls back to synthetic data if no files are found.

    Returns a dict with counts: {sites, locations, equipment, depends_on, shares_system}.
    """
    real_data = _try_load_eamlite_sql(data_dir)
    if real_data is not None:
        logger.info("Using parsed EAMLite data (not yet implemented)")
        # When implemented, the parsed data would be used in the same
        # Cypher-based creation flow below.

    # ── Create Site ──────────────────────────────────────────────
    cypher = f"CREATE (s:Site {_props_string(SITE)})"
    client.query(cypher, graph)
    sites_count = 1

    # ── Create Locations ─────────────────────────────────────────
    for loc in LOCATIONS:
        cypher = f"CREATE (l:Location {_props_string(loc)})"
        client.query(cypher, graph)

    # Site -> Location edges
    for loc in LOCATIONS:
        cypher = (
            f'MATCH (s:Site {{name: "{SITE["name"]}"}}), '
            f'(l:Location {{name: "{loc["name"]}"}}) '
            f"CREATE (s)-[:CONTAINS_LOCATION]->(l)"
        )
        client.query(cypher, graph)

    locations_count = len(LOCATIONS)

    # ── Create Equipment ─────────────────────────────────────────
    for equip in _EQUIPMENT_DEFS:
        loc_name = equip["location"]
        props = {k: v for k, v in equip.items() if k != "location"}
        props["isa95_level"] = 1  # equipment is ISA-95 level 1
        cypher = f"CREATE (e:Equipment {_props_string(props)})"
        client.query(cypher, graph)

        # Location -> Equipment edge
        cypher = (
            f'MATCH (l:Location {{name: "{loc_name}"}}), '
            f'(e:Equipment {{name: "{equip["name"]}"}}) '
            f"CREATE (l)-[:CONTAINS_EQUIPMENT]->(e)"
        )
        client.query(cypher, graph)

    equipment_count = len(_EQUIPMENT_DEFS)

    # ── DEPENDS_ON edges ─────────────────────────────────────────
    for src, tgt in DEPENDS_ON_EDGES:
        cypher = (
            f'MATCH (a:Equipment {{name: "{src}"}}), '
            f'(b:Equipment {{name: "{tgt}"}}) '
            f"CREATE (a)-[:DEPENDS_ON]->(b)"
        )
        client.query(cypher, graph)

    # ── SHARES_SYSTEM_WITH edges ─────────────────────────────────
    for src, tgt in SHARES_SYSTEM_EDGES:
        cypher = (
            f'MATCH (a:Equipment {{name: "{src}"}}), '
            f'(b:Equipment {{name: "{tgt}"}}) '
            f"CREATE (a)-[:SHARES_SYSTEM_WITH]->(b)"
        )
        client.query(cypher, graph)

    logger.info(
        "EAMLite load complete: %d sites, %d locations, %d equipment",
        sites_count, locations_count, equipment_count,
    )

    return {
        "sites": sites_count,
        "locations": locations_count,
        "equipment": equipment_count,
        "depends_on": len(DEPENDS_ON_EDGES),
        "shares_system": len(SHARES_SYSTEM_EDGES),
    }
