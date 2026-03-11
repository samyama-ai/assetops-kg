"""WorkOrder, Anomaly, MaintenanceWindow, and SparePart loader.

Generates synthetic data for maintenance-related nodes and edges that
are referenced by the benchmark scenarios (maintenance_optimization,
temporal_pattern) but not created by the other loaders.

Data is designed so that:
- WorkOrders span 2023-2024 for temporal pattern analysis
- Some equipment (Chiller-1, Boiler-1) has decreasing WO intervals
  (accelerating degradation pattern)
- Chillers get more WOs than pumps (realistic distribution)
- Anomalies link to sensors and trigger work orders
- SpareParts link to failure modes and work orders
- MaintenanceWindows cover planned and emergency outages
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from samyama import SamyamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _escape(val: Any) -> str:
    if isinstance(val, str):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


def _props_string(props: dict[str, Any]) -> str:
    parts = [f"{k}: {_escape(v)}" for k, v in props.items()]
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Static data definitions
# ---------------------------------------------------------------------------

# Equipment names from eamlite_loader grouped by class
_EQUIPMENT_BY_CLASS = {
    "chiller": ["Chiller-1", "Chiller-2", "Chiller-3", "Chiller-4"],
    "ahu": ["AHU-1", "AHU-2", "AHU-3", "AHU-4"],
    "pump": ["Pump-CW-1", "Pump-CW-2", "Pump-HW-1", "Pump-HW-2"],
    "motor": ["Motor-CH1", "Motor-AHU1", "Motor-P1", "Motor-BL1"],
    "boiler": ["Boiler-1", "Boiler-2", "Boiler-3", "Boiler-4"],
}

# Failure mode short names from fmsr_loader (first 3 meaningful words)
_FAILURE_MODES_BY_CLASS = {
    "chiller": [
        "Compressor-Overheating-Failed",
        "Heat-Exchangers-Fans",
        "Evaporator-Water-side",
        "Condenser-Water-side",
        "Condenser-Improper-water",
        "Purge-Unit-Excessive",
        "Refrigerant-Operated-Control",
    ],
    "ahu": [
        "Pressure-Regulators-Diaphragm",
        "Steam-Heating-Coils",
        "Belts-sheaves-Wear",
        "Improper-switch-position",
        "Solenoid-Valves-Bound",
    ],
    "pump": [
        "Impeller-erosion-cavitation",
        "Mechanical-seal-leakage",
        "Bearing-failure-due",
        "Coupling-wear-fatigue",
    ],
    "motor": [
        "Stator-winding-insulation",
        "Bearing-overheating-due",
        "Rotor-bar-cracking",
        "Shaft-misalignment-causing",
    ],
    "boiler": [
        "Tube-corrosion-and",
        "Scale-buildup-reducing",
        "Burner-flame-instability",
        "Safety-valve-malfunction",
        "Refractory-lining-degradation",
    ],
}

# ---------------------------------------------------------------------------
# WorkOrder definitions (25 total)
# ---------------------------------------------------------------------------
# Chillers get 8 WOs, Boilers get 6, AHUs get 4, Pumps get 4, Motors get 3
# Chiller-1 has decreasing intervals: Jan, May, Aug, Oct 2023 => degradation
# Boiler-1 has decreasing intervals: Mar, Jul, Sep, Nov 2023

WORK_ORDERS = [
    # -- Chiller-1: accelerating degradation (4 WOs, decreasing intervals) --
    {"wo_id": "WO-2023-0001", "description": "Chiller-1 compressor overheating - emergency repair",
     "status": "closed", "priority": 1, "cost": 12500.0, "duration_hours": 16.0,
     "created_date": "2023-01-15T08:00:00", "closed_date": "2023-01-16T00:00:00",
     "crew_size": 3, "equipment": "Chiller-1", "failure_mode": "Compressor-Overheating-Failed"},

    {"wo_id": "WO-2023-0005", "description": "Chiller-1 condenser fouling - planned cleaning",
     "status": "closed", "priority": 3, "cost": 3200.0, "duration_hours": 8.0,
     "created_date": "2023-05-10T07:00:00", "closed_date": "2023-05-10T15:00:00",
     "crew_size": 2, "equipment": "Chiller-1", "failure_mode": "Condenser-Water-side"},

    {"wo_id": "WO-2023-0010", "description": "Chiller-1 evaporator fouling detected",
     "status": "closed", "priority": 2, "cost": 5800.0, "duration_hours": 12.0,
     "created_date": "2023-08-02T06:00:00", "closed_date": "2023-08-02T18:00:00",
     "crew_size": 2, "equipment": "Chiller-1", "failure_mode": "Evaporator-Water-side"},

    {"wo_id": "WO-2023-0015", "description": "Chiller-1 refrigerant control valve spring failure",
     "status": "closed", "priority": 1, "cost": 8900.0, "duration_hours": 14.0,
     "created_date": "2023-10-05T09:00:00", "closed_date": "2023-10-06T23:00:00",
     "crew_size": 3, "equipment": "Chiller-1", "failure_mode": "Refrigerant-Operated-Control"},

    # -- Chiller-2 (2 WOs) --
    {"wo_id": "WO-2023-0003", "description": "Chiller-2 heat exchanger fan motor replacement",
     "status": "closed", "priority": 2, "cost": 6700.0, "duration_hours": 10.0,
     "created_date": "2023-03-20T08:00:00", "closed_date": "2023-03-20T18:00:00",
     "crew_size": 2, "equipment": "Chiller-2", "failure_mode": "Heat-Exchangers-Fans"},

    {"wo_id": "WO-2024-0002", "description": "Chiller-2 condenser water flow rate adjustment",
     "status": "closed", "priority": 3, "cost": 2100.0, "duration_hours": 4.0,
     "created_date": "2024-02-14T10:00:00", "closed_date": "2024-02-14T14:00:00",
     "crew_size": 1, "equipment": "Chiller-2", "failure_mode": "Condenser-Improper-water"},

    # -- Chiller-3 (1 WO) --
    {"wo_id": "WO-2024-0005", "description": "Chiller-3 purge unit excessive purge investigation",
     "status": "closed", "priority": 4, "cost": 1500.0, "duration_hours": 6.0,
     "created_date": "2024-05-20T07:00:00", "closed_date": "2024-05-20T13:00:00",
     "crew_size": 1, "equipment": "Chiller-3", "failure_mode": "Purge-Unit-Excessive"},

    # -- Chiller-4 (1 WO, open) --
    {"wo_id": "WO-2024-0010", "description": "Chiller-4 compressor overheating - scheduled PM",
     "status": "open", "priority": 2, "cost": 4500.0, "duration_hours": 8.0,
     "created_date": "2024-11-01T08:00:00", "closed_date": None,
     "crew_size": 2, "equipment": "Chiller-4", "failure_mode": "Compressor-Overheating-Failed"},

    # -- Boiler-1: accelerating degradation (4 WOs, decreasing intervals) --
    {"wo_id": "WO-2023-0002", "description": "Boiler-1 tube inspection - minor corrosion found",
     "status": "closed", "priority": 3, "cost": 4200.0, "duration_hours": 12.0,
     "created_date": "2023-03-10T06:00:00", "closed_date": "2023-03-10T18:00:00",
     "crew_size": 2, "equipment": "Boiler-1", "failure_mode": "Tube-corrosion-and"},

    {"wo_id": "WO-2023-0008", "description": "Boiler-1 scale buildup cleaning",
     "status": "closed", "priority": 2, "cost": 3800.0, "duration_hours": 8.0,
     "created_date": "2023-07-05T07:00:00", "closed_date": "2023-07-05T15:00:00",
     "crew_size": 2, "equipment": "Boiler-1", "failure_mode": "Scale-buildup-reducing"},

    {"wo_id": "WO-2023-0012", "description": "Boiler-1 burner instability - adjustment needed",
     "status": "closed", "priority": 1, "cost": 7600.0, "duration_hours": 10.0,
     "created_date": "2023-09-18T08:00:00", "closed_date": "2023-09-18T18:00:00",
     "crew_size": 3, "equipment": "Boiler-1", "failure_mode": "Burner-flame-instability"},

    {"wo_id": "WO-2023-0016", "description": "Boiler-1 safety valve malfunction - emergency",
     "status": "closed", "priority": 1, "cost": 15000.0, "duration_hours": 24.0,
     "created_date": "2023-11-02T05:00:00", "closed_date": "2023-11-03T05:00:00",
     "crew_size": 4, "equipment": "Boiler-1", "failure_mode": "Safety-valve-malfunction"},

    # -- Boiler-2 (2 WOs) --
    {"wo_id": "WO-2023-0006", "description": "Boiler-2 refractory lining inspection",
     "status": "closed", "priority": 3, "cost": 3500.0, "duration_hours": 8.0,
     "created_date": "2023-06-12T07:00:00", "closed_date": "2023-06-12T15:00:00",
     "crew_size": 2, "equipment": "Boiler-2", "failure_mode": "Refractory-lining-degradation"},

    {"wo_id": "WO-2024-0006", "description": "Boiler-2 tube corrosion - wall thinning detected",
     "status": "in_progress", "priority": 2, "cost": 8200.0, "duration_hours": 16.0,
     "created_date": "2024-06-15T08:00:00", "closed_date": None,
     "crew_size": 3, "equipment": "Boiler-2", "failure_mode": "Tube-corrosion-and"},

    # -- AHU-1 (2 WOs) --
    {"wo_id": "WO-2023-0004", "description": "AHU-1 belt and sheaves replacement",
     "status": "closed", "priority": 3, "cost": 1800.0, "duration_hours": 4.0,
     "created_date": "2023-04-18T08:00:00", "closed_date": "2023-04-18T12:00:00",
     "crew_size": 1, "equipment": "AHU-1", "failure_mode": "Belts-sheaves-Wear"},

    {"wo_id": "WO-2024-0003", "description": "AHU-1 pressure regulator diaphragm replacement",
     "status": "closed", "priority": 2, "cost": 4100.0, "duration_hours": 6.0,
     "created_date": "2024-03-22T09:00:00", "closed_date": "2024-03-22T15:00:00",
     "crew_size": 2, "equipment": "AHU-1", "failure_mode": "Pressure-Regulators-Diaphragm"},

    # -- AHU-3 (1 WO) --
    {"wo_id": "WO-2024-0007", "description": "AHU-3 solenoid valve seized - cleaning",
     "status": "closed", "priority": 3, "cost": 2200.0, "duration_hours": 5.0,
     "created_date": "2024-07-10T07:00:00", "closed_date": "2024-07-10T12:00:00",
     "crew_size": 1, "equipment": "AHU-3", "failure_mode": "Solenoid-Valves-Bound"},

    # -- AHU-2 (1 WO, open) --
    {"wo_id": "WO-2024-0011", "description": "AHU-2 steam coil fouling - PM scheduled",
     "status": "open", "priority": 3, "cost": 2800.0, "duration_hours": 6.0,
     "created_date": "2024-11-15T08:00:00", "closed_date": None,
     "crew_size": 2, "equipment": "AHU-2", "failure_mode": "Steam-Heating-Coils"},

    # -- Pump-CW-1 (2 WOs) --
    {"wo_id": "WO-2023-0007", "description": "Pump-CW-1 mechanical seal replacement",
     "status": "closed", "priority": 2, "cost": 5500.0, "duration_hours": 8.0,
     "created_date": "2023-07-25T06:00:00", "closed_date": "2023-07-25T14:00:00",
     "crew_size": 2, "equipment": "Pump-CW-1", "failure_mode": "Mechanical-seal-leakage"},

    {"wo_id": "WO-2024-0004", "description": "Pump-CW-1 impeller cavitation repair",
     "status": "closed", "priority": 1, "cost": 9200.0, "duration_hours": 12.0,
     "created_date": "2024-04-08T07:00:00", "closed_date": "2024-04-08T19:00:00",
     "crew_size": 3, "equipment": "Pump-CW-1", "failure_mode": "Impeller-erosion-cavitation"},

    # -- Pump-HW-1 (1 WO) --
    {"wo_id": "WO-2023-0011", "description": "Pump-HW-1 bearing realignment",
     "status": "closed", "priority": 2, "cost": 3900.0, "duration_hours": 6.0,
     "created_date": "2023-09-05T08:00:00", "closed_date": "2023-09-05T14:00:00",
     "crew_size": 2, "equipment": "Pump-HW-1", "failure_mode": "Bearing-failure-due"},

    # -- Pump-CW-2 (1 WO, open) --
    {"wo_id": "WO-2024-0012", "description": "Pump-CW-2 coupling fatigue inspection",
     "status": "open", "priority": 3, "cost": 2600.0, "duration_hours": 4.0,
     "created_date": "2024-12-01T08:00:00", "closed_date": None,
     "crew_size": 1, "equipment": "Pump-CW-2", "failure_mode": "Coupling-wear-fatigue"},

    # -- Motor-CH1 (1 WO) --
    {"wo_id": "WO-2023-0009", "description": "Motor-CH1 bearing lubrication and overheating fix",
     "status": "closed", "priority": 2, "cost": 4300.0, "duration_hours": 6.0,
     "created_date": "2023-08-14T07:00:00", "closed_date": "2023-08-14T13:00:00",
     "crew_size": 2, "equipment": "Motor-CH1", "failure_mode": "Bearing-overheating-due"},

    # -- Motor-P1 (1 WO) --
    {"wo_id": "WO-2024-0008", "description": "Motor-P1 shaft misalignment correction",
     "status": "closed", "priority": 2, "cost": 3700.0, "duration_hours": 5.0,
     "created_date": "2024-08-20T08:00:00", "closed_date": "2024-08-20T13:00:00",
     "crew_size": 2, "equipment": "Motor-P1", "failure_mode": "Shaft-misalignment-causing"},

    # -- Motor-AHU1 (1 WO, open) --
    {"wo_id": "WO-2024-0013", "description": "Motor-AHU1 stator winding insulation test",
     "status": "open", "priority": 2, "cost": 5100.0, "duration_hours": 8.0,
     "created_date": "2024-12-10T09:00:00", "closed_date": None,
     "crew_size": 2, "equipment": "Motor-AHU1", "failure_mode": "Stator-winding-insulation"},
]
# Total: 25 WorkOrders

# ---------------------------------------------------------------------------
# Anomaly definitions (18 total)
# ---------------------------------------------------------------------------
# Sensor names follow couchdb_loader pattern: {Equipment}-{Suffix}

ANOMALIES = [
    # Chiller anomalies (high frequency - 6)
    {"anomaly_id": "ANM-001", "description": "Chiller-1 condenser water return temperature spike above 105F",
     "severity": "high", "detected_at": "2023-01-14T22:30:00", "resolved": True,
     "anomaly_type": "temperature_spike", "sensor": "Chiller-1-CondWaterRetTemp",
     "triggered_wo": "WO-2023-0001"},

    {"anomaly_id": "ANM-002", "description": "Chiller-1 efficiency degradation below 0.3 kW/t",
     "severity": "medium", "detected_at": "2023-05-08T14:00:00", "resolved": True,
     "anomaly_type": "efficiency_drop", "sensor": "Chiller-1-Efficiency",
     "triggered_wo": "WO-2023-0005"},

    {"anomaly_id": "ANM-003", "description": "Chiller-1 evaporator supply temp rising above setpoint",
     "severity": "high", "detected_at": "2023-07-30T11:15:00", "resolved": True,
     "anomaly_type": "temperature_spike", "sensor": "Chiller-1-SupplyTemp",
     "triggered_wo": "WO-2023-0010"},

    {"anomaly_id": "ANM-004", "description": "Chiller-2 condenser water flow rate below minimum threshold",
     "severity": "medium", "detected_at": "2024-02-12T09:45:00", "resolved": True,
     "anomaly_type": "flow_drop", "sensor": "Chiller-2-CondWaterFlow",
     "triggered_wo": "WO-2024-0002"},

    {"anomaly_id": "ANM-005", "description": "Chiller-4 compressor discharge temperature trending high",
     "severity": "high", "detected_at": "2024-10-28T16:00:00", "resolved": False,
     "anomaly_type": "temperature_spike", "sensor": "Chiller-4-CondWaterRetTemp",
     "triggered_wo": "WO-2024-0010"},

    {"anomaly_id": "ANM-006", "description": "Chiller-3 excessive purge unit cycling detected",
     "severity": "low", "detected_at": "2024-05-18T08:30:00", "resolved": True,
     "anomaly_type": "cycle_anomaly", "sensor": "Chiller-3-CondWaterRetTemp",
     "triggered_wo": "WO-2024-0005"},

    # Boiler anomalies (4)
    {"anomaly_id": "ANM-007", "description": "Boiler-1 exhaust gas temperature spike indicating combustion issue",
     "severity": "high", "detected_at": "2023-09-16T03:20:00", "resolved": True,
     "anomaly_type": "temperature_spike", "sensor": "Boiler-1-ExhaustGasTemp",
     "triggered_wo": "WO-2023-0012"},

    {"anomaly_id": "ANM-008", "description": "Boiler-1 steam pressure oscillation beyond safe range",
     "severity": "critical", "detected_at": "2023-11-01T22:10:00", "resolved": True,
     "anomaly_type": "pressure_drop", "sensor": "Boiler-1-SteamPressure",
     "triggered_wo": "WO-2023-0016"},

    {"anomaly_id": "ANM-009", "description": "Boiler-2 water temperature below normal operating range",
     "severity": "medium", "detected_at": "2024-06-13T10:00:00", "resolved": False,
     "anomaly_type": "temperature_drop", "sensor": "Boiler-2-WaterTemp",
     "triggered_wo": "WO-2024-0006"},

    {"anomaly_id": "ANM-010", "description": "Boiler-2 exhaust gas temperature creeping above baseline",
     "severity": "low", "detected_at": "2023-06-10T15:30:00", "resolved": True,
     "anomaly_type": "temperature_spike", "sensor": "Boiler-2-ExhaustGasTemp",
     "triggered_wo": "WO-2023-0006"},

    # Pump anomalies (3)
    {"anomaly_id": "ANM-011", "description": "Pump-CW-1 vibration level exceeding ISO 10816 threshold",
     "severity": "high", "detected_at": "2023-07-23T04:45:00", "resolved": True,
     "anomaly_type": "vibration_increase", "sensor": "Pump-CW-1-Vibration",
     "triggered_wo": "WO-2023-0007"},

    {"anomaly_id": "ANM-012", "description": "Pump-CW-1 discharge pressure drop indicating cavitation",
     "severity": "high", "detected_at": "2024-04-06T12:20:00", "resolved": True,
     "anomaly_type": "pressure_drop", "sensor": "Pump-CW-1-DischargePressure",
     "triggered_wo": "WO-2024-0004"},

    {"anomaly_id": "ANM-013", "description": "Pump-HW-1 vibration spike due to bearing wear",
     "severity": "medium", "detected_at": "2023-09-03T07:00:00", "resolved": True,
     "anomaly_type": "vibration_increase", "sensor": "Pump-HW-1-Vibration",
     "triggered_wo": "WO-2023-0011"},

    # Motor anomalies (3)
    {"anomaly_id": "ANM-014", "description": "Motor-CH1 bearing temperature exceeding 200F threshold",
     "severity": "high", "detected_at": "2023-08-12T19:00:00", "resolved": True,
     "anomaly_type": "temperature_spike", "sensor": "Motor-CH1-BearingTemp",
     "triggered_wo": "WO-2023-0009"},

    {"anomaly_id": "ANM-015", "description": "Motor-P1 vibration pattern indicating misalignment",
     "severity": "medium", "detected_at": "2024-08-18T14:30:00", "resolved": True,
     "anomaly_type": "vibration_increase", "sensor": "Motor-P1-Vibration",
     "triggered_wo": "WO-2024-0008"},

    {"anomaly_id": "ANM-016", "description": "Motor-AHU1 current draw spike above rated amperage",
     "severity": "high", "detected_at": "2024-12-08T11:00:00", "resolved": False,
     "anomaly_type": "current_spike", "sensor": "Motor-AHU1-Current",
     "triggered_wo": "WO-2024-0013"},

    # AHU anomalies (2)
    {"anomaly_id": "ANM-017", "description": "AHU-1 supply air temperature oscillation outside control band",
     "severity": "medium", "detected_at": "2024-03-20T06:15:00", "resolved": True,
     "anomaly_type": "temperature_spike", "sensor": "AHU-1-SupplyAirTemp",
     "triggered_wo": "WO-2024-0003"},

    {"anomaly_id": "ANM-018", "description": "AHU-3 fan speed drop below commanded setpoint",
     "severity": "medium", "detected_at": "2024-07-08T13:00:00", "resolved": True,
     "anomaly_type": "speed_drop", "sensor": "AHU-3-FanSpeed",
     "triggered_wo": "WO-2024-0007"},
]

# ---------------------------------------------------------------------------
# MaintenanceWindow definitions (10 total)
# ---------------------------------------------------------------------------

MAINTENANCE_WINDOWS = [
    {"window_id": "MW-001", "name": "Q1-2023 Planned Outage",
     "start_date": "2023-01-20T06:00:00", "end_date": "2023-01-22T18:00:00",
     "type": "planned", "max_concurrent": 3, "shift": "day", "crew_size": 6,
     "notes": "Annual chiller inspection and pump maintenance"},

    {"window_id": "MW-002", "name": "Q2-2023 Spring Maintenance",
     "start_date": "2023-04-15T06:00:00", "end_date": "2023-04-17T18:00:00",
     "type": "planned", "max_concurrent": 4, "shift": "day", "crew_size": 8,
     "notes": "Pre-summer HVAC preparation and belt replacements"},

    {"window_id": "MW-003", "name": "Q3-2023 Summer Turnaround",
     "start_date": "2023-07-22T06:00:00", "end_date": "2023-07-25T18:00:00",
     "type": "planned", "max_concurrent": 5, "shift": "extended", "crew_size": 10,
     "notes": "Major summer turnaround for chillers and pumps"},

    {"window_id": "MW-004", "name": "Q3-2023 Emergency Boiler Repair",
     "start_date": "2023-09-17T00:00:00", "end_date": "2023-09-19T00:00:00",
     "type": "emergency", "max_concurrent": 2, "shift": "24hr", "crew_size": 4,
     "notes": "Emergency boiler-1 burner repair"},

    {"window_id": "MW-005", "name": "Q4-2023 Fall Shutdown",
     "start_date": "2023-11-01T06:00:00", "end_date": "2023-11-04T18:00:00",
     "type": "planned", "max_concurrent": 4, "shift": "day", "crew_size": 8,
     "notes": "Pre-winter boiler preparation and safety valve testing"},

    {"window_id": "MW-006", "name": "Q1-2024 Winter Maintenance",
     "start_date": "2024-02-12T06:00:00", "end_date": "2024-02-14T18:00:00",
     "type": "planned", "max_concurrent": 3, "shift": "day", "crew_size": 6,
     "notes": "Boiler tune-up and chiller off-season inspection"},

    {"window_id": "MW-007", "name": "Q2-2024 Spring Overhaul",
     "start_date": "2024-04-06T06:00:00", "end_date": "2024-04-10T18:00:00",
     "type": "planned", "max_concurrent": 5, "shift": "extended", "crew_size": 12,
     "notes": "Major spring overhaul across all systems"},

    {"window_id": "MW-008", "name": "Q3-2024 Mid-Year PM",
     "start_date": "2024-07-08T06:00:00", "end_date": "2024-07-10T18:00:00",
     "type": "planned", "max_concurrent": 3, "shift": "day", "crew_size": 6,
     "notes": "Routine preventive maintenance and AHU cleaning"},

    {"window_id": "MW-009", "name": "Q4-2024 Fall Shutdown",
     "start_date": "2024-11-01T06:00:00", "end_date": "2024-11-05T18:00:00",
     "type": "planned", "max_concurrent": 4, "shift": "day", "crew_size": 8,
     "notes": "Year-end comprehensive maintenance window"},

    {"window_id": "MW-010", "name": "Q4-2024 Emergency Motor Repair",
     "start_date": "2024-12-09T00:00:00", "end_date": "2024-12-11T00:00:00",
     "type": "emergency", "max_concurrent": 2, "shift": "24hr", "crew_size": 4,
     "notes": "Emergency motor insulation repair for Motor-AHU1"},
]

# WorkOrder -> MaintenanceWindow mapping (WO falls within or is assigned to a window)
_WO_TO_WINDOW = {
    "WO-2023-0001": "MW-001",
    "WO-2023-0003": "MW-001",
    "WO-2023-0004": "MW-002",
    "WO-2023-0007": "MW-003",
    "WO-2023-0009": "MW-003",
    "WO-2023-0012": "MW-004",
    "WO-2023-0015": "MW-005",
    "WO-2023-0016": "MW-005",
    "WO-2023-0006": "MW-002",
    "WO-2024-0002": "MW-006",
    "WO-2024-0003": "MW-007",
    "WO-2024-0004": "MW-007",
    "WO-2024-0007": "MW-008",
    "WO-2024-0010": "MW-009",
    "WO-2024-0011": "MW-009",
    "WO-2024-0012": "MW-009",
    "WO-2024-0013": "MW-010",
}

# ---------------------------------------------------------------------------
# SparePart definitions (18 total)
# ---------------------------------------------------------------------------

SPARE_PARTS = [
    # Chiller parts
    {"part_id": "SP-001", "name": "Compressor Bearing Assembly",
     "unit_cost": 2200.0, "lead_time_days": 14, "stock_level": 4, "reorder_point": 2,
     "description": "High-precision bearing for centrifugal compressor"},

    {"part_id": "SP-002", "name": "Compressor Gasket Set",
     "unit_cost": 450.0, "lead_time_days": 14, "stock_level": 2, "reorder_point": 3,
     "description": "Complete gasket set for compressor overhaul"},

    {"part_id": "SP-003", "name": "Condenser Tube Bundle",
     "unit_cost": 8500.0, "lead_time_days": 45, "stock_level": 1, "reorder_point": 1,
     "description": "Replacement copper tube bundle for shell-and-tube condenser"},

    {"part_id": "SP-004", "name": "Evaporator Tube Cleaning Kit",
     "unit_cost": 350.0, "lead_time_days": 7, "stock_level": 6, "reorder_point": 3,
     "description": "Brush and chemical cleaning kit for evaporator tubes"},

    {"part_id": "SP-005", "name": "Refrigerant Control Valve",
     "unit_cost": 1800.0, "lead_time_days": 21, "stock_level": 3, "reorder_point": 2,
     "description": "Electronic expansion valve for refrigerant circuit"},

    # Pump parts
    {"part_id": "SP-006", "name": "Mechanical Seal Assembly",
     "unit_cost": 1200.0, "lead_time_days": 10, "stock_level": 5, "reorder_point": 3,
     "description": "Double mechanical seal for centrifugal pump"},

    {"part_id": "SP-007", "name": "Pump Impeller",
     "unit_cost": 3500.0, "lead_time_days": 30, "stock_level": 2, "reorder_point": 1,
     "description": "Stainless steel impeller for chilled water pump"},

    {"part_id": "SP-008", "name": "Pump Bearing Kit",
     "unit_cost": 650.0, "lead_time_days": 7, "stock_level": 8, "reorder_point": 4,
     "description": "Bearing replacement kit with seals for CW pumps"},

    {"part_id": "SP-009", "name": "Flexible Coupling Insert",
     "unit_cost": 280.0, "lead_time_days": 5, "stock_level": 10, "reorder_point": 5,
     "description": "Elastomeric coupling insert for pump-motor connection"},

    # Motor parts
    {"part_id": "SP-010", "name": "Motor Bearing Set",
     "unit_cost": 550.0, "lead_time_days": 7, "stock_level": 6, "reorder_point": 3,
     "description": "Deep groove ball bearing set for electric motor"},

    {"part_id": "SP-011", "name": "Stator Winding Repair Kit",
     "unit_cost": 4200.0, "lead_time_days": 28, "stock_level": 1, "reorder_point": 1,
     "description": "Complete rewinding materials for 100HP motor stator"},

    # AHU parts
    {"part_id": "SP-012", "name": "V-Belt Set",
     "unit_cost": 120.0, "lead_time_days": 3, "stock_level": 12, "reorder_point": 6,
     "description": "Matched V-belt set for AHU fan drive"},

    {"part_id": "SP-013", "name": "Pressure Regulator Diaphragm",
     "unit_cost": 380.0, "lead_time_days": 10, "stock_level": 4, "reorder_point": 2,
     "description": "Replacement diaphragm for pneumatic pressure regulator"},

    {"part_id": "SP-014", "name": "Solenoid Valve Rebuild Kit",
     "unit_cost": 290.0, "lead_time_days": 7, "stock_level": 5, "reorder_point": 3,
     "description": "Rebuild kit with plunger, spring, and seals"},

    # Boiler parts
    {"part_id": "SP-015", "name": "Boiler Tube Section",
     "unit_cost": 1800.0, "lead_time_days": 21, "stock_level": 3, "reorder_point": 2,
     "description": "Pre-bent fire tube section for boiler replacement"},

    {"part_id": "SP-016", "name": "Safety Relief Valve",
     "unit_cost": 950.0, "lead_time_days": 14, "stock_level": 2, "reorder_point": 2,
     "description": "ASME-rated safety relief valve for steam boiler"},

    {"part_id": "SP-017", "name": "Burner Nozzle Assembly",
     "unit_cost": 720.0, "lead_time_days": 10, "stock_level": 4, "reorder_point": 2,
     "description": "Precision burner nozzle assembly with fuel atomizer"},

    {"part_id": "SP-018", "name": "Refractory Brick Set",
     "unit_cost": 1500.0, "lead_time_days": 14, "stock_level": 2, "reorder_point": 1,
     "description": "High-temperature refractory brick for combustion chamber lining"},
]

# FailureMode -> SparePart mapping (which parts each failure mode requires)
_FAILURE_MODE_PARTS = {
    "Compressor-Overheating-Failed": ["SP-001", "SP-002"],
    "Heat-Exchangers-Fans": ["SP-001"],
    "Evaporator-Water-side": ["SP-004"],
    "Condenser-Water-side": ["SP-003", "SP-004"],
    "Condenser-Improper-water": ["SP-004"],
    "Purge-Unit-Excessive": ["SP-005"],
    "Refrigerant-Operated-Control": ["SP-005"],
    "Impeller-erosion-cavitation": ["SP-007"],
    "Mechanical-seal-leakage": ["SP-006"],
    "Bearing-failure-due": ["SP-008"],
    "Coupling-wear-fatigue": ["SP-009"],
    "Stator-winding-insulation": ["SP-011"],
    "Bearing-overheating-due": ["SP-010"],
    "Rotor-bar-cracking": ["SP-010"],
    "Shaft-misalignment-causing": ["SP-009", "SP-010"],
    "Belts-sheaves-Wear": ["SP-012"],
    "Pressure-Regulators-Diaphragm": ["SP-013"],
    "Solenoid-Valves-Bound": ["SP-014"],
    "Tube-corrosion-and": ["SP-015"],
    "Scale-buildup-reducing": ["SP-004"],
    "Burner-flame-instability": ["SP-017"],
    "Safety-valve-malfunction": ["SP-016"],
    "Refractory-lining-degradation": ["SP-018"],
}

# WorkOrder -> SparePart mapping (which parts each WO uses)
_WO_PARTS = {
    "WO-2023-0001": ["SP-001", "SP-002"],
    "WO-2023-0003": ["SP-001"],
    "WO-2023-0005": ["SP-004"],
    "WO-2023-0007": ["SP-006"],
    "WO-2023-0009": ["SP-010"],
    "WO-2023-0010": ["SP-004"],
    "WO-2023-0012": ["SP-017"],
    "WO-2023-0015": ["SP-005"],
    "WO-2023-0016": ["SP-016"],
    "WO-2023-0006": ["SP-018"],
    "WO-2024-0002": ["SP-004"],
    "WO-2024-0003": ["SP-013"],
    "WO-2024-0004": ["SP-007"],
    "WO-2024-0005": ["SP-005"],
    "WO-2024-0006": ["SP-015"],
    "WO-2024-0007": ["SP-014"],
    "WO-2024-0008": ["SP-009", "SP-010"],
    "WO-2024-0010": ["SP-001", "SP-002"],
    "WO-2024-0011": ["SP-004"],
    "WO-2024-0013": ["SP-011"],
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_workorders(
    client: SamyamaClient, data_dir: str, graph: str = "industrial"
) -> Dict[str, int]:
    """Create WorkOrder, Anomaly, MaintenanceWindow, and SparePart nodes.

    Also creates all connecting edges:
      - (WorkOrder)-[:FOR_EQUIPMENT]->(Equipment)
      - (WorkOrder)-[:ADDRESSES]->(FailureMode)
      - (WorkOrder)-[:USES_PART]->(SparePart)
      - (WorkOrder)-[:FOLLOWS_PLAN]->(MaintenanceWindow)
      - (Sensor)-[:DETECTED_ANOMALY]->(Anomaly)
      - (Anomaly)-[:TRIGGERED]->(WorkOrder)
      - (FailureMode)-[:REQUIRES_PART]->(SparePart)

    Returns a stats dict with counts for each entity and edge type.
    """
    stats: Dict[str, int] = {
        "work_orders": 0,
        "anomalies": 0,
        "maintenance_windows": 0,
        "spare_parts": 0,
        "for_equipment_edges": 0,
        "addresses_edges": 0,
        "uses_part_edges": 0,
        "follows_plan_edges": 0,
        "detected_anomaly_edges": 0,
        "triggered_edges": 0,
        "requires_part_edges": 0,
    }

    # ── 1. Create MaintenanceWindow nodes ────────────────────────
    logger.info("Creating %d MaintenanceWindow nodes...", len(MAINTENANCE_WINDOWS))
    for mw in MAINTENANCE_WINDOWS:
        props = dict(mw)  # copy
        client.query(f"CREATE (mw:MaintenanceWindow {_props_string(props)})", graph)
        stats["maintenance_windows"] += 1

    # ── 2. Create SparePart nodes ────────────────────────────────
    logger.info("Creating %d SparePart nodes...", len(SPARE_PARTS))
    for sp in SPARE_PARTS:
        props = dict(sp)
        client.query(f"CREATE (sp:SparePart {_props_string(props)})", graph)
        stats["spare_parts"] += 1

    # ── 3. Create FailureMode -[:REQUIRES_PART]-> SparePart edges
    logger.info("Creating REQUIRES_PART edges...")
    for fm_name, part_ids in _FAILURE_MODE_PARTS.items():
        escaped_fm = fm_name.replace("\\", "\\\\").replace('"', '\\"')
        for part_id in part_ids:
            cypher = (
                f'MATCH (fm:FailureMode {{name: "{escaped_fm}"}}), '
                f'(sp:SparePart {{part_id: "{part_id}"}}) '
                f"CREATE (fm)-[:REQUIRES_PART]->(sp)"
            )
            try:
                client.query(cypher, graph)
                stats["requires_part_edges"] += 1
            except Exception as e:
                logger.debug("Could not create REQUIRES_PART %s -> %s: %s",
                             fm_name, part_id, e)

    # ── 4. Create WorkOrder nodes and edges ──────────────────────
    logger.info("Creating %d WorkOrder nodes...", len(WORK_ORDERS))
    for wo in WORK_ORDERS:
        # Build properties (exclude internal fields)
        props: dict[str, Any] = {
            "wo_id": wo["wo_id"],
            "description": wo["description"],
            "status": wo["status"],
            "priority": wo["priority"],
            "cost": wo["cost"],
            "duration_hours": wo["duration_hours"],
            "created_date": wo["created_date"],
            "crew_size": wo["crew_size"],
        }
        if wo["closed_date"] is not None:
            props["closed_date"] = wo["closed_date"]

        client.query(f"CREATE (wo:WorkOrder {_props_string(props)})", graph)
        stats["work_orders"] += 1

        wo_id = wo["wo_id"]

        # (WorkOrder)-[:FOR_EQUIPMENT]->(Equipment)
        equip_name = wo["equipment"]
        cypher = (
            f'MATCH (wo:WorkOrder {{wo_id: "{wo_id}"}}), '
            f'(e:Equipment {{name: "{equip_name}"}}) '
            f"CREATE (wo)-[:FOR_EQUIPMENT]->(e)"
        )
        try:
            client.query(cypher, graph)
            stats["for_equipment_edges"] += 1
        except Exception as e:
            logger.debug("Could not create FOR_EQUIPMENT %s -> %s: %s",
                         wo_id, equip_name, e)

        # (WorkOrder)-[:ADDRESSES]->(FailureMode)
        fm_name = wo["failure_mode"]
        escaped_fm = fm_name.replace("\\", "\\\\").replace('"', '\\"')
        cypher = (
            f'MATCH (wo:WorkOrder {{wo_id: "{wo_id}"}}), '
            f'(fm:FailureMode {{name: "{escaped_fm}"}}) '
            f"CREATE (wo)-[:ADDRESSES]->(fm)"
        )
        try:
            client.query(cypher, graph)
            stats["addresses_edges"] += 1
        except Exception as e:
            logger.debug("Could not create ADDRESSES %s -> %s: %s",
                         wo_id, fm_name, e)

        # (WorkOrder)-[:USES_PART]->(SparePart)
        if wo_id in _WO_PARTS:
            for part_id in _WO_PARTS[wo_id]:
                cypher = (
                    f'MATCH (wo:WorkOrder {{wo_id: "{wo_id}"}}), '
                    f'(sp:SparePart {{part_id: "{part_id}"}}) '
                    f"CREATE (wo)-[:USES_PART]->(sp)"
                )
                try:
                    client.query(cypher, graph)
                    stats["uses_part_edges"] += 1
                except Exception as e:
                    logger.debug("Could not create USES_PART %s -> %s: %s",
                                 wo_id, part_id, e)

        # (WorkOrder)-[:FOLLOWS_PLAN]->(MaintenanceWindow)
        if wo_id in _WO_TO_WINDOW:
            window_id = _WO_TO_WINDOW[wo_id]
            cypher = (
                f'MATCH (wo:WorkOrder {{wo_id: "{wo_id}"}}), '
                f'(mw:MaintenanceWindow {{window_id: "{window_id}"}}) '
                f"CREATE (wo)-[:FOLLOWS_PLAN]->(mw)"
            )
            try:
                client.query(cypher, graph)
                stats["follows_plan_edges"] += 1
            except Exception as e:
                logger.debug("Could not create FOLLOWS_PLAN %s -> %s: %s",
                             wo_id, window_id, e)

    # ── 5. Create Anomaly nodes and edges ────────────────────────
    logger.info("Creating %d Anomaly nodes...", len(ANOMALIES))
    for anm in ANOMALIES:
        props: dict[str, Any] = {
            "anomaly_id": anm["anomaly_id"],
            "description": anm["description"],
            "severity": anm["severity"],
            "detected_at": anm["detected_at"],
            "resolved": anm["resolved"],
            "anomaly_type": anm["anomaly_type"],
        }
        client.query(f"CREATE (a:Anomaly {_props_string(props)})", graph)
        stats["anomalies"] += 1

        anomaly_id = anm["anomaly_id"]

        # (Sensor)-[:DETECTED_ANOMALY]->(Anomaly)
        sensor_name = anm["sensor"]
        cypher = (
            f'MATCH (s:Sensor {{name: "{sensor_name}"}}), '
            f'(a:Anomaly {{anomaly_id: "{anomaly_id}"}}) '
            f"CREATE (s)-[:DETECTED_ANOMALY]->(a)"
        )
        try:
            client.query(cypher, graph)
            stats["detected_anomaly_edges"] += 1
        except Exception as e:
            logger.debug("Could not create DETECTED_ANOMALY %s -> %s: %s",
                         sensor_name, anomaly_id, e)

        # (Anomaly)-[:TRIGGERED]->(WorkOrder)
        triggered_wo = anm["triggered_wo"]
        cypher = (
            f'MATCH (a:Anomaly {{anomaly_id: "{anomaly_id}"}}), '
            f'(wo:WorkOrder {{wo_id: "{triggered_wo}"}}) '
            f"CREATE (a)-[:TRIGGERED]->(wo)"
        )
        try:
            client.query(cypher, graph)
            stats["triggered_edges"] += 1
        except Exception as e:
            logger.debug("Could not create TRIGGERED %s -> %s: %s",
                         anomaly_id, triggered_wo, e)

    logger.info(
        "WorkOrder loader complete: %d WOs, %d anomalies, "
        "%d windows, %d spare parts",
        stats["work_orders"], stats["anomalies"],
        stats["maintenance_windows"], stats["spare_parts"],
    )

    return stats
