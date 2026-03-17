// Industrial Asset Operations Knowledge Graph Schema
// Based on ISO 14224 (Equipment Reliability) + ISA-95 (Enterprise-Control Integration)
// Extended with AssetOpsBench HuggingFace scenarios (v2 — 467 scenarios)
//
// 14 Node Labels, 21 Edge Types

// ============================================================
// Node Labels
// ============================================================

// ISA-95 Level 3: Site
// CREATE (:Site {name, location, timezone, commissioning_date})

// ISA-95 Level 2: Functional Location
// CREATE (:Location {name, building, floor, zone, isa95_level})

// ISO 14224: Equipment
// CREATE (:Equipment {name, iso14224_class, isa95_level, criticality_score,
//                     mtbf_hours, install_date, manufacturer, model, serial_number,
//                     status})

// Sensor / IoT endpoint
// CREATE (:Sensor {name, type, unit, min_threshold, max_threshold, sampling_rate_hz})

// Failure mode with semantic embedding
// CREATE (:FailureMode {name, description, severity, iso14224_mechanism,
//                       category, embedding})  // embedding: 384-dim float[]

// Maintenance work order
// CREATE (:WorkOrder {wo_number, description, priority, status,
//                     created_date, due_date, completed_date,
//                     estimated_hours, actual_hours, cost})

// Spare part
// CREATE (:SparePart {part_number, name, description, unit_cost,
//                     stock_quantity, reorder_point, lead_time_days})

// Supplier
// CREATE (:Supplier {name, contact_email, phone, country, lead_time_days, rating})

// Time-series sensor reading
// CREATE (:SensorReading {timestamp, value, unit, quality_flag})

// Detected anomaly with semantic embedding
// CREATE (:Anomaly {description, severity, detected_at, resolved_at,
//                   root_cause, embedding})  // embedding: 384-dim float[]

// Maintenance window
// CREATE (:MaintenanceWindow {name, start_date, end_date, shift, crew_size, notes})

// ── New node labels added for AssetOpsBench HF expansion ───────

// Monitoring rule (from rule_logic config — AHU, CRAC, Pump, Boiler, HXU, Cooling Tower)
// CREATE (:MonitoringRule {rule_id, description, entity, category,
//                          deterministic, group})

// PHM scenario (Prognostics & Health Management — RUL, fault classification, etc.)
// CREATE (:PHMScenario {scenario_id, task_type, entity, description,
//                        deterministic, group})

// HF benchmark scenario (umbrella node linking all 467 HuggingFace rows)
// CREATE (:HFScenario {scenario_id, type, entity, category, group,
//                       deterministic, text})

// ── Extended Equipment iso14224_class values ────────────────────
// Original classes: chiller, ahu, pump, motor, boiler
// New FMSR classes: electric_motor, steam_turbine, aero_gas_turbine,
//   industrial_gas_turbine, power_transformer, compressor,
//   reciprocating_engine, fan, electric_generator
// New rule_logic classes: crac, hxu, cooling_tower
// New PHM classes: turbofan_engine, induction_motor, bearing, rotor,
//   gearbox, turbine
// New multiagent classes: hydraulic_pump


// ============================================================
// Edge Types (21)
// ============================================================

// Hierarchy
// (:Site)-[:CONTAINS_LOCATION]->(:Location)
// (:Location)-[:CONTAINS_EQUIPMENT]->(:Equipment)

// Monitoring
// (:Equipment)-[:HAS_SENSOR]->(:Sensor)
// (:FailureMode)-[:MONITORS]->(:Equipment)

// Failure & maintenance
// (:Equipment)-[:EXPERIENCED]->(:FailureMode)
// (:Sensor)-[:PRODUCED_READING]->(:SensorReading)
// (:Sensor)-[:DETECTED_ANOMALY]->(:Anomaly)

// Dependencies
// (:Equipment)-[:DEPENDS_ON]->(:Equipment)
// (:Equipment)-[:SHARES_SYSTEM_WITH]->(:Equipment)

// Work orders
// (:WorkOrder)-[:FOR_EQUIPMENT]->(:Equipment)
// (:WorkOrder)-[:ADDRESSES]->(:FailureMode)
// (:WorkOrder)-[:USES_PART]->(:SparePart)
// (:WorkOrder)-[:FOLLOWS_PLAN]->(:MaintenanceWindow)

// Supply chain
// (:Equipment)-[:REQUIRES_PART]->(:SparePart)
// (:SparePart)-[:SUPPLIED_BY]->(:Supplier)

// Anomaly trigger
// (:Anomaly)-[:TRIGGERED]->(:WorkOrder)

// ── New edge types for HF expansion ─────────────────────────────

// Equipment → MonitoringRule (rule applies to equipment type)
// (:Equipment)-[:HAS_RULE]->(:MonitoringRule)

// Equipment → PHMScenario (PHM task targets equipment)
// (:Equipment)-[:HAS_PHM_SCENARIO]->(:PHMScenario)

// Equipment → FailureMode (equipment experiences a known failure mode — from FMSR)
// NOTE: EXPERIENCED edge already defined above; reused for FMSR-sourced mappings

// Sensor → Equipment (sensor monitors equipment — from FMSR)
// NOTE: HAS_SENSOR already defined (Equipment → Sensor); MONITORS reused for FMSR

// HFScenario → Equipment (scenario references equipment)
// (:HFScenario)-[:TARGETS_EQUIPMENT]->(:Equipment)

// HFScenario → FailureMode (scenario involves failure mode)
// (:HFScenario)-[:INVOLVES_FAILURE_MODE]->(:FailureMode)

// HFScenario → MonitoringRule (scenario tests monitoring rule)
// (:HFScenario)-[:TESTS_RULE]->(:MonitoringRule)
