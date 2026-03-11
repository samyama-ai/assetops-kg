// Industrial Asset Operations Knowledge Graph Schema
// Based on ISO 14224 (Equipment Reliability) + ISA-95 (Enterprise-Control Integration)
//
// 11 Node Labels, 16 Edge Types

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


// ============================================================
// Edge Types (16)
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
