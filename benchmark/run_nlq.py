"""Run AssetOpsBench via NLQ: LLM generates Cypher, graph executes it.

Compares data layers while both approaches use LLM reasoning:
  - IBM's approach:  Question → GPT-4 → flat document search → LLM reasoning → answer
  - Our NLQ approach: Question → LLM → Cypher generation → graph traversal → answer

Note: IBM used GPT-4. Use --model gpt-4 for a true same-model comparison.
Default is gpt-4o (stronger model, so results are an upper bound on graph's contribution).

Usage:
    python -m benchmark.run_nlq --provider openai
    python -m benchmark.run_nlq --provider anthropic --model claude-sonnet-4-20250514
    python -m benchmark.run_nlq --category iot --output results/nlq_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from samyama import SamyamaClient

from etl.ibm_loader import load_ibm_data

GRAPH_NAME = "ibm_nlq"

DEFAULT_DATA_DIR = os.path.expanduser(
    "~/projects/Madhulatha-Sandeep/AssetOpsBench"
)

# Reuse scenario loading and evaluation from the existing benchmark
from benchmark.run_ibm_scenarios import (
    load_ibm_scenarios,
    evaluate_scenario,
    TSFM_TASKS,
    TSFM_MODELS,
)


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

async def _call_openai(
    messages: list[dict], model: str, max_tokens: int
) -> tuple[str, int]:
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text, tokens


async def _call_anthropic(
    messages: list[dict], model: str, max_tokens: int
) -> tuple[str, int]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system = ""
    user_msgs = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            user_msgs.append(m)

    resp = await client.messages.create(
        model=model, system=system, messages=user_msgs,
        max_tokens=max_tokens, temperature=0.0,
    )
    text = resp.content[0].text if resp.content else ""
    tokens = (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0
    return text, tokens


async def call_llm(
    messages: list[dict],
    provider: str = "openai",
    model: str | None = None,
    max_tokens: int = 2000,
) -> tuple[str, int]:
    if provider == "openai":
        return await _call_openai(messages, model or "gpt-4o", max_tokens)
    elif provider == "anthropic":
        return await _call_anthropic(messages, model or "claude-sonnet-4-20250514", max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Schema summary builder — explicit property catalog (no introspection)
# ---------------------------------------------------------------------------

# Hardcoded property catalog: the introspection-based approach returned
# ['id', 'labels', 'properties'] (node metadata) instead of actual property
# names, causing the LLM to generate fm.properties instead of fm.name.
NODE_PROPERTIES: dict[str, dict[str, str]] = {
    "Site": {
        "name": "e.g. 'MAIN'",
    },
    "Location": {
        "name": "e.g. 'Main Building'",
    },
    "Equipment": {
        "equipment_id": "CWC04xxx ID, e.g. 'CWC04006'",
        "name": "display name, e.g. 'Chiller 6'",
        "asset_type": "e.g. 'Chiller'",
    },
    "Sensor": {
        "name": "e.g. 'Chiller 6 Supply Temperature', 'Chiller 9 Tonnage'",
        "sensor_type": "e.g. 'temperature', 'flow', 'pressure', 'efficiency', 'power', 'vibration'",
        "equipment_id": "parent equipment CWC ID",
    },
    "FailureMode": {
        "name": "e.g. 'Compressor Overheating: Failed due to Normal wear, overheating'",
        "description": "same as name — full failure description",
        "asset_type": "e.g. 'Chiller', 'AHU'",
        "severity": "'critical', 'high', 'medium', or 'low'",
    },
    "WorkOrder": {
        "wo_id": "e.g. 'WO259747'",
        "wo_description": "work order description text",
        "equipment_id": "e.g. 'CWC04006'",
        "equipment_name": "e.g. 'Chiller 6'",
        "collection": "e.g. 'compressor'",
        "primary_code": "e.g. 'MT010'",
        "primary_code_description": "e.g. 'Oil Analysis'",
        "secondary_code": "e.g. 'MT010b'",
        "secondary_code_description": "e.g. 'Routine Oil Analysis'",
        "preventive": "'TRUE' or 'FALSE'",
        "work_priority": "e.g. '5'",
        "actual_finish": "format: 'M/D/YY H:MM' (e.g. '4/6/16 14:00')",
        "duration": "e.g. '3:00'",
        "actual_labor_hours": "e.g. '1:00'",
    },
    "AlertEvent": {
        "alert_key": "composite key",
        "equipment_id": "e.g. 'CWC04006'",
        "equipment_name": "e.g. 'Chiller 6'",
        "rule_id": "alert rule identifier",
        "start_time": "format: 'M/D/YY H:MM' (e.g. '11/24/20 19:00')",
        "end_time": "format: 'M/D/YY H:MM'",
    },
    "AnomalyEvent": {
        "anomaly_key": "e.g. 'anomaly_0'",
        "timestamp": "format: 'M/D/YY H:MM' (e.g. '4/26/20 14:14')",
        "kpi": "EXACT values: 'Cooling Load', 'Delta Setpoint', 'Delta Temperature', 'Flow Efficiency'",
        "asset_name": "e.g. 'Chiller 6' (NOT equipment_id — use name)",
        "anomaly_score": "float score",
    },
    "Event": {
        "event_id": "e.g. 'WO-16170', 'EVT-001'",
        "event_group": "UPPERCASE: 'WORK_ORDER', 'ALERT', or 'ANOMALY'",
        "event_category": "e.g. 'PM', 'maintenance', 'system'",
        "event_type": "e.g. 'MT013', 'repair', 'inspection'",
        "description": "e.g. 'Vibration Analysis', 'Oil Analysis'",
        "equipment_id": "e.g. 'CWC04006'",
        "equipment_name": "e.g. 'Chiller 6'",
        "event_time": "ISO format: 'YYYY-MM-DD HH:MM:SS' (e.g. '2010-06-22 14:12:00')",
    },
}

EDGE_TYPES = [
    ("CONTAINS_LOCATION", "Site", "Location"),
    ("CONTAINS_EQUIPMENT", "Location", "Equipment"),
    ("HAS_SENSOR", "Equipment", "Sensor"),
    ("MONITORS", "Sensor", "FailureMode"),
    ("EXPERIENCED", "Equipment", "FailureMode"),
    ("DEPENDS_ON", "Equipment", "Equipment"),
    ("SHARES_SYSTEM_WITH", "Equipment", "Equipment"),
    ("FOR_EQUIPMENT", "WorkOrder|Event|AlertEvent|AnomalyEvent", "Equipment"),
    ("DETECTED_ANOMALY", "Sensor", "Anomaly"),
    ("TRIGGERED", "Anomaly", "WorkOrder"),
    ("ADDRESSES", "WorkOrder", "FailureMode"),
    ("USES_PART", "WorkOrder", "SparePart"),
    ("SUPPLIED_BY", "SparePart", "Supplier"),
    ("REQUIRES_PART", "Equipment", "SparePart"),
    ("PRODUCED_READING", "Sensor", "SensorReading"),
    ("FOLLOWS_PLAN", "WorkOrder", "MaintenanceWindow"),
]


def build_schema_summary(client: SamyamaClient, graph: str) -> str:
    """Build a detailed schema summary with explicit property names and samples."""
    lines = ["Graph Schema (OpenCypher-compatible):\n"]
    lines.append("Node Labels and Properties:")
    lines.append("IMPORTANT: Access properties directly on the node variable, e.g. n.name, n.equipment_id.")
    lines.append("There is NO .properties bag — n.properties will return NULL.\n")

    for label, props in NODE_PROPERTIES.items():
        res = client.query_readonly(
            f"MATCH (n:{label}) RETURN count(n)", graph
        )
        count = res.records[0][0] if res.records else 0
        if count == 0:
            continue
        lines.append(f"  {label} ({count} nodes):")
        for prop_name, description in props.items():
            lines.append(f"    .{prop_name} — {description}")

    lines.append("\nEdge Types (direction matters — always source→target):")
    for etype, src, tgt in EDGE_TYPES:
        res = client.query_readonly(
            f"MATCH ()-[r:{etype}]->() RETURN count(r)", graph
        )
        count = res.records[0][0] if res.records else 0
        if count > 0:
            lines.append(f"  (:{src})-[:{etype}]->(:{tgt})  [{count} edges]")

    lines.append("\nKey Conventions:")
    lines.append("- Equipment IDs: CWC04xxx (CWC04006='Chiller 6', CWC04009='Chiller 9', CWC04013='Chiller 13')")
    lines.append("- Sensor names: 'Chiller N SensorType' (e.g. 'Chiller 9 Tonnage', 'Chiller 6 Supply Temperature')")
    lines.append("- There are NO vibration sensors in this dataset. All sensors for a chiller monitor all its failure modes.")
    lines.append("- Event.event_time is ISO: 'YYYY-MM-DD HH:MM:SS'. Filter years: ev.event_time STARTS WITH '2019'")
    lines.append("- Event.event_group is UPPERCASE: 'WORK_ORDER', 'ALERT', 'ANOMALY' (not lowercase)")
    lines.append("- WorkOrder.actual_finish is 'M/D/YY H:MM'. Filter years: wo.actual_finish CONTAINS '/19 ' for 2019")
    lines.append("- AlertEvent.start_time is 'M/D/YY H:MM'. Filter: a.start_time CONTAINS '/20 ' for 2020")
    lines.append("- AnomalyEvent.timestamp is 'M/D/YY H:MM'. Filter: a.timestamp CONTAINS '/20 ' for 2020")
    lines.append("- AnomalyEvent.kpi is one of: 'Cooling Load', 'Delta Setpoint', 'Delta Temperature', 'Flow Efficiency'")
    lines.append("- AnomalyEvent uses asset_name (e.g. 'Chiller 6'), NOT equipment_id. Query: a.asset_name = 'Chiller 6'")
    lines.append("- AnomalyEvent does NOT have FOR_EQUIPMENT edges. Filter by a.asset_name property instead.")
    lines.append("- FailureMode names are verbose: 'Compressor Overheating: Failed due to Normal wear, overheating'")
    lines.append("- Use CONTAINS for partial name matching: fm.name CONTAINS 'Overheating'")
    lines.append("- Site has only 'MAIN'. Locations include 'Main Building'.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cypher generation prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a Cypher query expert for an OpenCypher property graph database containing
industrial equipment maintenance data.

{schema}

CRITICAL RULES:
1. Access properties directly: n.name, n.equipment_id — NEVER use n.properties or n.properties.name (those return NULL).
2. Return a single valid Cypher query inside a ```cypher code block.
3. Use only property names listed in the schema above. Double-check before using any property.
4. For sensor-to-failure-mode queries, always traverse: (s:Sensor)-[:MONITORS]->(fm:FailureMode)
5. For equipment sensors, traverse: (e:Equipment)-[:HAS_SENSOR]->(s:Sensor)
6. For work orders on equipment: (wo:WorkOrder)-[:FOR_EQUIPMENT]->(e:Equipment)
7. For events on equipment: (ev:Event)-[:FOR_EQUIPMENT]->(e:Equipment)
8. Edge direction matters — always follow the arrow direction shown in the schema.
9. For time filtering on Event nodes, use STARTS WITH on event_time: ev.event_time STARTS WITH '2019'
10. For time filtering on WorkOrder, use CONTAINS on actual_finish: wo.actual_finish CONTAINS '/19 ' for 2019
11. RETURN named properties (e.g. RETURN e.name, e.equipment_id), NEVER RETURN raw node variables alone.
12. Use CONTAINS for partial string matching on names.
13. If the question is PURELY about ML models, forecasting capabilities, or time-series analysis types
    with NO graph-queryable entities, respond with KNOWLEDGE_QUERY followed by your best answer.
14. If the question asks to download data, export files, or retrieve raw sensor readings,
    generate a Cypher query that returns the relevant data — the system will format it.
15. ALWAYS attempt a Cypher query first if the question mentions specific equipment, sensors, KPIs,
    dates, or events that exist in the graph. Only use KNOWLEDGE_QUERY as a last resort.
16. For forecasting/prediction questions, respond with KNOWLEDGE_QUERY but include:
    the specific TSFM task (tsfm_forecasting, tsfm_forecasting_tune, tsfm_anomaly_detection),
    the model name (e.g., ttm_energy_96_28), context_length, forecast_length, and
    which sensor data would be used as input.
17. For anomaly detection questions, ALWAYS query the graph for AnomalyEvent or Event nodes.
    AnomalyEvent: filter by a.asset_name (NOT FOR_EQUIPMENT edges). Event: use FOR_EQUIPMENT edges.
18. When asked about causal linkages or reasoning about alerts, query the Event/AlertEvent/AnomalyEvent
    nodes and analyze the patterns — do NOT just give generic textbook advice.
"""

# Few-shot examples per scenario type — teaches the LLM correct patterns
FEW_SHOT_IOT = [
    {
        "role": "user",
        "content": "What IoT sites are available?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (s:Site)-[:CONTAINS_LOCATION]->(l:Location)-[:CONTAINS_EQUIPMENT]->(e:Equipment)\nRETURN s.name AS site, l.name AS location, collect(e.name) AS equipment\n```',
    },
    {
        "role": "user",
        "content": "List all the chillers/assets at the site.",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (s:Site)-[:CONTAINS_LOCATION]->(l:Location)-[:CONTAINS_EQUIPMENT]->(e:Equipment)\nRETURN e.name, e.equipment_id, e.asset_type\n```',
    },
    {
        "role": "user",
        "content": "What sensors are available for Chiller 9?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (e:Equipment {name: \'Chiller 9\'})-[:HAS_SENSOR]->(s:Sensor)\nRETURN s.name, s.sensor_type\n```',
    },
    {
        "role": "user",
        "content": "Get the latest supply humidity reading from AHU 1.",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (e:Equipment)-[:HAS_SENSOR]->(s:Sensor)\nWHERE e.name CONTAINS \'AHU\' AND s.name CONTAINS \'Humidity\'\nRETURN e.name, s.name, s.sensor_type\n```\nNote: Raw sensor readings are not stored in the graph. Returning sensor metadata.',
    },
]

FEW_SHOT_FMSR = [
    {
        "role": "user",
        "content": "What are the failure modes for chillers?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (fm:FailureMode)\nWHERE fm.asset_type = \'Chiller\'\nRETURN fm.name, fm.description, fm.severity\n```',
    },
    {
        "role": "user",
        "content": "Which sensors can detect Compressor Overheating?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (s:Sensor)-[:MONITORS]->(fm:FailureMode)\nWHERE fm.name CONTAINS \'Compressor Overheating\'\nRETURN s.name, s.sensor_type, fm.name\n```',
    },
    {
        "role": "user",
        "content": "For Chiller 6, what failure modes can be detected by its temperature sensors?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (e:Equipment {name: \'Chiller 6\'})-[:HAS_SENSOR]->(s:Sensor)-[:MONITORS]->(fm:FailureMode)\nWHERE s.name CONTAINS \'Temperature\'\nRETURN DISTINCT fm.name, fm.description, s.name\n```',
    },
    {
        "role": "user",
        "content": "What failure modes of Chiller 6 can be predicted by vibration sensor data?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (e:Equipment {name: \'Chiller 6\'})-[:HAS_SENSOR]->(s:Sensor)-[:MONITORS]->(fm:FailureMode)\nRETURN DISTINCT fm.name, fm.description, fm.severity, s.name\n```\nNote: There are no dedicated vibration sensors in this dataset. All sensors for a chiller monitor all its failure modes via MONITORS edges.',
    },
    {
        "role": "user",
        "content": "Which sensor should be prioritized for monitoring Compressor Overheating on Chiller 6?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (e:Equipment {name: \'Chiller 6\'})-[:HAS_SENSOR]->(s:Sensor)-[:MONITORS]->(fm:FailureMode)\nWHERE fm.name CONTAINS \'Compressor Overheating\'\nRETURN s.name, s.sensor_type\n```\nReturn ALL sensors. Temperature sensors (Supply Temperature, Return Temperature) are most relevant for thermal failure modes like overheating.',
    },
]

FEW_SHOT_WO = [
    {
        "role": "user",
        "content": "How many work order events were there for CWC04009 in 2019?",
    },
    {
        "role": "assistant",
        "content": "```cypher\nMATCH (ev:Event)-[:FOR_EQUIPMENT]->(e:Equipment {equipment_id: 'CWC04009'})\nWHERE ev.event_group = 'WORK_ORDER' AND ev.event_time STARTS WITH '2019'\nRETURN count(ev)\n```",
    },
    {
        "role": "user",
        "content": "List work orders for Chiller 13 related to oil analysis.",
    },
    {
        "role": "assistant",
        "content": "```cypher\nMATCH (wo:WorkOrder)-[:FOR_EQUIPMENT]->(e:Equipment {name: 'Chiller 13'})\nWHERE wo.wo_description CONTAINS 'Oil Analysis' OR wo.primary_code_description CONTAINS 'Oil Analysis'\nRETURN wo.wo_id, wo.wo_description, wo.primary_code_description, wo.actual_finish\n```",
    },
    {
        "role": "user",
        "content": "How many alert and anomaly events were there for CWC04009 in 2021?",
    },
    {
        "role": "assistant",
        "content": "```cypher\nMATCH (ev:Event)-[:FOR_EQUIPMENT]->(e:Equipment {equipment_id: 'CWC04009'})\nWHERE ev.event_group IN ['ALERT', 'ANOMALY'] AND ev.event_time STARTS WITH '2021'\nRETURN ev.event_group, count(ev)\n```\nNote: event_group values are UPPERCASE: 'WORK_ORDER', 'ALERT', 'ANOMALY'.",
    },
]

FEW_SHOT_MULTI = [
    {
        "role": "user",
        "content": "What sensors for Chiller 6 can detect failure modes, and what are those failure modes?",
    },
    {
        "role": "assistant",
        "content": '```cypher\nMATCH (e:Equipment {name: \'Chiller 6\'})-[:HAS_SENSOR]->(s:Sensor)-[:MONITORS]->(fm:FailureMode)\nRETURN s.name, s.sensor_type, fm.name, fm.severity\n```',
    },
    {
        "role": "user",
        "content": "Show me the anomaly events for Chiller 9 in April 2020.",
    },
    {
        "role": "assistant",
        "content": "```cypher\nMATCH (ev:Event)-[:FOR_EQUIPMENT]->(e:Equipment {name: 'Chiller 9'})\nWHERE ev.event_group = 'ANOMALY' AND ev.event_time STARTS WITH '2020-04'\nRETURN ev.event_id, ev.event_time, ev.description\n```",
    },
    {
        "role": "user",
        "content": "Are there anomalies in Chiller 6 Tonnage in the week of 2020-04-27?",
    },
    {
        "role": "assistant",
        "content": (
            "First query AnomalyEvent nodes (note: filter by asset_name, not FOR_EQUIPMENT edge):\n"
            "```cypher\n"
            "MATCH (a:AnomalyEvent)\n"
            "WHERE a.asset_name = 'Chiller 6' AND a.timestamp CONTAINS '/20 '\n"
            "RETURN a.kpi, a.timestamp, a.anomaly_score\n"
            "```\n"
            "Also check unified Event nodes:\n"
            "```cypher\n"
            "MATCH (ev:Event)-[:FOR_EQUIPMENT]->(e:Equipment {name: 'Chiller 6'})\n"
            "WHERE ev.event_group = 'ANOMALY' AND ev.event_time STARTS WITH '2020-04'\n"
            "RETURN ev.event_id, ev.event_time, ev.description\n"
            "```"
        ),
    },
    {
        "role": "user",
        "content": "Can you forecast Chiller 6 Tonnage for the next week starting from 2020-04-27?",
    },
    {
        "role": "assistant",
        "content": (
            "KNOWLEDGE_QUERY For time-series forecasting, the tsfm_forecasting task with model "
            "ttm_energy_96_28 (context_length: 96, forecast_length: 28) would be used. "
            "The Chiller 6 Tonnage sensor data from the week before 2020-04-27 would be used as "
            "input context. The model would forecast the next 28 time steps (approximately 1 week). "
            "The TTM (Tiny Time Mixer) model is suitable for energy-related KPIs like Tonnage. "
            "Required data: Chiller 6 Tonnage readings, timestamp column, target column. "
            "The forecast_length of 28 provides ~1 week of hourly predictions."
        ),
    },
]

TYPE_FEW_SHOTS: dict[str, list[dict]] = {
    "iot": FEW_SHOT_IOT,
    "fmsr": FEW_SHOT_FMSR,
    "wo": FEW_SHOT_WO,
    "multi": FEW_SHOT_MULTI,
}


def build_messages(question: str, schema: str, scenario_type: str) -> list[dict]:
    system = SYSTEM_PROMPT_TEMPLATE.format(schema=schema)

    if scenario_type == "tsfm":
        system += (
            "\n\nThis question is about time-series forecasting models (TSFM). "
            "These are domain knowledge questions about model capabilities, "
            "supported analysis types, and model checkpoints. "
            "Respond with KNOWLEDGE_QUERY followed by your answer.\n"
            f"Available TSFM tasks: {json.dumps(TSFM_TASKS)}\n"
            f"Available TSFM models: {json.dumps(TSFM_MODELS)}\n"
        )

    if scenario_type == "multi":
        system += (
            "\n\nThis is a multi-agent question that may combine IoT, FMSR, WO, "
            "and TSFM concerns. If it asks about forecasting or predictions, "
            "respond with KNOWLEDGE_QUERY including which TSFM model/task would "
            "be used, plus any graph data you can provide via Cypher.\n"
            f"Available TSFM tasks: {json.dumps(TSFM_TASKS)}\n"
            f"Available TSFM models: {json.dumps(TSFM_MODELS)}\n"
        )

    messages: list[dict] = [{"role": "system", "content": system}]

    # Add few-shot examples for the scenario type
    few_shots = TYPE_FEW_SHOTS.get(scenario_type, [])
    messages.extend(few_shots)

    messages.append({"role": "user", "content": question})
    return messages


# ---------------------------------------------------------------------------
# Cypher extraction and execution
# ---------------------------------------------------------------------------

def extract_cypher(llm_response: str) -> tuple[list[str], str | None]:
    """Extract Cypher queries from LLM response.

    Returns (cypher_queries, knowledge_text).
    If KNOWLEDGE_QUERY, returns ([], knowledge_text).
    """
    if "KNOWLEDGE_QUERY" in llm_response:
        text = re.sub(r"KNOWLEDGE_QUERY\s*", "", llm_response).strip()
        return [], text

    queries = re.findall(r"```(?:cypher)?\s*\n?(.*?)```", llm_response, re.DOTALL)
    queries = [q.strip() for q in queries if q.strip()]

    if not queries:
        # Try the whole response as a query if it looks like Cypher
        stripped = llm_response.strip()
        if stripped.upper().startswith(("MATCH", "RETURN", "WITH", "CALL", "CREATE")):
            queries = [stripped]
        else:
            return [], llm_response

    return queries, None


def execute_queries(
    client: SamyamaClient, graph: str, queries: list[str]
) -> list[dict]:
    """Execute Cypher queries and return results."""
    results = []
    for cypher in queries:
        start = time.perf_counter()
        try:
            res = client.query_readonly(cypher, graph)
            elapsed = (time.perf_counter() - start) * 1000
            records = []
            for row in res.records:
                records.append([str(v) for v in row])
            results.append({
                "cypher": cypher,
                "success": True,
                "columns": res.columns if hasattr(res, "columns") else [],
                "records": records,
                "record_count": len(records),
                "execution_ms": elapsed,
            })
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            results.append({
                "cypher": cypher,
                "success": False,
                "error": str(e),
                "execution_ms": elapsed,
            })
    return results


def format_query_results(query_results: list[dict]) -> str:
    """Format Cypher query results into text for answer synthesis."""
    parts = []
    for i, qr in enumerate(query_results):
        if not qr["success"]:
            parts.append(f"Query {i+1} failed: {qr.get('error', 'unknown')}")
            continue

        records = qr["records"]
        if not records:
            parts.append(f"Query {i+1}: No results returned.")
            continue

        parts.append(f"Query {i+1} ({qr['record_count']} rows):")
        for row in records[:50]:  # Cap at 50 rows
            parts.append("  " + " | ".join(row))
        if len(records) > 50:
            parts.append(f"  ... ({len(records) - 50} more rows)")

    return "\n".join(parts) if parts else "No query results."


SYNTH_SYSTEM = (
    "You are an industrial maintenance assistant. Given Cypher query results "
    "from a knowledge graph, synthesize a clear natural language answer. "
    "Include specific values, counts, and equipment names from the results. "
    "Be thorough — mention all items returned."
)


async def synthesize_answer(
    question: str,
    query_results_text: str,
    provider: str,
    model: str | None,
) -> tuple[str, int]:
    """Use LLM to synthesize a natural language answer from query results."""
    messages = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Graph query results:\n{query_results_text}\n\n"
                "Synthesize a complete answer."
            ),
        },
    ]
    return await call_llm(messages, provider=provider, model=model, max_tokens=1500)


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # retry up to 2 times on Cypher execution failure


async def run_nlq_scenario(
    client: SamyamaClient,
    graph: str,
    schema: str,
    scenario: dict[str, Any],
    provider: str,
    model: str | None,
) -> dict[str, Any]:
    """Run a single scenario via NLQ pipeline with retry on Cypher errors."""
    sid = scenario["id"]
    stype = scenario.get("scenario_type", "unknown")
    question = scenario.get("text", "")

    t0 = time.perf_counter()
    tokens_total = 0
    all_queries: list[str] = []
    retries_used = 0

    try:
        # Step 1: LLM generates Cypher
        messages = build_messages(question, schema, stype)
        gen_response, gen_tokens = await call_llm(
            messages, provider=provider, model=model
        )
        tokens_total += gen_tokens

        # Step 2: Extract and execute Cypher (with retry on failure)
        queries, knowledge_text = extract_cypher(gen_response)

        if knowledge_text is not None:
            response = knowledge_text
            cypher_results = []
            synth_tokens = 0
        elif not queries:
            response = gen_response
            cypher_results = []
            synth_tokens = 0
        else:
            cypher_results = execute_queries(client, graph, queries)
            all_queries.extend(queries)

            # Retry logic: if all queries failed or returned 0 rows with errors
            attempt = 0
            while attempt < MAX_RETRIES:
                has_failure = any(not r["success"] for r in cypher_results)
                all_empty = all(
                    r.get("record_count", 0) == 0 for r in cypher_results
                    if r["success"]
                )
                if not has_failure and not all_empty:
                    break  # at least one query succeeded with data

                # Build retry prompt with error feedback
                error_msgs = []
                for r in cypher_results:
                    if not r["success"]:
                        error_msgs.append(
                            f"Query: {r['cypher']}\nError: {r['error']}"
                        )
                    elif r.get("record_count", 0) == 0:
                        error_msgs.append(
                            f"Query: {r['cypher']}\nReturned 0 rows (empty result)"
                        )

                if not error_msgs:
                    break

                retry_content = (
                    f"The previous Cypher query had issues:\n\n"
                    + "\n\n".join(error_msgs)
                    + "\n\nPlease fix the query. Remember:\n"
                    "- Use n.name, n.equipment_id etc. — NEVER n.properties\n"
                    "- Check edge directions match the schema\n"
                    "- Use CONTAINS for partial name matching\n"
                    "- Property names are case-sensitive\n"
                    "Generate a corrected query."
                )
                messages.append({"role": "assistant", "content": gen_response})
                messages.append({"role": "user", "content": retry_content})

                gen_response, retry_tokens = await call_llm(
                    messages, provider=provider, model=model
                )
                tokens_total += retry_tokens

                queries, knowledge_text = extract_cypher(gen_response)
                if knowledge_text is not None:
                    response = knowledge_text
                    cypher_results = []
                    synth_tokens = 0
                    retries_used = attempt + 1
                    break
                if queries:
                    cypher_results = execute_queries(client, graph, queries)
                    all_queries.extend(queries)

                attempt += 1
                retries_used = attempt

            # Step 3: Synthesize answer from results
            if knowledge_text is not None:
                pass  # already set above
            else:
                results_text = format_query_results(cypher_results)
                response, synth_tokens = await synthesize_answer(
                    question, results_text, provider, model
                )
                tokens_total += synth_tokens

    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "id": sid, "type": stype,
            "category": scenario.get("category", ""),
            "question": question,
            "passed": False, "score": 0.0,
            "latency_ms": elapsed, "response": "",
            "rationale": "", "error": f"{type(e).__name__}: {e}",
            "nlq_details": {"provider": provider, "model": model or "default"},
        }

    elapsed = (time.perf_counter() - t0) * 1000

    # Step 4: Evaluate against characteristic_form
    passed, score, rationale = evaluate_scenario(scenario, response)

    return {
        "id": sid,
        "type": stype,
        "category": scenario.get("category", ""),
        "question": question,
        "passed": passed,
        "score": score,
        "latency_ms": elapsed,
        "response": response,
        "rationale": rationale,
        "error": None,
        "nlq_details": {
            "provider": provider,
            "model": model or ("gpt-4o" if provider == "openai" else "claude-sonnet-4-20250514"),
            "knowledge_only": knowledge_text is not None,
            "cypher_generated": all_queries if all_queries else [],
            "cypher_results": [
                {
                    "success": r["success"],
                    "record_count": r.get("record_count", 0),
                    "execution_ms": r.get("execution_ms", 0),
                }
                for r in cypher_results
            ] if cypher_results else [],
            "tokens_gen": gen_tokens,
            "tokens_total": tokens_total,
            "retries": retries_used,
        },
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_results_table(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    header = f"{'ID':<8} {'Type':<8} {'Pass':>5} {'Score':>6} {'Latency':>9} {'Tokens':>7} {'Cypher':>7}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for r in sorted(results, key=lambda x: x["id"]):
        status = "PASS" if r["passed"] else "FAIL"
        if r.get("error"):
            status = "ERR"
        latency = f"{r['latency_ms']:.0f}ms"
        tokens = r.get("nlq_details", {}).get("tokens_total", 0)
        n_cypher = len(r.get("nlq_details", {}).get("cypher_generated", []))
        knowledge = "K" if r.get("nlq_details", {}).get("knowledge_only") else ""
        lines.append(
            f"{r['id']:<8} {r['type']:<8} {status:>5} {r['score']:>6.3f} "
            f"{latency:>9} {tokens:>7} {n_cypher:>6}{knowledge}"
        )

    lines.append(sep)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / total if total else 0.0
    avg_tokens = sum(
        r.get("nlq_details", {}).get("tokens_total", 0) for r in results
    ) / total if total else 0
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0
    lines.append(
        f"Summary: {passed}/{total} passed ({passed/total*100:.0f}%), "
        f"avg score {avg_score:.3f}, avg latency {avg_latency:.0f}ms, "
        f"avg tokens {avg_tokens:.0f}"
    )

    # Per-type breakdown
    type_groups: dict[str, list] = defaultdict(list)
    for r in results:
        type_groups[r["type"]].append(r)

    lines.append("")
    lines.append("Per-type breakdown:")
    type_display = {"iot": "IoT", "fmsr": "FMSR", "wo": "WO", "tsfm": "TSFM", "multi": "Multi"}
    for tkey in ["iot", "fmsr", "wo", "tsfm", "multi"]:
        if tkey in type_groups:
            group = type_groups[tkey]
            gpassed = sum(1 for r in group if r["passed"])
            gavg = sum(r["score"] for r in group) / len(group) if group else 0.0
            gtokens = sum(
                r.get("nlq_details", {}).get("tokens_total", 0) for r in group
            ) / len(group) if group else 0
            display = type_display.get(tkey, tkey)
            lines.append(
                f"  {display:<8} {gpassed}/{len(group)} passed, "
                f"avg={gavg:.3f}, avg tokens={gtokens:.0f}"
            )

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_nlq_benchmark(
    data_dir: str,
    provider: str,
    model: str | None,
    category: str | None,
    output_path: str | None,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Run IBM 139 scenarios via NLQ pipeline."""
    # Load data
    client = SamyamaClient.embedded()
    print(f"Loading IBM data from {data_dir}...")
    stats = load_ibm_data(client, data_dir, GRAPH_NAME)
    print(f"Loaded: {stats}")

    # Build schema summary
    schema = build_schema_summary(client, GRAPH_NAME)
    print(f"\nSchema summary ({len(schema)} chars):")
    print(schema[:500] + "..." if len(schema) > 500 else schema)

    # Load scenarios
    scenarios = load_ibm_scenarios(data_dir, category)
    print(f"\nNLQ benchmark: {len(scenarios)} scenarios, provider={provider}, "
          f"model={model or 'default'}")

    # Run with concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def run_with_limit(s: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        async with semaphore:
            result = await run_nlq_scenario(
                client, GRAPH_NAME, schema, s, provider, model
            )
            completed += 1
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"  [{completed}/{len(scenarios)}] id={result['id']:<6} "
                f"type={result['type']:<6} {status} "
                f"score={result['score']:.3f} "
                f"latency={result['latency_ms']:.0f}ms "
                f"tokens={result.get('nlq_details', {}).get('tokens_total', 0)}"
            )
            return result

    results = await asyncio.gather(*[run_with_limit(s) for s in scenarios])
    results = list(results)

    print()
    print(f"=== NLQ RESULTS ({provider}, LLM → Cypher → graph) ===")
    print(format_results_table(results))

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults written to {output_path}")

    return results


def detect_provider() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise EnvironmentError(
        "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run the NLQ benchmark."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AssetOps-KG NLQ Benchmark (LLM → Cypher → graph)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="Path to AssetOpsBench repo root",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["openai", "anthropic"],
        help="LLM provider (auto-detected from API keys if not set)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: gpt-4o or claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["iot", "fmsr", "wo", "tsfm", "multi"],
        help="Run only one scenario type",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Max concurrent LLM calls (default: 3)",
    )
    args = parser.parse_args()

    provider = args.provider or detect_provider()
    asyncio.run(run_nlq_benchmark(
        data_dir=args.data_dir,
        provider=provider,
        model=args.model,
        category=args.category,
        output_path=args.output,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
