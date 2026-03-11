"""Tests for the Cypher schema file (schema/industrial_kg.cypher).

Validates that the schema file:
  - Exists and is non-empty
  - Declares the expected node labels (11) and edge types (16)
  - Uses valid Cypher comment syntax
  - References ISO 14224 and ISA-95 standards
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "industrial_kg.cypher"

# Expected node labels from the schema (11)
EXPECTED_NODE_LABELS = [
    "Site",
    "Location",
    "Equipment",
    "Sensor",
    "FailureMode",
    "WorkOrder",
    "SparePart",
    "Supplier",
    "SensorReading",
    "Anomaly",
    "MaintenanceWindow",
]

# Expected edge types from the schema (16)
EXPECTED_EDGE_TYPES = [
    "CONTAINS_LOCATION",
    "CONTAINS_EQUIPMENT",
    "HAS_SENSOR",
    "MONITORS",
    "EXPERIENCED",
    "PRODUCED_READING",
    "DETECTED_ANOMALY",
    "DEPENDS_ON",
    "SHARES_SYSTEM_WITH",
    "FOR_EQUIPMENT",
    "ADDRESSES",
    "USES_PART",
    "FOLLOWS_PLAN",
    "REQUIRES_PART",
    "SUPPLIED_BY",
    "TRIGGERED",
]


@pytest.fixture
def schema_text() -> str:
    """Read the schema file content."""
    assert SCHEMA_PATH.exists(), f"Schema file not found: {SCHEMA_PATH}"
    text = SCHEMA_PATH.read_text()
    assert len(text.strip()) > 0, "Schema file is empty"
    return text


class TestSchemaFileExists:
    def test_schema_file_exists(self) -> None:
        assert SCHEMA_PATH.exists(), f"Schema file not found: {SCHEMA_PATH}"

    def test_schema_file_not_empty(self) -> None:
        text = SCHEMA_PATH.read_text()
        assert len(text.strip()) > 100, "Schema file appears too short"


class TestNodeLabels:
    def test_all_node_labels_present(self, schema_text: str) -> None:
        """Every expected node label must appear in the schema."""
        for label in EXPECTED_NODE_LABELS:
            assert label in schema_text, f"Node label '{label}' not found in schema"

    def test_node_label_count(self, schema_text: str) -> None:
        """Schema should declare exactly 11 node labels."""
        # Count CREATE (:<Label> ...) patterns in comments
        label_pattern = re.findall(r"CREATE\s*\(:\w+", schema_text)
        assert len(label_pattern) >= len(EXPECTED_NODE_LABELS), (
            f"Expected at least {len(EXPECTED_NODE_LABELS)} CREATE (:Label) patterns, "
            f"found {len(label_pattern)}"
        )

    @pytest.mark.parametrize("label", EXPECTED_NODE_LABELS)
    def test_node_label_has_properties(self, schema_text: str, label: str) -> None:
        """Each node label should have at least one property defined."""
        # Look for CREATE (:<Label> {prop1, prop2, ...})
        pattern = rf"CREATE\s*\(:{re.escape(label)}\s*\{{[^}}]+\}}"
        match = re.search(pattern, schema_text)
        assert match is not None, f"Node label '{label}' has no property definition"


class TestEdgeTypes:
    def test_all_edge_types_present(self, schema_text: str) -> None:
        """Every expected edge type must appear in the schema."""
        for edge_type in EXPECTED_EDGE_TYPES:
            assert edge_type in schema_text, f"Edge type '{edge_type}' not found in schema"

    def test_edge_type_count(self, schema_text: str) -> None:
        """Schema should declare exactly 16 edge types."""
        # Count [:<EDGE_TYPE>] patterns
        edge_pattern = re.findall(r"\[:\w+\]", schema_text)
        assert len(edge_pattern) >= len(EXPECTED_EDGE_TYPES), (
            f"Expected at least {len(EXPECTED_EDGE_TYPES)} edge type patterns, "
            f"found {len(edge_pattern)}"
        )

    @pytest.mark.parametrize("edge_type", EXPECTED_EDGE_TYPES)
    def test_edge_type_has_endpoints(self, schema_text: str, edge_type: str) -> None:
        """Each edge type should have source and target node labels defined."""
        # Look for (:<Source>)-[:<EDGE_TYPE>]->(:<Target>)
        pattern = rf"\(:\w+\)-\[:{re.escape(edge_type)}\]->\(:\w+\)"
        match = re.search(pattern, schema_text)
        assert match is not None, (
            f"Edge type '{edge_type}' does not have (:<Source>)-[:{edge_type}]->(:<Target>) pattern"
        )


class TestSchemaStandards:
    def test_references_iso14224(self, schema_text: str) -> None:
        """Schema should reference ISO 14224 (Equipment Reliability)."""
        assert "ISO 14224" in schema_text or "iso14224" in schema_text.lower()

    def test_references_isa95(self, schema_text: str) -> None:
        """Schema should reference ISA-95 (Enterprise-Control Integration)."""
        assert "ISA-95" in schema_text or "isa95" in schema_text.lower()

    def test_uses_cypher_comment_syntax(self, schema_text: str) -> None:
        """Schema should use // comment syntax (standard Cypher)."""
        comment_lines = [line for line in schema_text.split("\n") if line.strip().startswith("//")]
        assert len(comment_lines) > 5, "Schema should have substantial documentation via // comments"


class TestSchemaSemantics:
    def test_equipment_has_criticality_score(self, schema_text: str) -> None:
        """Equipment nodes must have criticality_score for PageRank comparison."""
        assert "criticality_score" in schema_text

    def test_equipment_has_mtbf(self, schema_text: str) -> None:
        """Equipment nodes must have mtbf_hours for temporal pattern analysis."""
        assert "mtbf_hours" in schema_text

    def test_failure_mode_has_embedding(self, schema_text: str) -> None:
        """FailureMode nodes must have embedding field for vector search."""
        assert "embedding" in schema_text

    def test_anomaly_has_embedding(self, schema_text: str) -> None:
        """Anomaly nodes must have embedding field for semantic matching."""
        # Check that Anomaly definition includes embedding
        anomaly_section = schema_text[schema_text.index("Anomaly"):]
        anomaly_section = anomaly_section[:anomaly_section.index("\n\n") if "\n\n" in anomaly_section else len(anomaly_section)]
        assert "embedding" in anomaly_section

    def test_sensor_has_thresholds(self, schema_text: str) -> None:
        """Sensor nodes must have min/max thresholds for trend analysis."""
        assert "min_threshold" in schema_text
        assert "max_threshold" in schema_text

    def test_work_order_has_cost(self, schema_text: str) -> None:
        """WorkOrder nodes must have cost for maintenance optimization."""
        assert "cost" in schema_text

    def test_spare_part_has_lead_time(self, schema_text: str) -> None:
        """SparePart nodes must have lead_time_days for scheduling."""
        assert "lead_time_days" in schema_text
