"""Tests for scenario JSON files.

Validates that all scenario JSON files:
  - Parse as valid JSON
  - Contain arrays of scenario objects
  - Each scenario has all required fields with correct types
  - IDs are unique across all files
  - Categories match their file names
  - Difficulty values are from the allowed set
  - Expected tools reference known tool names
  - Total scenario count matches target (40)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "scenarios"

CATEGORY_FILES = {
    "multi_hop_dependency": "multi_hop_dependency.json",
    "cross_asset_correlation": "cross_asset_correlation.json",
    "failure_similarity": "failure_similarity.json",
    "criticality_analysis": "criticality_analysis.json",
    "maintenance_optimization": "maintenance_optimization.json",
    "root_cause_analysis": "root_cause_analysis.json",
    "temporal_pattern": "temporal_pattern.json",
}

EXPECTED_COUNTS = {
    "multi_hop_dependency": 8,
    "cross_asset_correlation": 6,
    "failure_similarity": 6,
    "criticality_analysis": 5,
    "maintenance_optimization": 5,
    "root_cause_analysis": 5,
    "temporal_pattern": 5,
}

TOTAL_SCENARIOS = 40

REQUIRED_FIELDS = {
    "id": str,
    "category": str,
    "description": str,
    "expected_tools": list,
    "expected_output_contains": list,
    "difficulty": str,
    "requires_graph": bool,
}

ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}

KNOWN_TOOLS = {
    "impact_analysis",
    "anomaly_correlation",
    "vector_search",
    "criticality_ranking",
    "maintenance_scheduler",
    "root_cause_trace",
    "sensor_trend",
    "cypher_query",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _load_all_scenarios() -> list[tuple[str, dict]]:
    """Load all scenarios as (filename, scenario_dict) pairs."""
    all_scenarios = []
    for category, fname in CATEGORY_FILES.items():
        fpath = SCENARIOS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for scenario in data:
                all_scenarios.append((fname, scenario))
    return all_scenarios


ALL_SCENARIOS = _load_all_scenarios()


@pytest.fixture
def all_scenarios() -> list[tuple[str, dict]]:
    return ALL_SCENARIOS


# ---------------------------------------------------------------------------
# File-level tests
# ---------------------------------------------------------------------------

class TestScenarioFilesExist:
    @pytest.mark.parametrize("category,filename", list(CATEGORY_FILES.items()))
    def test_file_exists(self, category: str, filename: str) -> None:
        fpath = SCENARIOS_DIR / filename
        assert fpath.exists(), f"Scenario file missing: {fpath}"

    @pytest.mark.parametrize("category,filename", list(CATEGORY_FILES.items()))
    def test_file_is_valid_json(self, category: str, filename: str) -> None:
        fpath = SCENARIOS_DIR / filename
        with open(fpath) as f:
            data = json.load(f)
        assert isinstance(data, list), f"{filename} should contain a JSON array"
        assert len(data) > 0, f"{filename} should not be empty"

    @pytest.mark.parametrize("category,filename", list(CATEGORY_FILES.items()))
    def test_file_scenario_count(self, category: str, filename: str) -> None:
        fpath = SCENARIOS_DIR / filename
        with open(fpath) as f:
            data = json.load(f)
        expected = EXPECTED_COUNTS[category]
        assert len(data) == expected, (
            f"{filename} has {len(data)} scenarios, expected {expected}"
        )


class TestTotalCount:
    def test_total_scenario_count(self, all_scenarios: list[tuple[str, dict]]) -> None:
        assert len(all_scenarios) == TOTAL_SCENARIOS, (
            f"Total scenarios: {len(all_scenarios)}, expected {TOTAL_SCENARIOS}"
        )


# ---------------------------------------------------------------------------
# Scenario-level tests
# ---------------------------------------------------------------------------

class TestScenarioFields:
    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_has_required_fields(self, filename: str, scenario: dict) -> None:
        for field, expected_type in REQUIRED_FIELDS.items():
            assert field in scenario, (
                f"Scenario {scenario.get('id', '?')} in {filename} missing field '{field}'"
            )
            assert isinstance(scenario[field], expected_type), (
                f"Scenario {scenario['id']} field '{field}' should be {expected_type.__name__}, "
                f"got {type(scenario[field]).__name__}"
            )

    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_id_format(self, filename: str, scenario: dict) -> None:
        """IDs should follow the pattern graph_<abbrev>_NNN."""
        sid = scenario["id"]
        assert sid.startswith("graph_"), f"ID '{sid}' should start with 'graph_'"
        parts = sid.split("_")
        assert len(parts) >= 3, f"ID '{sid}' should have at least 3 underscore-separated parts"
        assert parts[-1].isdigit(), f"ID '{sid}' should end with a numeric suffix"

    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_difficulty_valid(self, filename: str, scenario: dict) -> None:
        assert scenario["difficulty"] in ALLOWED_DIFFICULTIES, (
            f"Scenario {scenario['id']} has invalid difficulty '{scenario['difficulty']}'. "
            f"Allowed: {ALLOWED_DIFFICULTIES}"
        )

    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_category_matches_file(self, filename: str, scenario: dict) -> None:
        expected_category = filename.replace(".json", "")
        assert scenario["category"] == expected_category, (
            f"Scenario {scenario['id']} has category '{scenario['category']}' "
            f"but is in file '{filename}' (expected '{expected_category}')"
        )

    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_description_nonempty(self, filename: str, scenario: dict) -> None:
        assert len(scenario["description"].strip()) > 20, (
            f"Scenario {scenario['id']} description is too short"
        )

    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_expected_tools_are_known(self, filename: str, scenario: dict) -> None:
        for tool in scenario["expected_tools"]:
            assert tool in KNOWN_TOOLS, (
                f"Scenario {scenario['id']} references unknown tool '{tool}'. "
                f"Known tools: {sorted(KNOWN_TOOLS)}"
            )

    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_expected_output_contains_nonempty(self, filename: str, scenario: dict) -> None:
        assert len(scenario["expected_output_contains"]) > 0, (
            f"Scenario {scenario['id']} has empty expected_output_contains"
        )

    @pytest.mark.parametrize(
        "filename,scenario",
        ALL_SCENARIOS,
        ids=[f"{s[1].get('id', i)}" for i, s in enumerate(ALL_SCENARIOS)],
    )
    def test_requires_graph_is_true(self, filename: str, scenario: dict) -> None:
        """All scenarios in this benchmark should require graph capabilities."""
        assert scenario["requires_graph"] is True, (
            f"Scenario {scenario['id']} has requires_graph=False — "
            "all AssetOps-KG scenarios should require graph"
        )


class TestUniqueIds:
    def test_no_duplicate_ids(self, all_scenarios: list[tuple[str, dict]]) -> None:
        ids = [s["id"] for _, s in all_scenarios]
        duplicates = [sid for sid in ids if ids.count(sid) > 1]
        assert len(duplicates) == 0, f"Duplicate scenario IDs: {set(duplicates)}"

    def test_ids_are_sortable(self, all_scenarios: list[tuple[str, dict]]) -> None:
        """IDs within each category should sort correctly by numeric suffix."""
        for category in CATEGORY_FILES:
            cat_ids = sorted(
                [s["id"] for _, s in all_scenarios if s["category"] == category]
            )
            # Check numeric suffixes are sequential
            suffixes = [int(sid.split("_")[-1]) for sid in cat_ids]
            assert suffixes == sorted(suffixes), (
                f"Category '{category}' IDs have non-sequential suffixes: {suffixes}"
            )


class TestCategoryCoverage:
    def test_all_categories_have_scenarios(self, all_scenarios: list[tuple[str, dict]]) -> None:
        categories_found = {s["category"] for _, s in all_scenarios}
        expected_categories = set(CATEGORY_FILES.keys())
        missing = expected_categories - categories_found
        assert len(missing) == 0, f"Categories with no scenarios: {missing}"

    def test_difficulty_distribution(self, all_scenarios: list[tuple[str, dict]]) -> None:
        """Ensure a reasonable distribution of difficulties."""
        difficulties = [s["difficulty"] for _, s in all_scenarios]
        medium_count = difficulties.count("medium")
        hard_count = difficulties.count("hard")
        # Should have both medium and hard scenarios
        assert medium_count >= 5, f"Too few medium scenarios: {medium_count}"
        assert hard_count >= 5, f"Too few hard scenarios: {hard_count}"

    def test_tool_coverage(self, all_scenarios: list[tuple[str, dict]]) -> None:
        """All known tools should be referenced by at least one scenario."""
        all_tools_used: set[str] = set()
        for _, s in all_scenarios:
            all_tools_used.update(s["expected_tools"])

        # These core tools should be exercised
        core_tools = {
            "impact_analysis",
            "vector_search",
            "criticality_ranking",
            "cypher_query",
            "anomaly_correlation",
            "root_cause_trace",
            "maintenance_scheduler",
            "sensor_trend",
        }
        missing = core_tools - all_tools_used
        assert len(missing) == 0, f"Tools not exercised by any scenario: {missing}"
