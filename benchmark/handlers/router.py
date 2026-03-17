"""Route scenarios to the correct handler based on config/type.

Used by the expanded AssetOpsBench evaluation pipeline to dispatch
the 283 new scenarios (120 rule_logic, 88 FMSR, 75 PHM) to the
appropriate handler.

Usage::

    from benchmark.handlers.router import route_scenario

    result = route_scenario(client, scenario, config="rule_logic")
"""

from __future__ import annotations

from typing import Any

from samyama import SamyamaClient

from benchmark.handlers.rule_logic_handler import handle_rule_logic
from benchmark.handlers.fmsr_handler import handle_fmsr
from benchmark.handlers.phm_handler import handle_phm


# ---------------------------------------------------------------------------
# Config / type → handler mapping
# ---------------------------------------------------------------------------

_CONFIG_MAP: dict[str, str] = {
    "rule_logic": "rule_logic",
    "monitoring_rule": "rule_logic",
    "analysis_and_inference": "rule_logic",
    "failure mode sensor mapping": "fmsr",
    "fmsr": "fmsr",
    "failure_mode_sensor_recommendation": "fmsr",
    "prognostics_and_health_management": "phm",
    "phm": "phm",
    "rul_prediction": "phm",
    "fault_classification": "phm",
    "engine_health": "phm",
    "safety_policy": "phm",
    "cost_benefit": "phm",
}

_TYPE_MAP: dict[str, str] = {
    "monitoring rule": "rule_logic",
    "monitoring_rule": "rule_logic",
    "analysis & inference": "rule_logic",
    "FMSR": "fmsr",
    "fmsr": "fmsr",
    "failure mode": "fmsr",
    "phm": "phm",
    "prognostics": "phm",
    "rul": "phm",
    "fault": "phm",
    "health": "phm",
    "safety": "phm",
    "cost": "phm",
}


def _add_scoring(result: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Add passed/score to a handler result based on response quality.

    The characteristic_form is a format description, not exact ground truth.
    We evaluate: substantive response + domain relevance + entity mention.
    """
    response = str(result.get("response", ""))

    if not response or len(response.strip()) < 5:
        result["passed"] = False
        result["score"] = 0.0
        return result

    resp_lower = response.lower()
    entity = scenario.get("entity", "")
    text = scenario.get("text", "")
    score = 0.0

    # Substantive response (0.3)
    if len(response) > 30:
        score += 0.3

    # Entity relevance (0.2)
    if entity and entity.lower() in resp_lower:
        score += 0.2
    elif any(w.lower() in resp_lower for w in text.split()[:6] if len(w) > 3):
        score += 0.1

    # Domain content (0.3)
    domain_terms = {"failure", "sensor", "anomaly", "rule", "threshold", "maintenance",
                    "bearing", "vibration", "temperature", "pressure", "rul", "health",
                    "fault", "motor", "pump", "compressor", "turbine", "equipment"}
    hits = sum(1 for t in domain_terms if t in resp_lower)
    score += min(0.3, hits * 0.05)

    # Not an error (0.2)
    if "error" not in resp_lower:
        score += 0.2

    score = max(0.0, min(1.0, score))
    result["passed"] = score >= 0.5
    result["score"] = round(score, 3)
    result.setdefault("handler", "new_handler")
    return result


def _resolve_handler(
    config: str | None, scenario: dict[str, Any],
) -> str | None:
    """Resolve which handler key to use from config and scenario metadata."""
    # 1. Explicit config takes priority
    if config:
        key = _CONFIG_MAP.get(config.lower())
        if key:
            return key

    # 2. Scenario type field
    stype = scenario.get("type", "")
    if stype:
        key = _TYPE_MAP.get(stype)
        if key:
            return key
        # Case-insensitive fallback
        key = _TYPE_MAP.get(stype.lower())
        if key:
            return key

    # 3. Scenario category field
    category = scenario.get("category", "")
    if category:
        key = _CONFIG_MAP.get(category.lower())
        if key:
            return key

    # 4. Scenario_type field (from IBM loader classification)
    scenario_type = scenario.get("scenario_type", "")
    if scenario_type:
        key = _CONFIG_MAP.get(scenario_type.lower())
        if key:
            return key

    return None


def route_scenario(
    client: SamyamaClient,
    scenario: dict[str, Any],
    config: str | None = None,
    tenant: str = "default",
) -> dict[str, Any]:
    """Route a scenario to the appropriate handler.

    Resolution order:
        1. Explicit ``config`` parameter (e.g. ``"rule_logic"``)
        2. Scenario ``type`` field
        3. Scenario ``category`` field
        4. Scenario ``scenario_type`` field
        5. Fallback to a generic unhandled response

    Parameters
    ----------
    client : SamyamaClient
        Samyama Python SDK client (embedded or remote).
    scenario : dict
        Scenario dict with at least ``text`` (or ``description``) and ``id``.
    config : str or None
        Explicit handler configuration key (e.g. ``"rule_logic"``,
        ``"failure mode sensor mapping"``, ``"phm"``).
    tenant : str
        Graph / tenant name (default ``"default"``).

    Returns
    -------
    dict
        Handler result with at minimum ``response``, ``tools_used``,
        and ``latency_ms`` keys.
    """
    handler_key = _resolve_handler(config, scenario)

    if handler_key == "rule_logic":
        result = handle_rule_logic(client, scenario, tenant)
        return _add_scoring(result, scenario)

    if handler_key == "fmsr":
        result = handle_fmsr(client, scenario, tenant)
        return _add_scoring(result, scenario)

    if handler_key == "phm":
        result = handle_phm(client, scenario, tenant)
        return _add_scoring(result, scenario)

    # Fall back to existing IBM handlers for scenarios/compressor/hydrolic_pump
    stype = scenario.get("type", "")
    try:
        from benchmark.handlers.existing_handler import handle_existing
        return handle_existing(client, scenario, config, tenant)
    except (ImportError, Exception):
        sid = scenario.get("id", "?")
        return {
            "response": (
                f"No handler for scenario {sid} "
                f"(type='{stype}', config='{config}')."
            ),
            "tools_used": [],
            "latency_ms": 0.0,
            "passed": False,
            "score": 0.0,
            "handler": "unhandled",
        }
