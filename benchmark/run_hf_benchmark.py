#!/usr/bin/env python3
"""Full AssetOpsBench HuggingFace Benchmark Runner — 467 scenarios.

Runs all 6 HuggingFace configs against the Samyama KG, routing each
scenario to the appropriate handler, scoring results, and producing
a summary table suitable for the NeurIPS 2026 D&B paper.

Usage:
    python -m benchmark.run_hf_benchmark                          # all 467
    python -m benchmark.run_hf_benchmark --configs scenarios fmsr  # subset
    python -m benchmark.run_hf_benchmark --output results/hf_full.json
    python -m benchmark.run_hf_benchmark --dry-run                 # validate only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HF_DATA_DIR = Path(os.environ.get(
    "HF_DATA_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "AssetOpsBench" / "data"),
))

# Config name → file name mapping
CONFIG_FILES = {
    "scenarios": "hf_scenarios.json",
    "rule_logic": "hf_rule_logic.json",
    "fmsr": "hf_failure_mode_sensor_mapping.json",
    "compressor": "hf_compressor.json",
    "hydrolic_pump": "hf_hydrolic_pump.json",
    "phm": "hf_prognostics_and_health_management.json",
}

ALL_CONFIGS = list(CONFIG_FILES.keys())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_id: str
    config: str
    category: str
    text: str
    passed: bool
    score: float
    latency_ms: float
    handler: str
    tools_used: list[str] = field(default_factory=list)
    response: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Load scenarios from HF JSON dumps
# ---------------------------------------------------------------------------

def load_hf_scenarios(configs: list[str] | None = None) -> list[tuple[str, dict]]:
    """Load scenarios from HuggingFace JSON files.

    Returns list of (config_name, scenario_dict) tuples.
    """
    if configs is None:
        configs = ALL_CONFIGS

    scenarios = []
    for cfg in configs:
        fname = CONFIG_FILES.get(cfg)
        if not fname:
            print(f"[WARN] Unknown config: {cfg}", file=sys.stderr)
            continue
        fpath = HF_DATA_DIR / fname
        if not fpath.exists():
            print(f"[WARN] File not found: {fpath}", file=sys.stderr)
            print(f"  Run: python -c \"from datasets import load_dataset; "
                  f"load_dataset('ibm-research/AssetOpsBench', '{cfg}', split='train')"
                  f".to_json('{fpath}')\"", file=sys.stderr)
            continue
        with open(fpath) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            scenarios.append((cfg, row))
        print(f"  Loaded {len(rows):>4d} scenarios from {cfg}")

    return scenarios


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------

def route_and_execute(
    client,
    config: str,
    scenario: dict,
    tenant: str = "default",
    dry_run: bool = False,
) -> ScenarioResult:
    """Route a scenario to the correct handler and execute it."""

    sid = str(scenario.get("id", "?"))
    cat = scenario.get("category", scenario.get("type", "unknown"))
    text = scenario.get("text", "")
    stype = scenario.get("type", "")

    if dry_run:
        return ScenarioResult(
            scenario_id=f"{config}/{sid}",
            config=config,
            category=cat,
            text=text[:100],
            passed=True,
            score=0.0,
            latency_ms=0.0,
            handler="dry-run",
        )

    start = time.perf_counter()
    try:
        # Import handlers lazily
        from benchmark.handlers.router import route_scenario
        result = route_scenario(client, scenario, config, tenant)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return ScenarioResult(
            scenario_id=f"{config}/{sid}",
            config=config,
            category=cat,
            text=text[:100],
            passed=result.get("passed", False),
            score=result.get("score", 0.0),
            latency_ms=elapsed_ms,
            handler=result.get("handler", "unknown"),
            tools_used=result.get("tools_used", []),
            response=str(result.get("response", ""))[:500],
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ScenarioResult(
            scenario_id=f"{config}/{sid}",
            config=config,
            category=cat,
            text=text[:100],
            passed=False,
            score=0.0,
            latency_ms=elapsed_ms,
            handler="error",
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[ScenarioResult]):
    """Print a formatted summary table."""

    # Per-config stats
    by_config: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_config[r.config].append(r)

    print()
    print("=" * 80)
    print("AssetOpsBench Full Benchmark — Results Summary")
    print("=" * 80)
    print(f"{'Config':<12s} {'Total':>6s} {'Pass':>6s} {'Fail':>6s} {'Rate':>7s} {'Avg Score':>10s} {'Avg ms':>8s}")
    print("-" * 80)

    total_pass = 0
    total_fail = 0
    total_score = 0.0

    for cfg in ALL_CONFIGS:
        rs = by_config.get(cfg, [])
        if not rs:
            continue
        passed = sum(1 for r in rs if r.passed)
        failed = len(rs) - passed
        rate = passed / len(rs) * 100 if rs else 0
        avg_score = sum(r.score for r in rs) / len(rs) if rs else 0
        avg_ms = sum(r.latency_ms for r in rs) / len(rs) if rs else 0
        total_pass += passed
        total_fail += failed
        total_score += sum(r.score for r in rs)

        print(f"{cfg:<12s} {len(rs):>6d} {passed:>6d} {failed:>6d} {rate:>6.1f}% {avg_score:>10.3f} {avg_ms:>7.1f}")

    total = total_pass + total_fail
    overall_rate = total_pass / total * 100 if total else 0
    overall_avg = total_score / total if total else 0

    print("-" * 80)
    print(f"{'TOTAL':<12s} {total:>6d} {total_pass:>6d} {total_fail:>6d} {overall_rate:>6.1f}% {overall_avg:>10.3f}")
    print("=" * 80)
    print()

    # Per-category breakdown
    by_category: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    print(f"{'Category':<45s} {'Total':>6s} {'Pass':>6s} {'Rate':>7s}")
    print("-" * 65)
    for cat in sorted(by_category, key=lambda c: -len(by_category[c])):
        rs = by_category[cat]
        passed = sum(1 for r in rs if r.passed)
        rate = passed / len(rs) * 100
        print(f"{cat[:44]:<45s} {len(rs):>6d} {passed:>6d} {rate:>6.1f}%")
    print()

    # Errors
    errors = [r for r in results if r.error]
    if errors:
        print(f"Errors ({len(errors)}):")
        for r in errors[:10]:
            print(f"  [{r.scenario_id}] {r.error[:80]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AssetOpsBench Full HF Benchmark")
    parser.add_argument("--configs", nargs="+", default=None,
                        help=f"Configs to run (default: all). Choices: {ALL_CONFIGS}")
    parser.add_argument("--output", "-o", default=None, help="Save results to JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate loading only")
    parser.add_argument("--url", default="http://localhost:8080",
                        help="Samyama server URL (default: http://localhost:8080)")
    parser.add_argument("--tenant", default="default", help="Tenant/graph name")
    args = parser.parse_args()

    configs = args.configs or ALL_CONFIGS

    print("AssetOpsBench Full HF Benchmark")
    print(f"  Configs: {configs}")
    print(f"  Server: {args.url}")
    print(f"  Data dir: {HF_DATA_DIR}")
    print()

    # Load scenarios
    print("Loading scenarios...")
    scenarios = load_hf_scenarios(configs)
    print(f"  Total: {len(scenarios)} scenarios")
    print()

    if not scenarios:
        print("ERROR: No scenarios loaded. Download HF data first.", file=sys.stderr)
        sys.exit(1)

    # Connect to Samyama and load KG data
    client = None
    if not args.dry_run:
        try:
            from samyama import SamyamaClient
            client = SamyamaClient.embedded()
            print(f"  Client: embedded mode")

            # Load the industrial KG + HF expanded data
            print("  Loading KG data...")
            t_load = time.time()
            data_dir = str(Path(__file__).resolve().parent.parent.parent / "AssetOpsBench" / "src")
            try:
                from etl.eamlite_loader import load_eamlite
                from etl.couchdb_loader import load_couchdb
                from etl.fmsr_loader import load_fmsr
                from etl.workorder_loader import load_workorders

                eam = load_eamlite(client, data_dir, args.tenant)
                print(f"    EAMLite: {eam.get('equipment', 0)} equipment, {eam.get('sites', 0)} sites")
                cdb = load_couchdb(client, data_dir, args.tenant)
                print(f"    CouchDB: {cdb.get('sensors', 0)} sensors, {cdb.get('readings', 0)} readings")
                fmsr = load_fmsr(client, data_dir, args.tenant)
                print(f"    FMSR: {fmsr.get('failure_modes', 0)} failure modes")
                wo = load_workorders(client, data_dir, args.tenant)
                print(f"    WorkOrders: {wo.get('work_orders', 0)} work orders")
                print(f"    Base KG loaded ({time.time()-t_load:.1f}s)")
            except Exception as e:
                print(f"    Base KG load error: {e}")

            try:
                from etl.hf_loader import load_hf_scenarios as load_hf_data
                hf_data_dir = str(HF_DATA_DIR)
                stats = load_hf_data(client, data_dir=hf_data_dir, tenant=args.tenant)
                print(f"    HF data loaded: {stats} ({time.time()-t_load:.1f}s)")
            except Exception as e:
                print(f"    HF data load skipped: {e}")

            # Verify KG has data
            try:
                r = client.query("MATCH (n) RETURN count(n) AS cnt", args.tenant)
                cnt = r.records[0][0] if r.records else 0
                print(f"    Graph nodes: {cnt}")
            except Exception:
                pass

        except ImportError:
            print("  samyama package not available, using HTTP client stub")

    # Run scenarios
    print(f"\nRunning {len(scenarios)} scenarios...\n")
    results: list[ScenarioResult] = []
    t0 = time.time()

    for i, (cfg, scenario) in enumerate(scenarios):
        result = route_and_execute(client, cfg, scenario, args.tenant, args.dry_run)
        results.append(result)

        if (i + 1) % 50 == 0 or i + 1 == len(scenarios):
            elapsed = time.time() - t0
            passed = sum(1 for r in results if r.passed)
            rate = passed / len(results) * 100
            print(f"  [{i+1}/{len(scenarios)}] {passed}/{len(results)} passed "
                  f"({rate:.0f}%) — {elapsed:.1f}s")

    # Summary
    print_summary(results)

    # Save results
    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"Results saved to {outpath}")


if __name__ == "__main__":
    main()
