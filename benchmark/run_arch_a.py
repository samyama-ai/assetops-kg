#!/usr/bin/env python3
"""Architecture A, for real: IBM's agent over IBM's document data layer.

Protocol frozen in docs/ARCH-A-PROTOCOL.md BEFORE this file was written.

Why this exists
---------------
The paper's baseline -- the 65% that "+17 points, same model" is measured against --
was NEVER RUN by us on the 139. It is cited from IBM's paper: their harness, their
grader, their undisclosed model. And `benchmark/run_baseline.py`, the only thing that
ever ran under the name "Architecture A", loads no data at all -- it sends one LLM call
saying "answer based on general industrial knowledge".

This driver builds the baseline the paper needed and never had:

    IBM's ReAct agent-as-tool  (src/agent/openai_agent)
      over IBM's MCP servers   (iot / fmsr / wo / tsfm)
      over IBM's CouchDB documents
      driven by a NAMED model  (gpt-4o, pinned)
      graded by OUR grader     (evaluate_scenario -- byte-identical to Architecture B)

Everything is held fixed except the data layer. The gap between this and Architecture B
is the data-layer effect, isolated for the first time.

Usage:
    python -m benchmark.run_arch_a --model gpt-4o --output results/arch_a_gpt4o.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
AOB = ROOT.parent / "AssetOpsBench"

# IBM's agent lives in their repo; our grader lives in ours.
sys.path.insert(0, str(AOB / "src"))
sys.path.insert(0, str(ROOT))

# We import Architecture B's grader *verbatim* rather than copying it -- a copy would let
# the two arms silently drift apart, which is the whole defect we are here to fix.
# run_ibm_scenarios imports SamyamaClient at module scope (the graph client, absent from
# IBM's venv), but evaluate_scenario() is pure string matching and never touches it.
# Stub the unused import so we get the real grader, not a re-implementation of it.
if "samyama" not in sys.modules:
    import types
    _stub = types.ModuleType("samyama")
    _stub.SamyamaClient = object  # never instantiated on the grading path
    sys.modules["samyama"] = _stub

from benchmark.run_ibm_scenarios import evaluate_scenario  # noqa: E402  our grader, verbatim


def load_env() -> None:
    """Load AssetOpsBench/.env so the MCP servers see CouchDB + model config."""
    env = AOB / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def load_scenarios() -> list[dict]:
    """The same 139 scenarios Architecture B was scored on.

    Read from B's own committed result file so the two arms cannot drift apart:
    same ids, same questions, same characteristic_form.
    """
    b = json.loads((ROOT / "results/repro_nlq_gpt41_2026-06-29.json").read_text())
    hf = {}
    src = AOB / "data/hf_scenarios.json"
    if src.exists():
        for line in src.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                hf[s["id"]] = s
    out = []
    for r in b:
        sid = r["id"]
        s = hf.get(sid, {})
        out.append({
            "id": sid,
            "type": r.get("type", ""),
            "category": r.get("category", ""),
            "text": r["question"],
            "characteristic_form": s.get("characteristic_form", ""),
            "deterministic": s.get("deterministic", False),
        })
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--output", default="results/arch_a_gpt4o.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--ids-file", default=None,
                    help="JSON file with an 'ids' list; run only those scenario ids "
                         "(G-02 held-out split).")
    args = ap.parse_args()

    load_env()

    # The MCP servers are spawned as `uv run <entry-point>`, and uv resolves entry points
    # from the pyproject.toml in the CURRENT WORKING DIRECTORY. Run this from assetops-kg
    # and uv finds no such entry points, the server process dies immediately, and every
    # scenario fails with "McpError: Connection closed" -- a 0% baseline that would falsely
    # confirm the paper. The agent must run with AssetOpsBench as its cwd.
    os.chdir(AOB)

    # IBM's _build_mcp_servers() constructs MCPServerStdio without
    # client_session_timeout_seconds, so the SDK default of 5s applies. Each server is a
    # cold `uv run <entry-point>` and the TSFM one imports torch -- 5s is not survivable,
    # and every scenario dies with "Timed out waiting for response to ClientRequest".
    #
    # A timed-out baseline scores 0% and would FALSELY CONFIRM the paper's +17 claim.
    # That is the exact failure mode ARCH-A-PROTOCOL.md pre-registered. So: widen the
    # timeout here, in our driver, rather than editing IBM's repo.
    import agents.mcp.server as _mcp
    _Orig = _mcp.MCPServerStdio

    class _PatientMCPServerStdio(_Orig):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):
            kw.setdefault("client_session_timeout_seconds", 120)
            super().__init__(*a, **kw)

    _mcp.MCPServerStdio = _PatientMCPServerStdio
    import agent.openai_agent.runner as _r
    _r.MCPServerStdio = _PatientMCPServerStdio

    from agent.openai_agent.runner import OpenAIAgentRunner

    scenarios = load_scenarios()
    if args.ids_file:
        import json as _json
        keep = set(_json.loads(pathlib.Path(args.ids_file).read_text())["ids"])
        scenarios = [s for s in scenarios if s["id"] in keep]
        print(f"  held-out filter: {len(scenarios)} of the frozen split from {args.ids_file}")
    if args.limit:
        scenarios = scenarios[: args.limit]

    missing_cf = sum(1 for s in scenarios if not s["characteristic_form"])
    print(f"Architecture A (REAL): {len(scenarios)} scenarios, model={args.model}")
    print(f"  agent   : IBM openai-agent (ReAct agent-as-tool) over MCP + CouchDB")
    print(f"  grader  : evaluate_scenario() -- identical to Architecture B")
    if missing_cf:
        print(f"  WARNING : {missing_cf} scenarios have no characteristic_form "
              f"(grader returns pass by default -- reported separately)")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    done = 0

    async def one(s: dict) -> dict:
        nonlocal done
        async with sem:
            t0 = time.perf_counter()
            answer, err = "", None
            # With the full 4,249-row work-order set loaded, the agent pulls large payloads
            # into context and token throughput spikes -> OpenAI 429s. A rate-limited
            # scenario is scored 0, so 429s silently DEPRESS the baseline and manufacture a
            # data-layer effect. Retry with backoff; never let a 429 count as a failure.
            for attempt in range(6):
                try:
                    runner = OpenAIAgentRunner(model=args.model, max_turns=args.max_turns)
                    res = await runner.run(s["text"])
                    # AgentResult.answer (models.py:73). An earlier version read a
                    # non-existent `.final_answer` and fell back to str(res) -- the whole
                    # repr, trajectory included. The grader is keyword-overlap, so that
                    # INFLATED the baseline. Take the answer only.
                    answer, err = res.answer, None
                    break
                except Exception as e:  # noqa: BLE001
                    name = type(e).__name__
                    err = f"{name}: {e}"
                    # insufficient_quota is ALSO a 429 but is PERMANENT -- the account is
                    # out of credit. Retrying it just burns wall-clock and leaves every
                    # scenario scored 0, i.e. a fabricated 0% baseline. Abort loudly.
                    if "insufficient_quota" in str(e):
                        print("\n*** ABORT: OpenAI account is out of quota "
                              "(insufficient_quota). This is a billing failure, not a rate "
                              "limit. Every scenario would score 0 and understate the "
                              "baseline. Top up the account and re-run. ***",
                              file=sys.stderr, flush=True)
                        raise SystemExit(3)
                    if "RateLimit" in name or "429" in str(e):
                        await asyncio.sleep(min(60, 5 * 2 ** attempt))
                        continue
                    break  # real failure (e.g. context_length_exceeded) -- keep it
            ms = (time.perf_counter() - t0) * 1000

            if err:
                passed, score, why = False, 0.0, f"agent error: {err}"
            else:
                passed, score, why = evaluate_scenario(s, answer)

            done += 1
            if done % 5 == 0 or done == len(scenarios):
                print(f"  {done}/{len(scenarios)}", flush=True)
            return {
                "id": s["id"], "type": s["type"], "category": s["category"],
                "question": s["text"], "passed": bool(passed), "score": float(score),
                "latency_ms": ms, "response": answer, "rationale": why, "error": err,
            }

    results = list(await asyncio.gather(*[one(s) for s in scenarios]))
    results.sort(key=lambda r: r["id"])

    n = len(results) or 1
    p = sum(1 for r in results if r["passed"])
    errs = sum(1 for r in results if r["error"])
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    summary = {
        "architecture": "A (real): IBM agent over IBM CouchDB documents",
        "model": args.model,
        "grader": "evaluate_scenario() -- identical to Architecture B",
        "scenarios": len(results),
        "passed": p,
        "pass_pct": round(100 * p / n, 1),
        "avg_score": round(sum(r["score"] for r in results) / n, 3),
        "agent_errors": errs,
        "by_type": {
            t: {"n": len(v), "passed": sum(1 for r in v if r["passed"]),
                "pass_pct": round(100 * sum(1 for r in v if r["passed"]) / len(v), 1)}
            for t, v in sorted(by_type.items())
        },
    }

    # A 429 that survives retries scores 0 and DEPRESSES the baseline, which fabricates a
    # data-layer effect in our favour. Refuse to present such a run as a result.
    rl = sum(1 for r in results if r["error"] and "RateLimit" in r["error"])
    summary["rate_limited"] = rl
    summary["VALID"] = rl == 0

    dest = ROOT / args.output  # ROOT is absolute; safe after chdir
    dest.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {dest}")
    if rl:
        print(f"\n*** RUN INVALID: {rl} scenarios still rate-limited after retries. "
              f"These score 0 and understate the baseline. Lower --concurrency and re-run. "
              f"DO NOT REPORT THIS NUMBER. ***", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
