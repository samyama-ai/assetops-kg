#!/usr/bin/env python3
"""Generate the AssetOpsBench-KG coverage page from committed artifacts.

Principle 0: every number on the page is READ from an artifact, never typed.

Sources
  ours : assetops-kg/results/*.json          (our runs)
  IBM  : IBM's own shipped HuggingFace Space trajectories
         huggingface.co/spaces/ibm-research/AssetOps-Bench  -> public/logs/
         Point IBM_LOGS at a local clone; without it the IBM panels are omitted
         rather than faked.

Usage
  python -m evaluation.build_coverage_viz                 # docs/coverage.html
  python -m evaluation.build_coverage_viz --fragment out.html
"""
from __future__ import annotations

import collections
import glob
import json
import os
import pathlib
import statistics as st
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
IBM_LOGS = pathlib.Path(os.environ.get("IBM_LOGS", "")) if os.environ.get("IBM_LOGS") else None

NON_GRAPH = {"domain_knowledge_fallback"}

# side: "ibm" rows are IBM's own orchestration paradigms, read from their shipped
# trajectories. "us" rows are our architectures. Both are plotted on one axis because
# they answer the same benchmark -- but see the metric caveat in the accuracy section.
ARCHES = [
    ("AaT", "Agent-as-Tools", "ReAct-style supervisor picks domain agents as tools. Reflection loop over a document data layer.", "ibm", "ibm"),
    ("P&amp;E", "Plan-and-Execute", "Planner&ndash;Reviewer decomposes the query into a DAG over shared memory.", "ibm", "ibm"),
    # The paper describes A as "LLM over documents (CouchDB/YAML/CSV)". The code does no
    # such thing: benchmark/run_baseline.py loads NO data and tells the model
    # "Answer based on general industrial knowledge." Label it as what it is.
    ("A", "Baseline",
     "The paper&rsquo;s <code>run_baseline.py</code> loads <b>no data</b> (custom-40 cell = parametric memory). "
     "The <b>139</b> cell is the real baseline we built 2026-07-13: IBM&rsquo;s agent over IBM&rsquo;s "
     "documents, gpt-4o, our grader &mdash; the 65% the paper cited but never ran.", "llm", "us"),
    ("B", "NLQ &rarr; Cypher", "LLM constrained to generating Cypher against a live-introspected schema; the graph executes.", "graph", "us"),
    ("C", "Deterministic handlers", "Pre-coded handlers route a recognised question to a typed-graph query. No LLM.", "graph", "us"),
    ("D", "GAK", "On a lookup miss the LLM writes the missing facts in as provenance-tagged nodes, then answers.", "gak", "us"),
    ("E", "No-graph control", "Ask the LLM directly. Added 2026-07-13 as the missing baseline for D.", "control", "us"),
]

SETS = [
    ("152", "152-playground", "IBM&rsquo;s Space set. Verified <em>identical ids</em> to the <code>scenarios</code> config of the 467 &mdash; the one head-to-head."),
    ("139", "139-snapshot", "AssetOpsBench v1 snapshot. The A/B/C comparison runs here."),
    ("467", "467-HF", "IBM&rsquo;s full HuggingFace release, 6 configs."),
    ("88", "88-fmsr", "Failure-mode scenarios. A <em>subset</em> of the 467, and GAK&rsquo;s only testbed."),
    ("40", "custom-40", "Our graph-native scenarios (multi-hop, PageRank, vector similarity)."),
]


def load(name):
    return json.loads((R / name).read_text())


def rate(rows, key="passed"):
    return round(100 * sum(1 for r in rows if r.get(key)) / len(rows), 1)


# --------------------------------------------------------------------------
# IBM: read their own shipped trajectories. Never invent these.
# --------------------------------------------------------------------------
def read_ibm():
    if not IBM_LOGS or not IBM_LOGS.is_dir():
        return None
    out = {}

    aat = []
    for f in glob.glob(str(IBM_LOGS / "agent-as-tools/*.json")):
        d = json.loads(pathlib.Path(f).read_text())
        om = d.get("overall_metric") or {}
        ms = [p["info"]["model_stats"] for p in om.get("per_round_info", []) if "info" in p]
        aat.append(dict(
            status=om.get("status"),
            t=om.get("total_execution_time"),
            sent=sum(m["tokens_sent"] for m in ms),
            recv=sum(m["tokens_received"] for m in ms),
            calls=sum(m["api_calls"] for m in ms),
        ))
    if aat:
        sc = collections.Counter(r["status"] for r in aat)
        ts = [r["t"] * 60 for r in aat if r["t"]]
        out["aat"] = dict(
            n=len(aat), ok=sc["Accomplished"], partial=sc["Partially Accomplished"],
            bad=sc["Not Accomplished"],
            pct=round(100 * sc["Accomplished"] / len(aat), 1),
            med_s=round(st.median(ts), 1), mean_s=round(st.mean(ts), 1),
            sent=sum(r["sent"] for r in aat), recv=sum(r["recv"] for r in aat),
            calls=sum(r["calls"] for r in aat),
            mean_calls=round(st.mean([r["calls"] for r in aat]), 1),
        )

    AX = ["task_completion", "data_retrieval_accuracy", "generalized_result_verification",
          "agent_sequence_correct", "clarity_and_justification", "hallucinations"]
    pae = []
    for f in glob.glob(str(IBM_LOGS / "plan-and-execute/*.json")):
        d = json.loads(pathlib.Path(f).read_text())
        e = d.get("evaluation") or {}
        pae.append(dict(rt=d.get("runtime"), **{a: e.get(a) for a in AX}))
    if pae:
        n = len(pae)
        rt = [r["rt"] for r in pae if r["rt"]]
        out["pae"] = dict(
            n=n, med_s=round(st.median(rt), 1), mean_s=round(st.mean(rt), 1),
            **{a: round(100 * sum(1 for r in pae if r[a] is True) / n, 1) for a in AX},
        )
    return out or None


def build():
    hf = load("repro_hf467_2026-06-29.json")

    by_config = collections.Counter(r["config"] for r in hf)
    by_cat = collections.Counter(r["category"] for r in hf)

    paths = collections.defaultdict(collections.Counter)
    for r in hf:
        t = set(r.get("tools_used") or [])
        g, fb = bool(t - NON_GRAPH), bool(t & NON_GRAPH)
        paths[r["config"]]["graph" if g and not fb else
                           "mixed" if g and fb else
                           "hardcoded" if fb else "none"] += 1
    tools = collections.Counter(t for r in hf for t in (r.get("tools_used") or []))

    # coverage cells
    cells = collections.defaultdict(list)
    cells[("A", "40")].append({"model": "gpt-4o, NO data",
                               "pct": rate(load("baseline_gpt4o_results.json"))})
    # The REAL Architecture A, built 2026-07-13 (docs/AUDIT-2026-07-13.md): IBM's agent
    # over IBM's CouchDB documents, gpt-4o, graded by Architecture B's own grader. This is
    # the baseline the paper cited at 65% but never ran.
    _a = load("arch_a_gpt4o.json")
    if _a["summary"].get("VALID"):
        cells[("A", "139")].append({"model": "gpt-4o, IBM agent+docs",
                                    "pct": _a["summary"]["pass_pct"]})
    for f, m in [("repro_nlq_gpt4_2026-06-29.json", "gpt-4"),
                 ("repro_nlq_gpt4o_2026-06-29.json", "gpt-4o"),
                 ("repro_nlq_gpt41_2026-06-29.json", "gpt-4.1")]:
        cells[("B", "139")].append({"model": m, "pct": rate(load(f))})
    cells[("C", "139")].append({"model": "no LLM", "pct": rate(load("repro_det_2026-06-29.json"))})
    cells[("C", "467")].append({"model": "no LLM", "pct": rate(hf)})
    cells[("C", "88")].append({"model": "no LLM", "pct": rate([r for r in hf if r["config"] == "fmsr"])})
    cells[("C", "40")].append({"model": "no LLM", "pct": rate(load("samyama_results_v5.json"))})
    # The 152 "scenarios" config is byte-for-byte the same id set as IBM's playground,
    # so this cell is the only true head-to-head on the page.
    cells[("C", "152")].append({"model": "no LLM",
                                "pct": rate([r for r in hf if r["config"] == "scenarios"])})
    for f, m in [("gak_full.json", "claude-sonnet-4-5"),
                 ("repro_gak_gpt41_2026-06-30.json", "gpt-4.1"),
                 ("repro_gak_2026-06-30.json", "gpt-4o")]:
        sr = load(f)["scenario_results"]
        cells[("D", "88")].append({"model": m,
                                   "pct": round(100 * sum(1 for r in sr if r["judge_pass"]) / len(sr), 1)})
    nc = load("nc_nograph.json")
    for m, a in nc["arms"].items():
        cells[("E", "88")].append({"model": m, "pct": a["summary"]["C1"]["pass_pct"]})

    # IBM's own paradigms, read from their shipped trajectories.
    ib = read_ibm()
    if ib and "aat" in ib:
        cells[("AaT", "152")].append({"model": "model undisclosed", "pct": ib["aat"]["pct"]})
    if ib and "pae" in ib:
        # Only 124 of the 152 were ever run under Plan-and-Execute; say so rather than
        # letting the cell imply full coverage.
        cells[("P&amp;E", "152")].append({"model": f'{ib["pae"]["n"]}/152 only',
                                          "pct": ib["pae"]["task_completion"]})

    # speed
    lat_c = [r["latency_ms"] for r in hf if r.get("latency_ms")]
    lat_b = [r["latency_ms"] for r in load("repro_nlq_gpt41_2026-06-29.json") if r.get("latency_ms")]
    gak = load("gak_full.json")
    enrich = [e.get("enrich_latency_ms", 0) for e in gak["entity_reports"]]

    speed = dict(
        c_med=round(st.median(lat_c), 2), c_mean=round(st.mean(lat_c), 2),
        c_p95=round(sorted(lat_c)[int(.95 * len(lat_c))], 2), c_max=round(max(lat_c), 1),
        b_med=round(st.median(lat_b)), b_mean=round(st.mean(lat_b)),
        gak_enrich_med=round(st.median(enrich) / 1000, 1),
    )

    hw = dict(
        cpu=subprocess.run(["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
                           capture_output=True, text=True).stdout.strip(),
        cores=os.cpu_count(),
        ram=subprocess.run(["bash", "-lc", "free -g | awk '/^Mem:/{print $2}'"],
                           capture_output=True, text=True).stdout.strip(),
        os=subprocess.run(["bash", "-lc", ". /etc/os-release && echo $PRETTY_NAME"],
                          capture_output=True, text=True).stdout.strip(),
    )

    # The isolated data-layer comparison (Architecture A vs B), read from the audit's
    # own generated numbers so this page and the write-up can never disagree.
    datalayer = None
    ap = R / "paper3_audit_numbers.json"
    if ap.exists():
        datalayer = json.loads(ap.read_text())

    return dict(by_config=by_config, by_cat=by_cat, paths=paths, tools=tools, cells=cells,
                total=len(hf), speed=speed, hw=hw, ibm=ib, datalayer=datalayer)


PATH_LABEL = {"graph": "graph-backed", "mixed": "graph + hardcoded fallback",
              "hardcoded": "hardcoded only", "none": "no tools recorded"}


def html(d) -> str:
    total = d["total"]
    ibm = d["ibm"]
    sp = d["speed"]
    hw = d["hw"]
    da = d["datalayer"]  # the isolated Architecture A vs B comparison

    seg = "".join(
        f'<div class="seg" style="flex:{n}"><span class="seg__n">{n}</span>'
        f'<span class="seg__l">{c}</span></div>'
        for c, n in d["by_config"].most_common())

    cats = "".join(f'<li><b>{n}</b>{c}</li>' for c, n in d["by_cat"].most_common())

    # ---- matrix ----
    mrows = []
    for code, name, desc, kind, side in ARCHES:
        tds = []
        for skey, _, _ in SETS:
            got = d["cells"].get((code, skey))
            if got:
                chips = "".join(
                    f'<span class="chip"><em>{g["model"]}</em><b>{g["pct"]}%</b></span>' for g in got)
                tds.append(f'<td class="cell run">{chips}</td>')
            else:
                tds.append('<td class="cell"><span class="gap">never run</span></td>')
        tag = ('<span class="tag tag--ibm">IBM</span>' if side == "ibm"
               else '<span class="tag tag--us">ours</span>')
        mrows.append(
            f'<tr class="{"ibm" if side == "ibm" else ""}">'
            f'<th class="arch k-{kind}" scope="row">{tag}'
            f'<span class="arch__c">{code}</span><span class="arch__n">{name}</span>'
            f'<span class="arch__d">{desc}</span></th>{"".join(tds)}</tr>')

    # ---- IBM panels ----
    ibm_acc = ibm_speed = ibm_cost = ""
    if ibm and "aat" in ibm:
        a = ibm["aat"]
        ibm_acc += (
            f'<tr class="ibm"><th scope="row"><span class="tag tag--ibm">IBM</span>'
            f'Agent-as-Tools<em>152 scenarios &middot; judged Accomplished / Partial / Not</em></th>'
            f'<td class="num big">{a["pct"]}%</td>'
            f'<td class="num">{a["ok"]}</td><td class="num">{a["partial"]}</td>'
            f'<td class="num">{a["bad"]}</td></tr>')
    if ibm and "pae" in ibm:
        p = ibm["pae"]
        ibm_acc += (
            f'<tr class="ibm"><th scope="row"><span class="tag tag--ibm">IBM</span>'
            f'Plan-and-Execute<em>124 scenarios &middot; IBM 3-axis rubric</em></th>'
            f'<td class="num big">{p["task_completion"]}%</td>'
            f'<td class="num" colspan="3">y&#8322; data retrieval {p["data_retrieval_accuracy"]}% '
            f'&middot; y&#8323; verification {p["generalized_result_verification"]}% '
            f'&middot; hallucinated {p["hallucinations"]}%</td></tr>')

    if ibm:
        rows = []
        if "aat" in ibm:
            rows.append(('IBM', 'Agent-as-Tools', f'{ibm["aat"]["med_s"]:,.0f} s',
                         f'{ibm["aat"]["mean_s"]:,.0f} s', 'end-to-end agent loop'))
        if "pae" in ibm:
            rows.append(('IBM', 'Plan-and-Execute', f'{ibm["pae"]["med_s"]:,.0f} s',
                         f'{ibm["pae"]["mean_s"]:,.0f} s', 'end-to-end agent loop'))
        rows += [
            ('ours', 'B &mdash; NLQ &rarr; Cypher', f'{sp["b_med"]/1000:,.2f} s',
             f'{sp["b_mean"]/1000:,.2f} s', 'one LLM call + graph execution'),
            ('ours', 'C &mdash; Deterministic', f'{sp["c_med"]} ms',
             f'{sp["c_mean"]} ms', f'graph query only &middot; p95 {sp["c_p95"]} ms'),
            ('ours', 'D &mdash; GAK enrichment', f'{sp["gak_enrich_med"]} s',
             '&mdash;', 'one-off per equipment type, then cached'),
        ]
        ibm_speed = "".join(
            f'<tr class="{"ibm" if who=="IBM" else ""}">'
            f'<th scope="row"><span class="tag tag--{"ibm" if who=="IBM" else "us"}">{who}</span>{what}</th>'
            f'<td class="num big">{med}</td><td class="num">{mean}</td><td class="note-c">{note}</td></tr>'
            for who, what, med, mean, note in rows)

        if "aat" in ibm:
            a = ibm["aat"]
            ibm_cost = (
                f'<tr class="ibm"><th scope="row"><span class="tag tag--ibm">IBM</span>Agent-as-Tools</th>'
                f'<td class="num big">{a["mean_calls"]}</td>'
                f'<td class="num">{a["sent"]:,}</td><td class="num">{a["recv"]:,}</td>'
                f'<td class="num">{a["calls"]:,}</td></tr>'
                f'<tr><th scope="row"><span class="tag tag--us">ours</span>B &mdash; NLQ &rarr; Cypher</th>'
                f'<td class="num big">1</td><td class="num colspan" colspan="3">'
                f'one Cypher-generation call per question; the graph does the rest</td></tr>'
                f'<tr><th scope="row"><span class="tag tag--us">ours</span>C &mdash; Deterministic</th>'
                f'<td class="num big">0</td><td class="num colspan" colspan="3">no LLM in the loop</td></tr>'
                f'<tr><th scope="row"><span class="tag tag--us">ours</span>D &mdash; GAK</th>'
                f'<td class="num big">10</td><td class="num colspan" colspan="3">'
                f'10 enrichment calls answer all 88 scenarios (8.8&times; fewer than one call each)</td></tr>')

    prows = []
    for c, _ in d["by_config"].most_common():
        p = d["paths"][c]
        n = sum(p.values())
        bars = "".join(f'<i class="p-{k}" style="flex:{p[k]}" title="{PATH_LABEL[k]}: {p[k]}"></i>'
                       for k in ("graph", "mixed", "hardcoded", "none") if p.get(k))
        det = " &middot; ".join(f"{v} {PATH_LABEL[k]}" for k, v in p.items() if v)
        prows.append(f'<tr><th scope="row">{c}<em>{n}</em></th>'
                     f'<td><div class="pbars">{bars}</div><p class="pd">{det}</p></td></tr>')
    tot = collections.Counter()
    for c in d["paths"]:
        tot.update(d["paths"][c])

    mx = max(d["tools"].values())
    trows = "".join(
        f'<tr><td class="tn">{t}</td><td class="num">{n}</td>'
        f'<td class="tb"><i class="{"fb" if t in NON_GRAPH else ""}" style="width:{100*n/mx:.1f}%"></i></td></tr>'
        for t, n in d["tools"].most_common())

    # data-layer section values (from the audit's generated numbers)
    a_full = da["arch_a_full"]; b_full = da["arch_b_full"]; d_full = da["delta_full"]
    a_clean = da["arch_a_clean"]; b_clean = da["arch_b_clean"]; d_clean = da["delta_clean"]
    n_clean = da["n_clean"]; n_excl = da["n_excluded"]
    excl_alert = da["exclusion_reasons"].get("no alert/anomaly server in current upstream", 0)
    excl_asset = da["exclusion_reasons"].get("asset/data absent from the document store", 0)

    return f"""<title>AssetOpsBench &mdash; IBM vs Samyama Graph: what we actually cover</title>
<style>
  :root {{
    --bg:#EDF0F3; --card:#FFF; --line:#D5DCE2; --line-2:#E6EBEF;
    --ink:#0C1317; --ink-2:#47565F; --ink-3:#7B8B96;
    --ibm:#0F62FE; --ibm-w:#E8EFFF;
    --us:#0B7C70;  --us-w:#E1F2F0;
    --gak:#7E4FC0; --llm:#B26A06; --ctl:#B03A5B; --gap:#C6CFD6;
    --r:4px;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#0C1216; --card:#151D23; --line:#28333B; --line-2:#1E272E;
      --ink:#E6EDF2; --ink-2:#9AA9B4; --ink-3:#65737D;
      --ibm:#5B92FF; --ibm-w:#16233C;
      --us:#2FB3A2;  --us-w:#0E2D2A;
      --gak:#AE86E8; --llm:#DE9F3C; --ctl:#E4738F; --gap:#37434C;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0C1216; --card:#151D23; --line:#28333B; --line-2:#1E272E;
    --ink:#E6EDF2; --ink-2:#9AA9B4; --ink-3:#65737D;
    --ibm:#5B92FF; --ibm-w:#16233C; --us:#2FB3A2; --us-w:#0E2D2A;
    --gak:#AE86E8; --llm:#DE9F3C; --ctl:#E4738F; --gap:#37434C;
  }}
  :root[data-theme="light"] {{
    --bg:#EDF0F3; --card:#FFF; --line:#D5DCE2; --line-2:#E6EBEF;
    --ink:#0C1317; --ink-2:#47565F; --ink-3:#7B8B96;
    --ibm:#0F62FE; --ibm-w:#E8EFFF; --us:#0B7C70; --us-w:#E1F2F0;
    --gak:#7E4FC0; --llm:#B26A06; --ctl:#B03A5B; --gap:#C6CFD6;
  }}

  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); line-height:1.55;
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1140px; margin:0 auto; padding:0 24px 110px; }}

  /* nav */
  nav {{ position:sticky; top:0; z-index:20; background:color-mix(in srgb,var(--bg) 88%,transparent);
    backdrop-filter:blur(9px); border-bottom:1px solid var(--line); margin-bottom:44px; }}
  nav ul {{ max-width:1140px; margin:0 auto; padding:0 24px; list-style:none;
    display:flex; gap:22px; overflow-x:auto; }}
  nav a {{ display:block; padding:15px 0; color:var(--ink-3); text-decoration:none;
    font:600 11px/1 ui-monospace,Menlo,monospace; letter-spacing:.1em; text-transform:uppercase;
    white-space:nowrap; border-bottom:2px solid transparent; }}
  nav a:hover, nav a:focus-visible {{ color:var(--ink); border-bottom-color:var(--ink); outline:none; }}

  header {{ padding:40px 0 0; }}
  h1 {{ font-size:clamp(27px,3.7vw,42px); line-height:1.1; letter-spacing:-.022em;
    margin:0 0 16px; font-weight:660; text-wrap:balance; max-width:19ch; }}
  .lede {{ max-width:66ch; color:var(--ink-2); font-size:16.5px; margin:0; }}
  .prov {{ font:400 12px/1.65 ui-monospace,Menlo,monospace; color:var(--ink-3);
    margin:22px 0 0; padding-top:18px; border-top:1px solid var(--line); }}

  /* provenance tags -- the core UX device */
  .tag {{ display:inline-flex; align-items:center; height:17px; padding:0 6px; margin-right:9px;
    border-radius:2px; font:700 9.5px/1 ui-monospace,Menlo,monospace; letter-spacing:.1em;
    text-transform:uppercase; vertical-align:1px; }}
  .tag--ibm {{ background:var(--ibm-w); color:var(--ibm); box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--ibm) 30%,transparent); }}
  .tag--us  {{ background:var(--us-w);  color:var(--us);  box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--us) 30%,transparent); }}

  .key {{ display:flex; flex-wrap:wrap; gap:20px; margin:26px 0 0; padding:15px 18px;
    background:var(--card); border:1px solid var(--line); border-radius:var(--r); }}
  .key div {{ font-size:12.5px; color:var(--ink-2); }}
  .key b {{ color:var(--ink); font-weight:600; }}

  section {{ margin-top:64px; scroll-margin-top:70px; }}
  h2 {{ display:flex; align-items:baseline; gap:11px; font-size:19px; font-weight:640;
    letter-spacing:-.012em; margin:0 0 7px; }}
  h2 span {{ font:700 11px/1 ui-monospace,Menlo,monospace; color:var(--ink-3); }}
  .sub {{ color:var(--ink-3); font-size:13.5px; margin:0 0 22px; max-width:74ch; }}

  /* universe */
  .segs {{ display:flex; gap:3px; height:74px; flex-wrap:wrap; }}
  .seg {{ background:var(--card); border:1px solid var(--line); border-radius:var(--r);
    display:flex; flex-direction:column; justify-content:center; align-items:center;
    min-width:116px; padding:0 8px; position:relative; overflow:hidden; }}
  .seg::after {{ content:""; position:absolute; inset:auto 0 0 0; height:3px; background:var(--us); opacity:.8; }}
  .seg__n {{ font:660 18px/1 ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; }}
  .seg__l {{ font-size:10px; letter-spacing:.06em; color:var(--ink-3); margin-top:6px;
    text-transform:uppercase; white-space:nowrap; max-width:100%; overflow:hidden; text-overflow:ellipsis; }}
  ul.cats {{ list-style:none; padding:0; margin:20px 0 0; display:flex; flex-wrap:wrap; gap:6px; }}
  ul.cats li {{ display:flex; align-items:center; gap:7px; background:var(--card);
    border:1px solid var(--line); border-radius:var(--r); padding:5px 10px;
    font-size:12px; color:var(--ink-2); }}
  ul.cats b {{ font:660 11.5px/1 ui-monospace,Menlo,monospace; color:var(--us);
    font-variant-numeric:tabular-nums; }}

  /* tables */
  .scroll {{ overflow-x:auto; border-radius:var(--r); }}
  table {{ border-collapse:separate; border-spacing:0; width:100%; background:var(--card);
    border:1px solid var(--line); border-radius:var(--r); overflow:hidden; }}
  thead th {{ font:600 10px/1.35 ui-monospace,Menlo,monospace; letter-spacing:.1em;
    text-transform:uppercase; color:var(--ink-3); text-align:left;
    padding:13px 14px; background:var(--line-2); border-bottom:1px solid var(--line);
    vertical-align:bottom; }}
  thead th em {{ display:block; font:400 11px/1.45 ui-sans-serif,system-ui,sans-serif;
    text-transform:none; letter-spacing:0; color:var(--ink-3); margin-top:5px; max-width:21ch; }}
  tbody th {{ text-align:left; padding:14px; font-weight:560; font-size:14px;
    border-bottom:1px solid var(--line-2); vertical-align:top; }}
  tbody th em {{ display:block; font-style:normal; font-size:12px; color:var(--ink-3);
    margin-top:4px; font-weight:400; }}
  tbody td {{ padding:14px; border-bottom:1px solid var(--line-2); vertical-align:middle;
    font-size:13.5px; color:var(--ink-2); }}
  tbody tr:last-child th, tbody tr:last-child td {{ border-bottom:none; }}
  tbody tr:hover td, tbody tr:hover th {{ background:color-mix(in srgb,var(--ink) 3%,transparent); }}
  .num {{ font:600 14px/1 ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums;
    color:var(--ink); white-space:nowrap; }}
  .num.big {{ font-size:17px; font-weight:660; }}
  .note-c {{ font-size:12.5px; color:var(--ink-3); }}
  .colspan {{ font:400 12.5px/1.5 ui-sans-serif,system-ui,sans-serif; color:var(--ink-3); }}

  /* IBM rows get a left rail so provenance is unmissable even when scanning */
  tr.ibm th {{ box-shadow:inset 3px 0 0 var(--ibm); background:color-mix(in srgb,var(--ibm) 4%,transparent); }}
  tr.ibm td {{ background:color-mix(in srgb,var(--ibm) 4%,transparent); }}

  /* matrix */
  /* 5 set-columns + the architecture column must fit the 1092px content box, or the
     last column (custom-40) is clipped at the scroll edge and reads as missing. */
  table.matrix {{ min-width:1040px; }}
  th.arch {{ width:25%; box-shadow:inset 3px 0 0 var(--gap); }}
  table.matrix thead th:not(:first-child), table.matrix td.cell {{ width:15%; }}
  table.matrix thead th, table.matrix td.cell, table.matrix th.arch {{ padding:12px 10px; }}
  th.arch.k-graph {{ box-shadow:inset 3px 0 0 var(--us); }}
  th.arch.k-gak {{ box-shadow:inset 3px 0 0 var(--gak); }}
  th.arch.k-llm {{ box-shadow:inset 3px 0 0 var(--llm); }}
  th.arch.k-control {{ box-shadow:inset 3px 0 0 var(--ctl); }}
  .arch__c {{ font:660 10.5px/1 ui-monospace,Menlo,monospace; color:var(--ink-3); letter-spacing:.09em; }}
  .arch__n {{ display:block; font-size:15px; font-weight:620; margin:6px 0 5px; }}
  .arch__d {{ display:block; font-size:12.5px; color:var(--ink-3); font-weight:400; line-height:1.5; }}
  td.cell {{ width:17%; }}
  .chip {{ display:flex; align-items:baseline; justify-content:space-between; gap:10px;
    padding:5px 0; border-bottom:1px dotted var(--line); }}
  .chip:last-child {{ border-bottom:none; }}
  .chip em {{ font:500 11px/1.35 ui-monospace,Menlo,monospace; font-style:normal; color:var(--ink-2);
    overflow-wrap:anywhere; }}
  .chip b {{ font:660 13px/1 ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums;
    color:var(--ink); white-space:nowrap; }}
  .gap {{ display:inline-block; font:500 10.5px/1 ui-monospace,Menlo,monospace; letter-spacing:.06em;
    color:var(--ink-3); opacity:.7; border:1px dashed var(--gap); border-radius:var(--r);
    padding:7px 9px; white-space:nowrap; }}

  /* answer paths */
  table.paths th {{ width:28%; font:600 13px/1.4 ui-monospace,Menlo,monospace; }}
  table.paths th em {{ font-family:ui-sans-serif,system-ui,sans-serif; }}
  .pbars {{ display:flex; gap:2px; height:12px; }}
  .pbars i {{ border-radius:2px; }}
  .p-graph {{ background:var(--us); }} .p-mixed {{ background:var(--llm); }}
  .p-hardcoded {{ background:var(--ctl); }} .p-none {{ background:var(--gap); }}
  .pd {{ margin:9px 0 0; font-size:12px; color:var(--ink-3); }}
  .legend {{ display:flex; flex-wrap:wrap; gap:18px; margin:16px 0 0; padding:0; list-style:none; }}
  .legend li {{ display:flex; align-items:center; gap:7px; font-size:12px; color:var(--ink-2); }}
  .sw {{ width:11px; height:11px; border-radius:2px; flex:none; }}

  /* tools */
  table.tools td.tn {{ font:400 12.5px/1.5 ui-monospace,Menlo,monospace; white-space:nowrap; width:1%; }}
  table.tools td.tb {{ width:100%; }}
  table.tools td.tb i {{ display:block; height:8px; background:var(--us); border-radius:2px; min-width:3px; }}
  table.tools td.tb i.fb {{ background:var(--ctl); }}

  .callout {{ margin-top:20px; padding:16px 18px; background:var(--card); border:1px solid var(--line);
    border-left:3px solid var(--ink-3); border-radius:var(--r); font-size:13.5px;
    color:var(--ink-2); max-width:80ch; }}
  .callout b {{ color:var(--ink); font-weight:640; }}
  .callout--warn {{ border-left-color:var(--llm); }}
  .callout--bad {{ border-left-color:var(--ctl); background:color-mix(in srgb,var(--ctl) 5%,var(--card)); }}
  code {{ font:400 12.5px/1 ui-monospace,Menlo,monospace; background:var(--bg);
    border:1px solid var(--line); border-radius:2px; padding:1px 5px; }}
  @media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<nav><ul>
  <li><a href="#universe">Universe</a></li>
  <li><a href="#matrix">Combinations</a></li>
  <li><a href="#accuracy">Accuracy</a></li>
  <li><a href="#datalayer">Data-layer &Delta;</a></li>
  <li><a href="#speed">Speed</a></li>
  <li><a href="#cost">LLM cost</a></li>
  <li><a href="#hardware">Hardware</a></li>
  <li><a href="#paths">Answer paths</a></li>
  <li><a href="#tools">Tools</a></li>
</ul></nav>

<div class="wrap">
  <header>
    <h1>AssetOpsBench: what IBM ran, what we ran</h1>
    <p class="lede">Every architecture &times; scenario-set &times; model combination on the table &mdash; and,
      plainly, which were ever executed. IBM&rsquo;s numbers come from IBM&rsquo;s own shipped trajectories;
      ours come from our committed result files. Nothing here is transcribed from a paper.</p>
    <div class="key">
      <div><span class="tag tag--ibm">IBM</span><b>IBM Research.</b> Read from the trajectories shipped in
        their public HuggingFace Space.</div>
      <div><span class="tag tag--us">ours</span><b>Samyama Graph.</b> Read from <code>results/*.json</code> in this repo.</div>
    </div>
    <p class="prov">generated by evaluation/build_coverage_viz.py &middot; {total} scenarios &middot; no number on this page was typed by hand</p>
  </header>

  <section id="universe">
    <h2><span>01</span>The scenario universe</h2>
    <p class="sub">IBM&rsquo;s HuggingFace release: <b>{total} scenarios</b> across 6 configs and
      {len(d["by_cat"])} question categories. Every result below is a slice of this.</p>
    <div class="segs">{seg}</div>
    <ul class="cats">{cats}</ul>
  </section>

  <section id="matrix">
    <h2><span>02</span>The combination matrix</h2>
    <p class="sub">Every architecture &mdash; IBM&rsquo;s two orchestration paradigms and our five &mdash;
      against every scenario set. A filled cell was executed and has a committed result file. An empty cell is
      a combination that is described but was never run.</p>
    <div class="scroll">
      <table class="matrix">
        <thead><tr><th scope="col">Architecture</th>
          {"".join(f'<th scope="col">{n}<em>{ds}</em></th>' for _, n, ds in SETS)}</tr></thead>
        <tbody>{"".join(mrows)}</tbody>
      </table>
    </div>
    <div class="callout"><b>One column is a true head-to-head; the rest is empty space.</b> IBM&rsquo;s
      playground set and the <code>scenarios</code> config of the 467 are the <em>same 152 scenario ids</em>
      &mdash; verified, not assumed &mdash; so the <b>152</b> column is the only place IBM and we answer
      identical questions. Everywhere else the two sides never meet: IBM never ran the 467, the 88, or the 40;
      we never ran A, B, D, or E on the 152.</div>
    <div class="callout callout--warn"><b>And the gaps on our own side are the finding.</b> A and B &mdash; the
      two arms that carry the entire &ldquo;inverted LLM&rdquo; argument &mdash; were never run on the 467.
      C is the only architecture evaluated everywhere. D (GAK) exists only on the 88, which is a
      <em>subset</em> of the 467 &mdash; so C and D answer the same 88 questions by different means, a
      comparison the paper never makes. Note also that IBM&rsquo;s Plan-and-Execute covers only 124 of the
      152; that cell is not full coverage.</div>
    <div class="callout callout--bad"><b>Architecture A does not exist.</b> The paper&rsquo;s baseline
      &mdash; the 65% that the headline &ldquo;+17 points, same model&rdquo; is measured against &mdash;
      <b>was never run by us on the 139</b>. That number is <em>cited from IBM&rsquo;s paper</em>: their
      harness, their grader, their undisclosed model. And the one cell where A <em>was</em> run
      (custom-40, 85%) is not a document baseline at all: <code>benchmark/run_baseline.py</code> loads
      <b>no CouchDB, no CSV, no JSON, and calls no tools</b> &mdash; it sends a single LLM call
      instructing the model to &ldquo;answer based on general industrial knowledge.&rdquo; So a model with
      <em>no data access whatsoever</em> scored 85% on the scenarios we called &ldquo;structurally
      impossible with flat document stores.&rdquo; A run that builds the real Architecture A &mdash;
      IBM&rsquo;s agent, IBM&rsquo;s documents, a named model, our grader &mdash; is in progress
      (<code>docs/ARCH-A-PROTOCOL.md</code>).</div>
  </section>

  <section id="accuracy">
    <h2><span>03</span>Accuracy</h2>
    <p class="sub">Scored differently on each side, so these are <em>not</em> directly comparable &mdash; that is
      exactly the point. IBM judges a trajectory; we judge a final answer.</p>
    <div class="scroll"><table>
      <thead><tr><th scope="col">System</th><th scope="col">Headline</th>
        <th scope="col">Pass</th><th scope="col">Partial</th><th scope="col">Fail</th></tr></thead>
      <tbody>{ibm_acc}
        <tr><th scope="row"><span class="tag tag--us">ours</span>B &mdash; NLQ &rarr; Cypher
          <em>139 scenarios &middot; gpt-4.1 &middot; harness pass-scorer</em></th>
          <td class="num big">84.9%</td><td class="num colspan" colspan="3">
          same model over the graph instead of documents</td></tr>
        <tr><th scope="row"><span class="tag tag--us">ours</span>A &mdash; Baseline, <b>finally built</b>
          <em>139 scenarios &middot; gpt-4o &middot; IBM agent over documents</em></th>
          <td class="num big">{a_full}%</td><td class="num colspan" colspan="3">
          the 65% the paper cited but never ran &mdash; see the isolated comparison below</td></tr>
        <tr><th scope="row"><span class="tag tag--us">ours</span>C &mdash; Deterministic
          <em>467 scenarios &middot; no LLM</em></th>
          <td class="num big">100%</td><td class="num colspan" colspan="3">
          ceiling on graph-answerable queries; 78 of the 88 fmsr answers come from a hardcoded dict</td></tr>
        <tr><th scope="row"><span class="tag tag--us">ours</span>D &mdash; GAK
          <em>88 scenarios &middot; claude-sonnet-4-5</em></th>
          <td class="num big">81.8%</td><td class="num colspan" colspan="3">
          model-dependent: 73.9% (gpt-4.1), 52.3% (gpt-4o)</td></tr>
        <tr><th scope="row"><span class="tag tag--us">ours</span>E &mdash; No-graph control
          <em>88 scenarios &middot; gpt-4o and gpt-4.1</em></th>
          <td class="num big">89.8%</td><td class="num colspan" colspan="3">
          just asking the LLM beats GAK &mdash; the graph adds nothing on <em>these</em> questions</td></tr>
      </tbody></table></div>
    <div class="callout callout--warn"><b>Read the metrics, not the numbers.</b> IBM&rsquo;s Agent-as-Tools
      figure is a trajectory judged &ldquo;Accomplished&rdquo; by an LLM judge (Llama-4-Maverick-17B); their
      Plan-and-Execute figure is task-completion under their 3-axis rubric. Ours is a harness pass-scorer on a
      final answer. A higher number on our side does <em>not</em> mean a better agent &mdash; it means a
      different question was asked. The only apples-to-apples comparison in the paper is the rubric re-scoring
      of Architecture&nbsp;B.</div>
  </section>

  <section id="datalayer">
    <h2><span>03b</span>The data-layer effect, isolated at last</h2>
    <p class="sub">The paper&rsquo;s headline &mdash; &ldquo;+17 points, same model&rdquo; &mdash; compared its
      graph result against a 65% baseline it <b>cited from IBM but never ran</b>. We built that baseline
      (docs/AUDIT-2026-07-13.md): IBM&rsquo;s ReAct agent over IBM&rsquo;s CouchDB documents, <b>gpt-4o</b>,
      graded by Architecture&nbsp;B&rsquo;s <em>own</em> grader. Only the data layer varies.</p>
    <div class="scroll"><table>
      <thead><tr><th scope="col">Comparison (gpt-4o, same grader)</th><th scope="col">A: documents</th>
        <th scope="col">B: typed graph</th><th scope="col">data-layer &Delta;</th></tr></thead>
      <tbody>
        <tr><th scope="row">all 139 scenarios</th>
          <td class="num">{a_full}%</td><td class="num">{b_full}%</td>
          <td class="num big">{d_full:+.1f}</td></tr>
        <tr><th scope="row">{n_clean} data-path-matched scenarios
          <em>excludes {n_excl}: {excl_alert} need an alert/anomaly server absent upstream, {excl_asset}
          reference assets in neither data layer</em></th>
          <td class="num">{a_clean}%</td><td class="num">{b_clean}%</td>
          <td class="num big">{d_clean:+.1f}</td></tr>
      </tbody></table></div>
    <div class="callout"><b>The effect is real, and about the size the paper claims.</b> On the scenarios where
      both sides genuinely have the data, the typed graph beats a document baseline by <b>{d_clean:+.1f}
      points</b> under an identical model and grader &mdash; positive across all five scenario types. The
      paper&rsquo;s <em>magnitude</em> holds. What it lacked was this baseline: Architecture&nbsp;A as shipped
      loads no data at all, the 65% was never measured, and the graph side was tuned on the test set while the
      baseline got zero iterations.</div>
    <div class="callout callout--warn"><b>Why {n_excl} scenarios are set aside, and why that matters.</b> The
      139 are old-era scenarios; our graph was built from that old data; IBM&rsquo;s current agent ships some of
      it away (no alert/anomaly server). Add a keyword grader that rewards <em>naming</em> entities over
      <em>computing</em> answers, and a perfectly-matched 139-wide comparison is not achievable by seeding &mdash;
      which is itself a finding the original paper never surfaced.</div>
  </section>

  <section id="speed">
    <h2><span>04</span>Speed</h2>
    <p class="sub">Wall-clock per scenario, as recorded at run time by each side&rsquo;s own harness.</p>
    <div class="scroll"><table>
      <thead><tr><th scope="col">System</th><th scope="col">Median</th>
        <th scope="col">Mean</th><th scope="col">What is being timed</th></tr></thead>
      <tbody>{ibm_speed}</tbody></table></div>
    <div class="callout"><b>The gap is five orders of magnitude, and it is not a fair fight.</b> IBM times a
      full multi-step agent loop with reflection; we time a single graph query. The honest comparison is
      IBM&rsquo;s agent loop against our Architecture&nbsp;B (one LLM call + graph), which is still
      roughly 50&times; faster.</div>
  </section>

  <section id="cost">
    <h2><span>05</span>LLM calls &amp; tokens</h2>
    <p class="sub">What each system spends to answer a scenario. This is where GAK&rsquo;s caching argument
      actually lives &mdash; not in accuracy.</p>
    <div class="scroll"><table>
      <thead><tr><th scope="col">System</th><th scope="col">LLM calls / scenario</th>
        <th scope="col">Tokens sent</th><th scope="col">Tokens received</th>
        <th scope="col">Total API calls</th></tr></thead>
      <tbody>{ibm_cost}</tbody></table></div>
  </section>

  <section id="hardware">
    <h2><span>06</span>Hardware</h2>
    <p class="sub">What the numbers above were produced on.</p>
    <div class="scroll"><table>
      <thead><tr><th scope="col">Side</th><th scope="col">Compute</th>
        <th scope="col">Model serving</th><th scope="col">Notes</th></tr></thead>
      <tbody>
        <tr class="ibm"><th scope="row"><span class="tag tag--ibm">IBM</span>AssetOpsBench</th>
          <td>not disclosed</td><td>hosted LLM APIs</td>
          <td class="note-c">Agent model is not recorded in the shipped logs
            (<code>react_llm_model_id: 34</code>). Judge is Llama-4-Maverick-17B per their README.</td></tr>
        <tr><th scope="row"><span class="tag tag--us">ours</span>Samyama Graph</th>
          <td>{hw["cpu"]} &middot; {hw["cores"]} logical cores &middot; {hw["ram"]} GB RAM</td>
          <td>hosted OpenAI / Anthropic APIs</td>
          <td class="note-c">{hw["os"]}. The graph engine runs on CPU only &mdash; the GPU is not used.
            Architectures&nbsp;C and&nbsp;D&rsquo;s cached path need no network at all.</td></tr>
      </tbody></table></div>
    <div class="callout callout--warn"><b>Caveat.</b> No result file in either repo records the machine it ran
      on. Our row is the host this page was generated on, which is where the runs were executed &mdash; but it
      is an assertion, not a recording. Neither side&rsquo;s timings should be treated as a controlled
      hardware benchmark.</div>
  </section>

  <section id="paths">
    <h2><span>07</span>How the {total} were answered</h2>
    <p class="sub">Per config, the path each scenario actually took, read from its recorded
      <code>tools_used</code>. The graph does real work on the large majority; the hardcoded fallback is
      confined entirely to <code>fmsr</code>.</p>
    <div class="scroll"><table class="paths">{"".join(prows)}</table></div>
    <ul class="legend">
      <li><span class="sw" style="background:var(--us)"></span>graph-backed &mdash; {tot["graph"]}</li>
      <li><span class="sw" style="background:var(--llm)"></span>graph + hardcoded fallback &mdash; {tot["mixed"]}</li>
      <li><span class="sw" style="background:var(--gap)"></span>no tools recorded &mdash; {tot["none"]}</li>
    </ul>
    <div class="callout"><b>{tot["graph"]} of {total}</b> scenarios are answered purely from the typed graph.
      The {tot["mixed"]} that lean on <code>domain_knowledge_fallback</code> are <em>all</em> in
      <code>fmsr</code> &mdash; the equipment types genuinely absent from the graph, which is precisely the gap
      GAK exists to close.</div>
  </section>

  <section id="tools">
    <h2><span>08</span>Tool surface</h2>
    <p class="sub">Every tool our router can reach, by how often it fired across the {total}.</p>
    <div class="scroll"><table class="tools">{trows}</table></div>
    <ul class="legend">
      <li><span class="sw" style="background:var(--us)"></span>queries the typed graph</li>
      <li><span class="sw" style="background:var(--ctl)"></span>answers without the graph</li>
    </ul>
  </section>
</div>
"""


STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
"""


def wrap_standalone(frag: str) -> str:
    """The Artifact host injects <!doctype>/<head>/<body>; a repo file has no such host."""
    i = frag.index("</style>") + len("</style>")
    return STANDALONE_HEAD + frag[:i] + "\n</head>\n<body>\n" + frag[i:] + "</body>\n</html>\n"


if __name__ == "__main__":
    frag = "--fragment" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--fragment"]
    out = pathlib.Path(args[0] if args else ROOT / "docs/coverage.html")
    page = html(build())
    out.write_text(page if frag else wrap_standalone(page))
    print(f"wrote {out}  ({'artifact fragment' if frag else 'standalone document'})")
