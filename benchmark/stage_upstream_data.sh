#!/usr/bin/env bash
# Build a data-dir that loads the FULL AssetOpsBench graph (~12,647 nodes).
#
# IBM's `main` branch ships only a 135-node skeleton; the 4 CSVs that add the
# ~12.5k rows (workorders, alert_events, event, anomaly_events) live on the
# `graph-scenarios` branch, at the exact paths and schema the ETL expects. This
# extracts them read-only via `git show` (never modifies the IBM checkout) into a
# staging dir that symlinks everything else from the current checkout.
#
#   ./stage_upstream_data.sh [ASSETOPS_CHECKOUT] [STAGING_DIR]
#   default: ../AssetOpsBench  ->  $TMPDIR/assetops_full_graph
#
# Then:
#   python -m benchmark.empty_graph_control --data-dir "$STAGING_DIR" ...
set -euo pipefail
A="${1:-../AssetOpsBench}"
S="${2:-${TMPDIR:-/tmp}/assetops_full_graph}"
REF="${IBM_GRAPH_REF:-graph-scenarios}"
A="$(cd "$A" && pwd)"

rm -rf "$S"; mkdir -p "$S"
# symlink every top-level entry except 'src' (we overlay src/tmp into it).
for e in "$A"/*; do b=$(basename "$e"); [ "$b" = "src" ] && continue; ln -s "$e" "$S/$b"; done
mkdir -p "$S/src"
for e in "$A"/src/*; do ln -s "$e" "$S/src/$(basename "$e")"; done
mkdir -p "$S/src/tmp/assetopsbench/sample_data" "$S/aobench/datalayer/eamlite/db/data"

git -C "$A" show "$REF:aobench/datalayer/eamlite/db/data/workorders.csv" \
    > "$S/aobench/datalayer/eamlite/db/data/workorders.csv"
for f in alert_events event anomaly_events; do
  git -C "$A" show "$REF:src/tmp/assetopsbench/sample_data/$f.csv" \
      > "$S/src/tmp/assetopsbench/sample_data/$f.csv"
done

# Scenario utterance files (also relocated to graph-scenarios) — needed by
# load_ibm_scenarios in run_nlq.py / run_ibm_scenarios.py.
mkdir -p "$S/src/tmp/assetopsbench/scenarios/single_agent" \
         "$S/src/tmp/assetopsbench/scenarios/multi_agent"
for f in single_agent/iot_utterance_meta single_agent/fmsr_utterance \
         single_agent/wo_utterance single_agent/tsfm_utterance \
         multi_agent/end2end_utterance; do
  git -C "$A" show "$REF:src/tmp/assetopsbench/scenarios/$f.json" \
      > "$S/src/tmp/assetopsbench/scenarios/$f.json" 2>/dev/null || \
      echo "  [warn] scenario file $f.json not on $REF"
done

echo "staged full-graph data-dir at: $S"
echo "upstream ref: $REF ($(git -C "$A" rev-parse --short "$REF"))"
