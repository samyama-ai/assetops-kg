"""G-02 — freeze a held-out split of the scenario ids, reproducibly.

Stratified by scenario type, seeded, written once to
`scenarios/heldout_split.json`. The point of freezing it in a committed file is
that the held-out ids are chosen *blind* — before any evaluation — so a later run
cannot cherry-pick a flattering subset.

Honesty caveat (documented in the output): the graph tier's NLQ prompt was tuned
across the full scenario set BEFORE this split existed, so this is a held-out set
in *evaluation* order, not in *tuning* order. It tests whether the graph's
advantage is stable on a blind subset; a fully clean generalization test would
retune the NLQ prompt on the train ids only. The document baseline (Arch A)
received zero tuning iterations, so its held-out subset is already a fair
baseline.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SEED = 20260819              # fixed so the split is reproducible
HELDOUT_FRAC = 0.30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="../AssetOpsBench")
    ap.add_argument("--out", default="scenarios/heldout_split.json")
    a = ap.parse_args()

    # Freeze over the SAME 139-scenario set the runners iterate (load_ibm_scenarios),
    # so the --ids-file ids line up with run_arch_a.py / run_nlq.py exactly.
    from benchmark.run_ibm_scenarios import load_ibm_scenarios
    rows = load_ibm_scenarios(a.data_dir, None)
    by_type: dict[str, list[int]] = collections.defaultdict(list)
    for r in rows:
        by_type[r.get("scenario_type", r.get("type", "?"))].append(r["id"])

    rng = random.Random(SEED)
    heldout: list[int] = []
    for t in sorted(by_type):
        ids = sorted(by_type[t])
        rng.shuffle(ids)
        k = max(1, round(len(ids) * HELDOUT_FRAC))
        heldout.extend(ids[:k])
    heldout = sorted(heldout)
    all_ids = sorted(r["id"] for r in rows)
    train = [i for i in all_ids if i not in set(heldout)]

    out = {
        "goal": "G-02",
        "seed": SEED,
        "heldout_frac": HELDOUT_FRAC,
        "stratified_by": "type",
        "n_total": len(all_ids),
        "n_train": len(train),
        "n_heldout": len(heldout),
        "heldout_by_type": {t: sum(1 for i in by_type[t] if i in set(heldout)) for t in sorted(by_type)},
        "ids": heldout,           # the field the runners' --ids-file reads
        "train_ids": train,
        "protocol": (
            "Frozen blind (seed %d) before evaluation. Run Arch A (documents) and Arch B "
            "(graph NLQ) with --ids-file on THIS file; report the held-out delta. Caveat: the "
            "NLQ prompt was tuned on the full set before this split, so this is held-out in "
            "evaluation order, not tuning order — a fully clean test retunes NLQ on train_ids only." % SEED
        ),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"G-02 split: {len(heldout)} held-out / {len(train)} train (seed {SEED}) -> {a.out}")
    print(f"  held-out by type: {out['heldout_by_type']}")


if __name__ == "__main__":
    main()
