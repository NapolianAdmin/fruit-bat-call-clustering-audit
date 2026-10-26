"""Step 2a - sample design for the main audit (~15,000 annotated files, one annotation row per file).

Unit = annotated FILE (first annotation row of each FileID). The 2,360 baseline files are always included.
 1. floor: up to 20 files in every non-empty channel x month cell (seed 42)
 2. fill: uniform random draw from the remaining pool up to the target (== proportional allocation over cells)
 3. top-ups: every bat with >= 100 positive-emitter annotations in the corpus gets >= 100 sampled positive-emitter files
    (or all it has); every context gets >= 150 files (or all it has)
Writes data/features/claude/main_manifest.csv (NEW files only, for 02_extract_features.py), sample_all_manifest.csv (incl. baseline)
and results/step2_sample_design_*.csv / report.
Run: python code/claude/02a_sample_design.py [target_total=15000]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, RES, SEED, log_change  # noqa: E402

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
rg = np.random.default_rng(SEED)
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
ann["ann_row"] = np.arange(len(ann))
files = ann.drop_duplicates("FileID").copy()
files["cell"] = files.channel.astype(str) + "|" + files.month
files["em_class"] = np.where(files.Emitter == 0, "zero", np.where(files.Emitter < 0, "negative", "positive"))
base_ids = set(ann.loc[ann.ann_row.isin(pd.read_csv(FEAT / "claude" / "baseline_manifest.csv").ann_row), "FileID"])
EXTEND = len(sys.argv) > 2 and sys.argv[2] == "extend"  # keep every previously sampled file, add more (bigger bat/context minimums)
SUF = "2" if EXTEND else ""
if EXTEND:
    base_ids = set(pd.read_csv(FEAT / "claude" / "sample_all_manifest.csv").FileID)
NEED_BAT, NEED_CTX = (200, 500) if EXTEND else (100, 150)
files["in_baseline"] = files.FileID.isin(base_ids)
chosen = set(files.loc[files.in_baseline, "FileID"])
pool = files[~files.in_baseline]

for cell, g in pool.groupby("cell"):  # 1. floor
    have = files[(files.cell == cell) & files.FileID.isin(chosen)].shape[0]
    chosen |= set(rg.choice(g.FileID.values, min(max(0, 20 - have), len(g)), replace=False))
rest = pool[~pool.FileID.isin(chosen)]  # 2. fill
chosen |= set(rg.choice(rest.FileID.values, max(0, min(TARGET - len(chosen), len(rest))), replace=False))


def top_up(mask, need):
    global chosen
    have = files[mask & files.FileID.isin(chosen)].shape[0]
    avail = files[mask & ~files.FileID.isin(chosen)]
    add = min(max(0, need - have), len(avail))
    if add:
        chosen |= set(rg.choice(avail.FileID.values, add, replace=False))
    return have, add


tops = []
posc = files[files.Emitter > 0].Emitter.value_counts()
for b in posc[posc >= 100].index:  # 3a. bats
    tops.append((f"bat {b}",) + top_up((files.Emitter == b).values, NEED_BAT))
for c in files.context_name.unique():  # 3b. contexts
    tops.append((f"context {c}",) + top_up((files.context_name == c).values, NEED_CTX))
S = files[files.FileID.isin(chosen)].copy()
S.to_csv(FEAT / "claude" / f"sample_all{SUF}_manifest.csv", index=False, columns=["ann_row", "FileID", "in_baseline", "channel", "month", "folder", "Emitter", "context_name"])
S[~S.in_baseline][["ann_row", "FileID"]].to_csv(FEAT / "claude" / f"main{SUF}_manifest.csv", index=False)
pd.DataFrame(tops, columns=["group", "had_before_topup", "added"]).to_csv(RES / f"step2_sample_design{SUF}_topups.csv", index=False)

rows = []
for k, col in [("channel", "channel"), ("month", "month"), ("folder", "folder"), ("em_class", "em_class"), ("context", "context_name")]:
    a, b = S[col].astype(str).value_counts(), files[col].astype(str).value_counts()
    for lvl in b.index:
        rows.append(dict(variable=k, level=lvl, corpus_files=int(b[lvl]), sampled_files=int(a.get(lvl, 0)), sampled_share=a.get(lvl, 0) / b[lvl]))
pd.DataFrame(rows).to_csv(RES / f"step2_sample_design{SUF}_balance.csv", index=False)
pb = S[S.Emitter > 0].Emitter.value_counts().rename("sampled_positive_files").to_frame()
pb["corpus_positive_files"] = posc
pb.to_csv(RES / f"step2_sample_design{SUF}_per_bat.csv")
n_new = int((~S.in_baseline).sum())
rep = [f"# Sample design (seed {SEED})", f"- total files {len(S)} (baseline {int(S.in_baseline.sum())}, new {n_new}); target {TARGET}",
       f"- channels {S.channel.nunique()}/12, months {S.month.nunique()}, archives {S.folder.nunique()}/30, channel x month cells {S.cell.nunique()}",
       f"- positive-emitter files {int((S.Emitter > 0).sum())}, negative {int((S.Emitter < 0).sum())}, zero {int((S.Emitter == 0).sum())}",
       f"- estimated transfer for the new files: {n_new * 0.47 / 1000:.1f} GB (0.47 MB/file measured on the baseline), ~{n_new / 6.3 / 60:.0f} min at 6.3 files/s"]
(RES / f"step2_sample_design{SUF}_report.md").write_text("\n".join(rep), encoding="utf-8")
log_change(f"Step 2a sample design: {len(S)} files ({n_new} new); see results/step2_sample_design_*")
print("\n".join(rep))
print(pd.DataFrame(tops, columns=["group", "had", "added"]).query("added>0").to_string())
