"""Step 2b - turn raw extraction output into analysis sets: standardise + PCA(50), IDENTICAL rows in every set.

Unit = one annotated FILE (first annotation row of each FileID). Rows are dropped for all sets together when the voiced
statistics could not be computed (<3 voiced frames), so naive / voiced / MFCC sets are strictly comparable.
Writes data/features/claude/set_<prefix>_<key>_X.npy (N x 50), set_<prefix>_<key>_meta.csv and results/step2_sets_summary_<prefix>.csv.
Tags for load_set(): '<prefix>_<key>', e.g. baseline_voiced_ref1.

Run: python code/claude/05_build_sets.py <prefix> [key ...]      e.g.  05_build_sets.py baseline
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, RES, SEED, log_change  # noqa: E402

OUT = FEAT / "claude"
prefix = sys.argv[1]
if prefix.startswith("sub:"):  # e.g. sub:aves -> the 'all' features restricted to the files of aves_index.csv (same order); prefix 'avessub'
    ref = prefix[4:]
    base_idx = pd.read_csv(OUT / "all_index.csv")
    base_F = np.load(OUT / "all_features.npz")
    want = pd.read_csv(OUT / f"{ref}_index.csv").FileID  # join on FileID: ann_row can differ for files with several annotation rows
    pos = pd.Series(np.arange(len(base_idx)), index=base_idx.FileID).groupby(level=0).first().loc[want].values
    F = {k: base_F[k][pos] for k in base_F.files}
    idx = base_idx.iloc[pos].reset_index(drop=True)
    prefix = f"{ref}sub"
elif "+" in prefix:  # e.g. baseline+main -> merged prefix 'all' (rows concatenated, duplicate annotation rows dropped)
    parts = prefix.split("+")
    Fs = [np.load(OUT / f"{p}_features.npz") for p in parts]
    Is = [pd.read_csv(OUT / f"{p}_index.csv") for p in parts]
    common = [k for k in Fs[0].keys() if all(k in f for f in Fs)]
    F = {k: np.concatenate([f[k] for f in Fs]) for k in common}
    idx = pd.concat(Is, ignore_index=True)
    dup = idx.ann_row.duplicated().values
    F = {k: v[~dup] for k, v in F.items()}
    idx = idx[~dup].reset_index(drop=True)
    prefix = "all2" if "main2" in parts else "all"  # 'all2' = baseline + main + main2 (41k files); never overwrites 'all'
    idx.to_csv(OUT / f"{prefix}_index.csv", index=False)
    np.savez_compressed(OUT / f"{prefix}_features.npz", **F)
else:
    F = np.load(OUT / f"{prefix}_features.npz")
    idx = pd.read_csv(OUT / f"{prefix}_index.csv")
keys = sys.argv[2:] or list(F.keys())
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz", parse_dates=["rec_time"])
ann["ann_row"] = np.arange(len(ann))
keep_cols = ["ann_row", "Emitter", "Addressee", "Context", "context_name", "channel", "month", "date", "folder", "treatment_id", "rec_time"]
meta = idx.merge(ann[keep_cols], on="ann_row", how="left")
assert len(meta) == len(idx) and meta.channel.notna().all()

# one row per file, and a common validity mask
first = ~meta.FileID.duplicated().values
ok = first & (meta.too_short.values == 0)
for k in keys:
    if k.startswith(("voiced", "mfcc_voiced")):
        ok &= np.isfinite(F[k]).all(1)
meta = meta[ok].reset_index(drop=True)
meta["bat"] = meta.Emitter.abs().astype(str)
meta["stratum"] = meta.channel.astype(str) + "|" + meta.month
meta["em_class"] = np.where(meta.Emitter == 0, "zero", np.where(meta.Emitter < 0, "negative", "positive"))
meta["days_since_start"] = (meta.rec_time - pd.Timestamp("2012-06-01")).dt.total_seconds() / 86400
meta["channel"] = meta.channel.astype(str)
meta["treatment_id"] = meta.treatment_id.astype(str)
print(f"{len(idx)} extracted rows -> {len(meta)} unique files kept ({int((idx.too_short == 1).sum())} too short)", flush=True)

rows = []
for k in keys:
    M = F[k][ok]
    Z = StandardScaler().fit_transform(M)
    nc = min(50, M.shape[1], len(M) - 1)
    pca = PCA(n_components=nc, svd_solver="full", random_state=SEED)
    X = pca.fit_transform(Z)
    np.save(OUT / f"set_{prefix}_{k}_X.npy", X.astype(np.float32))
    meta.to_csv(OUT / f"set_{prefix}_{k}_meta.csv", index=False)
    rows.append(dict(set=f"{prefix}_{k}", N=len(M), d=M.shape[1], n_pcs=nc, explained_variance=float(pca.explained_variance_ratio_.sum()),
                     pc1_share=float(pca.explained_variance_ratio_[0])))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / f"step2_sets_summary_{prefix}.csv", index=False)
log_change(f"Step 2b analysis sets built from '{prefix}': {len(meta)} files; sets {', '.join(keys)}")
