"""Step 6a - syllable-level (per voiced segment) analysis sets.

Unit = one voiced segment (>= 3 frames, FileInfo segment inside the annotation span). Segments inherit the annotation's emitter / context /
channel / date. Three representations (all standardised, PCA <= 50), built for the SAME randomly sampled segments:
  shape   512 = log-mel pooled to 32 bands x 16 time bins, mean level removed: duration- and loudness-free spectro-temporal SHAPE
  mfcc    19 MFCC means (coefficient 0 = level dropped)
  spec    log10 duration, SNR, log10 centroid, log10 bandwidth, log10 peak frequency, log10 rolloff, log10 flatness (7 scalar descriptors)
Duration is deliberately NOT part of 'shape' and 'mfcc', so structure found there cannot be a duration artifact.
Writes data/features/claude/set_<name>_<rep>_X.npy / _meta.csv (tags '<name>_shape', '<name>_mfcc', '<name>_spec').
Run: python code/claude/16_build_segment_sets.py <seg_prefix> <name> [cap=30000]
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
prefix, name = sys.argv[1], sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
si = pd.read_csv(OUT / f"{prefix}_seg_index.csv")
A = np.load(OUT / f"{prefix}_seg_arrays.npz")
shape_all, mfcc_all = A["shape"], A["mfcc19"]
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz", parse_dates=["rec_time"])
ann["ann_row"] = np.arange(len(ann))
cols = ["ann_row", "Emitter", "context_name", "channel", "month", "date", "folder", "treatment_id", "rec_time"]
meta = si.merge(ann[cols], on="ann_row", how="left")
assert len(meta) == len(si) and meta.channel.notna().all()
print(f"{len(si)} segments from {si.ann_row.nunique()} annotation rows, {si.FileID.nunique()} files", flush=True)

ok = np.isfinite(shape_all).all(1) & np.isfinite(mfcc_all).all(1) & np.isfinite(si.snr_db.values) & (si.dur_s.values > 0)
rg = np.random.default_rng(SEED)
pool = np.where(ok)[0]
pick = np.sort(rg.choice(pool, min(cap, len(pool)), replace=False))
meta = meta.iloc[pick].reset_index(drop=True)
meta["bat"] = meta.Emitter.abs().astype(str)
meta["stratum"] = meta.channel.astype(str) + "|" + meta.month
meta["em_class"] = np.where(meta.Emitter == 0, "zero", np.where(meta.Emitter < 0, "negative", "positive"))
meta["days_since_start"] = (meta.rec_time - pd.Timestamp("2012-06-01")).dt.total_seconds() / 86400
meta["channel"] = meta.channel.astype(str)
meta["treatment_id"] = meta.treatment_id.astype(str)
meta["span_len_s"] = meta.dur_s  # segment duration (the 'length' covariate at this level)
meta["log10_dur"] = np.log10(meta.dur_s)

reps = {"shape": shape_all[pick], "mfcc": mfcc_all[pick],
        "spec": np.column_stack([np.log10(si.dur_s.values[pick]), si.snr_db.values[pick]] +
                                [np.log10(np.clip(si[c].values[pick], 1e-9, None)) for c in ("centroid_hz", "bandwidth_hz", "peak_hz", "rolloff_hz", "flatness")])}
rows = []
for rep, M in reps.items():
    Z = StandardScaler().fit_transform(M)
    nc = min(50, Z.shape[1])
    pca = PCA(nc, svd_solver="full", random_state=SEED)
    X = pca.fit_transform(Z)
    tag = f"{name}_{rep}"
    np.save(OUT / f"set_{tag}_X.npy", X.astype(np.float32))
    meta.to_csv(OUT / f"set_{tag}_meta.csv", index=False)
    rows.append(dict(set=tag, N=len(X), d=M.shape[1], n_pcs=nc, explained_variance=float(pca.explained_variance_ratio_.sum()), pc1_share=float(pca.explained_variance_ratio_[0])))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / f"step6_segment_sets_{name}.csv", index=False)
log_change(f"Step 6a segment sets '{name}': {len(meta)} of {int(ok.sum())} valid segments (cap {cap}); representations shape/mfcc/spec")
