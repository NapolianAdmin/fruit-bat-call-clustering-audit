"""Step 7c - do the headline NEGATIVE claims replicate on the 12,000 never-used files?

Projection as in 33_replication_analysis.py: scaler + 50-PC PCA fitted on the discovery files only, separately for the voiced 1,408 statistics and the whole-span (draft) 1,408 statistics.
  1. FCM c=2: FPC (discovery: whole-span 0.583, voiced 0.500), k-means k=2 silhouette
  2. the whole-span two-way FCM label vs silence fraction / voiced duration / span length (AUC, orientation-free; discovery silence AUC 0.887)
  3. voiced features: the lite k=1-allowed battery of 22_power_analysis.py (prediction strength and bootstrap Jaccard across bats, split-sample dip, FPC excess, silhouette) on the replication files
     against 10 Gaussian-copula null datasets of the same size (discovery: prediction strength 0.66 vs nulls 0.95 / 0.85)
Writes results/step7_replication_negative_claims.csv.
Run: python code/claude/35_replication_negative_claims.py [prefix=repl]     (prefix "rest": all analyses on a seeded 12,000-file subsample)
"""
import importlib.util
import os
import sys
import warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FEAT, RES, SEED, log_change  # noqa: E402
from clusterlib import auc_fast, fcm_best, fuzzy_indices, silhouette_fixed  # noqa: E402

_s22 = importlib.util.spec_from_file_location("p22", HERE / "22_power_analysis.py")
p22 = importlib.util.module_from_spec(_s22)
_s22.loader.exec_module(p22)
c03 = p22.c03
OUT = FEAT / "claude"
PFX = sys.argv[1] if len(sys.argv) > 1 else "repl"
TAG = "replication" if PFX == "repl" else PFX
N_SUB = 12000


def null_task(args):
    b, X, bat = args
    rg = np.random.default_rng(SEED + 5000 + b)
    return p22.lite_stats(c03.copula_sim(c03.copula_fit(X), len(X), rg), bat, SEED + b)


def obs_task(args):
    X, bat = args
    return p22.lite_stats(X, bat, SEED)


if __name__ == "__main__":
    smeta = pd.read_csv(OUT / "set_all2_voiced_ref1_meta.csv")
    F = np.load(OUT / "all2_features.npz")
    idx = pd.read_csv(OUT / "all2_index.csv")
    pos = pd.Series(np.arange(len(idx)), index=idx.ann_row).loc[smeta.ann_row].values
    R = np.load(OUT / f"{PFX}_features.npz")
    ri = pd.read_csv(OUT / f"{PFX}_index.csv")
    ok = (~ri.FileID.duplicated().values) & (ri.too_short.values == 0) & np.isfinite(R["voiced_ref1"]).all(1)
    ri = ri[ok].reset_index(drop=True)
    ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
    ann["ann_row"] = np.arange(len(ann))
    meta = ri.merge(ann[["ann_row", "Emitter"]], on="ann_row", how="left")
    bat = meta.Emitter.abs().astype(str).values
    proj = {}
    sel = np.sort(np.random.default_rng(SEED).choice(len(meta), N_SUB, replace=False)) if len(meta) > 15000 else np.arange(len(meta))
    meta, bat = meta.iloc[sel].reset_index(drop=True), bat[sel]
    for key in ("naive_ref1", "voiced_ref1"):
        sc = StandardScaler().fit(F[key][pos])
        pca = PCA(50, svd_solver="full", random_state=SEED).fit(sc.transform(F[key][pos]))
        proj[key] = pca.transform(sc.transform(R[key][ok]))[sel]
    rows = []
    for key, name in (("naive_ref1", "whole-span 1,408 (draft features)"), ("voiced_ref1", "voiced 1,408")):
        X = proj[key]
        r = fcm_best(X, 2, seed=SEED, n_init=5)
        fpc, xb = fuzzy_indices(X, r["U"], r["V"])
        lab = r["U"].argmax(0)
        km = KMeans(2, n_init=5, random_state=SEED).fit(X)
        row = dict(representation=name, n=len(X), FPC_c2=float(fpc), silhouette_fcm=float(silhouette_fixed(X, lab)),
                   kmeans_silhouette_k2=float(silhouette_score(X, km.labels_, sample_size=3000, random_state=SEED)))
        for v in ("silence_fraction", "voiced_dur_s", "span_len_s", "snr_db"):
            if v in meta and np.isfinite(meta[v]).all():
                a = float(auc_fast(lab, meta[v].values))
                row[f"AUC_label_vs_{v}"] = max(a, 1 - a)
        rows.append(row)
        print(row, flush=True)
    # k=1-allowed lite battery on the voiced replication features, observed vs 10 copula nulls
    Xv = proj["voiced_ref1"]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 4))) as ex:
        obs_f = ex.submit(obs_task, (Xv, bat))
        nulls = list(ex.map(null_task, [(b, Xv, bat) for b in range(10)], chunksize=1))
        obs = obs_f.result()
    N = pd.DataFrame(nulls)
    bat_rows = dict(representation="voiced 1,408: k=1-allowed battery (observed | null mean | null 95th pct | null min)")
    for s in ("FPC2_excess", "Silhouette2", "PS2", "Jaccard2", "DipSplit"):
        bat_rows[s] = f"{obs[s]:.4f} | {N[s].mean():.4f} | {N[s].quantile(.95):.4f} | {N[s].min():.4f}"
    bat_rows["DipSplit_median_p_observed"] = float(obs["DipSplit_median_p"])
    rows.append(bat_rows)
    D = pd.DataFrame(rows)
    D.to_csv(RES / f"step7_{TAG}_negative_claims.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(D.T.to_string())
    log_change("Step 7c replication of the negative claims (FCM collapse, silence/duration axis, k=1-allowed battery) on 12,000 unseen files: results/step7_replication_negative_claims.csv")
