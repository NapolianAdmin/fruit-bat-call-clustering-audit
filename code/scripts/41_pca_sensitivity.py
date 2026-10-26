"""Step 3n - does the number of principal components (50 throughout) or the scaling change the main conclusions?

Voiced 1,408 statistics of the discovery files (all2_voiced_ref1 rows).  For n_PC in {5, 10, 20, 50, 100} (standardise + PCA fitted on all discovery files) and for two other
preprocessings at 50 PCs (no standardisation; rank-based normal scores), on a seeded 12,000-file subsample:
  FPC of fuzzy c=2 minus 0.5,  k-means (k=2) silhouette,  prediction strength across bats (k=2, 10 splits),  split-sample dip statistic (median p),
  Isolation vs other contexts: bat-blocked logistic-regression AUC (fitted on ALL 41,116 files, 5 folds);
  what the fuzzy c=2 partition follows: orientation-free AUC of its label against silence fraction, SNR, voiced duration, noise floor, and the F1 of its best cluster for Isolation calls.
Writes results/step3n_pca_sensitivity.csv.
Run: python code/claude/41_pca_sensitivity.py
"""
import importlib.util
import os
import sys
import warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FEAT, RES, SEED, log_change  # noqa: E402
from clusterlib import auc_fast, fcm_best, fuzzy_indices  # noqa: E402

_spec = importlib.util.spec_from_file_location("c03", HERE / "03c_clusterability_old_matrix.py")
c03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c03)
OUT = FEAT / "claude"

smeta = pd.read_csv(OUT / "set_all2_voiced_ref1_meta.csv")
F = np.load(OUT / "all2_features.npz")
idx = pd.read_csv(OUT / "all2_index.csv")
pos = pd.Series(np.arange(len(idx)), index=idx.ann_row).loc[smeta.ann_row].values
M = F["voiced_ref1"][pos].astype(np.float64)
bat = smeta.bat.values
y = (smeta.context_name == "Isolation").astype(int).values
sub = np.sort(np.random.default_rng(SEED).choice(len(M), 12000, replace=False))


def isolation_auc(X):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, bat):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.5, class_weight="balanced")).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return float(auc_fast(y, s))


def prep(kind, n_pc):
    if kind == "standardise":
        Z = StandardScaler().fit_transform(M)
    elif kind == "no scaling":
        Z = M - M.mean(0)
    else:  # rank-based normal scores per feature
        Z = np.column_stack([norm.ppf((rankdata(M[:, j]) - 0.5) / len(M)) for j in range(M.shape[1])])
    return PCA(n_pc, svd_solver="randomized", random_state=SEED).fit_transform(Z)


rows = []
for kind, n_pc in [("standardise", 5), ("standardise", 10), ("standardise", 20), ("standardise", 50), ("standardise", 100), ("no scaling", 50), ("rank normal scores", 50)]:
    X = prep(kind, n_pc)
    Xs, bs = X[sub], bat[sub]
    r = fcm_best(Xs, 2, seed=SEED, n_init=3)
    fpc, _ = fuzzy_indices(Xs, r["U"], r["V"])
    km = KMeans(2, n_init=5, random_state=SEED).fit(Xs)
    ps = c03.pred_strength(Xs, bs, (2,), SEED, S=10)[2]
    d, pmed, pfrac = c03.split_dip(Xs, bs, SEED, S=10)
    lab = r["U"].argmax(0)
    cov = {}
    for cname in ("silence_fraction", "snr_db", "voiced_dur_s", "noise_floor_dbfs"):
        if cname in smeta and smeta[cname].notna().all():
            a_ = float(auc_fast(lab, smeta[cname].values[sub]))
            cov[f"AUC_label_vs_{cname}"] = max(a_, 1 - a_)
    f1s = []
    for c_ in (0, 1):
        m_ = lab == c_
        p_, r_ = y[sub][m_].mean() if m_.any() else 0, y[sub][m_].sum() / max(1, y[sub].sum())
        f1s.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ > 0 else 0)
    cov["isolation_F1_best_cluster"] = float(max(f1s))
    row = dict(preprocessing=kind, n_pc=n_pc, FPC2_excess=float(fpc - 0.5), kmeans_silhouette_k2=float(silhouette_score(Xs, km.labels_, sample_size=3000, random_state=SEED)),
               prediction_strength_k2=float(ps), dip_stat=float(d), dip_median_p=float(pmed), isolation_AUC_bat_blocked=isolation_auc(X), **cov)
    rows.append(row)
    print(row, flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "step3n_pca_sensitivity.csv", index=False)
log_change("Step 3n sensitivity of the voiced-feature conclusions to the number of PCs and the preprocessing: results/step3n_pca_sensitivity.csv")
pd.set_option("display.width", 250)
print(R.round(4).to_string())
