"""Step 3i (harder controls) - is the small UMAP cluster a discontinuity of the real data, or would a smooth continuum with the same variance spectrum also produce one?

The Gaussian-copula and session-mean nulls of 23_umap_hdbscan.py have no manifold structure; a reviewer can object that UMAP fragments them for that reason.  Two harder controls, run through the
IDENTICAL pipeline (UMAP 10-D, 30 neighbours, min_dist 0, HDBSCAN min size 30/100/300) on 12,000 rows:
  gradient   ONE skewed latent factor drives the leading PCs through t, He2(t), He3(t) - a curved, graded 1-D continuum - with the per-PC variances of the real voiced set (as in 03d); 6 datasets
  skewed     per-PC exp(0.6 z) marginals re-standardised to the real variances; 3 datasets
  real minus the small cluster (the 770 files of 26_umap_cluster_dissection.py): does anything else separate?
  real minus ALL Isolation-context files (label-based removal): same question
Writes results/step3i_umap_continuum_nulls.csv.
Run: python code/claude/34_umap_continuum_null.py
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

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FEAT, RES, SEED, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

_spec = importlib.util.spec_from_file_location("u23", HERE / "23_umap_hdbscan.py")
u23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u23)


def make(kind, var, n, r):
    d = len(var)
    z = r.standard_normal((n, d))
    if kind == "skewed":
        y = np.exp(0.6 * z)
        y = (y - y.mean(0)) / y.std(0)
        return y * np.sqrt(var)
    t = np.exp(0.5 * r.standard_normal(n))  # gradient: a curved, skewed 1-D continuum
    t = (t - t.mean()) / t.std()
    h = np.stack([t, (t ** 2 - 1) / np.sqrt(2), (t ** 3 - 3 * t) / np.sqrt(6)])
    h = (h - h.mean(1, keepdims=True)) / h.std(1, keepdims=True)
    rho = np.zeros(d)
    rho[:6] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    sig = np.zeros((n, d))
    for k in range(6):
        sig[:, k] = rho[k] * h[k % 3]
    return (sig + np.sqrt(1 - rho ** 2) * z) * np.sqrt(var)


def job(args):
    name, b, X = args
    E = u23.embed(X, 30, seed=SEED + b)
    row = dict(dataset=name, b=b, n=len(X))
    for m in u23.MCS:
        lab, s = u23.cluster_stats(E, m)
        sizes = sorted([int((lab == c).sum()) for c in set(lab[lab >= 0])], reverse=True)
        row.update({f"k_{m}": s["k"], f"noise_{m}": s["noise"], f"sil_{m}": s["sil"], f"sizes_{m}": str(sizes[:6])})
    return row


if __name__ == "__main__":
    X, M = load_set("all2_voiced_ref1")
    ii = np.sort(np.random.default_rng(SEED).choice(len(X), 12000, replace=False))
    X0, M0 = X[ii], M.iloc[ii].reset_index(drop=True)
    var = X.var(0)
    lab = np.load(FEAT / "claude" / "umap_labels_all2_voiced_ref1_mcs100.npy")
    small = min(set(lab[lab >= 0]), key=lambda c: (lab == c).sum())
    iso = (M0.context_name == "Isolation").values
    jobs = [("gradient_null", b, make("gradient", var, 12000, np.random.default_rng(900 + b))) for b in range(6)]
    jobs += [("skewed_null", b, make("skewed", var, 12000, np.random.default_rng(950 + b))) for b in range(3)]
    jobs += [("real_minus_small_cluster", 0, X0[lab != small]), ("real_minus_all_Isolation_files", 0, X0[~iso])]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 3))) as ex:
        res = list(ex.map(job, jobs, chunksize=1))
    D = pd.DataFrame(res)
    D.to_csv(RES / "step3i_umap_continuum_nulls.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(D.to_string())
    log_change("Step 3i harder controls: UMAP+HDBSCAN on curved-continuum / skewed nulls and on the real data minus the small cluster / minus Isolation files: results/step3i_umap_continuum_nulls.csv")
