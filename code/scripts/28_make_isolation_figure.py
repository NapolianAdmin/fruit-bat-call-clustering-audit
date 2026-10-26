"""Figure 7 - the one discrete-looking group: pups' Isolation calls.
 A  2-D UMAP of 12,000 voiced file-level feature vectors (the same rows as 23_umap_hdbscan.py), Isolation-context files highlighted (display only; no statistic uses the 2-D map)
 B  number of HDBSCAN clusters (min size 100) and share labelled noise: real data vs the Gaussian-copula and session-mean nulls (step 3i)
 C  held-out-bat score (log-odds of Isolation) for voiced file-level features, Isolation vs other contexts (leave-bats-out logistic regression; 41,116 files)
 D  the same for syllable shape (30,000 syllables)
Run: python code/claude/28_make_isolation_figure.py      -> figures/fig7_isolation_group.png
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, FIG, RES, SEED  # noqa: E402
from clusterlib import load_set  # noqa: E402

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap


def oof(X, y, g):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return s


X, M = load_set("all2_voiced_ref1")
ii = np.sort(np.random.default_rng(SEED).choice(len(X), 12000, replace=False))
E2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=SEED).fit_transform(X[ii])
iso = (M.context_name.values[ii] == "Isolation")
raw = pd.read_csv(RES / "step3i_umap_all2_voiced_ref1_raw.csv")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]
a.scatter(E2[~iso, 0], E2[~iso, 1], s=1, c="#9aa5b1", label="other contexts", rasterized=True)
a.scatter(E2[iso, 0], E2[iso, 1], s=2, c="#d1495b", label="Isolation context", rasterized=True)
a.set_xticks([]); a.set_yticks([])
a.set_title("A  2-D UMAP of voiced file features (display only)", fontsize=9, loc="left")
a.legend(fontsize=7, markerscale=5, loc="lower right")

a = ax[0, 1]
rng = np.random.default_rng(1)
cont = pd.read_csv(RES / "step3i_umap_continuum_nulls.csv")
groups = [("copula null", raw[raw.kind == "copula"], "#4c78a8"), ("session-mean null", raw[raw.kind == "session"], "#59a14f"),
          ("curved-continuum null", cont[cont.dataset == "gradient_null"], "#f28e2b"), ("skewed null", cont[cont.dataset == "skewed_null"], "#b07aa1"),
          ("real data", raw[(raw.kind == "obs") & (raw.nn == 30)], "#d1495b"), ("real minus small cluster", cont[cont.dataset == "real_minus_small_cluster"], "#7f7f7f"),
          ("real minus Isolation files", cont[cont.dataset == "real_minus_all_Isolation_files"], "#4d4d4d")]
for j, (lab_, v, colr) in enumerate(groups):
    a.scatter(np.full(len(v), j) + rng.uniform(-.12, .12, len(v)), v.k_100, c=colr, s=26 if "real" not in lab_ else 70, marker="D" if "real" in lab_ else "o", alpha=.85)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups], fontsize=6.5, rotation=30, ha="right")
a.set_ylabel("HDBSCAN clusters (min size 100)", fontsize=8)
top = a.get_ylim()[1]
for j, (lab_, v, colr) in enumerate(groups):
    a.text(j, top * 0.97, f"{v.noise_100.mean():.0%}", ha="center", fontsize=7)
a.set_title("B  UMAP + HDBSCAN: real data and controls (top: % noise)", fontsize=9, loc="left")

for a, tag, ttl in ((ax[1, 0], "all2_voiced_ref1", "C  held-out-bat Isolation score, voiced file features"), (ax[1, 1], "segall_shape", "D  held-out-bat Isolation score, syllable shape")):
    Xs, Ms = load_set(tag)
    y = (Ms.context_name.values == "Isolation").astype(int)
    s = oof(Xs, y, Ms.bat.values)
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 60)
    a.hist(s[y == 0], bins=bins, alpha=.65, density=True, color="#9aa5b1", label="other contexts")
    a.hist(s[y == 1], bins=bins, alpha=.65, density=True, color="#d1495b", label="Isolation")
    a.set_xlabel("log-odds of Isolation (model trained on other bats)", fontsize=8)
    a.set_title(ttl, fontsize=9, loc="left")
ax[1, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig7_isolation_group.png", dpi=150)
print("saved")
