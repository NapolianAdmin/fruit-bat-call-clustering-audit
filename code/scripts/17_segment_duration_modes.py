"""Step 6b - is syllable DURATION discrete or continuous, and does shape add context information beyond duration?

  duration modality   Hartigan dip on log10 segment duration (all segments, then per context), Gaussian-mixture BIC k=1..6 on log10 duration,
                      with a bat-level bootstrap of the dip statistic
  context decoding    leave-bats-out (GroupKFold 5) balanced accuracy for context from: log duration only | shape PCs | shape PCs + log duration
Writes results/step6_duration_modes.csv, step6_duration_gmm.csv, step6_context_decode_duration_vs_shape.csv, figures/fig5_segment_duration.png.
Run: python code/claude/17_segment_duration_modes.py <seg_prefix> <set_name>      e.g.  segall segall
"""
import sys
import warnings
from pathlib import Path

import diptest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, FIG, RES, SEED, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

warnings.filterwarnings("ignore")
prefix, name = sys.argv[1], sys.argv[2]
si = pd.read_csv(FEAT / "claude" / f"{prefix}_seg_index.csv")
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
ann["ann_row"] = np.arange(len(ann))
m = si.merge(ann[["ann_row", "Emitter", "context_name"]], on="ann_row", how="left")
m["ld"] = np.log10(m.dur_s)
m["bat"] = m.Emitter.abs().astype(str)

rows = []


def dip_row(label, d):
    x = d.ld.values
    dp, pv = diptest.diptest(x)
    rg = np.random.default_rng(SEED)
    bats = d.bat.unique()
    idx = {b: np.where(d.bat.values == b)[0] for b in bats}
    bs = []
    for _ in range(100):
        ii = np.concatenate([idx[b] for b in rg.choice(bats, len(bats), replace=True)])
        bs.append(diptest.diptest(x[ii])[0])
    rows.append(dict(subset=label, n_segments=len(x), n_bats=len(bats), dip=dp, dip_p_iid=pv, dip_bat_boot_q025=np.quantile(bs, .025), dip_bat_boot_q975=np.quantile(bs, .975),
                     median_dur_ms=1000 * float(np.median(d.dur_s)), q05_ms=1000 * float(d.dur_s.quantile(.05)), q95_ms=1000 * float(d.dur_s.quantile(.95))))


dip_row("all segments", m)
for c in ["Isolation", "Fighting", "Sleeping", "General", "Feeding", "Mating protest"]:
    d = m[m.context_name == c]
    if len(d) > 500:
        dip_row(f"context = {c}", d)
pd.DataFrame(rows).to_csv(RES / "step6_duration_modes.csv", index=False)

gm_rows = []
x = m.ld.values.reshape(-1, 1)
for k in range(1, 7):
    g = GaussianMixture(k, n_init=3, random_state=SEED).fit(x)
    gm_rows.append(dict(k=k, BIC=g.bic(x), means_ms=str(np.round(1000 * 10 ** np.sort(g.means_.ravel()), 1).tolist()), weights=str(np.round(g.weights_[np.argsort(g.means_.ravel())], 3).tolist())))
G = pd.DataFrame(gm_rows)
G["delta_BIC_vs_best"] = G.BIC - G.BIC.min()
G.to_csv(RES / "step6_duration_gmm.csv", index=False)

# decoding: duration only vs shape vs shape + duration (leave-bats-out), on the capped segment set
X, u = load_set(f"{name}_shape")
y, g_ = u.context_name.values, u.bat.values
feats = {"log duration only": u[["log10_dur"]].values, "shape PCs only (duration-free)": X, "shape PCs + log duration": np.column_stack([X, u.log10_dur.values])}
dec = []
for k, F in feats.items():
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(F, y, g_):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5)).fit(F[tr], y[tr])
        ok = np.isin(y[te], mdl.classes_)
        pred[te] = np.where(ok, mdl.predict(F[te]), "")
    keep = pred != ""
    dec.append(dict(features=k, n=int(keep.sum()), balanced_accuracy=balanced_accuracy_score(y[keep], pred[keep].astype(str)), accuracy=float((pred[keep] == y[keep]).mean()),
                    majority_baseline=float(pd.Series(y[keep]).value_counts(normalize=True).max()), classes=len(np.unique(y))))
pd.DataFrame(dec).to_csv(RES / "step6_context_decode_duration_vs_shape.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(m.ld, bins=120, color="#4c78a8", density=True)
xs = np.linspace(m.ld.min(), m.ld.max(), 400).reshape(-1, 1)
g2 = GaussianMixture(2, n_init=3, random_state=SEED).fit(x)
ax[0].plot(xs, np.exp(GaussianMixture(1).fit(x).score_samples(xs)), color="k", lw=1, label="1 Gaussian")
ax[0].plot(xs, np.exp(g2.score_samples(xs)), color="#e45756", lw=1.3, label="2 Gaussians")
ax[0].set_xlabel("log10 voiced-segment duration (s)")
ax[0].set_ylabel("density")
ax[0].legend(fontsize=7)
ax[0].set_title(f"A. Syllable duration, {len(m):,} segments", fontsize=9)
for c, col in [("Isolation", "#e45756"), ("Fighting", "#59a14f"), ("Sleeping", "#4c78a8"), ("General", "#9d9d9d")]:
    d = m[m.context_name == c].ld
    if len(d) > 200:
        ax[1].hist(d, bins=60, density=True, histtype="step", color=col, label=f"{c} (n={len(d):,})")
ax[1].set_xlabel("log10 voiced-segment duration (s)")
ax[1].legend(fontsize=7)
ax[1].set_title("B. By context", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig5_segment_duration.png", dpi=160)
log_change(f"Step 6b segment duration modes + duration-vs-shape decoding on '{prefix}' / '{name}'")
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(4).to_string())
print(G.to_string())
print(pd.DataFrame(dec).round(3).to_string())
