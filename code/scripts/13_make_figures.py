"""Memo figures, built only from results/*.csv (no recomputation).
 fig1_what_the_draft_pipeline_clusters_on.png  naive vs voiced: covariate AUCs and held-out-bat prediction of the cluster label
 fig2_fpc_excess_over_one_cluster.png          FPC - 1/c at c=2: simulated single populations vs real matrices / sets
 fig3_emitter_id_random_vs_blocked.png         emitter identification (36 bats) by representation and split
 fig4_isolation_cluster.png                    voiced-MFCC minority cluster: share by context, pup vs adult
Run: python code/claude/13_make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIG, RES  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY, GREEN = "#4c78a8", "#e45756", "#9d9d9d", "#59a14f"

# ---------------- fig 1
nv = pd.read_csv(RES / "step3a_all2_naive_ref1_continuous_auc.csv")
vo = pd.read_csv(RES / "step3a_all2_voiced_ref1_continuous_auc.csv")
pn = pd.read_csv(RES / "step3a_all2_naive_ref1_predictive_auc.csv")
pv = pd.read_csv(RES / "step3a_all2_voiced_ref1_predictive_auc.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
cov = [c for c in ["silence_fraction", "voiced_dur_s", "span_len_s", "n_voiced_seg", "noise_floor_dbfs", "snr_db", "days_since_start"] if c in set(nv.covariate)]
y = np.arange(len(cov))
for off, d, col, lab in [(-0.18, nv, RED, "draft pipeline (whole span, FCM labels)"), (0.18, vo, BLUE, "voiced frames only (forced k-means split)")]:
    dd = d.set_index("covariate").loc[cov]
    up = dd.AUC_for_cluster1 >= .5
    a = np.where(up, dd.AUC_for_cluster1, 1 - dd.AUC_for_cluster1)
    lo = np.where(up, dd.ci95_lo, 1 - dd.ci95_hi)
    hi = np.where(up, dd.ci95_hi, 1 - dd.ci95_lo)
    ax[0].barh(y + off, a, height=.34, color=col, label=lab)
    ax[0].errorbar(a, y + off, xerr=[a - lo, hi - a], fmt="none", ecolor="k", lw=.8)
ax[0].set_yticks(y)
ax[0].set_yticklabels(cov)
ax[0].axvline(.5, color="k", ls="--", lw=.8)
ax[0].set_xlim(.4, 1)
ax[0].set_xlabel("AUC of the covariate for the two-way label (orientation-free; 0.5 = none)")
ax[0].legend(fontsize=7, loc="lower right")
ax[0].set_title("A. Both splits track duration / silence / SNR", fontsize=9)
names = ["silence / length variables", "SNR + noise floor + clipping", "channel only", "all recording variables (no context, no emitter)", "context only"]
yy = np.arange(len(names))
for off, d, col in [(-0.18, pn, RED), (0.18, pv, BLUE)]:
    d = d[d.scheme == "leave-bats-out"].set_index("predictors").loc[names]
    ax[1].barh(yy + off, d.AUC, height=.34, color=col)
    ax[1].errorbar(d.AUC, yy + off, xerr=[d.AUC - d.ci95_lo, d.ci95_hi - d.AUC], fmt="none", ecolor="k", lw=.8)
ax[1].set_yticks(yy)
ax[1].set_yticklabels([n.replace(" (no context, no emitter)", "") for n in names])
ax[1].axvline(.5, color="k", ls="--", lw=.8)
ax[1].set_xlim(.4, 1)
ax[1].set_xlabel("held-out-bat AUC predicting the label from these variables only")
ax[1].set_title("B. Recording variables alone predict the label on unseen bats", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig1_what_the_draft_pipeline_clusters_on.png", dpi=170)

# ---------------- fig 2
sim = pd.read_csv(RES / "step3d_control_summary.csv")
items = [(f"simulated one-population\n({r.kind})", r.FPC_c2_mean - .5, GREY) for r in sim.itertuples() if r.kind in ("gauss", "t5", "skewed", "gradient")]
items.append(("draft matrix\n(saved array)", 0.5786 - .5, RED))
M = pd.read_csv(RES / "step3b_methods_all2.csv")
for s, lab, col in [("all2_naive_ref1", "naive features\n(re-extracted)", RED), ("all2_voiced_ref1", "voiced frames\n(1,408)", BLUE), ("all2_mfcc_voiced", "voiced MFCC", BLUE)]:
    items.append((lab, float(M[(M.set == s) & (M.method == "FCM") & (M.k == 2)].FPC_excess_over_1_over_c.iloc[0]), col))
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.bar(range(len(items)), [i[1] for i in items], color=[i[2] for i in items])
ax.set_xticks(range(len(items)))
ax.set_xticklabels([i[0] for i in items], fontsize=7, rotation=35, ha="right")
ax.set_ylabel("FPC at c=2 minus 0.5\n(0 = the one-cluster solution)")
ax.set_title("Fuzzy c-means finds structure only in whole-span features; voiced-frame sets collapse to the one-cluster value", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_fpc_excess_over_one_cluster.png", dpi=170)

# ---------------- fig 3
D = pd.read_csv(RES / "step3b_decode_all2.csv")
D = D[D.task == "emitter(+)"]
sets = [("all2_naive_ref1", "naive 1,408"), ("all2_voiced_ref1", "voiced 1,408"), ("all2_mfcc_naive", "MFCC naive"), ("all2_mfcc_voiced", "MFCC voiced")]
splits = [("random 5-fold", BLUE), ("blocked by day", GREEN), ("blocked by channel", RED)]
fig, ax = plt.subplots(figsize=(8, 3.6))
w = .26
for j, (sp, col) in enumerate(splits):
    v = [float(D[(D.set == s) & (D.split == sp)].accuracy.iloc[0]) for s, _ in sets]
    ax.bar(np.arange(len(sets)) + (j - 1) * w, v, w, color=col, label=sp)
mb = float(D[D.split == "random 5-fold"].majority_baseline.iloc[0])
ax.axhline(mb, color="k", ls="--", lw=.8)
ax.text(len(sets) - .5, mb + .01, "majority class (random split)", ha="right", fontsize=7)
ax.set_xticks(range(len(sets)))
ax.set_xticklabels([n for _, n in sets])
ax.set_ylabel(f"accuracy, {int(D.classes.iloc[0])} bats (positive codes)")
ax.legend(fontsize=8)
ax.set_title("Emitter identification collapses when whole channels are held out", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig3_emitter_id_random_vs_blocked.png", dpi=170)

# ---------------- fig 4
ct = pd.read_csv(RES / "step3e_all2_mfcc_voiced_context_share.csv").sort_values("share_minority")
pa = pd.read_csv(RES / "step3e_all2_mfcc_voiced_pup_adult_by_isolation.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
ax[0].barh(ct.context_name, ct.share_minority, color=[RED if c == "Isolation" else BLUE for c in ct.context_name])
ax[0].set_xlabel("share of files in the small voiced-MFCC cluster (17% of all files)")
ax[0].set_title("A. The small cluster is the Isolation context", fontsize=9)
lab = [f"{r.status.split(' ')[0]}\n{r.isolation_context}" for r in pa.itertuples()]
ax[1].bar(range(len(pa)), pa.share_minority, color=[RED if "Isolation" == r.isolation_context else BLUE for r in pa.itertuples()])
ax[1].set_xticks(range(len(pa)))
ax[1].set_xticklabels(lab, fontsize=7)
for i, r in enumerate(pa.itertuples()):
    ax[1].text(i, r.share_minority + .02, f"n={r.files}", ha="center", fontsize=7)
ax[1].set_ylabel("share in the small cluster")
ax[1].set_title("B. Pups' non-Isolation calls: 17%; adults': 3%", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig4_isolation_cluster.png", dpi=170)
print("figures written:", sorted(p.name for p in FIG.glob("fig*.png")))
