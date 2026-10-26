"""Version-4 manuscript figures: the same data, plot types, colours, panel layout and legends as the originals, redrawn at the final printed size.

Reads ONLY the saved result files in results/ (no analysis is rerun) and writes vector PDFs that are included at natural size
(text width 17 cm = 6.69 in; axis text 7-8 pt, panel letters 9 pt):
  fig1_v4.pdf  <- 13_make_figures.py, figure 1 (step3a_all2_*_continuous/predictive_auc.csv)
  fig2_v4.pdf  <- 22_power_analysis.py, power figure (step3h_power_all2_voiced_ref1_power.csv)
  fig3_v4.pdf  <- 38_all_methods.py summary() (step3m_all_methods_*.csv)
  fig6_v4.pdf  <- 13_make_figures.py, figure 3 (step3b_decode_all2.csv)
Not redrawn here: the Isolation figure (28_make_isolation_figure.py) and the replication figure (33_replication_analysis.py) need a 2-D UMAP embedding and
out-of-fold scores that are not saved, so redrawing them would recompute.
Run: python code/claude/43_figures_v4.py
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

W = 6.69  # in; text width at 2 cm margins on A4
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6})
BLUE, RED, GREY, GREEN = "#4c78a8", "#e45756", "#9d9d9d", "#59a14f"
OUT = Path(__file__).resolve().parents[2] / "paper" / "figures"


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    fig.savefig(OUT / name)
    fig.savefig(FIG / name)
    print("saved", name, [round(v, 2) for v in fig.get_size_inches()])


# ------------------------------------------------ figure 1
nv = pd.read_csv(RES / "step3a_all2_naive_ref1_continuous_auc.csv")
vo = pd.read_csv(RES / "step3a_all2_voiced_ref1_continuous_auc.csv")
pn = pd.read_csv(RES / "step3a_all2_naive_ref1_predictive_auc.csv")
pv = pd.read_csv(RES / "step3a_all2_voiced_ref1_predictive_auc.csv")
fig, ax = plt.subplots(1, 2, figsize=(W, 2.75), layout="constrained")
cov = [c for c in ["silence_fraction", "voiced_dur_s", "span_len_s", "n_voiced_seg", "noise_floor_dbfs", "snr_db", "days_since_start"] if c in set(nv.covariate)]
y = np.arange(len(cov))
for off, d, col, lab in [(-0.18, nv, RED, "draft pipeline (whole span, FCM labels)"), (0.18, vo, BLUE, "voiced frames only (forced k-means split)")]:
    dd = d.set_index("covariate").loc[cov]
    up = dd.AUC_for_cluster1 >= .5
    a = np.where(up, dd.AUC_for_cluster1, 1 - dd.AUC_for_cluster1)
    lo = np.where(up, dd.ci95_lo, 1 - dd.ci95_hi)
    hi = np.where(up, dd.ci95_hi, 1 - dd.ci95_lo)
    ax[0].barh(y + off, a, height=.34, color=col, label=lab)
    ax[0].errorbar(a, y + off, xerr=[a - lo, hi - a], fmt="none", ecolor="k", lw=.6)
ax[0].set_yticks(y)
ax[0].set_yticklabels(cov)
ax[0].axvline(.5, color="k", ls="--", lw=.6)
ax[0].set_xlim(.4, 1)
ax[0].set_xlabel("AUC of the covariate for the two-way label\n(orientation-free; 0.5 = none)")
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=2, frameon=False, handlelength=1.2)
ax[0].set_title("Both splits track duration / silence / SNR", loc="left", fontsize=7.5)
letter(ax[0], "A")
names = ["silence / length variables", "SNR + noise floor + clipping", "channel only", "all recording variables (no context, no emitter)", "context only"]
yy = np.arange(len(names))
for off, d, col in [(-0.18, pn, RED), (0.18, pv, BLUE)]:
    d = d[d.scheme == "leave-bats-out"].set_index("predictors").loc[names]
    ax[1].barh(yy + off, d.AUC, height=.34, color=col)
    ax[1].errorbar(d.AUC, yy + off, xerr=[d.AUC - d.ci95_lo, d.ci95_hi - d.AUC], fmt="none", ecolor="k", lw=.6)
ax[1].set_yticks(yy)
ax[1].set_yticklabels([n.replace(" (no context, no emitter)", "").replace("silence / length variables", "silence / length\nvariables").replace("SNR + noise floor + clipping", "SNR + noise floor\n+ clipping").replace("all recording variables", "all recording\nvariables") for n in names])
ax[1].axvline(.5, color="k", ls="--", lw=.6)
ax[1].set_xlim(.4, 1)
ax[1].set_xlabel("held-out-bat AUC predicting the label\nfrom these variables only")
ax[1].set_title("Recording variables alone predict the label\non unseen bats", loc="left", fontsize=7.5)
letter(ax[1], "B")
save(fig, "fig1_v4.pdf")
plt.close(fig)

# ------------------------------------------------ figure 2 (power) and Fig. S5 (power for syllable shapes); the 200-replicate run is used when present
PIS = (0.5, 0.2, 0.05)
cols = {"power_FPC_excess>0.01": "FPC excess > 0.01", "power_Silhouette2": "silhouette", "power_PS2": "prediction strength",
        "power_Jaccard2": "bootstrap Jaccard", "power_DipSplit": "split-sample dip", "power_dip_p<.05": "dip p < 0.05"}
for tag, out in (("all2_voiced_ref1", "fig2_v4.pdf"), ("segall_shape", "fig_s5_v4.pdf")):
    suf = "_R200" if (RES / f"step3h_power_{tag}_R200_power.csv").exists() else ""
    P = pd.read_csv(RES / f"step3h_power_{tag}{suf}_power.csv").sort_values(["pi", "delta"])
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.55), sharey=True, layout="constrained")
    for a, pi in zip(ax, PIS):
        g = P[np.isclose(P.pi, pi)]
        for c, lab in cols.items():
            a.plot(g.delta, g[c], marker="o", ms=2.5, lw=1, label=lab)
        a.set_title(f"minority share {int(pi * 100)}%", loc="left", fontsize=7.5)
        a.set_xlabel("separation of the two types (SD units)")
        a.axhline(0.05, color="grey", lw=.5, ls=":")
        a.set_ylim(-0.02, 1.02)
    ax[0].set_ylabel("detection rate")
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=6, frameon=False, handlelength=1.2, columnspacing=0.8, fontsize=7)
    fig.suptitle(f"Power of the one-cluster-allowed battery - spiked mixtures in a null of set {tag} (n=6000; {int(P.n_reps.iloc[0])} replicates per point)", fontsize=7.5)
    save(fig, out)
    plt.close(fig)

# ------------------------------------------------ figure 3 (every algorithm x every representation)
DEFAULT_SETS = ["all2_naive_ref1", "all2_voiced_ref1", "all2_mfcc_naive", "all2_mfcc_voiced", "segall_shape", "segall_mfcc", "segall_spec", "segall_vae8_b0.1", "segall_pca8",
                "B_mic_voiced_ref1", "B_al_voiced_ref1"] + [f"aves2{m}_aves_{v}_{p}" for m in ("bio", "all") for v in ("ds16", "te2") for p in ("naive", "voiced")]
D = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("step3m_all_methods_*.csv")) if "summary" not in f.name], ignore_index=True)
best = D[(D.n_major >= 2) & D.silhouette.notna()].sort_values("silhouette", ascending=False).groupby(["set", "family"]).head(1)
fam_order = list(D.family.drop_duplicates())
fams = sorted(set(best.family), key=lambda f: fam_order.index(f))
sets = [s for s in DEFAULT_SETS if s in set(best.set)]
fig = plt.figure(figsize=(W, 4.6), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.035])
ax = [fig.add_subplot(gs[0, 0])]
ax += [fig.add_subplot(gs[0, k], sharey=ax[0]) for k in (1, 2)]
for a, col, ttl, cm, k in ((ax[0], "silhouette", "silhouette of the best\nconfiguration", "viridis", 0), (ax[1], "iso_best_f1", "best cluster vs pups' Isolation\ncalls (F1)", "magma", 1),
                           (ax[2], "bat_transfer_ari", "bat-transfer stability\n(ARI)", "cividis", 2)):
    Pv = best.pivot(index="family", columns="set", values=col).reindex(index=fams, columns=sets)
    im = a.imshow(Pv.values.astype(float), aspect="auto", cmap=cm, vmin=0, vmax=0.6 if col == "silhouette" else 1)
    a.set_xticks(range(len(sets)))
    a.set_xticklabels(sets, rotation=90, ha="center", fontsize=7)
    a.set_title(ttl, loc="left", fontsize=7.5, pad=3)
    for s_ in a.spines.values():
        s_.set_visible(False)
    cax = fig.add_subplot(gs[1, k])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, width=.5, length=2)
    cb.outline.set_linewidth(.4)
ax[0].set_yticks(range(len(fams)))
ax[0].set_yticklabels(fams, fontsize=7)
for a in ax[1:]:
    plt.setp(a.get_yticklabels(), visible=False)
save(fig, "fig3_v4.pdf")
plt.close(fig)

# ------------------------------------------------ figure 6 (emitter identification)
Dd = pd.read_csv(RES / "step3b_decode_all2.csv")
Dd = Dd[Dd.task == "emitter(+)"]
sets6 = [("all2_naive_ref1", "naive 1,408"), ("all2_voiced_ref1", "voiced 1,408"), ("all2_mfcc_naive", "MFCC naive"), ("all2_mfcc_voiced", "MFCC voiced")]
splits = [("random 5-fold", BLUE), ("blocked by day", GREEN), ("blocked by channel", RED)]
fig, ax = plt.subplots(figsize=(4.7, 2.45), layout="constrained")
w = .26
for j, (sp, col) in enumerate(splits):
    v = [float(Dd[(Dd.set == s) & (Dd.split == sp)].accuracy.iloc[0]) for s, _ in sets6]
    ax.bar(np.arange(len(sets6)) + (j - 1) * w, v, w, color=col, label=sp)
mb = float(Dd[Dd.split == "random 5-fold"].majority_baseline.iloc[0])
ax.axhline(mb, color="k", ls="--", lw=.6, label="majority class (random split)")
ax.set_xticks(range(len(sets6)))
ax.set_xticklabels([n for _, n in sets6])
ax.set_ylabel(f"accuracy, {int(Dd.classes.iloc[0])} bats (positive codes)")
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, handlelength=1.2)
ax.set_title("Emitter identification collapses when whole channels are held out", loc="left", fontsize=7.5)
save(fig, "fig6_v4.pdf")
plt.close(fig)
