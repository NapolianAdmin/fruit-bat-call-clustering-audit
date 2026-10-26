"""Step 3f - redo the draft's own association tests with the labels that actually belong to its audio.

The draft's 3,897 rows are audio of Annotations.csv ROW n (n = the integer in the filename; reproduces X_pca_train.npy exactly, see
07_baseline_reproduction.py hrowfull). The draft's emitter / context arrays were joined by FileID = n, i.e. from OTHER files. Here:
  A  cluster x context     V with the paper's (mismatched) labels vs V with the TRUE labels of the audio
  B  cluster x emitter     the draft's convention (emitter 0 dropped, negative codes kept as levels) for both label sets, plus positive-only
  C  true-context V given channel x month strata (2,000 within-stratum permutations on unique annotation rows)
  D  the paper's Table 2 duration contrast, recomputed on the audio the clusters really came from
  E  agreement of the paper's labels with the truth
Run: python code/claude/14_draft_true_labels.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, FIG, OLD, RES, SEED, cramers_v, log_change  # noqa: E402
from clusterlib import boot_groups, fcm_best  # noqa: E402

Xp = np.load(OLD / "X_pca_train.npy").astype(np.float64)
fn = np.load(OLD / "filenames_train.npy", allow_pickle=True)
em_paper = np.load(OLD / "emitter_train.npy", allow_pickle=True).astype(int)
cx_paper = np.load(OLD / "context_train.npy", allow_pickle=True)
n = np.array([int(s.split(".")[0]) for s in fn])
U = fcm_best(Xp, 2, seed=SEED, n_init=20)["U"].argmax(0)
if (U == 0).sum() > (U == 1).sum():
    U = 1 - U
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
t = ann.iloc[n].reset_index(drop=True)
d = pd.DataFrame(dict(cluster=U, em_paper=em_paper, cx_paper=cx_paper, em_true=t.Emitter.values, cx_true=t.context_name.values,
                      channel=t.channel.astype(str).values, month=t.month.values, span=t.annot_len_s.values, silence=1 - t.voiced_fraction_of_annot.values,
                      nseg=t.n_voice_seg_in_annot.values, ann_row=n))
d["bat_true"] = np.abs(d.em_true).astype(str)
S = {}

S["agreement_of_paper_labels_with_truth"] = dict(emitter=float((d.em_paper == d.em_true).mean()), context=float((d.cx_paper == d.cx_true).mean()), n_rows=len(d))


def vrow(name, df_, col):
    v, _, p, n_, r_, k_ = cramers_v(df_.cluster, df_[col], False)
    vb = cramers_v(df_.cluster, df_[col], True)[0]
    lo, hi = boot_groups(df_, lambda z: cramers_v(z.cluster, z[col], False)[0], group_col="bat_true", n_boot=300)
    return dict(analysis=name, n=int(n_), levels=int(k_), V_plain=v, V_biascorrected=vb, ci95_lo=lo, ci95_hi=hi, chi2_p=p)


rows = [vrow("cluster x context, PAPER labels (mismatched files)", d, "cx_paper"),
        vrow("cluster x context, TRUE labels of the audio", d, "cx_true"),
        vrow("cluster x emitter (draft convention: 0 dropped, negatives as levels), PAPER labels", d[d.em_paper != 0], "em_paper"),
        vrow("cluster x emitter (draft convention), TRUE labels", d[d.em_true != 0], "em_true"),
        vrow("cluster x emitter, TRUE labels, positive codes only", d[d.em_true > 0], "em_true")]
A = pd.DataFrame(rows)
A.to_csv(RES / "step3f_draft_true_labels_V.csv", index=False)

# ---- C  true context given channel x month (unique annotation rows)
u = d.drop_duplicates("ann_row").reset_index(drop=True)
u["stratum"] = u.channel + "|" + u.month
obs = cramers_v(u.cluster, u.cx_true, True)[0]
idx = [np.where(u.stratum.values == s)[0] for s in u.stratum.unique()]
rg = np.random.default_rng(SEED + 5)
null = []
for _ in range(2000):
    p_ = u.cluster.values.copy()
    for ii in idx:
        p_[ii] = rg.permutation(p_[ii])
    null.append(cramers_v(p_, u.cx_true.values, True)[0])
null = np.array(null)
ge = int((null >= obs).sum())
S["true_context_given_channel_month"] = dict(n_unique_rows=len(u), n_strata=len(idx), V_obs_biascorr=obs, null_mean=float(null.mean()), null_q025=float(np.quantile(null, .025)),
                                             null_q975=float(np.quantile(null, .975)), p=(f"<{1/2001:.4f} (0/2000)" if ge == 0 else f"{(1+ge)/2001:.4f} ({ge}/2000)"))

# ---- D  paper Table 2 duration contrast on the real audio
med = d.groupby("cluster")[["span", "silence", "nseg"]].median()
S["table2_duration_check"] = dict(paper_median_s=dict(cluster0=1.966, cluster1=1.314), paper_point_biserial_r=-0.422,
                                  recomputed_median_span_s={f"cluster{k}": float(v) for k, v in med.span.items()},
                                  recomputed_point_biserial_r=float(np.corrcoef(d.cluster, d.span)[0, 1]),
                                  recomputed_median_silence=med.silence.round(3).to_dict(), recomputed_median_n_voiced_segments=med.nseg.to_dict())

ct = pd.crosstab(d.cx_true, d.cluster)
ct["total"] = ct.sum(axis=1)
ct["cluster1_share"] = ct[1] / ct.total
ct.to_csv(RES / "step3f_draft_true_labels_context_table.csv")
S["draft_rows"] = dict(rows=len(d), unique_annotation_rows=len(u), unique_FileIDs=int(t.FileID.nunique()))
(RES / "step3f_draft_true_labels_summary.json").write_text(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 3.4))
lab = ["context\n(paper labels)", "context\n(true labels)", "emitter\n(paper labels)", "emitter\n(true labels)"]
vals = A.iloc[:4]
y = vals.V_plain.values
ax.bar(range(4), y, color=["#9d9d9d", "#4c78a8", "#9d9d9d", "#4c78a8"])
ax.errorbar(range(4), y, yerr=[y - vals.ci95_lo.values, vals.ci95_hi.values - y], fmt="none", ecolor="k", lw=.8)
ax.set_xticks(range(4))
ax.set_xticklabels(lab)
ax.set_ylabel("Cramer's V with the draft's two clusters")
ax.set_title("The draft's cluster associations rested on labels from other files", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "step3f_draft_true_labels.png", dpi=160)
log_change("Step 3f draft associations redone with the true labels of its audio; results/step3f_draft_true_labels_*")
pd.set_option("display.width", 250)
print(A.round(4).to_string())
print(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
