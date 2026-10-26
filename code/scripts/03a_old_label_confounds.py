"""Step 3a (early version, NO new audio needed) - is the DRAFT's cluster label predicted by recording variables?

Reproduces the draft's c=2 partition from the saved 50-PC matrix (own FCM, 20 restarts, seed 42), joins
each row to its annotation / FileInfo metadata, and asks, on held-out BATS where possible:
  * continuous covariates (silence fraction from FileInfo voiced segments, annotation length, #voiced segments,
    days since first recording): AUC of the covariate for the cluster label, bat-cluster-bootstrap CI
  * categorical covariates (channel, month, archive folder, treatment id, context, emitter class): Cramer's V
    (bias-corrected AND the draft's plain V), bat-cluster-bootstrap CI
  * leave-bats-out logistic prediction of the cluster label from recording-only variables vs random-split
  * stratified permutation: does context stay associated with the cluster label once channel x month is held fixed?
Unit of analysis: unique FileID (the draft counted 1,537 files twice). All-rows version reported as a sensitivity row.
SNR and clipping need audio and are NOT included yet (Codex phase 2b).
Run: python code/claude/03a_old_label_confounds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, FIG, OLD, RES, SEED, cramers_v, log_change  # noqa: E402
from clusterlib import fcm_best, fuzzy_indices  # noqa: E402

N_BOOT, N_PERM = 500, 2000

# ------------------------------------------------------------------ labels from the saved matrix
Xp = np.load(OLD / "X_pca_train.npy").astype(np.float64)
fn = np.load(OLD / "filenames_train.npy", allow_pickle=True)
em = np.load(OLD / "emitter_train.npy", allow_pickle=True).astype(int)
cx = np.load(OLD / "context_train.npy", allow_pickle=True)
fit = fcm_best(Xp, 2, seed=SEED, n_init=20)
lab = fit["U"].argmax(0)
if (lab == 0).sum() > (lab == 1).sum():  # orient: cluster 1 = larger (draft: 1,860 / 2,037)
    lab = 1 - lab
fpc, xb = fuzzy_indices(Xp, fit["U"], fit["V"])
S = dict(fcm_c2=dict(FPC=fpc, XieBeni=xb, sizes=np.bincount(lab).tolist(), draft_sizes=[1860, 2037]))

rows = pd.DataFrame(dict(FileID=[int(s.split(".")[0]) for s in fn], Emitter=em, context_name=cx, cluster=lab))
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz", parse_dates=["rec_time"])
ann = ann.drop_duplicates(["FileID", "Emitter", "context_name"])
m = rows.merge(ann[["FileID", "Emitter", "context_name", "channel", "month", "folder", "treatment_id", "rec_time",
                    "voiced_fraction_of_annot", "annot_len_s", "n_voice_seg_in_annot"]],
               on=["FileID", "Emitter", "context_name"], how="left")
assert len(m) == len(rows) and m.channel.notna().all(), "row-to-annotation join failed"
m["silence_fraction"] = 1 - m.voiced_fraction_of_annot
m["days_since_start"] = (m.rec_time - pd.Timestamp("2012-06-01")).dt.total_seconds() / 86400
m["bat"] = m.Emitter.abs()  # grouping unit for resampling: |emitter| code, 0 = its own group
m["em_class"] = np.where(m.Emitter == 0, "zero", np.where(m.Emitter < 0, "negative", "positive"))
for c in ("channel", "treatment_id"):
    m[c] = m[c].astype(str)

# draft check: plain V of cluster vs context on all 3,897 rows (draft: 0.1448, p 7.5e-14)
v_plain, _, p_plain, n_, r_, k_ = cramers_v(m.cluster, m.context_name, bias_correct=False)
S["draft_check_context_V_all_rows"] = dict(V=v_plain, p=p_plain, n=n_, draft_V=0.1448, draft_p=7.54e-14)

u = m.drop_duplicates("FileID").reset_index(drop=True)  # unique files = primary unit
S["n_rows"], S["n_unique_files"] = len(m), len(u)
S["pct_duplicate_pairs_with_different_label"] = float((m.groupby("FileID").cluster.nunique() > 1).sum() / m.FileID.nunique() * 100)


def auc_fast(y, x):
    r = rankdata(x)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_groups(df, fun, n_boot=N_BOOT, seed=0):
    """Percentile CI of fun(df_resampled) resampling whole bats (|emitter| groups), never rows."""
    g = df.bat.values
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    rg = np.random.default_rng(SEED + seed)
    vals = []
    for _ in range(n_boot):
        pick = rg.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx[k] for k in pick])
        try:
            vals.append(fun(df.iloc[ii]))
        except Exception:
            pass
    return np.nanpercentile(vals, [2.5, 97.5])


# ------------------------------------------------------------------ continuous covariates
cont = ["silence_fraction", "annot_len_s", "n_voice_seg_in_annot", "days_since_start"]
rows_c = []
for tag, d in [("unique files (primary)", u), ("all rows (sensitivity)", m)]:
    for c in cont:
        a = auc_fast(d.cluster.values, d[c].values)
        lo, hi = boot_groups(d, lambda z, c=c: auc_fast(z.cluster.values, z[c].values))
        rows_c.append(dict(unit=tag, covariate=c, n=len(d), median_cluster0=d[d.cluster == 0][c].median(),
                           median_cluster1=d[d.cluster == 1][c].median(), AUC_for_cluster1=a, ci95_lo=lo, ci95_hi=hi))
cont_df = pd.DataFrame(rows_c)
cont_df.to_csv(RES / "step3a_continuous_auc.csv", index=False)

# ------------------------------------------------------------------ categorical covariates
cats = ["channel", "month", "folder", "treatment_id", "context_name", "em_class"]
rows_k = []
for tag, d in [("unique files (primary)", u), ("all rows (sensitivity)", m)]:
    for c in cats:
        vb, _, pb, n_, r_, k_ = cramers_v(d.cluster, d[c], bias_correct=True)
        vp = cramers_v(d.cluster, d[c], bias_correct=False)[0]
        lo, hi = boot_groups(d, lambda z, c=c: cramers_v(z.cluster, z[c], True)[0], n_boot=300)
        rows_k.append(dict(unit=tag, covariate=c, n=n_, levels=k_, V_biascorrected=vb, V_plain=vp, ci95_lo=lo, ci95_hi=hi, chi2_p=pb))
cat_df = pd.DataFrame(rows_k)
cat_df.to_csv(RES / "step3a_categorical_v.csv", index=False)


# ------------------------------------------------------------------ held-out-bat prediction of the cluster label
def design(cols_num, cols_cat):
    tr = []
    if cols_num:
        tr.append(("num", StandardScaler(), cols_num))
    if cols_cat:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat))
    return make_pipeline(ColumnTransformer(tr), LogisticRegression(max_iter=3000, C=1.0))


NUM = ["silence_fraction", "annot_len_s", "n_voice_seg_in_annot", "days_since_start"]
sets = {
    "silence fraction only": (["silence_fraction"], []),
    "silence + annot length + #segments": (["silence_fraction", "annot_len_s", "n_voice_seg_in_annot"], []),
    "channel only": ([], ["channel"]),
    "date only (days + month)": (["days_since_start"], ["month"]),
    "all recording variables (no context, no emitter)": (NUM, ["channel", "month", "treatment_id"]),
    "context only": ([], ["context_name"]),
    "recording variables + context": (NUM, ["channel", "month", "treatment_id", "context_name"]),
}
rows_p = []
y = u.cluster.values
for name, (cn, cc) in sets.items():
    for scheme in ("leave-bats-out (GroupKFold 5, grouped by |emitter|)", "random 5-fold over files"):
        oof = np.zeros(len(u))
        splitter = GroupKFold(5).split(u, y, u.bat) if scheme.startswith("leave") else StratifiedKFold(5, shuffle=True, random_state=SEED).split(u, y)
        for tr_i, te_i in splitter:
            mdl = design(cn, cc).fit(u.iloc[tr_i], y[tr_i])
            oof[te_i] = mdl.predict_proba(u.iloc[te_i])[:, 1]
        auc = roc_auc_score(y, oof)
        ug = u.bat.unique()
        idx = {k: np.where(u.bat.values == k)[0] for k in ug}
        rg = np.random.default_rng(SEED + 7)
        bs = []
        for _ in range(300):
            ii = np.concatenate([idx[k] for k in rg.choice(ug, len(ug), replace=True)])
            if len(np.unique(y[ii])) == 2:
                bs.append(roc_auc_score(y[ii], oof[ii]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows_p.append(dict(predictors=name, scheme=scheme, n_files=len(u), AUC=auc, ci95_lo=lo, ci95_hi=hi))
        print(f"{name:52s} {scheme[:14]:14s} AUC {auc:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
pred_df = pd.DataFrame(rows_p)
pred_df.to_csv(RES / "step3a_predictive_auc.csv", index=False)

# ------------------------------------------------------------------ does context association survive holding channel x month fixed?
u["stratum"] = u.channel + "|" + u.month
obs = cramers_v(u.cluster, u.context_name, bias_correct=True)[0]
obs_plain = cramers_v(u.cluster, u.context_name, bias_correct=False)[0]
idx_s = [np.where(u.stratum.values == s)[0] for s in u.stratum.unique()]
rp = np.random.default_rng(SEED + 99)
null_b, null_p = [], []
cl = u.cluster.values.copy()
for _ in range(N_PERM):
    perm = cl.copy()
    for ii in idx_s:
        perm[ii] = rp.permutation(cl[ii])
    null_b.append(cramers_v(perm, u.context_name.values, True)[0])
    null_p.append(cramers_v(perm, u.context_name.values, False)[0])
null_b, null_p = np.array(null_b), np.array(null_p)
n_ge = int((null_b >= obs).sum())
perm_row = dict(n_files=len(u), n_strata=len(idx_s), V_obs_biascorr=obs, null_mean=np.nanmean(null_b),
                null_q025=np.nanquantile(null_b, .025), null_q975=np.nanquantile(null_b, .975),
                p=(1 + n_ge) / (1 + N_PERM), n_perm=N_PERM, null_ge_obs=n_ge,
                V_obs_plain=obs_plain, plain_null_mean=null_p.mean())
pd.DataFrame([perm_row]).to_csv(RES / "step3a_context_given_channel_month_perm.csv", index=False)
S["context_given_channel_month"] = perm_row
S["reported_p"] = f"p<{1/(1+N_PERM):.4f} (0/{N_PERM} null replicates >= observed)" if n_ge == 0 else f"p={perm_row['p']:.4f} ({n_ge}/{N_PERM} null replicates >= observed)"

(RES / "step3a_summary.json").write_text(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

# ------------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for k, colr in [(0, "#4c78a8"), (1, "#e45756")]:
    ax[0].hist(u[u.cluster == k].silence_fraction, bins=40, alpha=.55, color=colr, label=f"cluster {k} (n={int((u.cluster == k).sum())})", density=True)
ax[0].set_xlabel("silence fraction of annotated span (1 - voiced, from FileInfo)")
ax[0].set_ylabel("density")
ax[0].set_title("Draft cluster label vs silence fraction (unique files)", fontsize=9)
ax[0].legend(fontsize=7)
pp = pred_df[pred_df.scheme.str.startswith("leave")]
ax[1].barh(pp.predictors, pp.AUC, xerr=[pp.AUC - pp.ci95_lo, pp.ci95_hi - pp.AUC], color="#4c78a8")
ax[1].axvline(.5, color="k", lw=.8, ls="--")
ax[1].set_xlim(.4, 1)
ax[1].set_xlabel("held-out-bat AUC for the cluster label (0.5 = chance)")
ax[1].set_title("Predicting the draft cluster from recording variables", fontsize=9)
ax[1].tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "step3a_old_labels.png", dpi=150)
log_change("Step 3a (early, no audio): draft cluster labels vs FileInfo recording variables; results/step3a_*")
print(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(cont_df.round(3).to_string())
print(cat_df.round(3).to_string())
