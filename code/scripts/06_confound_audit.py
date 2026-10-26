"""Step 3a on any analysis set (tag-aware version of 03a, now WITH audio-derived covariates).

Cluster label = FCM c=2 on the set's 50 PCs (20 restarts, seed 42; cluster 1 = the larger one). Then, on unique files:
  * continuous covariates: silence fraction, SNR (dB), noise floor (dBFS), clipping fraction, voiced duration, span length,
    #voiced segments, days since start -> AUC for the cluster label with 95% CI resampling BATS
  * categorical covariates: channel, month, archive folder, treatment id, context, emitter class -> Cramer's V
    (bias-corrected and plain) with 95% CI resampling bats
  * leave-bats-out (GroupKFold 5 by |emitter|) and random-5-fold logistic prediction of the cluster label from recording
    variables vs context; AUC with CI
  * context-cluster association with channel x month strata held fixed (2,000 within-stratum permutations)
Run: python code/claude/06_confound_audit.py <tag>      e.g. baseline_naive_ref1 | baseline_voiced_ref1
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RES, SEED, cramers_v, log_change  # noqa: E402
from clusterlib import auc_fast, boot_groups, fcm_best, fuzzy_indices, load_set  # noqa: E402

TAG = sys.argv[1]
N_PERM = 2000
X, u = load_set(TAG)
fit = fcm_best(X, 2, seed=SEED, n_init=20)
lab = fit["U"].argmax(0)
if (lab == 0).sum() > (lab == 1).sum():
    lab = 1 - lab
fpc, xb = fuzzy_indices(X, fit["U"], fit["V"])
label_method = "FCM c=2 (argmax membership)"
if fpc < 0.51:  # FCM collapsed to its k=1 solution (all memberships ~0.5): its hard labels are arbitrary -> forced k-means split
    from sklearn.cluster import KMeans
    lab = KMeans(2, n_init=10, random_state=SEED).fit_predict(X)
    if (lab == 0).sum() > (lab == 1).sum():
        lab = 1 - lab
    label_method = "k-means k=2 (FCM collapsed, FPC<0.51: forced two-way split, NOT a detected structure)"
if "cluster_draft" in u.columns:  # the draft matrix: use the draft's own partition (fitted on its 3,897 rows)
    lab = u.cluster_draft.values
    label_method = "the draft's FCM c=2 partition (all 3,897 rows), unique annotation rows"
u["cluster"] = lab
S = dict(tag=TAG, label_method=label_method, n_files=len(u), n_bats=int(u.bat.nunique()),
         fcm_c2=dict(FPC=fpc, XieBeni=xb, sizes=np.bincount(fit["U"].argmax(0)).tolist()), label_sizes=np.bincount(lab).tolist())

cont = [c for c in ["silence_fraction", "snr_db", "noise_floor_dbfs", "clip_frac", "voiced_dur_s", "span_len_s", "n_voiced_seg", "days_since_start"] if c in u.columns]
rows = []
for c in cont:
    d = u[u[c].notna()]
    a = auc_fast(d.cluster.values, d[c].values)
    lo, hi = boot_groups(d, lambda z, c=c: auc_fast(z.cluster.values, z[c].values), n_boot=500)
    rows.append(dict(covariate=c, n=len(d), median_cluster0=d[d.cluster == 0][c].median(), median_cluster1=d[d.cluster == 1][c].median(),
                     AUC_for_cluster1=a, ci95_lo=lo, ci95_hi=hi))
cont_df = pd.DataFrame(rows)
cont_df.to_csv(RES / f"step3a_{TAG}_continuous_auc.csv", index=False)

cats = ["channel", "month", "folder", "treatment_id", "context_name", "em_class"]
rows = []
for c in cats:
    vb, _, pb, n_, r_, k_ = cramers_v(u.cluster, u[c], True)
    vp = cramers_v(u.cluster, u[c], False)[0]
    lo, hi = boot_groups(u, lambda z, c=c: cramers_v(z.cluster, z[c], True)[0], n_boot=300)
    rows.append(dict(covariate=c, n=n_, levels=k_, V_biascorrected=vb, V_plain=vp, ci95_lo=lo, ci95_hi=hi, chi2_p=pb))
cat_df = pd.DataFrame(rows)
cat_df.to_csv(RES / f"step3a_{TAG}_categorical_v.csv", index=False)


def design(num, cat):
    tr = []
    if num:
        tr.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return make_pipeline(ColumnTransformer(tr), LogisticRegression(max_iter=3000))


have = lambda L: [c for c in L if c in u.columns]
AUDIO = have(["snr_db", "noise_floor_dbfs", "clip_frac"])
SIL = have(["silence_fraction", "span_len_s", "n_voiced_seg", "voiced_dur_s"])
ALLREC = SIL + AUDIO + ["days_since_start"]
sets = {
    "silence / length variables": (SIL, []),
    "SNR + noise floor + clipping": (AUDIO, []),
    "channel only": ([], ["channel"]),
    "date only (days + month)": (["days_since_start"], ["month"]),
    "all recording variables (no context, no emitter)": (ALLREC, ["channel", "month", "treatment_id"]),
    "context only": ([], ["context_name"]),
    "recording variables + context": (ALLREC, ["channel", "month", "treatment_id", "context_name"]),
}
rows, y = [], u.cluster.values
for name, (cn, cc) in sets.items():
    if not cn and not cc:
        continue
    for scheme in ("leave-bats-out", "random 5-fold"):
        oof = np.zeros(len(u))
        sp = GroupKFold(5).split(u, y, u.bat) if scheme == "leave-bats-out" else StratifiedKFold(5, shuffle=True, random_state=SEED).split(u, y)
        for tr, te in sp:
            oof[te] = design(cn, cc).fit(u.iloc[tr], y[tr]).predict_proba(u.iloc[te])[:, 1]
        auc = roc_auc_score(y, oof)
        ug = u.bat.unique()
        ix = {k: np.where(u.bat.values == k)[0] for k in ug}
        rg = np.random.default_rng(SEED + 7)
        bs = []
        for _ in range(300):
            ii = np.concatenate([ix[k] for k in rg.choice(ug, len(ug), replace=True)])
            if len(np.unique(y[ii])) == 2:
                bs.append(roc_auc_score(y[ii], oof[ii]))
        rows.append(dict(predictors=name, scheme=scheme, n_files=len(u), AUC=auc, ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5)))
        print(f"{name:50s} {scheme:15s} AUC {auc:.3f} [{rows[-1]['ci95_lo']:.3f},{rows[-1]['ci95_hi']:.3f}]", flush=True)
pred_df = pd.DataFrame(rows)
pred_df.to_csv(RES / f"step3a_{TAG}_predictive_auc.csv", index=False)

idx_s = [np.where(u.stratum.values == s)[0] for s in u.stratum.unique()]
obs = cramers_v(u.cluster, u.context_name, True)[0]
rp = np.random.default_rng(SEED + 99)
null = []
for _ in range(N_PERM):
    perm = u.cluster.values.copy()
    for ii in idx_s:
        perm[ii] = rp.permutation(perm[ii])
    null.append(cramers_v(perm, u.context_name.values, True)[0])
null = np.array(null)
ge = int((null >= obs).sum())
S["context_given_channel_month"] = dict(V_obs_biascorr=obs, V_plain=cramers_v(u.cluster, u.context_name, False)[0], null_mean=float(np.nanmean(null)),
                                        null_q025=float(np.nanquantile(null, .025)), null_q975=float(np.nanquantile(null, .975)), n_strata=len(idx_s),
                                        p=(f"<{1/(1+N_PERM):.4f} (0/{N_PERM})" if ge == 0 else f"{(1+ge)/(1+N_PERM):.4f} ({ge}/{N_PERM})"))
(RES / f"step3a_{TAG}_summary.json").write_text(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
cv = "snr_db" if "snr_db" in u.columns and u.snr_db.notna().sum() > 50 else "silence_fraction"
for k, colr in [(0, "#4c78a8"), (1, "#e45756")]:
    ax[0].hist(u[u.cluster == k][cv].dropna(), bins=40, alpha=.55, color=colr, density=True, label=f"cluster {k} (n={int((u.cluster == k).sum())})")
ax[0].set_xlabel(cv)
ax[0].legend(fontsize=7)
ax[0].set_title(f"{TAG}: cluster vs {cv}", fontsize=9)
pp = pred_df[pred_df.scheme == "leave-bats-out"]
ax[1].barh(pp.predictors, pp.AUC, xerr=[pp.AUC - pp.ci95_lo, pp.ci95_hi - pp.AUC], color="#4c78a8")
ax[1].axvline(.5, color="k", lw=.8, ls="--")
ax[1].set_xlim(.4, 1)
ax[1].set_xlabel("held-out-bat AUC for the cluster label")
ax[1].tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(FIG / f"step3a_{TAG}.png", dpi=150)
log_change(f"Step 3a confound audit on '{TAG}': results/step3a_{TAG}_*")
print(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(cont_df.round(3).to_string())
print(cat_df.round(3).to_string())
