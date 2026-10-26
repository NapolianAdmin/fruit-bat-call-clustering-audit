"""Step 3k - are pups' ISOLATION calls a separable call type, or the end of a continuum?

UMAP + HDBSCAN (23/26) found a stable small cluster that is 86% Isolation-context files; the earlier k-means two-way split could not isolate a 6% group.  A direct test, bat-held-out:
  * logistic regression (balanced classes) on the 50 PCs, GroupKFold(5) over bats (|emitter|): out-of-fold score for Isolation vs all other files
  * AUC with bat-level bootstrap CI; the standardised separation d' of the two score distributions (pooled SD); the share of non-Isolation files scoring above the 5th percentile of Isolation
  * the sharper control: the SAME analysis among pups' files only (positive-coded pups, Isolation vs their calls in other contexts) - same bats, same age window;
    and a within-stratum version (mean of per-(channel x month) AUCs over strata with >= 15 files of each class)
  * a gap check: Hartigan dip test of the pooled out-of-fold score and of the non-Isolation scores alone (a continuum end would leave the non-Isolation scores skewed but unimodal,
    with Isolation overlapping their tail rather than separated by a density valley)
Sets: file-level voiced (all2_voiced_ref1), file-level voiced MFCC (all2_mfcc_voiced), syllable shape (segall_shape), syllable MFCC (segall_mfcc).
Writes results/step3k_isolation_separability.csv and figures/step3k_isolation_scores.png.
Run: python code/claude/27_isolation_separability.py <tag> [<tag> ...]
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
from common import FIG, RES, SEED, log_change  # noqa: E402
from clusterlib import auc_fast, boot_groups, load_set  # noqa: E402

warnings.filterwarnings("ignore")
import diptest  # noqa: E402

BIRTH = {103, 105, 108, 111, 113, 115, 120, 202, 207, 208, 210, 211, 214, 215, 216, 220, 221, 222, 230, 231}


def oof_scores(X, y, groups):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return s


def summarise(name, tag, d, score):
    y = d.iso.values
    a = float(auc_fast(y, score))
    lo, hi = boot_groups(d.assign(score=score), lambda z: auc_fast(z.iso.values, z.score.values), n_boot=200)
    sd = np.sqrt((score[y == 1].var() * (y == 1).sum() + score[y == 0].var() * (y == 0).sum()) / len(y))
    dprime = float(abs(score[y == 1].mean() - score[y == 0].mean()) / sd)
    thr = np.percentile(score[y == 1], 5)
    row = dict(set=tag, analysis=name, n=len(d), n_isolation=int(y.sum()), n_bats=int(d.bat.nunique()), AUC=a, AUC_lo=float(lo), AUC_hi=float(hi), d_prime=dprime,
               share_nonIso_above_Iso_p5=float((score[y == 0] > thr).mean()), dip_pooled=float(diptest.diptest(score)[0]), dip_pooled_p=float(diptest.diptest(score)[1]),
               dip_nonIso=float(diptest.diptest(score[y == 0])[0]), dip_nonIso_p=float(diptest.diptest(score[y == 0])[1]))
    return row


rows, store = [], {}
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    u = u.reset_index(drop=True)
    u["iso"] = (u.context_name == "Isolation").astype(int)
    s = oof_scores(X, u.iso.values, u.bat.values)
    store[tag] = (u.iso.values, s)
    rows.append(summarise("all files, leave-bats-out", tag, u, s))
    print(rows[-1], flush=True)
    pups = u[(u.Emitter > 0) & u.Emitter.isin(BIRTH)].copy()
    if pups.iso.nunique() == 2:
        sp = oof_scores(X[pups.index.values], pups.iso.values, pups.bat.values)
        rows.append(summarise("pups only (Isolation vs their other contexts), leave-bats-out", tag, pups, sp))
        print(rows[-1], flush=True)
        pups["score"] = sp
        aucs, ws = [], []
        for st, g in pups.groupby("stratum"):
            if (g.iso == 1).sum() >= 15 and (g.iso == 0).sum() >= 15:
                aucs.append(auc_fast(g.iso.values, g.score.values))
                ws.append(len(g))
        if aucs:
            rows.append(dict(set=tag, analysis="pups only, within channel x month strata (weighted mean AUC; AUC_lo/hi = min/max over strata; dip_nonIso_p = number of strata)", n=int(sum(ws)),
                             AUC=float(np.average(aucs, weights=ws)), AUC_lo=float(min(aucs)), AUC_hi=float(max(aucs)), dip_nonIso_p=float(len(aucs))))
            print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "step3k_isolation_separability.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(store), figsize=(4.2 * len(store), 3.4), squeeze=False)
for a, (tag, (y, s)) in zip(ax[0], store.items()):
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 60)
    a.hist(s[y == 0], bins=bins, alpha=.6, density=True, label="other contexts")
    a.hist(s[y == 1], bins=bins, alpha=.6, density=True, label="Isolation")
    a.set_title(tag, fontsize=9)
    a.set_xlabel("held-out-bat score (log-odds of Isolation)")
ax[0][0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "step3k_isolation_scores.png", dpi=150)
log_change(f"Step 3k Isolation separability (leave-bats-out, pups-only, within-stratum) on {', '.join(sys.argv[1:])}: results/step3k_isolation_separability.csv")
pd.set_option("display.width", 250)
print(R.round(3).to_string())
