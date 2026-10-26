"""Step 3k (robustness) - is the Isolation separability carried by one pup?

One pup contributes ~57% of all Isolation files.  Two checks on the leave-bats-out Isolation score of 27_isolation_separability.py:
  1. per-pup AUC (Isolation vs the SAME pup's other contexts) for every positive-coded bat with >= 10 files of each class, using the out-of-fold score (the model never saw that bat)
  2. the whole analysis repeated after REMOVING the bat with the most Isolation files (from training and test): AUC with a bat-level bootstrap CI
Writes results/step3k_isolation_per_bat.csv and results/step3k_isolation_without_top_bat.csv.
Run: python code/claude/29_isolation_per_bat.py <tag> [<tag> ...]     e.g. all2_voiced_ref1 segall_shape
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
from common import RES, log_change  # noqa: E402
from clusterlib import auc_fast, boot_groups, load_set  # noqa: E402

warnings.filterwarnings("ignore")
BIRTH = {103, 105, 108, 111, 113, 115, 120, 202, 207, 208, 210, 211, 214, 215, 216, 220, 221, 222, 230, 231}


def oof(X, y, g):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return s


per, nobat = [], []
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    u = u.reset_index(drop=True)
    u["iso"] = (u.context_name == "Isolation").astype(int)
    u["score"] = oof(X, u.iso.values, u.bat.values)
    iso_by_bat = u[u.iso == 1].groupby("bat").size().sort_values(ascending=False)
    top = iso_by_bat.index[0]
    print(tag, "Isolation files per bat (top 5):", iso_by_bat.head(5).to_dict(), "share of top bat", round(iso_by_bat.iloc[0] / iso_by_bat.sum(), 3), flush=True)
    for b, g in u[u.Emitter > 0].groupby("bat"):
        if (g.iso == 1).sum() >= 10 and (g.iso == 0).sum() >= 10:
            per.append(dict(set=tag, bat=b, pup=int(int(b) in BIRTH), n_isolation=int((g.iso == 1).sum()), n_other=int((g.iso == 0).sum()), AUC=float(auc_fast(g.iso.values, g.score.values))))
    keep = (u.bat != top).values
    d2 = u[keep].reset_index(drop=True)
    d2["score"] = oof(X[keep], d2.iso.values, d2.bat.values)
    a = float(auc_fast(d2.iso.values, d2.score.values))
    lo, hi = boot_groups(d2, lambda z: auc_fast(z.iso.values, z.score.values), n_boot=200)
    nobat.append(dict(set=tag, removed_bat=top, removed_isolation_files=int(iso_by_bat.iloc[0]), remaining_isolation_files=int(d2.iso.sum()), remaining_bats_with_isolation=int(d2[d2.iso == 1].bat.nunique()),
                      AUC=a, AUC_lo=float(lo), AUC_hi=float(hi)))
    print(nobat[-1], flush=True)
P = pd.DataFrame(per)
P.to_csv(RES / "step3k_isolation_per_bat.csv", index=False)
N = pd.DataFrame(nobat)
N.to_csv(RES / "step3k_isolation_without_top_bat.csv", index=False)
pd.set_option("display.width", 250)
print(P.round(3).to_string())
for t, g in P.groupby("set"):
    print(t, f"{len(g)} bats with >=10 files per class: median AUC {g.AUC.median():.3f}, min {g.AUC.min():.3f}, max {g.AUC.max():.3f}, share >0.9: {(g.AUC > 0.9).mean():.2f}")
log_change(f"Step 3k Isolation robustness (per-bat AUC; without the top Isolation bat) on {', '.join(sys.argv[1:])}")
