"""Step 3k (comparison) - does a pretrained embedding (AVES) separate pups' Isolation calls as well as handcrafted features do?

On the 1,000-file AVES subset (114 Isolation files from 14 bats; identical files for every representation) a balanced logistic regression predicts Isolation vs all other contexts with GroupKFold(5)
over bats; AUC with a bat-level bootstrap CI.  With only 1,000 files the intervals are wide; this is a comparison on identical files, not a replacement for the 41,116-file result.
Writes results/step3k_isolation_aves.csv.
Run: python code/claude/31_isolation_aves.py <tag> [<tag> ...]
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
rows = []
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    u = u.reset_index(drop=True)
    y = (u.context_name == "Isolation").astype(int).values
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, u.bat.values):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.1, class_weight="balanced")).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    u["iso"], u["score"] = y, s
    lo, hi = boot_groups(u, lambda z: auc_fast(z.iso.values, z.score.values), n_boot=200)
    rows.append(dict(set=tag, n=len(u), n_isolation=int(y.sum()), n_bats=int(u.bat.nunique()), AUC=float(auc_fast(y, s)), AUC_lo=float(lo), AUC_hi=float(hi)))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "step3k_isolation_aves.csv", index=False)
log_change("Step 3k Isolation separability with AVES vs handcrafted on the 1,000-file subset: results/step3k_isolation_aves.csv")
pd.set_option("display.width", 250)
print(R.round(3).to_string())
