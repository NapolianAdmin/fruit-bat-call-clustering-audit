"""Step 3k (robustness) - is the Isolation separability just a pup AGE effect?

Isolation calls are produced by young pups; calls of any context change with age.  Among positive-coded pups with a known birth date (Metadata.pdf Table 2, as in 12_pup_adult_check.py) the
leave-bats-out Isolation score of the 50 PCs (balanced logistic regression, GroupKFold(5) over bats; fitted on ALL files) is evaluated WITHIN age windows (days since birth):
AUC of Isolation vs the same pups' other contexts in each window with >= 15 files of each class, and the file-weighted mean.  Also the age distributions of the two classes.
Writes results/step3k_isolation_age_matched.csv.
Run: python code/claude/36_isolation_age_matched.py <tag> [<tag> ...]     e.g. all2_voiced_ref1 segall_shape
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
from clusterlib import auc_fast, load_set  # noqa: E402

warnings.filterwarnings("ignore")
BIRTH = {103: "2012-06-28", 105: "2012-05-19", 108: "2012-06-05", 111: "2012-05-09", 113: "2012-06-10", 115: "2012-05-20", 120: "2012-06-05",
         202: "2012-09-25", 207: "2012-10-02", 208: "2012-09-22", 210: "2012-09-30", 211: "2012-09-22", 214: "2012-09-12", 215: "2012-10-04",
         216: "2012-09-12", 220: "2012-10-14", 221: "2012-09-26", 222: "2012-09-25", 230: "2012-10-14", 231: "2012-09-22"}
EDGES = [0, 20, 40, 60, 90, 150, 400]
rows = []
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    u = u.reset_index(drop=True)
    y = (u.context_name == "Isolation").astype(int).values
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, u.bat.values):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    u["iso"], u["score"] = y, s
    p = u[(u.Emitter > 0) & u.Emitter.isin(list(BIRTH))].copy()
    p["age"] = (pd.to_datetime(p.date) - p.Emitter.map(lambda e: pd.Timestamp(BIRTH[e]))).dt.days
    p["window"] = pd.cut(p.age, EDGES, right=False)
    print(tag, "pups' files:", len(p), "Isolation:", int(p.iso.sum()), "| median age Isolation", p[p.iso == 1].age.median(), "other", p[p.iso == 0].age.median(), flush=True)
    aucs, ws = [], []
    for w, g in p.groupby("window"):
        n1, n0 = int((g.iso == 1).sum()), int((g.iso == 0).sum())
        row = dict(set=tag, age_window_days=str(w), n_isolation=n1, n_other=n0, bats=int(g.bat.nunique()))
        if n1 >= 15 and n0 >= 15:
            row["AUC"] = float(auc_fast(g.iso.values, g.score.values))
            aucs.append(row["AUC"])
            ws.append(len(g))
        rows.append(row)
        print(row, flush=True)
    if aucs:
        rows.append(dict(set=tag, age_window_days="WEIGHTED MEAN over windows with >= 15 per class", n_isolation=int(p.iso.sum()), n_other=int((p.iso == 0).sum()), bats=int(p.bat.nunique()),
                         AUC=float(np.average(aucs, weights=ws))))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "step3k_isolation_age_matched.csv", index=False)
log_change(f"Step 3k Isolation vs age: within-age-window AUC for pups on {', '.join(sys.argv[1:])}: results/step3k_isolation_age_matched.csv")
pd.set_option("display.width", 250)
print(R.round(3).to_string())
