"""Step 3k (supplement) - sample sizes and bat-level bootstrap intervals for the age-stratified and channel/treatment-stratified Isolation AUCs.

Re-runs the SAME models as 36_isolation_age_matched.py (leave-bats-out scores, GroupKFold(5) over bats) and 30_isolation_channel_blocked.py (leave-one-group-out), checks that the AUCs equal the stored ones
(results/step3k_isolation_age_matched.csv, step3k_isolation_channel_blocked.csv) and adds, per stratum: number of recordings, positives (Isolation), negatives (other contexts), unique pups/bats, and a 95% percentile
bootstrap interval that resamples whole bats (pups), never recordings (1,000 resamples; seed 42).
Writes results/step3k_isolation_strata_ci.csv.   Run: python code/claude/46_isolation_strata_ci.py
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
from common import RES  # noqa: E402
from clusterlib import auc_fast, boot_groups, load_set  # noqa: E402

warnings.filterwarnings("ignore")
BIRTH = {103: "2012-06-28", 105: "2012-05-19", 108: "2012-06-05", 111: "2012-05-09", 113: "2012-06-10", 115: "2012-05-20", 120: "2012-06-05",
         202: "2012-09-25", 207: "2012-10-02", 208: "2012-09-22", 210: "2012-09-30", 211: "2012-09-22", 214: "2012-09-12", 215: "2012-10-04",
         216: "2012-09-12", 220: "2012-10-14", 221: "2012-09-26", 222: "2012-09-25", 230: "2012-10-14", 231: "2012-09-22"}
EDGES = [0, 20, 40, 60, 90, 150, 400]
NB = 1000
age_stored = pd.read_csv(RES / "step3k_isolation_age_matched.csv")
ch_stored = pd.read_csv(RES / "step3k_isolation_channel_blocked.csv")
rows = []


def ci(d):
    lo, hi = boot_groups(d, lambda z: auc_fast(z.iso.values, z.score.values), n_boot=NB)
    return float(lo), float(hi)


for tag in ("all2_voiced_ref1", "segall_shape"):
    X, u = load_set(tag)
    u = u.reset_index(drop=True)
    y = (u.context_name == "Isolation").astype(int).values
    u["iso"] = y
    # ---- age windows (same model as 36_isolation_age_matched.py)
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, u.bat.values):
        s[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[tr], y[tr]).decision_function(X[te])
    u["score"] = s
    p = u[(u.Emitter > 0) & u.Emitter.isin(list(BIRTH))].copy()
    p["age"] = (pd.to_datetime(p.date) - p.Emitter.map(lambda e: pd.Timestamp(BIRTH[e]))).dt.days
    p["window"] = pd.cut(p.age, EDGES, right=False)
    for w, g in p.groupby("window"):
        n1, n0 = int((g.iso == 1).sum()), int((g.iso == 0).sum())
        row = dict(set=tag, stratum="pup age (days)", group=str(w), recordings=len(g), isolation=n1, other=n0, bats=int(g.bat.nunique()))
        if n1 >= 15 and n0 >= 15:
            row["AUC"] = float(auc_fast(g.iso.values, g.score.values))
            row["AUC_lo"], row["AUC_hi"] = ci(g)
            ref = age_stored[(age_stored.set == tag) & (age_stored.age_window_days == str(w))].AUC.iloc[0]
            assert abs(row["AUC"] - ref) < 1e-9, (tag, w, row["AUC"], ref)
        rows.append(row)
        print(row, flush=True)
    # ---- leave-one-channel / treatment out (same model as 30_isolation_channel_blocked.py)
    for key, lab in (("channel", "channel held out"), ("treatment_id", "treatment held out")):
        gg = u[key].astype(str).values
        for v in np.unique(gg):
            te = gg == v
            if (y[te] == 1).sum() < 20 or (y[te] == 0).sum() < 20:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[~te], y[~te])
            d = u[te].copy()
            d["score"] = m.decision_function(X[te])
            row = dict(set=tag, stratum=lab, group=v, recordings=int(te.sum()), isolation=int(y[te].sum()), other=int((y[te] == 0).sum()), bats=int(d.bat.nunique()), AUC=float(auc_fast(d.iso.values, d.score.values)))
            row["AUC_lo"], row["AUC_hi"] = ci(d)
            ref = ch_stored[(ch_stored.set == tag) & (ch_stored.held_out == key) & (ch_stored.group.astype(str) == v)].AUC.iloc[0]
            assert abs(row["AUC"] - ref) < 1e-9, (tag, key, v, row["AUC"], ref)
            rows.append(row)
            print(row, flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "step3k_isolation_strata_ci.csv", index=False)
print("saved; every AUC equals the stored value")
