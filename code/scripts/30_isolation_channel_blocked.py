"""Step 3k (robustness) - is the Isolation separability a property of one recording channel or treatment (chamber)?

Leave-one-GROUP-out: for every channel (and, separately, every treatment id) that holds >= 20 Isolation files and >= 20 other files, a logistic regression (balanced classes) is fitted on all
OTHER channels (treatments) and scored on the held-out one; the AUC of Isolation vs the other contexts is reported per held-out group and pooled (weighted by group size).
Because the held-out channel is never in training, a channel-specific recording signature cannot explain the score.
Writes results/step3k_isolation_channel_blocked.csv.
Run: python code/claude/30_isolation_channel_blocked.py <tag> [<tag> ...]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, log_change  # noqa: E402
from clusterlib import auc_fast, load_set  # noqa: E402

warnings.filterwarnings("ignore")
rows = []
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    u = u.reset_index(drop=True)
    y = (u.context_name == "Isolation").astype(int).values
    for key in ("channel", "treatment_id"):
        g = u[key].astype(str).values
        ok = [v for v in np.unique(g) if (y[g == v] == 1).sum() >= 20 and (y[g == v] == 0).sum() >= 20]
        aucs, ws = [], []
        for v in ok:
            te = g == v
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[~te], y[~te])
            a = float(auc_fast(y[te], m.decision_function(X[te])))
            aucs.append(a)
            ws.append(int(te.sum()))
            rows.append(dict(set=tag, held_out=key, group=v, n=int(te.sum()), n_isolation=int(y[te].sum()), bats=int(u.bat[te].nunique()), AUC=a))
            print(rows[-1], flush=True)
        rows.append(dict(set=tag, held_out=key, group="POOLED (weighted mean over groups)", n=int(sum(ws)), AUC=float(np.average(aucs, weights=ws)),
                         AUC_min=float(min(aucs)), AUC_max=float(max(aucs)), n_groups=len(aucs)))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "step3k_isolation_channel_blocked.csv", index=False)
log_change(f"Step 3k Isolation leave-one-channel-out and leave-one-treatment-out AUC on {', '.join(sys.argv[1:])}: results/step3k_isolation_channel_blocked.csv")
pd.set_option("display.width", 250)
print(R.round(3).to_string())
