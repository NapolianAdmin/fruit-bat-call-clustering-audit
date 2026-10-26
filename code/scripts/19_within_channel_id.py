"""Step 4c - emitter identification WITHIN a single recording channel (channel cannot help), blocked by day.

For every channel with >= 3 positive-coded bats of >= 40 files each: multinomial logistic regression on the 50 PCs, GroupKFold(5) over recording
DAYS inside that channel; accuracy vs the within-channel majority-class rate. Pooled over channels with a day-block bootstrap CI.
Run: python code/claude/19_within_channel_id.py <tag>        e.g. all_voiced_ref1
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
from common import RES, SEED, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

warnings.filterwarnings("ignore")
TAG = sys.argv[1]
X, u = load_set(TAG)
u = u.reset_index(drop=True)
u["day"] = u.date.astype(str)
rows, corr_all, blocks_all, base_all = [], [], [], []
for ch, g in u[u.Emitter > 0].groupby(u.channel.astype(str)):
    vc = g.Emitter.value_counts()
    keep = vc[vc >= 40].index
    g = g[g.Emitter.isin(keep)]
    if len(keep) < 3 or g.day.nunique() < 5:
        continue
    Xg, y, days = X[g.index.values], g.Emitter.values, g.day.values
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(min(5, g.day.nunique())).split(Xg, y, days):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)).fit(Xg[tr], y[tr])
        ok = np.isin(y[te], mdl.classes_)
        pred[te[ok]] = mdl.predict(Xg[te[ok]])
    kept = pred != -1
    corr = pred[kept] == y[kept]
    maj = pd.Series(y[kept]).value_counts(normalize=True).max()
    rows.append(dict(channel=ch, bats=len(keep), files=int(kept.sum()), days=g.day.nunique(), accuracy=float(corr.mean()), majority_rate=float(maj), uniform_chance=1 / len(keep)))
    corr_all.append(corr)
    blocks_all.append(np.array([f"{ch}|{d}" for d in days[kept]]))
    base_all.append(np.full(int(kept.sum()), maj))
R = pd.DataFrame(rows)
corr, blocks, base = np.concatenate(corr_all), np.concatenate(blocks_all), np.concatenate(base_all)
ub = np.unique(blocks)
ix = {b: np.where(blocks == b)[0] for b in ub}
rg = np.random.default_rng(SEED)
bs = []
for _ in range(500):
    ii = np.concatenate([ix[b] for b in rg.choice(ub, len(ub), replace=True)])
    bs.append(corr[ii].mean() - base[ii].mean())
R = pd.concat([R, pd.DataFrame([dict(channel="POOLED", bats=int(R.bats.sum()), files=len(corr), days=len(ub), accuracy=float(corr.mean()), majority_rate=float(base.mean()),
                                     uniform_chance=np.nan, accuracy_minus_majority=float(corr.mean() - base.mean()), ci95_lo=float(np.quantile(bs, .025)), ci95_hi=float(np.quantile(bs, .975)))])], ignore_index=True)
R.to_csv(RES / f"step4c_within_channel_{TAG}.csv", index=False)
log_change(f"Step 4c within-channel day-blocked emitter ID on '{TAG}': results/step4c_within_channel_{TAG}.csv")
pd.set_option("display.width", 250)
print(R.round(3).to_string())
