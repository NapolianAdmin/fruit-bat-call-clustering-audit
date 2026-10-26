"""Step 4d - does CHANNEL NORMALISATION rescue cross-channel identification?

Prat et al. (2016) mean-subtracted their MFCC features per recording channel before classifying emitters.  Step 4 found that identification collapses when whole channels are
held out (accuracy 0.20 vs 0.59 with random splits, 40 bats).  Here the same evaluation is repeated after four normalisations, all computed on ALL files of the set (labelled or not),
so no emitter label is used for the normalisation:
    raw            the 50 PCs as they are
    chan_mean      subtract the per-channel mean
    chan_z         per-channel mean and SD
    chanmonth_mean subtract the per-(channel x month) mean (session-level normalisation; the strongest)
Evaluation (positive-coded bats with >= 30 files; multinomial logistic regression):
    random 5-fold over files, day-blocked, channel-blocked (test bats absent from training are dropped; share kept reported)
and, for the channel-blocked scheme, a permutation null: emitter labels shuffled WITHIN channel x month strata (keeps who-was-recorded-where composition, removes voice), 10 shuffles.
Caveat: a bat that is the main occupant of a channel is partly normalised away with that channel's mean; the permutation null and the random-split accuracy show how much.
Run: python code/claude/25_channel_normalisation.py <tag> [<tag> ...]     e.g. all2_voiced_ref1 all2_naive_ref1 segall_shape
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SEED, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

warnings.filterwarnings("ignore")
N_PERM = 10


def normalise(X, u, how):
    X = X.copy()
    if how == "raw":
        return X
    keys = {"chan_mean": u.channel.astype(str), "chan_z": u.channel.astype(str), "chanmonth_mean": u.channel.astype(str) + "|" + u.month.astype(str)}[how]
    df = pd.DataFrame(X)
    mu = df.groupby(keys.values).transform("mean").values
    X = X - mu
    if how == "chan_z":
        X = X / (df.groupby(keys.values).transform("std").values + 1e-9)
    return X


def run(Xe, ye, blocks, kind):
    if kind == "random":
        splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xe, ye)
    else:
        splits = GroupKFold(min(5, len(np.unique(blocks)))).split(Xe, ye, blocks)
    pred, kept = np.full(len(ye), -1), np.zeros(len(ye), bool)
    for tr, te in splits:
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0)).fit(Xe[tr], ye[tr])
        ok = np.isin(ye[te], mdl.classes_)
        if ok.any():
            pred[te[ok]] = mdl.predict(Xe[te[ok]])
            kept[te[ok]] = True
    return float((pred == ye)[kept].mean()), float(balanced_accuracy_score(ye[kept], pred[kept])), float(kept.mean())


rows = []
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    u = u.reset_index(drop=True)
    pos = u[u.Emitter > 0]
    vc = pos.Emitter.value_counts()
    pos = pos[pos.Emitter.isin(vc[vc >= 30].index)]
    idx = pos.index.values
    ye = pos.Emitter.values
    major = float(pd.Series(ye).value_counts(normalize=True).max())
    strat = (pos.channel.astype(str) + "|" + pos.month.astype(str)).values
    for how in ("raw", "chan_mean", "chan_z", "chanmonth_mean"):
        Xn = normalise(X, u, how)[idx]
        r = dict(set=tag, normalisation=how, n_files=len(idx), n_bats=len(np.unique(ye)), majority=major)
        r["acc_random"], r["bal_random"], _ = run(Xn, ye, None, "random")
        r["acc_day_blocked"], r["bal_day_blocked"], _ = run(Xn, ye, pos.date.astype(str).values, "block")
        r["acc_channel_blocked"], r["bal_channel_blocked"], r["share_kept_channel_blocked"] = run(Xn, ye, pos.channel.astype(str).values, "block")
        rg = np.random.default_rng(SEED)
        nulls, nulls_bal = [], []
        for _ in range(N_PERM):
            ysh = ye.copy()
            for s in np.unique(strat):
                m = strat == s
                ysh[m] = rg.permutation(ye[m])
            a_, b_, _ = run(Xn, ysh, pos.channel.astype(str).values, "block")
            nulls.append(a_)
            nulls_bal.append(b_)
        r["perm_null_channel_blocked_mean"], r["perm_null_channel_blocked_max"] = float(np.mean(nulls)), float(np.max(nulls))
        r["perm_null_bal_channel_blocked_mean"], r["perm_null_bal_channel_blocked_max"] = float(np.mean(nulls_bal)), float(np.max(nulls_bal))
        r["uniform_chance"] = 1 / len(np.unique(ye))
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame([x for x in rows if x["set"] == tag]).to_csv(RES / f"step4d_channel_norm_{tag}.csv", index=False)
log_change(f"Step 4d channel normalisation vs cross-channel emitter ID on {', '.join(sys.argv[1:])}: results/step4d_channel_norm_*.csv")
