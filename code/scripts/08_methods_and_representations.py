"""Step 3b/4 - several clustering methods x several representations: what they find, how long they take, how stable they are
across held-out bats, and what they line up with (channel vs context vs bat); plus supervised decodability of emitter /
context / channel from each representation under random vs blocked splits.

Per set (tag):
  methods    FCM c=2..6 (FPC excess, XB, silhouette), k-means k=1..6 (inertia, silhouette), GMM-diag k=1..6 (BIC, ICL),
             HDBSCAN (min_cluster_size 25 and 100): #clusters, noise share.  wall seconds for every fit.
  stability  held-out-bat ARI: model fitted on a random half of the bats vs model fitted on the other half, compared on the
             second half (k-means k=2, GMM k=2, FCM c=2); mean of 10 splits.
  nmi        NMI of the k=2 labels with channel, month, context, bat(+ only), silence-bin and SNR-bin (when available)
  decode     logistic regression on the 50 PCs: emitter (positive bats >= 30 files; random 5-fold vs blocked by day/channel),
             context (random vs leave-bats-out), channel (leave-bats-out). accuracy, balanced accuracy, majority baseline, seconds.
Run: python code/claude/08_methods_and_representations.py <name> <tag> [<tag> ...]
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SEED, log_change  # noqa: E402
from clusterlib import fcm_best, fuzzy_indices, load_set, silhouette_fixed  # noqa: E402

warnings.filterwarnings("ignore")
NAME, TAGS = sys.argv[1], sys.argv[2:]


def memberships(X, V, m=2.0):
    d2 = np.maximum(((X[:, None, :] - V[None]) ** 2).sum(-1), 1e-12)
    u = d2 ** (-1 / (m - 1))
    return u / u.sum(1, keepdims=True)


def bat_halves(bat, rg):
    ub = np.unique(bat)
    for _ in range(60):
        p = rg.permutation(ub)
        A = np.isin(bat, p[: len(p) // 2])
        if 0.35 <= A.mean() <= 0.65:
            return A
    return A


M, ST, NM, DC = [], [], [], []
for tag in TAGS:
    X, u = load_set(tag)
    n = len(X)
    t_all = time.perf_counter()
    # ---------------- methods
    for c in range(2, 7):
        t = time.perf_counter()
        r = fcm_best(X, c, seed=SEED, n_init=5)
        fpc, xb = fuzzy_indices(X, r["U"], r["V"])
        M.append(dict(set=tag, n=n, method="FCM", k=c, FPC=fpc, FPC_excess_over_1_over_c=fpc - 1 / c, XieBeni=xb,
                      silhouette=silhouette_fixed(X, r["U"].argmax(0)), seconds=time.perf_counter() - t))
    for k in range(1, 7):
        t = time.perf_counter()
        km = KMeans(k, n_init=5, random_state=SEED).fit(X)
        sil = silhouette_fixed(X, km.labels_) if k > 1 else np.nan
        M.append(dict(set=tag, n=n, method="KMeans", k=k, inertia=km.inertia_, silhouette=sil, seconds=time.perf_counter() - t))
    for k in range(1, 7):
        t = time.perf_counter()
        g = GaussianMixture(k, covariance_type="diag", n_init=2, random_state=SEED, reg_covar=1e-4).fit(X)
        p = g.predict_proba(X)
        icl = g.bic(X) + 2 * (-(p * np.log(np.clip(p, 1e-12, None))).sum())
        M.append(dict(set=tag, n=n, method="GMM-diag", k=k, BIC=g.bic(X), ICL=icl, seconds=time.perf_counter() - t))
    for mcs in (25, 100):
        t = time.perf_counter()
        lab = HDBSCAN(min_cluster_size=mcs, n_jobs=2).fit_predict(X)
        kk = len(set(lab)) - (1 if -1 in lab else 0)
        sil = silhouette_fixed(X[lab >= 0], lab[lab >= 0]) if kk >= 2 else np.nan
        M.append(dict(set=tag, n=n, method=f"HDBSCAN(min_cluster_size={mcs})", k=kk, noise_share=float((lab == -1).mean()), silhouette=sil, seconds=time.perf_counter() - t))
    # ---------------- stability across held-out bats
    rg = np.random.default_rng(SEED + 3)
    for meth in ("KMeans", "GMM-diag", "FCM"):
        aris = []
        t = time.perf_counter()
        for s in range(10):
            A = bat_halves(u.bat.values, rg)
            lab = {}
            for part, mask in (("A", A), ("B", ~A)):
                Xt = X[mask]
                if meth == "KMeans":
                    mdl = KMeans(2, n_init=3, random_state=SEED + s).fit(Xt)
                    lab[part] = (mdl.labels_, mdl.predict)
                elif meth == "GMM-diag":
                    mdl = GaussianMixture(2, covariance_type="diag", n_init=2, random_state=SEED + s, reg_covar=1e-4).fit(Xt)
                    lab[part] = (mdl.predict(Xt), mdl.predict)
                else:
                    r = fcm_best(Xt, 2, seed=SEED + s, n_init=3)
                    lab[part] = (r["U"].argmax(0), lambda Z, V=r["V"]: memberships(Z, V).argmax(1))
            aris.append(0.5 * (adjusted_rand_score(lab["B"][0], lab["A"][1](X[~A])) + adjusted_rand_score(lab["A"][0], lab["B"][1](X[A]))))
        ST.append(dict(set=tag, method=meth, k=2, heldout_bat_ARI_mean=np.mean(aris), ci95_lo=np.percentile(aris, 2.5), ci95_hi=np.percentile(aris, 97.5),
                       n_splits=10, seconds=time.perf_counter() - t))
    # ---------------- what do the k=2 labels line up with?
    labs = {"KMeans": KMeans(2, n_init=5, random_state=SEED).fit_predict(X),
            "GMM-diag": GaussianMixture(2, covariance_type="diag", n_init=2, random_state=SEED, reg_covar=1e-4).fit(X).predict(X),
            "FCM": fcm_best(X, 2, seed=SEED, n_init=5)["U"].argmax(0)}
    ext = {"channel": u.channel.astype(str), "month": u.month, "context": u.context_name, "bat(+ only)": u.bat.where(u.Emitter > 0, "NA")}
    if "silence_fraction" in u:
        ext["silence_fraction bin (quintile)"] = pd.qcut(u.silence_fraction, 5, duplicates="drop").astype(str)
    if "snr_db" in u and u.snr_db.notna().sum() > 100:
        ext["SNR bin (quintile)"] = pd.qcut(u.snr_db, 5, duplicates="drop").astype(str)
    for meth, lab in labs.items():
        for k_, g in ext.items():
            ok = ((g != "NA") & g.notna()).values
            NM.append(dict(set=tag, method=meth, k=2, aligned_with=k_, NMI=normalized_mutual_info_score(g[ok].astype(str), lab[ok]), n=int(ok.sum())))
    # ---------------- decodability
    pos = (u.Emitter > 0).values
    vc = u[pos].Emitter.value_counts()
    keep = pos & u.Emitter.isin(vc[vc >= 30].index).values
    tasks = [("emitter(+)", u.Emitter.values, keep, {"random 5-fold": None, "blocked by day": u.date.astype(str).values, "blocked by channel": u.channel.astype(str).values}),
             ("context", u.context_name.values, np.ones(n, bool), {"random 5-fold": None, "leave-bats-out": u.bat.values}),
             ("channel", u.channel.astype(str).values, np.ones(n, bool), {"random 5-fold": None, "leave-bats-out": u.bat.values})]
    for task, y_all, msk, splits in tasks:
        if msk.sum() < 150:  # too few eligible files (e.g. emitter task on a 500-file subset)
            continue
        idx = np.where(msk)[0]
        Xs, y = X[idx], np.asarray(y_all)[idx]
        for sname, grp in splits.items():
            t = time.perf_counter()
            if grp is None:
                ystrat = np.where(pd.Series(y).map(pd.Series(y).value_counts()).values >= 5, y, "rare")
                sp = StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, ystrat)
            else:
                g_ = np.asarray(grp)[idx]
                sp = GroupKFold(min(5, len(np.unique(g_)))).split(Xs, y, g_)
            pred, kept = np.full(len(idx), "", dtype=object), np.zeros(len(idx), bool)
            for tr, te in sp:
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)).fit(Xs[tr], y[tr])
                ok = np.isin(y[te], mdl.classes_)
                if ok.any():
                    pred[te[ok]] = mdl.predict(Xs[te[ok]])
                    kept[te[ok]] = True
            yk, pk = y[kept].astype(str), pred[kept].astype(str)
            maj = pd.Series(yk).value_counts(normalize=True).max()
            DC.append(dict(set=tag, task=task, split=sname, n=int(kept.sum()), classes=int(len(np.unique(y))), accuracy=float((pk == yk).mean()),
                           balanced_accuracy=balanced_accuracy_score(yk, pk), majority_baseline=float(maj), seconds=time.perf_counter() - t))
    print(f"{tag}: done in {time.perf_counter()-t_all:.0f}s", flush=True)

pd.DataFrame(M).to_csv(RES / f"step3b_methods_{NAME}.csv", index=False)
pd.DataFrame(ST).to_csv(RES / f"step3b_stability_{NAME}.csv", index=False)
pd.DataFrame(NM).to_csv(RES / f"step3b_nmi_{NAME}.csv", index=False)
pd.DataFrame(DC).to_csv(RES / f"step3b_decode_{NAME}.csv", index=False)
log_change(f"Step 3b methods x representations '{NAME}' on sets {', '.join(TAGS)}; results/step3b_*_{NAME}.csv")
pd.set_option("display.width", 250)
print(pd.DataFrame(ST).round(3).to_string())
print(pd.DataFrame(DC).round(3).to_string())
