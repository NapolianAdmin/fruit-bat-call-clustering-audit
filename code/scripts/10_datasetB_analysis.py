"""Step 5b - Dataset B (Elie 2024) analysis: sets, mic-vs-piezo agreement, individual / hearing decodability, method timing.

Builds analysis sets set_B_<mic|al>_<key>_X.npy/_meta.csv (PCA 50, IDENTICAL events in every set: both sensors valid, >= 3 voiced
frames in both). meta has bat (= individual), stratum (= recording date), rec_time (= date), hearing, sex, source, event_id.

Outputs
  results/stepB_sets_summary.csv
  results/stepB_mic_vs_al_agreement.csv   ARI/NMI between the k-means partitions of the SAME events from the ambient mic and from
                                          the throat piezo (k=2..5), with a shuffled-pairing null; if clusters were call structure
                                          the two sensors should agree, if they were sensor noise they should not
  results/stepB_decode.csv                individual (random vs blocked-by-day), hearing status and sex (leave-individuals-out)
  results/stepB_methods.csv               FCM / GMM-diag / HDBSCAN counts and wall time per set
Run: python code/claude/10_datasetB_analysis.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, RES, SEED, log_change  # noqa: E402
from clusterlib import fcm_best, fuzzy_indices, silhouette_fixed  # noqa: E402

warnings.filterwarnings("ignore")
OUT = FEAT / "claude"
F = np.load(OUT / "B_features.npz")
idx = pd.read_csv(OUT / "B_index.csv", parse_dates=["date"])
KEYS = ["voiced_ref1", "naive_ref1", "mfcc_voiced", "mfcc_naive"]

ok = (idx.too_short == 0).to_numpy().copy()  # writable copy (pandas 3 returns read-only views)
for k in ("voiced_ref1", "mfcc_voiced"):
    ok &= np.isfinite(F[k]).all(1)
valid = idx.assign(ok=ok).groupby("event_id").ok.transform("all") & (idx.groupby("event_id").name.transform("size") == 2)
meta_all = idx[valid.values].copy()  # index labels = row positions in B_features.npz
meta_all["bat"] = meta_all.individual
meta_all["stratum"] = meta_all.date.dt.strftime("%Y-%m-%d")
meta_all["rec_time"] = meta_all.date
meta_all["date"] = meta_all.stratum
meta_all = meta_all.sort_values(["source", "event_id"])
rows, sets = [], {}
for src in ("Mic", "AL"):
    mm = meta_all[meta_all.source == src]
    assert mm.event_id.is_unique
    for k in KEYS:
        M = F[k][mm.index.values]
        X = PCA(50, svd_solver="full", random_state=SEED).fit_transform(StandardScaler().fit_transform(M))
        tag = f"B_{src.lower()}_{k}"
        np.save(OUT / f"set_{tag}_X.npy", X.astype(np.float32))
        mm.reset_index(drop=True).to_csv(OUT / f"set_{tag}_meta.csv", index=False)
        sets[tag] = (X, mm.reset_index(drop=True))
        rows.append(dict(set=tag, N=len(M), d=M.shape[1], individuals=mm.individual.nunique(), days=mm.stratum.nunique()))
pd.DataFrame(rows).to_csv(RES / "stepB_sets_summary.csv", index=False)
print(pd.DataFrame(rows).to_string(), flush=True)

# ---------------- mic vs piezo agreement of partitions of the same events
rg = np.random.default_rng(SEED)
ag = []
for k in KEYS:
    Xm, mm = sets[f"B_mic_{k}"]
    Xa, ma = sets[f"B_al_{k}"]
    assert (mm.event_id.values == ma.event_id.values).all()
    for kk in (2, 3, 4, 5):
        lm = KMeans(kk, n_init=5, random_state=SEED).fit_predict(Xm)
        la = KMeans(kk, n_init=5, random_state=SEED).fit_predict(Xa)
        null = [adjusted_rand_score(lm, rg.permutation(la)) for _ in range(100)]
        ag.append(dict(representation=k, k=kk, n_events=len(lm), ARI_mic_vs_piezo=adjusted_rand_score(lm, la), NMI_mic_vs_piezo=normalized_mutual_info_score(lm, la),
                       ARI_shuffled_pairing_mean=np.mean(null), ARI_shuffled_pairing_q975=np.quantile(null, .975)))
A = pd.DataFrame(ag)
A.to_csv(RES / "stepB_mic_vs_al_agreement.csv", index=False)
print(A.round(3).to_string(), flush=True)

# ---------------- decodability
DC = []
for tag, (X, m) in sets.items():
    for task, y, splits in [("individual", m.individual.values, {"random 5-fold": None, "blocked by day": m.stratum.values}),
                            ("hearing status", m.hearing.values, {"leave-individuals-out": m.individual.values}),
                            ("sex", m.sex.values, {"leave-individuals-out": m.individual.values})]:
        for sname, grp in splits.items():
            t = time.perf_counter()
            sp = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y) if grp is None else GroupKFold(5).split(X, y, grp)
            pred = np.empty(len(y), dtype=object)
            for tr, te in sp:
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)).fit(X[tr], y[tr])
                pred[te] = mdl.predict(X[te])
            DC.append(dict(set=tag, task=task, split=sname, n=len(y), classes=len(np.unique(y)), accuracy=float((pred == y).mean()),
                           balanced_accuracy=balanced_accuracy_score(y, pred.astype(str)), majority_baseline=float(pd.Series(y).value_counts(normalize=True).max()),
                           seconds=time.perf_counter() - t))
D = pd.DataFrame(DC)
D.to_csv(RES / "stepB_decode.csv", index=False)
print(D.round(3).to_string(), flush=True)

# ---------------- methods
MT = []
for tag, (X, m) in sets.items():
    for c in (2, 3, 4):
        t = time.perf_counter()
        r = fcm_best(X, c, seed=SEED, n_init=5)
        fpc, xb = fuzzy_indices(X, r["U"], r["V"])
        MT.append(dict(set=tag, method="FCM", k=c, FPC=fpc, FPC_excess=fpc - 1 / c, XieBeni=xb, silhouette=silhouette_fixed(X, r["U"].argmax(0)), seconds=time.perf_counter() - t))
    for k in (1, 2, 3, 4, 5, 6):
        t = time.perf_counter()
        g = GaussianMixture(k, covariance_type="diag", n_init=2, random_state=SEED, reg_covar=1e-4).fit(X)
        MT.append(dict(set=tag, method="GMM-diag", k=k, BIC=g.bic(X), seconds=time.perf_counter() - t))
    for mcs in (50, 200):
        t = time.perf_counter()
        lab = HDBSCAN(min_cluster_size=mcs, n_jobs=2).fit_predict(X)
        MT.append(dict(set=tag, method=f"HDBSCAN(min_cluster_size={mcs})", k=len(set(lab)) - (1 if -1 in lab else 0), noise_share=float((lab == -1).mean()), seconds=time.perf_counter() - t))
pd.DataFrame(MT).to_csv(RES / "stepB_methods.csv", index=False)
log_change("Dataset B analysis: sets, mic-vs-piezo agreement, decodability, methods; results/stepB_*")
print(pd.DataFrame(MT).round(3).to_string())
