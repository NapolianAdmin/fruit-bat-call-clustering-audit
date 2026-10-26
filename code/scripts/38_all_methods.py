"""Step 3m - EVERY clustering family on EVERY representation (the objective 'all the clustering methods should be used').

Families (configurations in brackets), each fitted on a seeded 6,000-point subsample (all points for the 1,000-file AVES sets):
  centroid / model      k-means [k=2..6]; fuzzy c-means [c=2..6, FPC]; GMM with spherical, diagonal, tied and full covariance [k=1..6, BIC];
                        Bayesian GMM with a Dirichlet-process prior [10 components, effective clusters = weight > 1%]; k-medoids (PAM) [k=2..6]
  hierarchical          Ward, average, complete, single linkage [k=2..6]
  graph / spectral      spectral clustering with a 15-NN affinity [k=2..6]; Louvain and Leiden community detection on the 15-NN graph [resolution 1.0; Leiden also 0.1]
  density / mode        DBSCAN [eps = 50/75/90th percentile of the 10-NN distance, min 10]; OPTICS [xi 0.05]; HDBSCAN [min size 25, 100]; mean-shift [estimated bandwidth]
  other                 BIRCH [k=2..6]; affinity propagation [damping 0.9]; self-organising map [3x3 grid]
Per configuration: clusters found (all / holding >= 1% of points), noise share, silhouette (non-noise points), NMI with recording channel and with silence-fraction quintile (when known),
the best cluster for pups' Isolation calls (precision, recall, F1; 'found' = precision >= 0.7 and recall >= 0.5), seconds.  For the best-silhouette configuration of every family (among configurations with >= 2 clusters of >= 1% of the points):
bat-transfer stability = adjusted Rand index between a fit on one random half of the bats applied (5-NN transfer) to the other half and a fit on that half itself (2 splits).
Run:  python code/claude/38_all_methods.py <tag> [<tag> ...]    (writes results/step3m_all_methods_<tag>.csv)
      python code/claude/38_all_methods.py summary                (pools them: results/step3m_all_methods_summary.csv + figures/step3m_all_methods.png)
"""
import os
import sys
import time
import warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, AffinityPropagation, Birch, KMeans, MeanShift, SpectralClustering, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances, silhouette_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, kneighbors_graph

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, RES, SEED, log_change  # noqa: E402
from clusterlib import fcm_best, fuzzy_indices, load_set  # noqa: E402

N_SUB = 6000
DEFAULT_SETS = ["all2_naive_ref1", "all2_voiced_ref1", "all2_mfcc_naive", "all2_mfcc_voiced", "segall_shape", "segall_mfcc", "segall_spec", "segall_vae8_b0.1", "segall_pca8",
                "B_mic_voiced_ref1", "B_al_voiced_ref1"] + [f"aves2{m}_aves_{v}_{p}" for m in ("bio", "all") for v in ("ds16", "te2") for p in ("naive", "voiced")]
_cache = {}


def _lab(x):
    return np.asarray(x, dtype=int)


def hier(method, k):
    def f(X, s):
        key = (id(X), method)
        if key not in _cache:
            for old in [o for o in _cache if o[1] == method]:
                del _cache[old]
            _cache[key] = linkage(X, method=method)
        return fcluster(_cache[key], k, criterion="maxclust") - 1
    return f


def knn_graph(X, k=15):
    A = kneighbors_graph(X, k, mode="connectivity", include_self=False)
    return ((A + A.T) > 0).astype(int).tocoo()


def louvain(X, s):
    import networkx as nx
    A = knn_graph(X)
    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))
    G.add_edges_from(zip(A.row.tolist(), A.col.tolist()))
    lab = np.zeros(X.shape[0], int)
    for c, nodes in enumerate(nx.community.louvain_communities(G, resolution=1.0, seed=int(s))):
        lab[list(nodes)] = c
    return lab


def leiden(res):
    def f(X, s):
        import igraph as ig
        import leidenalg
        A = knn_graph(X)
        g = ig.Graph(n=X.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())))
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=int(s))
        return np.array(part.membership)
    return f


def som(X, s):
    from minisom import MiniSom
    m = MiniSom(3, 3, X.shape[1], sigma=1.0, learning_rate=0.5, random_seed=int(s))
    m.random_weights_init(X)
    m.train_random(X, 3000)
    return np.array([i * 3 + j for i, j in (m.winner(x) for x in X)])


def pam(k):
    def f(X, s):
        D = pairwise_distances(X.astype(np.float32))
        rg = np.random.default_rng(int(s))
        med = [int(rg.integers(len(X)))]
        for _ in range(k - 1):
            d = D[:, med].min(1).astype(np.float64) ** 2
            med.append(int(rg.choice(len(X), p=d / d.sum())))
        med = np.array(med)
        for _ in range(20):
            lab = D[:, med].argmin(1)
            new = med.copy()
            for c in range(k):
                idx = np.where(lab == c)[0]
                if len(idx):
                    new[c] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
            if (new == med).all():
                break
            med = new
        return D[:, med].argmin(1)
    return f


def affinity(X, s):
    rg = np.random.default_rng(int(s))
    sub = rg.choice(len(X), min(3000, len(X)), replace=False)
    ap = AffinityPropagation(damping=0.9, max_iter=200, convergence_iter=15, random_state=int(s)).fit(X[sub])
    return ap.predict(X)


def meanshift(X, s):
    bw = estimate_bandwidth(X, quantile=0.2, n_samples=500, random_state=int(s))
    return MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=1, max_iter=100).fit_predict(X)


def gmm(cov, k):
    def f(X, s):
        g = GaussianMixture(k, covariance_type=cov, random_state=int(s), reg_covar=1e-4, max_iter=200).fit(X)
        return g.predict(X), dict(bic=float(g.bic(X)))
    return f


def bgmm(X, s):
    g = BayesianGaussianMixture(n_components=10, covariance_type="diag", weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=0.01,
                                random_state=int(s), max_iter=300, reg_covar=1e-4).fit(X)
    return g.predict(X), dict(effective_components=int((g.weights_ > 0.01).sum()))


def fcm(c):
    def f(X, s):
        r = fcm_best(X, c, seed=int(s), n_init=3)
        fpc, _ = fuzzy_indices(X, r["U"], r["V"])
        return r["U"].argmax(0), dict(fpc=float(fpc))
    return f


def configs(X):
    d10 = NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)[0][:, -1]
    C = []
    for k in range(2, 7):
        C.append(("k-means", k, lambda X, s, k=k: KMeans(k, n_init=5, random_state=int(s)).fit_predict(X)))
        C.append(("fuzzy c-means", k, fcm(k)))
        C.append(("k-medoids (PAM)", k, pam(k)))
        for m in ("ward", "average", "complete", "single"):
            C.append((f"{m} linkage", k, hier(m, k)))
        C.append(("spectral (15-NN)", k, lambda X, s, k=k: SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, assign_labels="kmeans", random_state=int(s)).fit_predict(X)))
        C.append(("BIRCH", k, lambda X, s, k=k: Birch(n_clusters=k, threshold=float(np.sqrt(X.var(0).sum()) * 0.15)).fit_predict(X)))
    for cov in ("spherical", "diag", "tied", "full"):
        for k in range(1, 7):
            C.append((f"GMM-{cov}", k, gmm(cov, k)))
    C.append(("Bayesian GMM (Dirichlet process)", 10, bgmm))
    for q in (50, 75, 90):
        eps = float(np.percentile(d10, q))
        C.append((f"DBSCAN eps=p{q}", 0, lambda X, s, eps=eps: DBSCAN(eps=eps, min_samples=10).fit_predict(X)))
    C.append(("OPTICS (xi 0.05)", 0, lambda X, s: OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.01, n_jobs=1).fit_predict(X)))
    for m in (25, 100):
        C.append((f"HDBSCAN min {m}", 0, lambda X, s, m=m: HDBSCAN(min_cluster_size=m, min_samples=10).fit_predict(X)))
    C += [("mean-shift", 0, meanshift), ("affinity propagation", 0, affinity), ("Louvain (15-NN graph)", 0, louvain), ("Leiden res 1.0 (15-NN graph)", 0, leiden(1.0)),
          ("Leiden res 0.1 (15-NN graph)", 0, leiden(0.1)), ("self-organising map 3x3", 0, som)]
    return C


def balanced_halves(bat, rg):
    ub = np.unique(bat)
    for _ in range(60):
        A = np.isin(bat, rg.permutation(ub)[: len(ub) // 2])
        if 0.3 <= A.mean() <= 0.7:
            return A
    return A


def _fit_labels(fn, X, s):
    out = fn(X, s)
    return _lab(out[0] if isinstance(out, tuple) else out)


def transfer_ari(fn, X, bat, seed, reps=2):
    rg = np.random.default_rng(seed)
    out = []
    for r in range(reps):
        A = balanced_halves(bat, rg)
        try:
            la = _fit_labels(fn, X[A], seed + r)
            lb = _fit_labels(fn, X[~A], seed + r + 50)
            pred = KNeighborsClassifier(5).fit(X[A], la).predict(X[~A])
            out.append(adjusted_rand_score(lb, pred))
        except Exception:
            out.append(np.nan)
    return float(np.nanmean(out)) if not np.all(np.isnan(out)) else np.nan


def score(X, lab, ch, silbin, iso):
    n = len(lab)
    ok = lab >= 0
    ids = [c for c in np.unique(lab) if c >= 0]
    sizes = np.array([(lab == c).sum() for c in ids]) if ids else np.array([])
    r = dict(n_clusters=len(ids), n_major=int((sizes >= 0.01 * n).sum()) if len(ids) else 0, noise=float(1 - ok.mean()), largest_share=float(sizes.max() / n) if len(ids) else np.nan)
    r["silhouette"] = np.nan
    if len(ids) >= 2 and ok.sum() > 50:
        rg = np.random.default_rng(SEED)
        sub = rg.choice(np.where(ok)[0], min(3000, ok.sum()), replace=False)
        if len(np.unique(lab[sub])) >= 2:
            r["silhouette"] = float(silhouette_score(X[sub], lab[sub]))
    r["nmi_channel"] = float(normalized_mutual_info_score(ch, lab)) if ch is not None else np.nan
    r["nmi_silence_quintile"] = float(normalized_mutual_info_score(silbin, lab)) if silbin is not None else np.nan
    r.update(iso_best_f1=np.nan, iso_precision=np.nan, iso_recall=np.nan, iso_found=np.nan)
    if iso is not None and iso.sum() >= 10 and len(ids):
        best = (0, 0, 0)
        for c in ids:
            m = lab == c
            p, rc = iso[m].mean(), iso[m].sum() / iso.sum()
            f1 = 2 * p * rc / (p + rc) if p + rc > 0 else 0
            if f1 > best[0]:
                best = (f1, p, rc)
        r.update(iso_best_f1=float(best[0]), iso_precision=float(best[1]), iso_recall=float(best[2]), iso_found=bool(best[1] >= 0.7 and best[2] >= 0.5))
    return r


def run_set(tag):
    X, M = load_set(tag)
    ii = np.sort(np.random.default_rng(SEED).choice(len(X), min(N_SUB, len(X)), replace=False))
    X, M = X[ii].astype(np.float64), M.iloc[ii].reset_index(drop=True)
    bat = M.bat.values if "bat" in M else np.zeros(len(M))
    ch = M.channel.astype(str).values if "channel" in M else None
    iso = (M.context_name.values == "Isolation") if "context_name" in M else None
    silbin = pd.qcut(M.silence_fraction, 5, labels=False, duplicates="drop").values if "silence_fraction" in M and M.silence_fraction.notna().all() else None
    rows, fns = [], {}
    for fam, k, fn in configs(X):
        t = time.perf_counter()
        try:
            out = fn(X, SEED)
            lab, extra = (out if isinstance(out, tuple) else (out, {}))
            r = dict(set=tag, n=len(X), family=fam, param=k, **score(X, _lab(lab), ch, silbin, iso), **extra)
        except Exception as e:
            r = dict(set=tag, n=len(X), family=fam, param=k, error=repr(e)[:80])
        r["seconds"] = time.perf_counter() - t
        rows.append(r)
        fns[(fam, k)] = fn
        print(f"{tag} | {fam} {k}: clusters {r.get('n_clusters')} sil {r.get('silhouette')} {r['seconds']:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R["bat_transfer_ari"] = np.nan
    for fam, g in R.groupby("family"):
        g = g[(g.n_major >= 2) & g.silhouette.notna()]  # a partition needs >= 2 clusters holding >= 1% of the points (single linkage 'wins' on silhouette by peeling off singletons)
        if g.empty:
            continue
        b = g.loc[g.silhouette.idxmax()]
        if b.seconds > 90:
            continue
        R.loc[(R.family == fam) & (R.param == b.param), "bat_transfer_ari"] = transfer_ari(fns[(fam, b.param)], X, bat, SEED)
    R.to_csv(RES / f"step3m_all_methods_{tag}.csv", index=False)
    return tag


def summary():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("step3m_all_methods_*.csv")) if "summary" not in f.name], ignore_index=True)
    best = D[(D.n_major >= 2) & D.silhouette.notna()].sort_values("silhouette", ascending=False).groupby(["set", "family"]).head(1)
    best.to_csv(RES / "step3m_all_methods_summary.csv", index=False)
    fam_order = list(D.family.drop_duplicates())
    fams = sorted(set(best.family), key=lambda f: fam_order.index(f))
    sets = [s for s in DEFAULT_SETS if s in set(best.set)]
    fig, ax = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    for a, col, ttl, cm in ((ax[0], "silhouette", "silhouette of the best configuration", "viridis"), (ax[1], "iso_best_f1", "best cluster vs pups' Isolation calls (F1)", "magma"),
                            (ax[2], "bat_transfer_ari", "bat-transfer stability (ARI)", "cividis")):
        P = best.pivot(index="family", columns="set", values=col).reindex(index=fams, columns=sets)
        im = a.imshow(P.values.astype(float), aspect="auto", cmap=cm, vmin=0, vmax=0.6 if col == "silhouette" else 1)
        a.set_xticks(range(len(sets)))
        a.set_xticklabels(sets, rotation=75, ha="right", fontsize=6)
        a.set_title(ttl, fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.03)
    ax[0].set_yticks(range(len(fams)))
    ax[0].set_yticklabels(fams, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "step3m_all_methods.png", dpi=140)
    found = D[D.iso_found == True].groupby("family").set.nunique().sort_values(ascending=False)  # noqa: E712
    print("families that isolate pups' Isolation calls (precision >= 0.7, recall >= 0.5) in at least one configuration, by number of representations:")
    print(found.to_string())
    log_change(f"Step 3m every clustering family on every representation: {D.family.nunique()} families, {D.set.nunique()} sets; results/step3m_all_methods_*.csv")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        tags = sys.argv[1:]
        with ProcessPoolExecutor(max_workers=min(int(os.environ.get("WORKERS", 3)), len(tags))) as ex:
            for t in ex.map(run_set, tags):
                print("done", t, flush=True)
