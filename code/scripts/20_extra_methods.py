"""Step 3b (extra) - the remaining standard clustering families on a 6,000-point subsample per set (quadratic-memory methods cannot run on 40k):
  Ward agglomerative (k=2..6), spectral clustering with a 15-NN affinity (k=2..6), BIRCH (k=2..6), DBSCAN (eps at the 50/75/90th percentile of the
  10-NN distance, min_samples=10).  For each: silhouette (hard labels), clusters found, noise share, wall seconds.
The k that maximises the silhouette is NOT evidence for k clusters (k=1 has no silhouette); it is reported next to the one-cluster-null comparison
in 03c.  Run: python code/claude/20_extra_methods.py <tag> [<tag> ...]
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, Birch, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SEED, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

warnings.filterwarnings("ignore")
N_SUB = 6000
rows = []
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    rg = np.random.default_rng(SEED)
    ii = np.sort(rg.choice(len(X), min(N_SUB, len(X)), replace=False))
    Xs = X[ii]
    thr = float(np.sqrt(Xs.var(0).sum()) * 0.15)
    print(tag, Xs.shape, flush=True)
    for k in range(2, 7):
        for name, mk in [("Ward", lambda k=k: AgglomerativeClustering(n_clusters=k, linkage="ward")),
                         ("Spectral(15-NN)", lambda k=k: SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=15, random_state=SEED, assign_labels="kmeans")),
                         ("BIRCH", lambda k=k: Birch(n_clusters=k, threshold=thr))]:
            t = time.perf_counter()
            try:
                lab = mk().fit_predict(Xs)
                sil = silhouette_score(Xs, lab) if len(set(lab)) > 1 else np.nan
                rows.append(dict(set=tag, n=len(Xs), method=name, k=k, clusters_found=len(set(lab)), silhouette=sil, noise_share=0.0,
                                 smallest_cluster_share=float(np.bincount(lab).min() / len(lab)), seconds=time.perf_counter() - t))
            except Exception as e:
                rows.append(dict(set=tag, n=len(Xs), method=name, k=k, clusters_found=np.nan, silhouette=np.nan, noise_share=np.nan, smallest_cluster_share=np.nan,
                                 seconds=time.perf_counter() - t, note=repr(e)[:80]))
    d10 = NearestNeighbors(n_neighbors=10).fit(Xs).kneighbors(Xs)[0][:, -1]
    for q in (50, 75, 90):
        eps = float(np.percentile(d10, q))
        t = time.perf_counter()
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Xs)
        kk = len(set(lab)) - (1 if -1 in lab else 0)
        ok = lab >= 0
        sil = silhouette_score(Xs[ok], lab[ok]) if kk > 1 and ok.sum() > 50 else np.nan
        rows.append(dict(set=tag, n=len(Xs), method=f"DBSCAN(eps=p{q} of 10-NN dist)", k=kk, clusters_found=kk, silhouette=sil, noise_share=float((~ok).mean()),
                         smallest_cluster_share=(float(np.bincount(lab[ok]).min() / ok.sum()) if kk else np.nan), seconds=time.perf_counter() - t))
R = pd.DataFrame(rows)
R.to_csv(RES / "step3b_extra_methods.csv", index=False)
log_change(f"Step 3b extra methods (Ward, spectral, BIRCH, DBSCAN) on {', '.join(sys.argv[1:])}: results/step3b_extra_methods.csv")
pd.set_option("display.width", 250)
print(R.round(3).to_string())
