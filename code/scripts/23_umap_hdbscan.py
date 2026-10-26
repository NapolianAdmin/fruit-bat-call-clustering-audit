"""Step 3i - the standard bioacoustic pipeline (UMAP latent space -> HDBSCAN, as in Sainburg et al. 2020 / Thomas et al. 2022) WITH the controls it usually lacks.

UMAP + density clustering nearly always returns clusters, because neighbourhood graphs of a continuum are locally clumpy.  Here the same pipeline is run on
  (1) the real data,
  (2) B Gaussian-copula null datasets and B session-mean null datasets (no discrete structure, same size, same bat / stratum labels), and
  (3) bat bootstraps (rows of resampled bats; clusters of the full run matched to the best bootstrap cluster by Jaccard).
Statistics per HDBSCAN minimum cluster size (30 / 100 / 300): number of clusters, share of points labelled noise, silhouette of the non-noise points in the 10-D embedding;
for the real data also the bat-bootstrap Jaccard stability and the association of the clusters with recording variables (Cramer's V with channel, medians of voiced duration
and silence).  UMAP settings: 10 components, n_neighbors 30, min_dist 0.0, euclidean, random_state = SEED; sensitivity runs n_neighbors 15 and 100 (real data only).
A 2-D UMAP of the real data is drawn for the figure (visual only; not used for any statistic).
Run: python code/claude/23_umap_hdbscan.py <tag> [n_rows=12000] [B_null=12] [B_boot=8]
"""
import importlib.util
import os
import warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FIG, RES, SEED, cramers_v, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

_spec = importlib.util.spec_from_file_location("c03", HERE / "03c_clusterability_old_matrix.py")
c03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c03)

MCS = (30, 100, 300)


def embed(X, nn=30, nc=10, seed=SEED):
    import umap
    return umap.UMAP(n_components=nc, n_neighbors=nn, min_dist=0.0, metric="euclidean", random_state=seed).fit_transform(X)


def cluster_stats(E, mcs):
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=10).fit_predict(E)
    ok = lab >= 0
    k = len(set(lab[ok]))
    sil = np.nan
    if k > 1 and ok.sum() > 100:
        rg = np.random.default_rng(SEED)
        sub = rg.choice(np.where(ok)[0], min(3000, ok.sum()), replace=False)
        sil = float(silhouette_score(E[sub], lab[sub])) if len(set(lab[sub])) > 1 else np.nan
    return lab, dict(k=k, noise=float((~ok).mean()), sil=sil)


def jaccard_match(lab, lab_b, in_b):
    out = []
    for i in sorted(set(lab[lab >= 0])):
        Ci = (lab == i) & in_b
        best = 0.0
        for j in sorted(set(lab_b[lab_b >= 0])):
            Cj = (lab_b == j)
            best = max(best, (Ci & Cj).sum() / max(1, (Ci | Cj).sum()))
        out.append(best)
    return float(np.mean(out)) if out else np.nan


def task(args):
    kind, b, nn, X, bat, strata, labs_full = args
    seed = SEED + 1000 * {"obs": 0, "boot": 1, "copula": 2, "session": 3}[kind] + b
    rg = np.random.default_rng(seed)
    t = time.perf_counter()
    row = dict(kind=kind, b=b, nn=nn)
    if kind in ("obs", "boot"):
        if kind == "boot":
            ub = np.unique(bat)
            pick = rg.choice(ub, len(ub), replace=True)
            in_b = np.isin(bat, np.unique(pick))
            Xs = X[in_b]
        else:
            in_b, Xs = np.ones(len(X), bool), X
        E = embed(Xs, nn, seed=seed if kind == "boot" else SEED)
    else:
        if kind == "copula":
            Xs = c03.copula_sim(c03.copula_fit(X), len(X), rg)
        else:
            mu = pd.DataFrame(X).groupby(strata).transform("mean").values
            Xs = mu + c03.copula_sim(c03.copula_fit(X - mu), len(X), rg)
        E = embed(Xs, nn, seed=seed)
    for m in MCS:
        lab, s = cluster_stats(E, m)
        for k_, v in s.items():
            row[f"{k_}_{m}"] = v
        if kind == "boot":
            full = np.full(len(X), -1)
            full[in_b] = lab
            row[f"jac_{m}"] = jaccard_match(labs_full[m], full, in_b) if labs_full is not None else np.nan
        if kind == "obs" and nn == 30:
            row[f"labels_{m}"] = lab
    row["seconds"] = time.perf_counter() - t
    return row


if __name__ == "__main__":
    tag = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    B_NULL = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    B_BOOT = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    X, meta = load_set(tag)
    ii = np.sort(np.random.default_rng(SEED).choice(len(X), min(n, len(X)), replace=False))
    X0, M = X[ii], meta.iloc[ii].reset_index(drop=True)
    bat, strata = M.bat.values, M.stratum.values
    DUR = "voiced_dur_s" if "voiced_dur_s" in M else "span_len_s"  # syllable sets carry the segment duration as span_len_s
    SIL = M["silence_fraction"].values if "silence_fraction" in M else np.full(len(M), np.nan)
    W = int(os.environ.get("WORKERS", 4))
    print(f"{tag}: {X0.shape}, bats {len(np.unique(bat))}; B_null={B_NULL} B_boot={B_BOOT} workers={W}", flush=True)

    # 1. real data (+ sensitivity of n_neighbors); first, because the bootstraps need its labels
    with ProcessPoolExecutor(max_workers=min(W, 3)) as ex:
        obs = list(ex.map(task, [("obs", 0, nn, X0, bat, strata, None) for nn in (30, 15, 100)]))
    ob30 = obs[0]
    labs_full = {m: ob30[f"labels_{m}"] for m in MCS}
    print("real data:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ob30.items() if not k.startswith("labels_")}, flush=True)

    jobs = [("boot", b, 30, X0, bat, strata, labs_full) for b in range(B_BOOT)] + \
           [(k, b, 30, X0, bat, strata, None) for k in ("copula", "session") for b in range(B_NULL)]
    res = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, r in enumerate(ex.map(task, jobs, chunksize=1), 1):
            res.append(r)
            print(f"{i}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("labels_")} for r in obs + res])
    df.to_csv(RES / f"step3i_umap_{tag}_raw.csv", index=False)

    summ = []
    for m in MCS:
        for s in ("k", "noise", "sil"):
            col = f"{s}_{m}"
            row = dict(min_cluster_size=m, statistic=s, observed=float(df[(df.kind == "obs") & (df.nn == 30)][col].iloc[0]),
                       obs_nn15=float(df[(df.kind == "obs") & (df.nn == 15)][col].iloc[0]), obs_nn100=float(df[(df.kind == "obs") & (df.nn == 100)][col].iloc[0]))
            for nk in ("copula", "session"):
                v = df[df.kind == nk][col].dropna().values
                row.update({f"{nk}_mean": float(np.mean(v)) if len(v) else np.nan, f"{nk}_min": float(np.min(v)) if len(v) else np.nan,
                            f"{nk}_max": float(np.max(v)) if len(v) else np.nan, f"{nk}_n": len(v)})
            summ.append(row)
        jb = df[df.kind == "boot"][f"jac_{m}"].dropna()
        summ.append(dict(min_cluster_size=m, statistic="bat_bootstrap_jaccard_mean_and_range", observed=float(jb.mean()) if len(jb) else np.nan,
                         copula_min=float(jb.min()) if len(jb) else np.nan, copula_max=float(jb.max()) if len(jb) else np.nan))
    S = pd.DataFrame(summ)
    S.to_csv(RES / f"step3i_umap_{tag}_summary.csv", index=False)

    assoc = []
    for m in MCS:
        lab = labs_full[m]
        ok = lab >= 0
        for c in sorted(set(lab[ok])):
            g = lab == c
            vc = M.channel[g].value_counts(normalize=True)
            assoc.append(dict(min_cluster_size=m, cluster=int(c), n=int(g.sum()), share=float(g.mean()), n_bats=int(len(np.unique(bat[g]))),
                              voiced_dur_median_in=float(np.nanmedian(M[DUR].values[g])), voiced_dur_median_out=float(np.nanmedian(M[DUR].values[~g])),
                              silence_median_in=float(np.nanmedian(SIL[g])), silence_median_out=float(np.nanmedian(SIL[~g])),
                              top_channel=str(vc.index[0]), top_channel_share=float(vc.iloc[0])))
        if len(set(lab[ok])) > 1:
            assoc.append(dict(min_cluster_size=m, cluster="ALL non-noise (Cramer V with channel)", n=int(ok.sum()), share=float(ok.mean()),
                              cramers_v_channel=float(cramers_v(lab[ok], M.channel.values[ok])[0])))
    A = pd.DataFrame(assoc)
    A.to_csv(RES / f"step3i_umap_{tag}_clusters.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    print(A.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E2 = embed(X0, 30, nc=2)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    lab = labs_full[100]
    ax[0].scatter(E2[:, 0], E2[:, 1], c=np.log10(M[DUR].values + 1e-3), s=1, cmap="viridis", rasterized=True)
    ax[0].set_title("log10 voiced duration", fontsize=9)
    ax[1].scatter(E2[:, 0], E2[:, 1], c=pd.factorize(M.channel)[0], s=1, cmap="tab20", rasterized=True)
    ax[1].set_title("recording channel", fontsize=9)
    ax[2].scatter(E2[:, 0], E2[:, 1], c=np.where(lab < 0, -1, lab), s=1, cmap="tab10", rasterized=True)
    ax[2].set_title(f"HDBSCAN (10-D UMAP, min size 100): {len(set(lab[lab >= 0]))} clusters, {np.mean(lab < 0):.0%} noise", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"UMAP of set {tag} (n={len(X0)}); the 2-D map is for display only", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"step3i_umap_{tag}.png", dpi=150)
    log_change(f"Step 3i UMAP+HDBSCAN on '{tag}' (n={len(X0)}, B_null={B_NULL}, B_boot={B_BOOT}): results/step3i_umap_{tag}_*")
