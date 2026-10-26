"""Step 3c on the DRAFT's saved matrix (X_pca_train.npy, unique files) - clusterability tests that allow k=1,
resampled by BAT, against two nulls that keep the marginals:

  statistics (each: larger = more cluster-like)
    FPC2_excess     fuzzy partition coefficient at c=2 minus 0.5 (0.5 = the k=1 collapse value)
    Silhouette2     silhouette of the c=2 FCM hard labels (3,000-row subsample rule)
    Gap21           gap(2) - gap(1)  (Tibshirani gap statistic, k-means, PCA-box uniform reference, B=10)
    PS2             prediction strength at k=2, k-means, train/test halves split by BAT (mean of 20 splits, both directions)
    Jaccard2        bootstrap Jaccard stability of k-means k=2, bats resampled with replacement (50 resamples)
    DipSplit        mean Hartigan dip of the held-out half projected on a centroid axis learned on the OTHER half of bats
    DipSplit_p<.05  fraction of splits with dip p < 0.05
  also reported: k chosen by the gap rule, PS at k=3, median dip p

  nulls (each B simulated datasets, bat/strata labels kept on the same rows)
    copula   Gaussian copula: empirical marginals of every PC + the rank-correlation matrix; no other structure
    session  residuals about channel x month stratum means are copula-simulated, stratum means added back
             (keeps between-session shifts; nothing else)
  p = (1 + #null >= observed) / (1 + B), reported as p<1/(B+1) when no null replicate reaches the observed value.
  CI for the observed value: resample bats with replacement (30 resamples).

Run: python code/claude/03c_clusterability_old_matrix.py [B_null=100] [B_boot=30]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, FIG, OLD, RES, SEED, log_change  # noqa: E402
from clusterlib import fcm_best, fuzzy_indices, load_set, silhouette_fixed  # noqa: E402


def km(X, k, seed):
    return KMeans(n_clusters=k, n_init=3, random_state=int(seed)).fit(X)


def gap_stat(X, seed, kmax=6, B=10):
    rg = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    logW = np.array([np.log(km(X, k, seed).inertia_) for k in range(1, kmax + 1)])
    ref = np.zeros((B, kmax))
    for b in range(B):
        R = rg.uniform(lo, hi, size=X.shape)
        for k in range(1, kmax + 1):
            ref[b, k - 1] = np.log(km(R, k, seed + b).inertia_)
    gap = ref.mean(0) - logW
    s = ref.std(0) * np.sqrt(1 + 1 / B)
    kc = next((k for k in range(1, kmax) if gap[k - 1] >= gap[k] - s[k]), kmax)  # smallest k with gap(k) >= gap(k+1) - s(k+1)
    return gap, kc


def _balanced_halves(bat, rg, tries=60):
    ub = np.unique(bat)
    for _ in range(tries):
        perm = rg.permutation(ub)
        A = np.isin(bat, perm[: len(perm) // 2])
        if 0.35 <= A.mean() <= 0.65:
            return A
    return A


def pred_strength(X, bat, ks, seed, S=20):
    rg = np.random.default_rng(seed)
    out = {k: [] for k in ks}
    for s in range(S):
        A = _balanced_halves(bat, rg)
        for k in ks:
            ps = []
            for tr, te in ((A, ~A), (~A, A)):
                lt = km(X[tr], k, seed + s).predict(X[te])
                lc = km(X[te], k, seed + s + 1).labels_
                m = []
                for j in range(k):
                    ii = np.where(lc == j)[0]
                    nj = len(ii)
                    if nj < 2:
                        continue
                    same = (lt[ii][:, None] == lt[ii][None, :]).sum() - nj
                    m.append(same / (nj * (nj - 1)))
                ps.append(min(m) if m else np.nan)
            out[k].append(np.nanmean(ps))
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def boot_jaccard(X, bat, k, seed, B=50):
    rg = np.random.default_rng(seed)
    lab = km(X, k, seed).labels_
    ub = np.unique(bat)
    idx = {b: np.where(bat == b)[0] for b in ub}
    J = []
    for b in range(B):
        ii = np.concatenate([idx[p] for p in rg.choice(ub, len(ub), replace=True)])
        lb = np.full(len(X), -1)
        lb[ii] = km(X[ii], k, seed + b).labels_
        inb = lb >= 0
        for i in range(k):
            Ci = (lab == i) & inb
            J.append(max(((Ci & (lb == j)).sum() / max(1, (Ci | (lb == j)).sum())) for j in range(k)))
    return float(np.mean(J))


def split_dip(X, bat, seed, S=20):
    import diptest
    rg = np.random.default_rng(seed)
    dips, ps = [], []
    for s in range(S):
        A = _balanced_halves(bat, rg)
        for tr, te in ((A, ~A), (~A, A)):
            c = km(X[tr], 2, seed + s).cluster_centers_
            v = c[1] - c[0]
            v /= np.linalg.norm(v)
            d, p = diptest.diptest(X[te] @ v)
            dips.append(d)
            ps.append(p)
    return float(np.mean(dips)), float(np.median(ps)), float(np.mean(np.array(ps) < 0.05))


def all_stats(X, bat, seed):
    r = fcm_best(X, 2, seed=seed, n_init=5)
    fpc, xb = fuzzy_indices(X, r["U"], r["V"])
    gap, kc = gap_stat(X, seed)
    ps = pred_strength(X, bat, (2, 3), seed)
    d, pmed, pfrac = split_dip(X, bat, seed)
    return {"FPC2_excess": fpc - 0.5, "Silhouette2": silhouette_fixed(X, r["U"].argmax(0)), "Gap21": float(gap[1] - gap[0]),
            "gap_k_chosen": kc, "PS2": ps[2], "PS3": ps[3], "Jaccard2": boot_jaccard(X, bat, 2, seed),
            "DipSplit": d, "DipSplit_median_p": pmed, "DipSplit_frac_p<.05": pfrac}


# ---------------------------------------------------------------- nulls
def copula_fit(X):
    n, d = X.shape
    ranks = np.argsort(np.argsort(X, 0), 0) + 1
    Z = norm.ppf(ranks / (n + 1))
    L = np.linalg.cholesky(np.corrcoef(Z.T) + 1e-9 * np.eye(d))
    return L, np.sort(X, 0)


def copula_sim(fit, n, rg):
    L, sx = fit
    U = norm.cdf(rg.standard_normal((n, L.shape[0])) @ L.T)
    grid = (np.arange(sx.shape[0]) + 0.5) / sx.shape[0]
    return np.column_stack([np.interp(U[:, j], grid, sx[:, j]) for j in range(sx.shape[1])])


def task(args):
    kind, b, X, bat, strata = args
    seed = SEED + 1000 * {"obs": 0, "boot": 1, "copula": 2, "session": 3}[kind] + b
    rg = np.random.default_rng(seed)
    t = time.perf_counter()
    if kind == "obs":
        Xs, bs = X, bat
    elif kind == "boot":
        ub = np.unique(bat)
        idx = {u: np.where(bat == u)[0] for u in ub}
        pick = rg.choice(ub, len(ub), replace=True)
        ii = [idx[p] for p in pick]
        Xs = X[np.concatenate(ii)]
        bs = np.concatenate([np.full(len(a), j) for j, a in enumerate(ii)])
    elif kind == "copula":
        Xs, bs = copula_sim(copula_fit(X), len(X), rg), bat
    else:
        mu = pd.DataFrame(X).groupby(strata).transform("mean").values
        Xs, bs = mu + copula_sim(copula_fit(X - mu), len(X), rg), bat
    out = all_stats(Xs, bs, seed)
    out.update(kind=kind, b=b, seconds=time.perf_counter() - t)
    return out


if __name__ == "__main__":
    B_NULL = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    B_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    TAG = sys.argv[3] if len(sys.argv) > 3 else "old"  # 'old' = draft matrix; else e.g. baseline_voiced_ref1 (see 05_build_sets.py)
    NAME = "old_matrix" if TAG == "old" else TAG
    X, meta = load_set(TAG)
    bat, strata = meta.bat.values, meta.stratum.values
    print(f"unique files {len(X)}, bats(|emitter| groups) {len(np.unique(bat))}, strata {len(np.unique(strata))}; B_null={B_NULL}, B_boot={B_BOOT}", flush=True)

    jobs = [("obs", 0, X, bat, strata)] + [("boot", b, X, bat, strata) for b in range(B_BOOT)] + \
           [(k, b, X, bat, strata) for k in ("copula", "session") for b in range(B_NULL)]
    t0 = time.perf_counter()
    res = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(task, jobs, chunksize=1), 1):
            res.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} done, {time.perf_counter()-t0:.0f}s", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(RES / f"step3c_{NAME}_raw.csv", index=False)

    stats = ["FPC2_excess", "Silhouette2", "Gap21", "PS2", "Jaccard2", "DipSplit", "DipSplit_frac_p<.05"]
    obs = df[df.kind == "obs"].iloc[0]
    out = []
    for s in stats:
        bt = df[df.kind == "boot"][s]
        row = dict(statistic=s, observed=obs[s], boot_by_bat_q025=bt.quantile(.025), boot_by_bat_q975=bt.quantile(.975), n_boot=len(bt))
        for nk in ("copula", "session"):
            nv = df[df.kind == nk][s].values
            ge = int((nv >= obs[s]).sum())
            row.update({f"{nk}_null_mean": nv.mean(), f"{nk}_null_q025": np.quantile(nv, .025), f"{nk}_null_q975": np.quantile(nv, .975),
                        f"{nk}_n_ge_obs": ge, f"{nk}_p": (f"<{1/(len(nv)+1):.4f} (0/{len(nv)})" if ge == 0 else f"{(1+ge)/(1+len(nv)):.4f} ({ge}/{len(nv)})")})
        out.append(row)
    T = pd.DataFrame(out)
    T.to_csv(RES / f"step3c_{NAME}_summary.csv", index=False)
    extra = pd.DataFrame([{"kind": k, "gap_k_chosen_counts": str(df[df.kind == k].gap_k_chosen.value_counts().sort_index().to_dict()),
                           "PS3_mean": df[df.kind == k].PS3.mean(), "median_dip_p_mean": df[df.kind == k]["DipSplit_median_p"].mean(),
                           "mean_seconds_per_task": df[df.kind == k].seconds.mean()} for k in ("obs", "boot", "copula", "session")])
    extra.to_csv(RES / f"step3c_{NAME}_extra.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string())
    print(extra.to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(15, 6))
    for a, s in zip(ax.ravel(), stats):
        for nk, colr in (("copula", "#4c78a8"), ("session", "#59a14f")):
            a.hist(df[df.kind == nk][s], bins=20, alpha=.55, color=colr, label=f"{nk} null")
        a.axvline(obs[s], color="#e45756", lw=2, label="observed (draft matrix)")
        a.set_title(s, fontsize=9)
    ax.ravel()[0].legend(fontsize=7)
    ax.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.suptitle(f"set: {TAG}", fontsize=9)
    fig.savefig(FIG / f"step3c_{NAME}.png", dpi=150)
    log_change(f"Step 3c clusterability on set '{TAG}': n={len(X)}, B_null={B_NULL}, B_boot={B_BOOT}; results/step3c_{NAME}_*")
