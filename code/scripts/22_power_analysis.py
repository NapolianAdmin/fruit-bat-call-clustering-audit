"""Step 3h - POWER of the k=1-allowed battery: can it see a discrete structure if one is there?

A negative result (no stable clusters, no dip) is only informative if the same tests would have detected a real discrete mixture of
the same size in the same data.  Positive control: take a null dataset with NO discrete structure (session-mean null of step 3c: stratum means
kept, residuals simulated from a Gaussian copula of the real marginals/rank correlations; bat and stratum labels kept on the rows), and spike in a
two-type mixture:
    a random fraction `pi` of the rows (chosen independently of bat) is shifted by  delta * sigma_u * u,
    u = random unit vector in the span of the top-10 principal components, sigma_u = SD of the base data along u,
so `delta` is the separation between the two types in SD units along u (delta = 0 is the null).  The statistics are those of
03c (FPC excess, silhouette, prediction strength across bats, bootstrap Jaccard across bats, split-sample dip across bats), with lighter
settings (see LITE below).  Detection of a statistic = value above the 95th percentile of its delta=0 distribution (alpha = 0.05 by construction);
for the dip test also the plain rule 'median split-sample dip p < 0.05', and for FPC the rule 'FPC excess > 0.01'.
Output: results/step3h_power_<tag>_raw.csv, _power.csv, figures/step3h_power_<tag>.png.

Run: python code/claude/22_power_analysis.py <tag> [n_rows=6000] [R=20] [R0=40] [suffix]
"""
import importlib.util
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FIG, RES, SEED, log_change  # noqa: E402
from clusterlib import fcm_best, fuzzy_indices, load_set, silhouette_fixed  # noqa: E402

_spec = importlib.util.spec_from_file_location("c03", HERE / "03c_clusterability_old_matrix.py")
c03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c03)  # defines km, pred_strength, boot_jaccard, split_dip, copula_fit, copula_sim (main block is guarded)

LITE = dict(fcm_init=3, ps_splits=10, boot=20, dip_splits=10)
PIS = (0.5, 0.2, 0.05)
DELTAS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
STATS = ["FPC2_excess", "Silhouette2", "PS2", "Jaccard2", "DipSplit"]


def lite_stats(X, bat, seed):
    r = fcm_best(X, 2, seed=seed, n_init=LITE["fcm_init"])
    fpc, _ = fuzzy_indices(X, r["U"], r["V"])
    d, pmed, pfrac = c03.split_dip(X, bat, seed, S=LITE["dip_splits"])
    return {"FPC2_excess": fpc - 0.5, "Silhouette2": silhouette_fixed(X, r["U"].argmax(0)),
            "PS2": c03.pred_strength(X, bat, (2,), seed, S=LITE["ps_splits"])[2], "Jaccard2": c03.boot_jaccard(X, bat, 2, seed, B=LITE["boot"]),
            "DipSplit": d, "DipSplit_median_p": pmed}


def task(args):
    pi, delta, rep, X0, bat, strata = args
    seed = SEED + 7919 * rep + int(1000 * delta) + int(100000 * pi)
    rg = np.random.default_rng(seed)
    t = time.perf_counter()
    mu = pd.DataFrame(X0).groupby(strata).transform("mean").values
    Xb = mu + c03.copula_sim(c03.copula_fit(X0 - mu), len(X0), rg)
    if delta > 0:
        w, V = np.linalg.eigh(np.cov(Xb.T))
        top = V[:, np.argsort(w)[::-1][:10]]
        u = top @ rg.standard_normal(10)
        u /= np.linalg.norm(u)
        sig = float(np.sqrt(u @ np.cov(Xb.T) @ u))
        z = rg.random(len(Xb)) < pi
        Xb = Xb + np.outer(z, delta * sig * u)
    out = lite_stats(Xb, bat, seed)
    out.update(pi=pi, delta=delta, rep=rep, seconds=time.perf_counter() - t)
    return out


if __name__ == "__main__":
    tag = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    R0 = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    SUF = sys.argv[5] if len(sys.argv) > 5 else ""  # output-name suffix, e.g. _R200, so that a larger run does not overwrite the first one
    X, meta = load_set(tag)
    ii = np.sort(np.random.default_rng(SEED).choice(len(X), min(n, len(X)), replace=False))
    X0, bat, strata = X[ii], meta.bat.values[ii], meta.stratum.values[ii]
    print(f"{tag}: template {X0.shape}, bats {len(np.unique(bat))}, strata {len(np.unique(strata))}; R={R} per cell, R0={R0} nulls", flush=True)
    jobs = [(0.0, 0.0, r, X0, bat, strata) for r in range(R0)] + [(pi, d, r, X0, bat, strata) for pi in PIS for d in DELTAS[1:] for r in range(R)]
    t0 = time.perf_counter()
    res = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 4))) as ex:
        for i, r in enumerate(ex.map(task, jobs, chunksize=1), 1):
            res.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} done, {time.perf_counter() - t0:.0f}s", flush=True)
                pd.DataFrame(res).to_csv(RES / f"step3h_power_{tag}{SUF}_raw.csv", index=False)
    df = pd.DataFrame(res)
    df.to_csv(RES / f"step3h_power_{tag}{SUF}_raw.csv", index=False)

    null = df[df.delta == 0]
    thr = {s: float(null[s].quantile(0.95)) for s in STATS}
    rows = []
    for (pi, d), g in df[df.delta > 0].groupby(["pi", "delta"]):
        row = dict(pi=pi, delta=d, n_reps=len(g))
        for s in STATS:
            row[f"power_{s}"] = float((g[s] > thr[s]).mean())
        row["power_dip_p<.05"] = float((g.DipSplit_median_p < 0.05).mean())
        row["power_FPC_excess>0.01"] = float((g.FPC2_excess > 0.01).mean())
        rows.append(row)
    P = pd.DataFrame(rows).sort_values(["pi", "delta"])
    P.to_csv(RES / f"step3h_power_{tag}{SUF}_power.csv", index=False)
    pd.set_option("display.width", 250)
    print("null thresholds (95th pct of delta=0):", {k: round(v, 4) for k, v in thr.items()})
    print("null means:", null[STATS].mean().round(4).to_dict())
    print(P.round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    cols = {"power_FPC_excess>0.01": "FPC excess > 0.01", "power_Silhouette2": "silhouette", "power_PS2": "prediction strength",
            "power_Jaccard2": "bootstrap Jaccard", "power_DipSplit": "split-sample dip", "power_dip_p<.05": "dip p < 0.05"}
    for a, pi in zip(ax, PIS):
        g = P[P.pi == pi]
        for c, lab in cols.items():
            a.plot(g.delta, g[c], marker="o", ms=3, label=lab)
        a.set_title(f"minority share {int(pi * 100)}%", fontsize=9)
        a.set_xlabel("separation of the two types (SD units)")
        a.axhline(0.05, color="grey", lw=.6, ls=":")
        a.set_ylim(-0.02, 1.02)
    ax[0].set_ylabel("detection rate")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Power of the one-cluster-allowed battery - spiked mixtures in a null of set {tag} (n={len(X0)})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"step3h_power_{tag}{SUF}.png", dpi=150)
    log_change(f"Step 3h power analysis on '{tag}' (n={len(X0)}, R={R}/cell, R0={R0}): results/step3h_power_{tag}{SUF}_*; figures/step3h_power_{tag}{SUF}.png")
