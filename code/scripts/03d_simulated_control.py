"""Step 3d - single-cluster control: do FPC / Xie-Beni / silhouette pick c=2 on data with NO cluster structure?

Simulated datasets have one population and the SAME per-component variances as the draft's saved
50-PC training matrix (X_pca_train.npy), so the spectrum decays exactly as in the real data.
Four single-population families (all unimodal):
  gradient - ONE skewed latent factor drives the leading PCs through t, He2(t), He3(t): a curved, graded continuum
  gauss    - multivariate normal (diagonal in PC space)
  t5       - multivariate Student-t, 5 d.f. (heavy tails)
  skewed   - per-component exp(0.6 z) transform, re-standardised to the same variances (skewed marginals)
Output: results/step3d_control_sweeps.csv (every dataset x c), results/step3d_control_summary.csv,
        figures/step3d_control.png.   Also the real-data sweep for side-by-side comparison.
Run: python code/claude/03d_simulated_control.py [n_datasets] [n_init]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, OLD, RES, SEED, log_change, rng  # noqa: E402
from clusterlib import sweep  # noqa: E402

N_DS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
N_INIT = int(sys.argv[2]) if len(sys.argv) > 2 else 5
CS = range(2, 7)
KIND_OFFSET = {"gauss": 10_000, "t5": 20_000, "skewed": 30_000, "gradient": 40_000}  # fixed (hash() is salted per process)
KINDS = tuple(KIND_OFFSET)

X = np.load(OLD / "X_pca_train.npy").astype(np.float64)
n, d = X.shape
var = X.var(0)
print(f"real matrix {X.shape}; PC variances: first {var[0]:.2f}, 10th {var[9]:.2f}, 50th {var[-1]:.3f}; ratio first/last {var[0]/var[-1]:.0f}")


def make(kind, r):
    z = r.standard_normal((n, d))
    if kind == "gauss":
        return z * np.sqrt(var)
    if kind == "t5":
        w = r.chisquare(5, size=(n, 1)) / 5.0
        return z / np.sqrt(w) * np.sqrt(var) * np.sqrt(3.0 / 5.0)  # rescale to unit variance (t5 var = 5/3)
    if kind == "skewed":
        y = np.exp(0.6 * z)
        y = (y - y.mean(0)) / y.std(0)
        return y * np.sqrt(var)
    if kind == "gradient":
        # ONE skewed latent factor t drives the leading PCs through t, He2(t), He3(t) (a curved 1-D continuum);
        # per-PC variances still equal the real ones. Unimodal, graded, non-Gaussian - the realistic "graded" rival.
        t = np.exp(0.5 * r.standard_normal(n))
        t = (t - t.mean()) / t.std()
        h = np.stack([t, (t ** 2 - 1) / np.sqrt(2), (t ** 3 - 3 * t) / np.sqrt(6)])
        h = (h - h.mean(1, keepdims=True)) / h.std(1, keepdims=True)
        rho = np.zeros(d)
        rho[:6] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        sig = np.zeros((n, d))
        for k in range(6):
            sig[:, k] = rho[k] * h[k % 3]
        sig = sig + np.sqrt(1 - rho ** 2) * z
        return sig * np.sqrt(var)
    raise ValueError(kind)


rows = []
t0 = time.perf_counter()
real = sweep(X, CS, n_init=20, seed=SEED)  # 20 restarts as in the draft
for r_ in real:
    rows.append(dict(kind="REAL (draft, 3,897 rows incl. duplicates)", ds=0, **{k: r_[k] for k in ("c", "FPC", "XieBeni", "Silhouette", "J", "seconds")}))
print("real sweep (c, FPC, XB, silhouette):", [(r_["c"], round(r_["FPC"], 4), round(r_["XieBeni"], 3), round(r_["Silhouette"], 4)) for r_ in real], flush=True)
for kind in KINDS:
    for ds in range(N_DS):
        Xs = make(kind, rng(KIND_OFFSET[kind] + ds))
        for r_ in sweep(Xs, CS, n_init=N_INIT, seed=ds):
            rows.append(dict(kind=kind, ds=ds, **{k: r_[k] for k in ("c", "FPC", "XieBeni", "Silhouette", "J", "seconds")}))
    print(f"{kind} done, elapsed {time.perf_counter()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "step3d_control_sweeps.csv", index=False)

summ = []
for kind in KINDS:
    g = df[df.kind == kind]
    wins = {"FPC": [], "XieBeni": [], "Silhouette": []}
    for ds, h in g.groupby("ds"):
        wins["FPC"].append(int(h.loc[h.FPC.idxmax(), "c"]))
        wins["XieBeni"].append(int(h.loc[h.XieBeni.idxmin(), "c"]))
        wins["Silhouette"].append(int(h.loc[h.Silhouette.idxmax(), "c"]))
    c2 = g[g.c == 2]
    summ.append(dict(kind=kind, n_datasets=g.ds.nunique(), n_init=N_INIT,
                     pct_FPC_picks_c2=100 * np.mean(np.array(wins["FPC"]) == 2),
                     pct_XB_picks_c2=100 * np.mean(np.array(wins["XieBeni"]) == 2),
                     pct_Silhouette_picks_c2=100 * np.mean(np.array(wins["Silhouette"]) == 2),
                     FPC_c2_mean=c2.FPC.mean(), FPC_c2_q025=c2.FPC.quantile(.025), FPC_c2_q975=c2.FPC.quantile(.975),
                     XB_c2_mean=c2.XieBeni.mean(), XB_c2_q025=c2.XieBeni.quantile(.025), XB_c2_q975=c2.XieBeni.quantile(.975),
                     Sil_c2_mean=c2.Silhouette.mean(), Sil_c2_q025=c2.Silhouette.quantile(.025), Sil_c2_q975=c2.Silhouette.quantile(.975)))
rc2 = [r_ for r_ in real if r_["c"] == 2][0]
summ.append(dict(kind="REAL draft matrix", n_datasets=1, n_init=20,
                 pct_FPC_picks_c2=100 * (max(real, key=lambda r_: r_["FPC"])["c"] == 2),
                 pct_XB_picks_c2=100 * (min(real, key=lambda r_: r_["XieBeni"])["c"] == 2),
                 pct_Silhouette_picks_c2=100 * (max(real, key=lambda r_: r_["Silhouette"])["c"] == 2),
                 FPC_c2_mean=rc2["FPC"], XB_c2_mean=rc2["XieBeni"], Sil_c2_mean=rc2["Silhouette"]))
S = pd.DataFrame(summ)
S.to_csv(RES / "step3d_control_summary.csv", index=False)
print(S.round(4).to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for a, (col, ttl) in zip(ax, [("FPC", "Fuzzy partition coefficient (higher=better)"), ("XieBeni", "Xie-Beni (lower=better)"), ("Silhouette", "Silhouette (higher=better)")]):
    for kind, colr in [("gauss", "#4c78a8"), ("t5", "#59a14f"), ("skewed", "#b07aa1"), ("gradient", "#f28e2b")]:
        g = df[df.kind == kind].groupby("c")[col]
        a.plot(g.mean().index, g.mean().values, "-o", color=colr, label=f"simulated one-population: {kind}", ms=4)
        a.fill_between(g.mean().index, g.quantile(.025).values, g.quantile(.975).values, color=colr, alpha=.15)
    rr = df[df.kind.str.startswith("REAL")]
    a.plot(rr.c, rr[col], "-s", color="#e45756", label="real draft matrix")
    a.set_title(ttl, fontsize=9)
    a.set_xlabel("c")
    if col == "XieBeni":
        a.set_yscale("log")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "step3d_control.png", dpi=150)
log_change(f"Step 3d control run: {N_DS} datasets/kind, {N_INIT} restarts; results/step3d_control_*.csv")
