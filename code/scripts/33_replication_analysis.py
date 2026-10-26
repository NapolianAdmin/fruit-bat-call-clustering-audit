"""Step 7b - out-of-sample REPLICATION of the Isolation finding on files never used before (see 32_replication_sample.py).

Criteria fixed in logs/analysis_log.md BEFORE the replication data were analysed:
  C1 supervised   a logistic regression (balanced, C=0.5) trained on ALL discovery files scores the replication files; Isolation vs other contexts AUC >= 0.90
  C2 unsupervised UMAP (10-D, 30 neighbours, min_dist 0, seed 42) + HDBSCAN (min size 100) fitted on the replication files ALONE returns a cluster of 3-12% of files in which >= 70% are Isolation-context
  C3 null contrast the same pipeline on 6 Gaussian-copula null datasets of the same size returns more clusters and a higher noise share than the real data
Features of the replication files are projected with the scaler + 50-PC PCA FITTED ON THE DISCOVERY SET ONLY (voiced 1,408 statistics, the set all2_voiced_ref1).
Writes results/step7_replication_summary.csv, step7_replication_clusters.csv, step7_replication_nulls.csv and figures/fig8_replication.png.
Run: python code/claude/33_replication_analysis.py [prefix=repl]     (prefix "rest" = the remaining 33,557 annotated files; UMAP part on a seeded 12,000-file subsample)
"""
import importlib.util
import os
import sys
import warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FEAT, FIG, RES, SEED, log_change  # noqa: E402
from clusterlib import auc_fast, boot_groups  # noqa: E402

_spec = importlib.util.spec_from_file_location("u23", HERE / "23_umap_hdbscan.py")
u23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u23)
OUT = FEAT / "claude"
PFX = sys.argv[1] if len(sys.argv) > 1 else "repl"
TAG = "replication" if PFX == "repl" else PFX
N_UMAP = 12000
BIRTH = {103, 105, 108, 111, 113, 115, 120, 202, 207, 208, 210, 211, 214, 215, 216, 220, 221, 222, 230, 231}


def run_task(args):
    return u23.task(args)


if __name__ == "__main__":
    # ---------------- discovery: refit the SAME scaler + PCA as 05_build_sets.py and check against the stored set
    smeta = pd.read_csv(OUT / "set_all2_voiced_ref1_meta.csv")
    F = np.load(OUT / "all2_features.npz")
    idx = pd.read_csv(OUT / "all2_index.csv")
    pos = pd.Series(np.arange(len(idx)), index=idx.ann_row).loc[smeta.ann_row].values
    M = F["voiced_ref1"][pos]
    scaler = StandardScaler().fit(M)
    pca = PCA(n_components=50, svd_solver="full", random_state=SEED).fit(scaler.transform(M))
    Xd = pca.transform(scaler.transform(M))
    Xs = np.load(OUT / "set_all2_voiced_ref1_X.npy")
    corr = [abs(np.corrcoef(Xd[:, j], Xs[:, j])[0, 1]) for j in range(5)]
    print("discovery PCA reproduces the stored set: |r| of PC1-5 =", np.round(corr, 4), flush=True)
    assert min(corr) > 0.999, "scaler/PCA refit does not reproduce the stored discovery set"
    yd = (smeta.context_name == "Isolation").astype(int).values

    # ---------------- replication files
    R = np.load(OUT / f"{PFX}_features.npz")
    ri = pd.read_csv(OUT / f"{PFX}_index.csv")
    ok = (~ri.FileID.duplicated().values) & (ri.too_short.values == 0) & np.isfinite(R["voiced_ref1"]).all(1)
    ri = ri[ok].reset_index(drop=True)
    Xr = pca.transform(scaler.transform(R["voiced_ref1"][ok]))
    ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
    ann["ann_row"] = np.arange(len(ann))
    meta = ri.merge(ann[["ann_row", "Emitter", "context_name", "channel", "month"]], on="ann_row", how="left")
    meta["bat"] = meta.Emitter.abs().astype(str)
    meta["stratum"] = meta.channel.astype(str) + "|" + meta.month
    meta["iso"] = (meta.context_name == "Isolation").astype(int)
    print(f"replication: {len(meta)} files, {meta.iso.sum()} Isolation, {meta.bat.nunique()} emitter groups", flush=True)

    summary = {}
    # ---------------- C1 supervised
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(Xd, yd)
    sc = clf.decision_function(Xr)
    meta["score"] = sc
    auc = float(auc_fast(meta.iso.values, sc))
    lo, hi = boot_groups(meta, lambda z: auc_fast(z.iso.values, z.score.values), n_boot=300)
    y = meta.iso.values
    sd = np.sqrt((sc[y == 1].var() * (y == 1).sum() + sc[y == 0].var() * (y == 0).sum()) / len(y))
    pups = meta[(meta.Emitter > 0) & meta.Emitter.isin(BIRTH)]
    auc_pups = float(auc_fast(pups.iso.values, pups.score.values)) if pups.iso.nunique() == 2 else np.nan
    summary.update(C1_AUC=auc, C1_AUC_lo=float(lo), C1_AUC_hi=float(hi), C1_d_prime=float(abs(sc[y == 1].mean() - sc[y == 0].mean()) / sd),
                   C1_share_nonIso_above_Iso_p5=float((sc[y == 0] > np.percentile(sc[y == 1], 5)).mean()), C1_AUC_pups_only=auc_pups, C1_pass=bool(auc >= 0.90),
                   n_files=len(meta), n_isolation=int(y.sum()))
    print("C1:", {k: v for k, v in summary.items() if k.startswith("C1")}, flush=True)

    # ---------------- C2 / C3 UMAP + HDBSCAN on replication files alone, real data + 6 copula nulls
    sel = np.sort(np.random.default_rng(SEED).choice(len(Xr), N_UMAP, replace=False)) if len(Xr) > 15000 else np.arange(len(Xr))
    Xu, yu, meta_u = Xr[sel], y[sel], meta.iloc[sel].reset_index(drop=True)
    bat, strata = meta_u.bat.values, meta_u.stratum.values
    jobs = [("obs", 0, 30, Xu, bat, strata, None)] + [("copula", b, 30, Xu, bat, strata, None) for b in range(6)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 3))) as ex:
        res = list(ex.map(run_task, jobs, chunksize=1))
    obs, nulls = res[0], res[1:]
    rows = []
    for mcs in u23.MCS:
        lab = obs[f"labels_{mcs}"]
        for c in sorted(set(lab[lab >= 0])):
            g = lab == c
            rows.append(dict(min_cluster_size=mcs, cluster=int(c), n=int(g.sum()), share=float(g.mean()), isolation_share_within=float(yu[g].mean()), isolation_recall=float(yu[g].sum() / max(1, yu.sum())),
                             n_bats=int(len(np.unique(bat[g])))))
    C = pd.DataFrame(rows)
    C.to_csv(RES / f"step7_{TAG}_clusters.csv", index=False)
    print(C.round(3).to_string(), flush=True)
    c2 = C[(C.min_cluster_size == 100) & (C.share >= 0.03) & (C.share <= 0.12) & (C.isolation_share_within >= 0.70)]
    N = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("labels_")} for r in res])
    N.to_csv(RES / f"step7_{TAG}_nulls.csv", index=False)
    nul = N[N.kind == "copula"]
    summary.update(C2_pass=bool(len(c2) > 0), C2_cluster_share=float(c2.share.iloc[0]) if len(c2) else np.nan, C2_isolation_share_within=float(c2.isolation_share_within.iloc[0]) if len(c2) else np.nan,
                   C2_isolation_recall=float(c2.isolation_recall.iloc[0]) if len(c2) else np.nan, real_k_100=int(obs["k_100"]), real_noise_100=float(obs["noise_100"]),
                   null_k_100_mean=float(nul.k_100.mean()), null_noise_100_mean=float(nul.noise_100.mean()), null_k_100_min=int(nul.k_100.min()), null_noise_100_min=float(nul.noise_100.min()),
                   C3_pass=bool(obs["k_100"] < nul.k_100.mean() and obs["noise_100"] < nul.noise_100.min()))
    pd.DataFrame([summary]).to_csv(RES / f"step7_{TAG}_summary.csv", index=False)
    print("SUMMARY", summary, flush=True)

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E2 = u23.embed(Xu, 30, nc=2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(E2[yu == 0, 0], E2[yu == 0, 1], s=1, c="#9aa5b1", rasterized=True, label="other contexts")
    ax[0].scatter(E2[yu == 1, 0], E2[yu == 1, 1], s=2, c="#d1495b", rasterized=True, label="Isolation context")
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].legend(fontsize=7, markerscale=5)
    ax[0].set_title("A  replication files: 2-D UMAP (display only)", fontsize=9, loc="left")
    bins = np.linspace(np.percentile(sc, 0.5), np.percentile(sc, 99.5), 60)
    ax[1].hist(sc[y == 0], bins=bins, alpha=.65, density=True, color="#9aa5b1", label="other contexts")
    ax[1].hist(sc[y == 1], bins=bins, alpha=.65, density=True, color="#d1495b", label="Isolation")
    ax[1].set_xlabel("score from a model trained on the discovery files only", fontsize=8)
    ax[1].set_title(f"B  replication files, AUC {auc:.3f}", fontsize=9, loc="left")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / ("fig8_replication.png" if PFX == "repl" else f"fig8_{PFX}.png"), dpi=150)
    log_change(f"Step 7b replication ({PFX}) of the Isolation finding on {len(meta)} unseen files: C1 AUC {auc:.3f}, C2 pass {bool(len(c2) > 0)}, C3 pass {summary['C3_pass']}; results/step7_{TAG}_*")
