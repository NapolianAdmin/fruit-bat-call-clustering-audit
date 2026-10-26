"""Step 3j - what IS the small stable cluster that UMAP + HDBSCAN finds in the voiced file-level features?

23_umap_hdbscan.py found, in the voiced 1,408-feature space (12,000 files), two clusters (minimum size 100/300): one holding 93% of files and one of ~6.4% (770 files, 30 bats, spread over
channels, no difference in voiced duration or silence), stable across bat bootstraps (Jaccard 0.975).  The linear tests (FCM, prediction strength, dip along the k-means axis) had little
power for a minority this small (step 3h).  This script characterises that cluster:
  1. recomputes the same UMAP/HDBSCAN solution (same rows, seed, settings) and saves the labels
  2. context table, P(cluster | Isolation), P(Isolation | cluster), pup vs adult x Isolation (positive-coded bats; Metadata.pdf Table 2 birth dates, as in 12_pup_adult_check.py)
  3. recording covariates (AUC of cluster membership; bias-corrected Cramer's V with channel, month, bat)
  4. stratified permutation of the cluster x context association (labels shuffled within channel x month strata)
  5. HELD-OUT-BAT bimodality: an LDA axis learned from the cluster labels of one random half of the bats, the other half projected on it -> Hartigan dip test (20 balanced bat splits),
     and the standardised separation d' of the two clusters on that held-out axis (comparable with delta in the power analysis)
Run: python code/claude/26_umap_cluster_dissection.py <tag> [n_rows=12000] [mcs=100]
"""
import importlib.util
import os
import sys
import warnings
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FEAT, RES, SEED, cramers_v, log_change  # noqa: E402
from clusterlib import auc_fast, load_set  # noqa: E402

_spec = importlib.util.spec_from_file_location("u23", HERE / "23_umap_hdbscan.py")
u23 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u23)

BIRTH = {103, 105, 108, 111, 113, 115, 120, 202, 207, 208, 210, 211, 214, 215, 216, 220, 221, 222, 230, 231}  # bats born in the setup (Metadata.pdf Table 2)

tag = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
mcs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
X, meta = load_set(tag)
ii = np.sort(np.random.default_rng(SEED).choice(len(X), min(n, len(X)), replace=False))
X0, M = X[ii], meta.iloc[ii].reset_index(drop=True)
E = u23.embed(X0, 30)
lab, st = u23.cluster_stats(E, mcs)
print(tag, "UMAP/HDBSCAN:", st, flush=True)
np.save(FEAT / "claude" / f"umap_labels_{tag}_mcs{mcs}.npy", lab)
cl = [c for c in sorted(set(lab[lab >= 0]))]
sizes = {c: int((lab == c).sum()) for c in cl}
small = min(cl, key=lambda c: sizes[c]) if len(cl) > 1 else None
print("cluster sizes:", sizes, "-> small cluster:", small, flush=True)
if small is None:
    sys.exit("only one cluster")
M["small"] = (lab == small).astype(int)
M["isolation"] = (M.context_name == "Isolation").astype(int)

# 2. context
ct = pd.crosstab(M.context_name, M.small)
ct["share_in_small_cluster"] = ct[1] / (ct[0] + ct[1])
ct.to_csv(RES / f"step3j_umap_dissection_{tag}_context.csv")
print(ct.round(3).to_string())
p_c_given_iso = float(M[M.isolation == 1].small.mean())
p_iso_given_c = float(M[M.small == 1].isolation.mean())
print(f"P(small cluster | Isolation) = {p_c_given_iso:.3f}; P(Isolation | small cluster) = {p_iso_given_c:.3f}; Isolation share of all files {M.isolation.mean():.3f}")
pos = M[M.Emitter > 0].copy()
pos["status"] = np.where(pos.Emitter.isin(BIRTH), "pup", "adult")
pa = pos.groupby(["status", pos.isolation.map({1: "Isolation", 0: "other contexts"})]).agg(files=("small", "size"), share_small=("small", "mean"), bats=("bat", "nunique")).reset_index()
pa.to_csv(RES / f"step3j_umap_dissection_{tag}_pup_adult.csv", index=False)
print(pa.round(3).to_string())

# 3. covariates
rows = []
for v in ("snr_db", "noise_floor_dbfs", "voiced_dur_s", "silence_fraction", "span_len_s", "n_voiced_seg", "clip_frac"):
    if v in M and M[v].notna().all() and M.small.nunique() == 2:
        rows.append(dict(covariate=v, AUC_for_small_cluster=float(auc_fast(M.small.values, M[v].values)), median_small=float(M[M.small == 1][v].median()), median_rest=float(M[M.small == 0][v].median())))
for v in ("channel", "month", "bat", "stratum"):
    r = cramers_v(M.small.values, M[v].values)
    rows.append(dict(covariate=f"Cramer V with {v}", AUC_for_small_cluster=float(r[0])))
pd.DataFrame(rows).to_csv(RES / f"step3j_umap_dissection_{tag}_covariates.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())

# 4. stratified permutation of the association with Isolation
from sklearn.metrics import normalized_mutual_info_score as nmi
obs = nmi(M.small.values, M.isolation.values)
rg = np.random.default_rng(SEED)
strata = M.stratum.values
null = []
for _ in range(1000):
    sh = M.isolation.values.copy()
    for s in np.unique(strata):
        m = strata == s
        sh[m] = rg.permutation(sh[m])
    null.append(nmi(M.small.values, sh))
print(f"NMI(small cluster, Isolation) = {obs:.3f}; within-stratum permutation null mean {np.mean(null):.4f}, 97.5th {np.quantile(null, .975):.4f}, max {np.max(null):.4f}; 0 of 1000 reach observed: {int(np.sum(np.array(null) >= obs)) == 0}", flush=True)

# 5. held-out-bat bimodality on an LDA axis learned from the cluster labels
import diptest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
bat = M.bat.values
ub = np.unique(bat)
rg = np.random.default_rng(SEED + 5)
out = []
for s in range(20):
    for _ in range(60):
        half = rg.permutation(ub)[: len(ub) // 2]
        A = np.isin(bat, half)
        if 0.35 <= A.mean() <= 0.65:
            break
    for tr, te in ((A, ~A), (~A, A)):
        if M.small[tr].sum() < 20 or M.small[te].sum() < 20:
            continue
        lda = LinearDiscriminantAnalysis().fit(X0[tr], M.small.values[tr])
        proj = X0[te] @ lda.coef_[0]
        yy = M.small.values[te]
        d, p = diptest.diptest(proj)
        sd = np.sqrt((proj[yy == 1].var() * (yy == 1).sum() + proj[yy == 0].var() * (yy == 0).sum()) / len(yy))
        dprime = float(abs(proj[yy == 1].mean() - proj[yy == 0].mean()) / sd)
        out.append(dict(split=s, dip=float(d), dip_p=float(p), d_prime_heldout=dprime, minority_share_heldout=float(yy.mean())))
D = pd.DataFrame(out)
D.to_csv(RES / f"step3j_umap_dissection_{tag}_heldout_axis.csv", index=False)
print(f"held-out-bat LDA axis ({len(D)} fits): median dip p {D.dip_p.median():.3g}, fraction p<0.05 {(D.dip_p < 0.05).mean():.2f}, mean d' {D.d_prime_heldout.mean():.2f} (range {D.d_prime_heldout.min():.2f}-{D.d_prime_heldout.max():.2f})")
summary = dict(tag=tag, n=len(M), mcs=mcs, small_cluster_files=sizes[small], small_cluster_share=sizes[small] / len(M), n_bats_in_small=int(M[M.small == 1].bat.nunique()),
               P_small_given_isolation=p_c_given_iso, P_isolation_given_small=p_iso_given_c, isolation_share_all=float(M.isolation.mean()),
               nmi_isolation=float(obs), nmi_null_mean=float(np.mean(null)), nmi_null_max=float(np.max(null)),
               heldout_dip_median_p=float(D.dip_p.median()), heldout_dip_frac_p05=float((D.dip_p < 0.05).mean()), heldout_dprime_mean=float(D.d_prime_heldout.mean()))
pd.DataFrame([summary]).to_csv(RES / f"step3j_umap_dissection_{tag}_summary.csv", index=False)
log_change(f"Step 3j dissection of the small UMAP/HDBSCAN cluster in '{tag}': results/step3j_umap_dissection_{tag}_*")
