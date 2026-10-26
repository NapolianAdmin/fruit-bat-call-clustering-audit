"""Display data for the replication figure (new Fig. S6): out-of-sample scores and a seeded 2-D UMAP of the replication and completion samples.
Same code path as 33_replication_analysis.py, including its single-threaded BLAS setting (OMP/MKL/OPENBLAS_NUM_THREADS=1), which the stored C1 values depend on
at the 1e-5 level.  Checks the recomputed C1 AUC and d' against results/step7_{replication,rest}_summary.csv, then saves
data/features/claude/figv4_fig5.npz and results/step_v4_rerun_check_fig5.csv.   Called by 44_figures_v4_umap.py.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FEAT, RES, SEED  # noqa: E402
from clusterlib import auc_fast  # noqa: E402

warnings.filterwarnings("ignore")
checks = []


def dprime(s, y):
    sd = np.sqrt((s[y == 1].var() * (y == 1).sum() + s[y == 0].var() * (y == 0).sum()) / len(y))
    return float(abs(s[y == 1].mean() - s[y == 0].mean()) / sd)


def check(name, got, want):
    d = abs(got - want)
    checks.append(dict(item=name, recomputed=got, stored=want, abs_diff=d))
    print(f"{name}: recomputed {got:.6f}  stored {want:.6f}  diff {d:.2e}", flush=True)


spec = importlib.util.spec_from_file_location("u23", HERE / "23_umap_hdbscan.py")
u23 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u23)
OF = FEAT / "claude"
smeta = pd.read_csv(OF / "set_all2_voiced_ref1_meta.csv")
F = np.load(OF / "all2_features.npz")
idx = pd.read_csv(OF / "all2_index.csv")
pos = pd.Series(np.arange(len(idx)), index=idx.ann_row).loc[smeta.ann_row].values
Mf = F["voiced_ref1"][pos]
scaler = StandardScaler().fit(Mf)
pca = PCA(n_components=50, svd_solver="full", random_state=SEED).fit(scaler.transform(Mf))
Xd = pca.transform(scaler.transform(Mf))
yd = (smeta.context_name == "Isolation").astype(int).values
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(Xd, yd)
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
ann["ann_row"] = np.arange(len(ann))
rep = {}
for pfx, tag in (("repl", "replication"), ("rest", "rest")):
    R = np.load(OF / f"{pfx}_features.npz")
    ri = pd.read_csv(OF / f"{pfx}_index.csv")
    ok = (~ri.FileID.duplicated().values) & (ri.too_short.values == 0) & np.isfinite(R["voiced_ref1"]).all(1)
    ri = ri[ok].reset_index(drop=True)
    Xr = pca.transform(scaler.transform(R["voiced_ref1"][ok]))
    meta = ri.merge(ann[["ann_row", "context_name"]], on="ann_row", how="left")
    y = (meta.context_name == "Isolation").astype(int).values
    sc = clf.decision_function(Xr)
    sel = np.sort(np.random.default_rng(SEED).choice(len(Xr), 12000, replace=False)) if len(Xr) > 15000 else np.arange(len(Xr))
    E = u23.embed(Xr[sel], 30, nc=2)
    st = pd.read_csv(RES / f"step7_{tag}_summary.csv").iloc[0]
    check(f"{tag} C1 AUC", float(auc_fast(y, sc)), float(st.C1_AUC))
    check(f"{tag} C1 d'", dprime(sc, y), float(st.C1_d_prime))
    rep[tag] = (E, y[sel], sc, y, float(auc_fast(y, sc)))
pd.DataFrame(checks).to_csv(RES / "step_v4_rerun_check_fig5.csv", index=False)
out = {}
for t, v in rep.items():
    out.update({t + "_E": v[0], t + "_yu": v[1], t + "_sc": v[2], t + "_y": v[3], t + "_auc": v[4]})
if max(c["abs_diff"] for c in checks) > 1e-9:
    print("MISMATCH")
    sys.exit(1)
np.savez(OF / "figv4_fig5.npz", **out)
print("saved figv4_fig5.npz")
