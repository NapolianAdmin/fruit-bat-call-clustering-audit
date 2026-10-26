"""Version-4 redraw of the Isolation figure (old Fig. 4 / 28_make_isolation_figure.py) and the replication figure (old Fig. 5 / 33_replication_analysis.py).

These two figures need display-only quantities that were never saved: a seeded 2-D UMAP embedding and the out-of-fold / transfer scores.  They are recomputed here with the ORIGINAL
code paths and seeds (nothing is changed in any analysis), and the script first checks that the recomputed statistics equal the stored ones
(results/step3k_isolation_separability.csv, results/step7_replication_summary.csv, results/step7_rest_summary.csv); it stops before drawing if any value differs.
Everything else in the panels (7 HDBSCAN-count groups) is read from the saved csv files.
Writes paper/figures/fig4_v4.pdf, fig5_v4.pdf and results/step_v4_rerun_check.csv.   Run: python code/claude/44_figures_v4_umap.py
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
from common import FEAT, FIG, RES, SEED  # noqa: E402
from clusterlib import auc_fast, load_set  # noqa: E402

warnings.filterwarnings("ignore")
import diptest  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import umap  # noqa: E402

W = 6.69
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.linewidth": .6})
OUT = HERE.parents[1] / "paper" / "figures"
OUT.mkdir(exist_ok=True)
checks = []


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def oof(X, y, g):  # identical to 28_make_isolation_figure.oof / 27_isolation_separability.oof_scores
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return s


def dprime(s, y):
    sd = np.sqrt((s[y == 1].var() * (y == 1).sum() + s[y == 0].var() * (y == 0).sum()) / len(y))
    return float(abs(s[y == 1].mean() - s[y == 0].mean()) / sd)


def check(name, got, want):
    d = abs(got - want)
    checks.append(dict(item=name, recomputed=got, stored=want, abs_diff=d))
    print(f"{name}: recomputed {got:.6f}  stored {want:.6f}  diff {d:.2e}", flush=True)


# ====================================================================== Fig. 4 (Isolation group)
sep = pd.read_csv(RES / "step3k_isolation_separability.csv")
sep = sep[sep.analysis == "all files, leave-bats-out"].set_index("set")
panels = {}
for tag in ("all2_voiced_ref1", "segall_shape"):
    Xs, Ms = load_set(tag)
    y = (Ms.context_name.values == "Isolation").astype(int)
    s = oof(Xs, y, Ms.bat.values)
    panels[tag] = (s, y)
    check(f"{tag} AUC", float(auc_fast(y, s)), float(sep.loc[tag, "AUC"]))
    check(f"{tag} d'", dprime(s, y), float(sep.loc[tag, "d_prime"]))
    check(f"{tag} dip p (pooled)", float(diptest.diptest(s)[1]), float(sep.loc[tag, "dip_pooled_p"]))
    check(f"{tag} dip p (non-Isolation)", float(diptest.diptest(s[y == 0])[1]), float(sep.loc[tag, "dip_nonIso_p"]))
X, M = load_set("all2_voiced_ref1")
ii = np.sort(np.random.default_rng(SEED).choice(len(X), 12000, replace=False))
E2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=SEED).fit_transform(X[ii])
iso = (M.context_name.values[ii] == "Isolation")
print("UMAP files", len(ii), "Isolation", int(iso.sum()), flush=True)

# ====================================================================== Fig. 5 (replication) data: computed by 45_fig5_data.py (single-threaded BLAS, as in 33_replication_analysis.py)
import subprocess
subprocess.run([sys.executable, str(HERE / '45_fig5_data.py')], check=True)
Z = np.load(HERE.parents[1] / 'data' / 'features' / 'claude' / 'figv4_fig5.npz')
rep = {t: (Z[t + '_E'], Z[t + '_yu'], Z[t + '_sc'], Z[t + '_y'], float(Z[t + '_auc'])) for t in ('replication', 'rest')}
checks += pd.read_csv(RES / 'step_v4_rerun_check_fig5.csv').to_dict('records')
chk = pd.DataFrame(checks)
chk.to_csv(RES / "step_v4_rerun_check.csv", index=False)
if (chk.abs_diff > 1e-9).any():
    print("MISMATCH - not drawing figures:\n", chk[chk.abs_diff > 1e-9])
    sys.exit(1)
print("all recomputed values equal the stored values (max diff %.1e)" % chk.abs_diff.max(), flush=True)

# ====================================================================== draw Fig. 4
fig, ax = plt.subplots(2, 2, figsize=(W, 5.0), layout="constrained")
a = ax[0, 0]
a.scatter(E2[~iso, 0], E2[~iso, 1], s=1, c="#9aa5b1", label="other contexts", rasterized=True)
a.scatter(E2[iso, 0], E2[iso, 1], s=2, c="#d1495b", label="Isolation context", rasterized=True)
a.set_xticks([]); a.set_yticks([])
a.set_title("2-D UMAP of voiced file features (display only)", loc="left")
a.legend(markerscale=5, loc="lower right", frameon=False)
letter(a, "A")
a = ax[0, 1]
raw = pd.read_csv(RES / "step3i_umap_all2_voiced_ref1_raw.csv")
cont = pd.read_csv(RES / "step3i_umap_continuum_nulls.csv")
rng = np.random.default_rng(1)
groups = [("copula null", raw[raw.kind == "copula"], "#4c78a8"), ("session-mean null", raw[raw.kind == "session"], "#59a14f"),
          ("curved-continuum null", cont[cont.dataset == "gradient_null"], "#f28e2b"), ("skewed null", cont[cont.dataset == "skewed_null"], "#b07aa1"),
          ("real data", raw[(raw.kind == "obs") & (raw.nn == 30)], "#d1495b"), ("real minus small cluster", cont[cont.dataset == "real_minus_small_cluster"], "#7f7f7f"),
          ("real minus Isolation files", cont[cont.dataset == "real_minus_all_Isolation_files"], "#4d4d4d")]
for j, (lab_, v, colr) in enumerate(groups):
    a.scatter(np.full(len(v), j) + rng.uniform(-.12, .12, len(v)), v.k_100, c=colr, s=14 if "real" not in lab_ else 40, marker="D" if "real" in lab_ else "o", alpha=.85, linewidths=0)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups], rotation=35, ha="right")
a.set_ylabel("HDBSCAN clusters (min size 100)")
top = a.get_ylim()[1]
for j, (lab_, v, colr) in enumerate(groups):
    a.text(j, top * 0.97, f"{v.noise_100.mean():.0%}", ha="center", fontsize=7)
a.set_title("UMAP + HDBSCAN: real data and controls (top: % noise)", loc="left")
letter(a, "B")
for a, tag, ttl, L in ((ax[1, 0], "all2_voiced_ref1", "held-out-bat Isolation score, voiced file features", "C"), (ax[1, 1], "segall_shape", "held-out-bat Isolation score, syllable shape", "D")):
    s, y = panels[tag]
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 60)
    a.hist(s[y == 0], bins=bins, alpha=.65, density=True, color="#9aa5b1", label="other contexts")
    a.hist(s[y == 1], bins=bins, alpha=.65, density=True, color="#d1495b", label="Isolation")
    a.set_xlabel("log-odds of Isolation (model trained on other bats)")
    a.set_title(ttl, loc="left")
    letter(a, L)
ax[1, 0].legend(frameon=False)
fig.savefig(OUT / "fig4_v4.pdf")
fig.savefig(FIG / "fig4_v4.pdf")
print("fig4 size", fig.get_size_inches())
plt.close(fig)

# ====================================================================== draw Fig. 5 (top row: replication sample, bottom row: completion sample)
fig, ax = plt.subplots(2, 2, figsize=(W, 4.1), layout="constrained")
for r, (tag, nm) in enumerate((("replication", "replication files"), ("rest", "completion files"))):
    E, yu, sc, y, auc = rep[tag]
    a = ax[r, 0]
    a.scatter(E[yu == 0, 0], E[yu == 0, 1], s=1, c="#9aa5b1", rasterized=True, label="other contexts")
    a.scatter(E[yu == 1, 0], E[yu == 1, 1], s=2, c="#d1495b", rasterized=True, label="Isolation context")
    a.set_xticks([]); a.set_yticks([]); a.legend(markerscale=5, frameon=False, loc="lower right")
    a.set_title(f"{nm}: 2-D UMAP (display only)", loc="left")
    letter(a, "A" if r == 0 else "C")
    a = ax[r, 1]
    bins = np.linspace(np.percentile(sc, 0.5), np.percentile(sc, 99.5), 60)
    a.hist(sc[y == 0], bins=bins, alpha=.65, density=True, color="#9aa5b1", label="other contexts")
    a.hist(sc[y == 1], bins=bins, alpha=.65, density=True, color="#d1495b", label="Isolation")
    a.set_xlabel("score from a model trained on the discovery files only")
    a.set_title(f"{nm}, AUC {auc:.3f}", loc="left")
    a.legend(frameon=False)
    letter(a, "B" if r == 0 else "D")
fig.savefig(OUT / "fig5_v4.pdf")
fig.savefig(FIG / "fig5_v4.pdf")
print("fig5 size", fig.get_size_inches())
