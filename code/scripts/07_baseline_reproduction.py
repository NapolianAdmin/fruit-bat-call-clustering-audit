"""Step 2c - does re-extracting the draft's 1,408 whole-span features from audio reproduce the draft's numbers?

Builds the 3,897-row matrix (duplicates included, same row order as the draft) from the 'baseline' extraction for each
dB-reference variant, standardises (fit on those rows), PCA(50), and compares with the saved X_pca_train.npy:
   |r| between new and old PC k (k=1..10), FCM c=2..6 metrics (20 restarts, seed 42), cluster sizes, ARI vs the labels
   obtained from FCM on the OLD matrix.
Acceptance (same tolerances as the Codex hand-off): |r| >= 0.95 for PCs 1-5, ARI >= 0.90, FPC within 0.002,
Xie-Beni within 0.01, silhouette within 0.005 of the draft (0.5786 / 1.2498 / 0.2195).
Writes results/step2_baseline_variants.csv and results/step2_baseline_report.md.
Run: python code/claude/07_baseline_reproduction.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, OLD, RES, SEED, log_change  # noqa: E402
from clusterlib import fcm_best, sweep  # noqa: E402

OUT = FEAT / "claude"
P = sys.argv[1] if len(sys.argv) > 1 else "baseline"  # 'baseline' = FileID reading; 'hrow' = filename number read as Annotations.csv ROW index
SUF = "" if P == "baseline" else f"_{P}"
F = np.load(OUT / f"{P}_features.npz")
idx = pd.read_csv(OUT / f"{P}_index.csv")
rmap = pd.read_csv(OUT / f"{P}_row_map.csv")
pos = pd.Series(np.arange(len(idx)), index=idx.ann_row)
take = pos.loc[rmap.ann_row].values  # feature row for each of the draft's 3,897 rows
Xold = np.load(OLD / "X_pca_train.npy").astype(np.float64)
fit_old = fcm_best(Xold, 2, seed=SEED, n_init=20)
lab_old = fit_old["U"].argmax(0)

rows = []
for k in [k for k in F.keys() if k.startswith("naive_")]:
    M = F[k][take].astype(np.float64)
    Xn = PCA(50, svd_solver="full", random_state=SEED).fit_transform(StandardScaler().fit_transform(M))
    r = [abs(np.corrcoef(Xn[:, j], Xold[:, j])[0, 1]) for j in range(10)]
    sw = sweep(Xn, range(2, 7), n_init=20, seed=SEED)
    c2 = sw[0]
    fit = fcm_best(Xn, 2, seed=SEED, n_init=20)
    ari = adjusted_rand_score(lab_old, fit["U"].argmax(0))
    row = dict(variant=k, **{f"absr_PC{j+1}": r[j] for j in range(10)}, FPC=c2["FPC"], XieBeni=c2["XieBeni"], Silhouette=c2["Silhouette"],
               sizes=str(sorted(c2["sizes"])), ARI_vs_old_labels=ari, draft_FPC=0.5786, draft_XB=1.2498, draft_Sil=0.2195)
    row["PASS"] = bool(min(r[:5]) >= .95 and ari >= .90 and abs(c2["FPC"] - .5786) <= .002 and abs(c2["XieBeni"] - 1.2498) <= .01 and abs(c2["Silhouette"] - .2195) <= .005)
    rows.append(row)
    print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items() if a in ("variant", "absr_PC1", "absr_PC2", "absr_PC5", "FPC", "XieBeni", "Silhouette", "sizes", "ARI_vs_old_labels", "PASS")}, flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / f"step2_baseline_variants{SUF}.csv", index=False)

oob = int(idx.voice_oob.sum())
best = R.sort_values("ARI_vs_old_labels", ascending=False).iloc[0]
md = [f"# Step 2c - baseline reproduction from audio ({len(take)} rows, {len(idx)} unique annotation rows)\n",
      "| variant | |r| PC1 | |r| PC2 | |r| PC5 | FPC | XB | silhouette | sizes | ARI vs old | pass |", "|---|---|---|---|---|---|---|---|---|---|"]
for _, a in R.iterrows():
    md.append(f"| {a.variant} | {a.absr_PC1:.3f} | {a.absr_PC2:.3f} | {a.absr_PC5:.3f} | {a.FPC:.4f} | {a.XieBeni:.4f} | {a.Silhouette:.4f} | {a.sizes} | {a.ARI_vs_old_labels:.3f} | {a.PASS} |")
md += ["\nDraft: FPC 0.5786, XB 1.2498, silhouette 0.2195, sizes 1,860/2,037.",
       f"Best variant by ARI: **{best.variant}** (PASS={best.PASS}).",
       f"FileInfo voice-sample indices beyond the WAV length (1-based inclusive assumed): {oob} of {len(idx)} rows.",
       f"Header sample rates seen: {sorted(idx.header_sr.unique().tolist())}."]
(RES / f"step2_baseline_report{SUF}.md").write_text("\n".join(md), encoding="utf-8")
log_change(f"Step 2c baseline reproduction ({P}): best variant {best.variant}, PASS={best.PASS}, ARI {best.ARI_vs_old_labels:.3f}")
print("\n".join(md))
