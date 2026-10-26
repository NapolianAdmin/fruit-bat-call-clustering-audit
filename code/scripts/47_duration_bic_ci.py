"""Step 6b (supplement) - uncertainty of the one- versus two-Gaussian comparison for log10 syllable duration.

Same segments as 17_segment_duration_modes.py (prefix segall).  Fits 1- and 2-component Gaussian mixtures (n_init 3, seed 42), reports BIC(1) - BIC(2) and the weights/means of the two-component fit, and a 95% percentile
interval of BIC(1) - BIC(2) over 200 bootstrap resamples of whole bats (never of syllables).  A positive difference favours two components.  Checks the point estimate against results/step6_duration_gmm.csv.
Writes results/step6_duration_bic_ci.csv.   Run: python code/claude/47_duration_bic_ci.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, RES, SEED  # noqa: E402

warnings.filterwarnings("ignore")
si = pd.read_csv(FEAT / "claude" / "segall_seg_index.csv")
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
ann["ann_row"] = np.arange(len(ann))
m = si.merge(ann[["ann_row", "Emitter", "context_name"]], on="ann_row", how="left")
x = np.log10(m.dur_s.values).reshape(-1, 1)
bat = m.Emitter.abs().astype(str).values


def dbic(v):
    g1 = GaussianMixture(1, n_init=3, random_state=SEED).fit(v)
    g2 = GaussianMixture(2, n_init=3, random_state=SEED).fit(v)
    return g1.bic(v) - g2.bic(v), g2


d0, g2 = dbic(x)
ref = pd.read_csv(RES / "step6_duration_gmm.csv").set_index("k").BIC
assert abs(d0 - (ref[1] - ref[2])) < 1e-6, (d0, ref[1] - ref[2])
rg = np.random.default_rng(SEED)
ub = np.unique(bat)
idx = {b: np.where(bat == b)[0] for b in ub}
bs = []
for _ in range(200):
    ii = np.concatenate([idx[b] for b in rg.choice(ub, len(ub), replace=True)])
    bs.append(dbic(x[ii])[0])
o = np.argsort(g2.means_.ravel())
row = dict(n_syllables=len(x), n_bat_groups=len(ub), delta_BIC_1_minus_2=float(d0), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)), n_boot=200,
           means_ms=str(np.round(1000 * 10 ** g2.means_.ravel()[o], 1).tolist()), weights=str(np.round(g2.weights_[o], 3).tolist()))
pd.DataFrame([row]).to_csv(RES / "step6_duration_bic_ci.csv", index=False)
print(row)
