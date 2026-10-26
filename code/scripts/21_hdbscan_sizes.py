"""Step 3b (extra) - what are the HDBSCAN clusters? 08_methods_and_representations.py reports only the number of clusters and the noise share.
For each set and minimum cluster size this lists the size of every cluster found, the noise share, and for each cluster the median of the recording
covariates (silence fraction, voiced duration, span length, SNR) against the rest, so that a 'cluster' that is just a pile of near-identical short files
or one end of the duration axis can be recognised.  Writes results/step3b_hdbscan_sizes.csv.
Run: python code/claude/21_hdbscan_sizes.py <tag> [<tag> ...]      e.g.  all2_naive_ref1 all2_voiced_ref1
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, log_change  # noqa: E402
from clusterlib import load_set  # noqa: E402

COV = ["silence_fraction", "voiced_dur_s", "span_len_s", "snr_db"]
rows = []
for tag in sys.argv[1:]:
    X, u = load_set(tag)
    for mcs in (25, 100):
        t = time.perf_counter()
        lab = HDBSCAN(min_cluster_size=mcs, n_jobs=2).fit_predict(X)
        print(tag, mcs, "clusters", sorted(set(lab) - {-1}), f"{time.perf_counter() - t:.0f}s", flush=True)
        for c in sorted(set(lab)):
            m = lab == c
            r = dict(set=tag, min_cluster_size=mcs, cluster=int(c), n=int(m.sum()), share=float(m.mean()))
            for v in COV:
                if v in u.columns:
                    r[f"{v}_median_in"] = float(np.nanmedian(u[v].values[m]))
                    r[f"{v}_median_out"] = float(np.nanmedian(u[v].values[~m])) if (~m).any() else np.nan
            rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(RES / "step3b_hdbscan_sizes.csv", index=False)
log_change(f"Step 3b extra: HDBSCAN cluster sizes and covariate medians for {', '.join(sys.argv[1:])}: results/step3b_hdbscan_sizes.csv")
pd.set_option("display.width", 250)
print(R.round(3).to_string())
