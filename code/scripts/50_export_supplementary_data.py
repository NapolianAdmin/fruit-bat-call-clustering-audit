"""Export the machine-readable Supplementary Data files SD1-SD7 (paper/supplementary_data/), each copied or assembled from the result files without any recomputation.
  SD1 all clustering configurations (step3m_all_methods_*.csv)          SD2 best configuration per algorithm and representation (step3m_all_methods_summary.csv)
  SD3 power-analysis detection rates with 95% Wilson intervals           SD4 observed and null values of every clusterability statistic (step3c_*_summary.csv)
  SD5 Isolation AUCs and bat-level intervals by age window / channel / treatment (step3k_isolation_strata_ci.csv)
  SD6 sample accounting and bat/channel/day overlap                      SD7 package list of the analysis environment (logs/pip_freeze.txt)
Run: python code/claude/50_export_supplementary_data.py
"""
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RES  # noqa: E402

OUT = RES.parent / "paper" / "supplementary_data"
OUT.mkdir(exist_ok=True)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


files = [x for x in sorted(glob.glob(str(RES / "step3m_all_methods_*.csv"))) if "summary" not in x]
sd1 = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
sd1.to_csv(OUT / "SD1_all_methods_configurations.csv", index=False)
pd.read_csv(RES / "step3m_all_methods_summary.csv").to_csv(OUT / "SD2_all_methods_best_configurations.csv", index=False)
rows = []
STATS = {"power_PS2": "prediction strength", "power_Silhouette2": "silhouette", "power_Jaccard2": "bootstrap Jaccard", "power_DipSplit": "split-sample dip statistic",
         "power_dip_p<.05": "median dip p < 0.05", "power_FPC_excess>0.01": "FPC excess > 0.01"}
for tag, nm in (("all2_voiced_ref1", "voiced file-level features"), ("segall_shape", "syllable shapes")):
    suf = "_R200" if (RES / f"step3h_power_{tag}_R200_power.csv").exists() else ""
    d = pd.read_csv(RES / f"step3h_power_{tag}{suf}_power.csv")
    raw = pd.read_csv(RES / f"step3h_power_{tag}{suf}_raw.csv")
    for _, r in d.iterrows():
        for c, lab in STATS.items():
            n = int(r.n_reps)
            k = int(round(r[c] * n))
            lo, hi = wilson(k, n)
            rows.append(dict(data=nm, minority_share=r["pi"], separation_sd=r["delta"], statistic=lab, detections=k, replicates=n, detection_rate=r[c], wilson_lo=lo, wilson_hi=hi,
                             null_datasets=int((raw.delta == 0).sum()), run="200 replicates" if suf else "20 replicates"))
pd.DataFrame(rows).to_csv(OUT / "SD3_power_detection_rates.csv", index=False)
parts = []
for x in sorted(glob.glob(str(RES / "step3c_*_summary.csv"))):
    d = pd.read_csv(x)
    d.insert(0, "representation", Path(x).name[len("step3c_"):-len("_summary.csv")])
    parts.append(d)
pd.concat(parts, ignore_index=True).to_csv(OUT / "SD4_clusterability_observed_and_null.csv", index=False)
pd.read_csv(RES / "step3k_isolation_strata_ci.csv").to_csv(OUT / "SD5_isolation_strata_ci.csv", index=False)
acc = pd.read_csv(RES / "step7_sample_accounting.csv")
acc.to_csv(OUT / "SD6a_sample_accounting.csv", index=False)
pd.read_csv(RES / "step7_sample_overlap.csv").to_csv(OUT / "SD6b_sample_overlap.csv", index=False)
shutil.copy(RES.parent / "logs" / "pip_freeze.txt", OUT / "SD7_environment_pip_freeze.txt")
(OUT / "README.txt").write_text(
    "Supplementary Data for 'Discrete or graded? A confound-controlled re-analysis of clustering in Egyptian fruit bat social calls'.\n"
    "All files are derived from the analysis scripts' result files without recomputation (script 50).\n"
    "SD1  all 1,558 clustering configurations: algorithm family, parameter, representation, number of clusters, noise share, silhouette, NMI with channel/silence, Isolation precision/recall/F1, run time.\n"
    "SD2  the best admissible configuration (at least two clusters holding >= 1% of points, highest silhouette) per algorithm and representation, with bat-transfer ARI.\n"
    "SD3  detection rates of the one-cluster-allowed battery for spiked two-type mixtures, with 95% Wilson intervals.\n"
    "SD4  observed value, null mean, null 2.5th/97.5th percentiles and empirical counts (null replicates >= observed) of each clusterability statistic, per representation.\n"
    "SD5  Isolation AUC with bat-level bootstrap intervals, sample sizes and bats by pup-age window, held-out channel and held-out treatment.\n"
    "SD6a/b  data partitions (files, Isolation files, bats, channels, days) and the overlap of held-out bats, channels and days with the discovery sample.\n"
    "SD7  Python package versions of the analysis environment.\n", encoding="utf-8")
print(sorted(p.name for p in OUT.iterdir()), len(sd1), "configurations")
