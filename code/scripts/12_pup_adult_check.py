"""Step 3e - is the small cluster that lines up with the 'Isolation' context a call type, or just pups' voices?

Uses a forced k-means k=2 split of a voiced-only representation (default all_mfcc_voiced), then asks:
  1. how the minority cluster is distributed over contexts
  2. among POSITIVE-emitter files (real bat identity): pup vs adult (Metadata.pdf Table 2: pups = born in the setup, with a birth date)
     x Isolation vs other contexts -> share in the minority cluster. If pups' non-isolation calls also sit in the minority cluster
     it is a pup-voice effect; if only Isolation-context calls do, it is context-linked.
  3. within pups: age at recording (days) vs cluster (AUC), i.e. a developmental gradient
  4. stratified permutation: is the cluster's alignment with the Isolation context beyond what channel x month strata produce?
Writes results/step3e_<tag>_*.csv.
Run: python code/claude/12_pup_adult_check.py [tag=all_mfcc_voiced]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RES, SEED, log_change  # noqa: E402
from clusterlib import auc_fast, boot_groups, load_set  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "all_mfcc_voiced"
# Metadata.pdf Table 2: bats with a Mother ID were born in the experimental setup (date of birth given); all others are wild-caught adults
BIRTH = {103: "2012-06-28", 105: "2012-05-19", 108: "2012-06-05", 111: "2012-05-09", 113: "2012-06-10", 115: "2012-05-20", 120: "2012-06-05",
         202: "2012-09-25", 207: "2012-10-02", 208: "2012-09-22", 210: "2012-09-30", 211: "2012-09-22", 214: "2012-09-12", 215: "2012-10-04",
         216: "2012-09-12", 220: "2012-10-14", 221: "2012-09-26", 222: "2012-09-25", 230: "2012-10-14", 231: "2012-09-22"}
X, u = load_set(TAG)
lab = KMeans(2, n_init=10, random_state=SEED).fit_predict(X)
if (lab == 0).sum() > (lab == 1).sum():
    lab = 1 - lab
u["minority"] = (lab == 0).astype(int)  # cluster 0 = minority
print(f"{TAG}: minority cluster {int(u.minority.sum())} / {len(u)} files")

ct = pd.crosstab(u.context_name, u.minority)
ct["share_minority"] = ct[1] / (ct[0] + ct[1])
ct.to_csv(RES / f"step3e_{TAG}_context_share.csv")
print(ct.round(3).to_string())

pos = u[u.Emitter > 0].copy()
pos["status"] = np.where(pos.Emitter.isin(list(BIRTH)), "pup (born in setup)", "adult (wild-caught)")
pos["isolation_context"] = np.where(pos.context_name == "Isolation", "Isolation", "other contexts")
g = pos.groupby(["status", "isolation_context"]).agg(files=("minority", "size"), share_minority=("minority", "mean"), bats=("bat", "nunique")).reset_index()
g.to_csv(RES / f"step3e_{TAG}_pup_adult_by_isolation.csv", index=False)
print(g.round(3).to_string())

pups = pos[pos.status.str.startswith("pup")].copy()
pups["age_days"] = (pd.to_datetime(pups.date) - pups.Emitter.map(lambda e: pd.Timestamp(BIRTH[e]))).dt.days
rows = []
for sub, d in [("pups, all contexts", pups), ("pups, Isolation context only", pups[pups.context_name == "Isolation"]), ("pups, other contexts", pups[pups.context_name != "Isolation"])]:
    d = d[d.age_days.notna()]
    if d.minority.nunique() == 2 and len(d) > 30:
        a = auc_fast(d.minority.values, d.age_days.values)
        lo, hi = boot_groups(d, lambda z: auc_fast(z.minority.values, z.age_days.values), n_boot=300)
        rows.append(dict(subset=sub, n=len(d), bats=d.bat.nunique(), AUC_age_for_minority_cluster=a, ci95_lo=lo, ci95_hi=hi,
                         median_age_minority=d[d.minority == 1].age_days.median(), median_age_majority=d[d.minority == 0].age_days.median()))
pd.DataFrame(rows).to_csv(RES / f"step3e_{TAG}_age_vs_cluster.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())

iso = u[u.context_name == "Isolation"].copy()
iso["em_class"] = np.where(iso.Emitter > 0, "positive", np.where(iso.Emitter < 0, "negative(pair)", "zero"))
iso.groupby("em_class").agg(files=("minority", "size"), share_minority=("minority", "mean")).to_csv(RES / f"step3e_{TAG}_isolation_by_emitter_class.csv")

iso_bin = (u.context_name == "Isolation").astype(int).values
obs = normalized_mutual_info_score(iso_bin, u.minority.values)
rg = np.random.default_rng(SEED + 31)
idx = [np.where(u.stratum.values == s)[0] for s in u.stratum.unique()]
null = []
for _ in range(1000):
    p = u.minority.values.copy()
    for ii in idx:
        p[ii] = rg.permutation(p[ii])
    null.append(normalized_mutual_info_score(iso_bin, p))
null = np.array(null)
ge = int((null >= obs).sum())
res = dict(NMI_cluster_vs_isolation_context=obs, null_mean=null.mean(), null_q975=np.quantile(null, .975), n_null_ge_obs=ge, n_perm=1000,
           p=(f"<{1/1001:.4f} (0/1000)" if ge == 0 else f"{(1+ge)/1001:.4f}"))
pd.DataFrame([res]).to_csv(RES / f"step3e_{TAG}_strata_perm.csv", index=False)
print(res)
log_change(f"Step 3e pup/adult check on '{TAG}': results/step3e_{TAG}_*")
