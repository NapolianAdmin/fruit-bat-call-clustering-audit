"""Step 3g - audit of the JUNE draft's call-level table (16,000 rows = 10 bats x 1,600, 'T1-T6' battery).

Findings checked here (filename number = 0-based row of Annotations.csv; emitter labels are then exactly right for that audio):
  1  pseudo-replication: distinct rows per bat vs the 'balanced' 1,600; share of rows that are repeats
  2  T1 cluster x emitter V on all 16,000 rows (as published) vs ONE row per distinct annotation row, with bat-level bootstrap
  3  what the June cluster is: AUC against real silence / span length / #voiced segments of the audio
  4  T1 repeated within channel x month strata (stratified permutation, unique rows)
  5  T6 entropy: pairwise KS between bats on all rows vs unique rows; entropy vs span length
Run: python code/claude/15_june_check.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT, RES, SEED, cramers_v, log_change  # noqa: E402
from clusterlib import auc_fast, boot_groups  # noqa: E402

j = pd.read_csv(RES.parent / "data" / "raw" / "june_outputs" / "Bat vocalization" / "call_level_data.csv")
j["n"] = j.filename.str.replace(".wav", "", regex=False).astype(int)
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
t = ann.iloc[j.n.values].reset_index(drop=True)
j["true_emitter"] = t.Emitter.values
j["channel"], j["month"] = t.channel.astype(str).values, t.month.values
j["span"], j["silence"], j["nseg"] = t.annot_len_s.values, 1 - t.voiced_fraction_of_annot.values, t.n_voice_seg_in_annot.values
j["bat"] = j.emitter.astype(str)
S = {"emitter_equals_annotation_emitter_of_row_n": float((j.emitter == j.true_emitter).mean())}

per = j.groupby("emitter").agg(rows=("n", "size"), distinct_rows=("n", "nunique"))
per["repeat_share"] = 1 - per.distinct_rows / per.rows
per.to_csv(RES / "step3g_june_replication_by_bat.csv")
S["pseudo_replication"] = dict(rows=len(j), distinct_annotation_rows=int(j.n.nunique()), repeat_rows=int(len(j) - j.n.nunique()),
                               distinct_per_bat_min=int(per.distinct_rows.min()), distinct_per_bat_max=int(per.distinct_rows.max()))

u = j.drop_duplicates("n").reset_index(drop=True)
rows = []
for name, d in [("all 16,000 rows (as published)", j), ("one row per distinct annotation row", u)]:
    v, chi2, p, n_, r_, k_ = cramers_v(d.cluster, d.emitter, False)
    lo, hi = boot_groups(d, lambda z: cramers_v(z.cluster, z.emitter, False)[0], group_col="bat", n_boot=300)
    rows.append(dict(rows_used=name, n=int(n_), V_plain=v, ci95_lo=lo, ci95_hi=hi, chi2=chi2, p=p))
pd.DataFrame(rows).to_csv(RES / "step3g_june_T1_V.csv", index=False)

S["cluster_vs_real_audio_properties(unique rows)"] = {c: float(auc_fast(u.cluster.values, u[c].values)) for c in ["silence", "span", "nseg"]}

u["stratum"] = u.channel + "|" + u.month
obs = cramers_v(u.cluster, u.emitter, False)[0]
idx = [np.where(u.stratum.values == s)[0] for s in u.stratum.unique()]
rg = np.random.default_rng(SEED + 9)
null = []
for _ in range(1000):
    p_ = u.cluster.values.copy()
    for ii in idx:
        p_[ii] = rg.permutation(p_[ii])
    null.append(cramers_v(p_, u.emitter.values, False)[0])
null = np.array(null)
ge = int((null >= obs).sum())
S["T1_given_channel_month"] = dict(V_obs=obs, null_mean=float(null.mean()), null_q975=float(np.quantile(null, .975)), n_strata=len(idx),
                                   p=(f"<{1/1001:.4f} (0/1000)" if ge == 0 else f"{(1+ge)/1001:.4f}"))

ks = []
for name, d in [("all rows", j), ("unique rows", u)]:
    pv = [ks_2samp(d[d.emitter == a].entropy, d[d.emitter == b].entropy).pvalue for a, b in combinations(sorted(d.emitter.unique()), 2)]
    ks.append(dict(rows=name, pairs=len(pv), significant_005=int((np.array(pv) < .05).sum()), significant_bonferroni=int((np.array(pv) < .05 / len(pv)).sum())))
pd.DataFrame(ks).to_csv(RES / "step3g_june_T6_ks.csv", index=False)
rho, p = spearmanr(u.entropy, u.span)
S["entropy_vs_span_length_spearman"] = dict(rho=float(rho), p=float(p), n=len(u))
(RES / "step3g_june_summary.json").write_text(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
log_change("Step 3g June draft audit: replication, T1 V (all vs unique rows), cluster vs real silence, KS; results/step3g_june_*")
pd.set_option("display.width", 220)
print(per.to_string())
print(pd.DataFrame(rows).round(4).to_string())
print(pd.DataFrame(ks).to_string())
print(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
