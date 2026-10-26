"""Step 3g (addendum) - the June table's 'individual specialists': are they bats recorded under different conditions?

Same mapping as 15_june_check.py: the June filename integer is the 0-based row of Annotations.csv, so the true treatment, month, channel, context and the file's span / silence
(from FileInfo voice segments; metadata only, no audio) are known for every June call.  Per June bat (10 bats): share of calls in June cluster 1, median annotated span (s),
median silence fraction, share of Sleeping-context calls, treatment ids and months.  Spearman correlation across the 10 bats of the cluster-1 share with median silence and with median span
(distinct annotation rows only, one row per annotation).
Writes results/step3g_june_specialists.csv and prints the correlations.
Run: python code/claude/37_june_specialists.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT, RES, log_change  # noqa: E402

j = pd.read_csv(RES.parent / "data" / "raw" / "june_outputs" / "Bat vocalization" / "call_level_data.csv")
j["n"] = j.filename.str.replace(".wav", "", regex=False).astype(int)
ann = pd.read_csv(FEAT / "annotations_merged.csv.gz")
t = ann.iloc[j.n.values].reset_index(drop=True)
j["span"], j["silence"] = t.annot_len_s.values, 1 - t.voiced_fraction_of_annot.values
j["treatment"], j["month"], j["channel"], j["context"] = t.treatment_id.astype(str).values, t.month.values, t.channel.astype(str).values, t.context_name.values
u = j.drop_duplicates("n")
rows = []
for b, g in u.groupby("emitter"):
    rows.append(dict(emitter=b, distinct_rows=len(g), share_cluster1=float((g.cluster == 1).mean()), median_span_s=float(g.span.median()), median_silence=float(g.silence.median()),
                     share_sleeping=float((g.context == "Sleeping").mean()), treatments="/".join(sorted(g.treatment.unique(), key=int)), n_months=int(g.month.nunique()),
                     channels="/".join(sorted(g.channel.unique(), key=int))))
P = pd.DataFrame(rows).sort_values("share_cluster1", ascending=False)
r1 = spearmanr(P.share_cluster1, P.median_silence)
r2 = spearmanr(P.share_cluster1, P.median_span_s)
P.to_csv(RES / "step3g_june_specialists.csv", index=False)
pd.set_option("display.width", 250)
print(P.round(3).to_string())
print(f"Spearman cluster-1 share vs median silence: rho {r1.statistic:.3f}, p {r1.pvalue:.2g}; vs median span: rho {r2.statistic:.3f}, p {r2.pvalue:.2g}  (n = {len(P)} bats)")
pd.DataFrame([dict(statistic="spearman cluster1_share vs median_silence", rho=r1.statistic, p=r1.pvalue, n_bats=len(P)),
              dict(statistic="spearman cluster1_share vs median_span", rho=r2.statistic, p=r2.pvalue, n_bats=len(P))]).to_csv(RES / "step3g_june_specialists_spearman.csv", index=False)
log_change("Step 3g addendum: June specialist bats - per-bat cluster-1 share, span, silence, treatment, Spearman across 10 bats: results/step3g_june_specialists*.csv")
