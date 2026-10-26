"""Step 9 - write the supplementary tables of the paper (paper/tables/*.tex) DIRECTLY from the result files, so that no number in them is typed by hand.
Missing input files are skipped with a message.   Run: python code/claude/42_make_supplement_tables.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RES, log_change  # noqa: E402

TAB = RES.parent / "paper" / "tables"
TAB.mkdir(exist_ok=True)
BS = chr(92)


def tex(s):
    return str(s).replace("&", BS + "&").replace("%", BS + "%").replace("_", BS + "_")


def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write(name, caption, label, header, rows, colfmt, size="scriptsize"):
    lines = [f"{BS}begin{{table}}[htbp]{BS}centering{BS}{size}", f"{BS}caption{{{BS}textbf{{Table {name}.}} {caption}}}{BS}label{{{label}}}", f"{BS}begin{{adjustbox}}{{max width={BS}linewidth}}", f"{BS}begin{{tabular}}{{{colfmt}}}", BS + "toprule", " & ".join(header) + " " + BS + BS, BS + "midrule"]
    lines += [" & ".join(r) + " " + BS + BS for r in rows]
    lines += [BS + "bottomrule", f"{BS}end{{tabular}}", f"{BS}end{{adjustbox}}", f"{BS}end{{table}}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name, len(rows), "rows")


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rd(name):
    p = RES / name
    if not p.exists():
        print("MISSING", name)
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------- S1: clusterability battery by representation
SETS = [("original matrix (draft, 3,897 rows)", "step3c_old_matrix_summary.csv"), ("whole-span 1,408 (15,992 files)", "step3c_all_naive_ref1_summary.csv"),
        ("voiced 1,408 (15,992 files)", "step3c_all_voiced_ref1_summary.csv"), ("voiced MFCC (15,992 files)", "step3c_all_mfcc_voiced_summary.csv"),
        ("syllable shape (30,000)", "step3c_segall_shape_summary.csv"), ("VAE latent, unseen bats (13,510)", "step3c_segall_vae8_b0.1_summary.csv"),
        ("8-PC baseline, unseen bats (13,510)", "step3c_segall_pca8_summary.csv"), ("Dataset B microphone (12,417)", "step3c_B_mic_voiced_ref1_summary.csv"),
        ("Dataset B piezo (12,417)", "step3c_B_al_voiced_ref1_summary.csv")]
rowsA, rowsB = [], []
for lab, fn in SETS:
    d = rd(fn)
    if d is None:
        continue
    d = d.set_index("statistic")

    def tri(s, dec=3, ci=False):
        r = d.loc[s]
        o = f(r.observed, dec) + (f" [{f(r.boot_by_bat_q025, 2)}, {f(r.boot_by_bat_q975, 2)}]" if ci else "")
        return [o, f(r.copula_null_mean, dec), f(r.session_null_mean, dec)]
    rowsA.append([tex(lab)] + tri("PS2", 3, True) + tri("Jaccard2", 3))
    rowsB.append([tex(lab)] + tri("FPC2_excess") + tri("Silhouette2") + tri("DipSplit", 4))
write("S1a", "Prediction strength (PS; bat-level 95\\% interval in brackets) and bootstrap Jaccard stability at $k=2$ (definitions in Table S1): observed value, mean of the Gaussian-copula null (cop.) and of the session-mean null (ses.), each over 50--100 simulated datasets. Higher = more stable. Representations: file-level features of the 15,992-file development set unless stated; unseen bats = the 21 bats not used to train the VAE.", "tab:S1a",
      ["representation", "PS obs.", "PS cop.", "PS ses.", "Jaccard obs.", "cop.", "ses."], rowsA, "lcccccc")
write("S1b", "Fuzzy partition coefficient (FPC) excess over 0.5, silhouette and split-sample Hartigan dip statistic at $c=k=2$ (Table S1): observed value, mean of the Gaussian-copula null (cop.) and of the session-mean null (ses.). Higher = more cluster-like for all three statistics.", "tab:S1b",
      ["representation", "FPC$-$0.5 obs.", "cop.", "ses.", "sil. obs.", "cop.", "ses.", "dip obs.", "cop.", "ses."], rowsB, "lccccccccc")

# S1c: position of the observed value in each null distribution (empirical, one-sided)
rowsC = []
STAT = [("FPC2_excess", "FPC$-$0.5"), ("Silhouette2", "silhouette"), ("PS2", "PS"), ("Jaccard2", "Jaccard"), ("DipSplit", "dip")]
for lab, fn in SETS:
    d = rd(fn)
    if d is None:
        continue
    d = d.set_index("statistic")
    cells = []
    for st, _ in STAT:
        for nul in ("copula", "session"):
            r = d.loc[st]
            n = int(str(r[f"{nul}_p"]).split("/")[1].rstrip(")"))
            cells.append(f"{int(r[f'{nul}_n_ge_obs'])}/{n}")
    rowsC.append([tex(lab)] + cells)
write("S1c", "Position of the observed statistic in each null distribution: number of null replicates (of $N$, 50--100) whose value is at least as large as the observed one, for the Gaussian-copula (cop.) and session-mean (ses.) nulls. The empirical one-sided $p$ is $(1+k)/(N+1)$, so $0/50$ means $p<1/51$; $N/N$ means the observed value lies below every null replicate (the stability statistics PS and Jaccard are then lower than expected without structure). Dip = split-sample dip statistic; other abbreviations as in Table S1. Where the FPC excess is numerically zero (all voiced and syllable data sets; collapse to one cluster) it is zero to below $10^{-7}$ in data and nulls alike, so its count carries no information; a small count for the silhouette means the observed silhouette exceeds the null, not that the excess is large (Table S1b).", "tab:S1c",
      ["representation"] + [f"{b} {n}" for _, b in STAT for n in ("cop.", "ses.")], rowsC, "l" + "c" * 10)

# ---------------------------------------------------------------- S2: power (Wilson 95% intervals; the 200-replicate run is used when present)
for tag, nm, cap in (("all2_voiced_ref1", "S2a", "voiced file-level features"), ("segall_shape", "S2b", "syllable shapes")):
    suf = "_R200" if (RES / f"step3h_power_{tag}_R200_raw.csv").exists() else ""
    d = rd(f"step3h_power_{tag}{suf}_power.csv")
    raw = rd(f"step3h_power_{tag}{suf}_raw.csv")
    if d is None or raw is None:
        continue
    n0 = int((raw.delta == 0).sum())
    out = []
    for _, r in d.iterrows():
        n = int(r["n_reps"])
        cells = []
        for c in ("power_PS2", "power_Silhouette2", "power_Jaccard2", "power_DipSplit", "power_dip_p<.05", "power_FPC_excess>0.01"):
            k = int(round(r[c] * n))
            lo, hi = wilson(k, n)
            cells.append(f"{r[c]:.2f} [{lo:.2f}, {hi:.2f}]")
        out.append([f"{int(round(r['pi'] * 100))}\\%", f"{r['delta']:g}"] + cells)
    reps = int(d.n_reps.iloc[0])
    write(nm, f"Detection rate of the one-cluster-allowed battery for a two-type mixture spiked into a null of {cap}, with 95\\% Wilson intervals ({reps} simulated datasets per row). Minority = share of rows shifted; separation = shift in standard deviations (SD) along a random direction in the top ten principal components. Detection = statistic above the 95th percentile of the no-structure null (fixed once per data set from {n0} independent null datasets, not re-estimated per row); PS = prediction strength, dip = split-sample Hartigan dip (statistic, or median $p<0.05$), FPC$>0.01$ = fuzzy partition coefficient excess above 0.01.", f"tab:{nm}",
          ["minority", "separation (SD)", "PS", "silhouette", "Jaccard", "dip stat.", "dip $p<.05$", "FPC$>0.01$"], out, "llcccccc")

# ---------------------------------------------------------------- S3: identity (ONE table; accuracy, balanced accuracy and baselines kept apart)
FOLDS = [("random 5-fold (files)", "random 5-fold"), ("blocked by day", "day-blocked"), ("blocked by channel", "channel-blocked"), ("blocked by month", "month-blocked"), ("blocked by archive folder", "archive-blocked")]
rows = []
for lab, fn, tagn in (("voiced 1,408", "step4_emitter_id_random_vs_blocked_all2_voiced_ref1.csv", "all2_voiced_ref1"), ("whole-span 1,408", "step4_emitter_id_random_vs_blocked_all2_naive_ref1.csv", "all2_naive_ref1")):
    d, cn = rd(fn), rd(f"step4d_channel_norm_{tagn}.csv")
    if d is None or cn is None:
        continue
    d = d.set_index("scheme")
    for sch, nm in FOLDS:
        if sch not in d.index:
            continue
        r = d.loc[sch]
        maj = 0.152 if nm == "channel-blocked" else 0.145
        nul = f(float(cn[cn.normalisation == "raw"].perm_null_bal_channel_blocked_mean.iloc[0]), 3) if nm == "channel-blocked" else "--"
        rows.append([tex(lab), "none", nm, f"{f(r.accuracy)} [{f(r.ci95_lo)}, {f(r.ci95_hi)}]", f(r.balanced_accuracy), f(maj), nul])
    for _, r in cn[cn.normalisation != "raw"].iterrows():
        for nm, c, nul in (("random-5-fold", "bal_random", "--"), ("day-blocked", "bal_day_blocked", "--"), ("channel-blocked", "bal_channel_blocked", f(r.perm_null_bal_channel_blocked_mean, 3))):
            rows.append([tex(lab), tex(r.normalisation), nm.replace("random-5-fold", "random 5-fold"), "--", f(r[c]), "--", nul])
cn = rd("step4d_channel_norm_segall_shape.csv")
if cn is not None:
    for _, r in cn.iterrows():
        for nm, c, nul in (("random 5-fold", "bal_random", "--"), ("day-blocked", "bal_day_blocked", "--"), ("channel-blocked", "bal_channel_blocked", f(r.perm_null_bal_channel_blocked_mean, 3))):
            rows.append(["syllable shape (36 bats)", tex("none" if r.normalisation == "raw" else r.normalisation), nm, "--", f(r[c]), "--", nul])
write("S3", "Emitter identification from file-level features (40 bats) and syllable shapes (36 bats): accuracy with 95\\% bat-level bootstrap interval, balanced accuracy, the majority-class accuracy baseline, and, for channel-blocked folds, the mean balanced accuracy over permutations of the labels within channel$\\times$month strata (null). Uniform chance (balanced accuracy of a random guess) is 0.025 for 40 bats and 0.028 for 36. 40 bats = the 40 of the 41 positive-coded bats with at least 30 files in the 41,116-file discovery sample (the 41st has 23); 36 = bats with at least 30 of the 30,000 sampled syllables. Rows of bats absent from the training folds are dropped (test rows kept: 100\\% except 96\\% for channel-blocked). Normalisations use no labels: chan\\_mean = per-channel mean subtracted, chan\\_z = per-channel standardisation, chanmonth\\_mean = per-channel$\\times$month mean subtracted; accuracy was not recorded for them (--). Model: multinomial logistic regression on 50 principal components.", "tab:S3",
      ["features", "normalisation", "folds", "accuracy [95\\% CI]", "balanced acc.", "majority acc.", "null bal. acc."], rows, "lllcccc")

# ---------------------------------------------------------------- S4: Isolation robustness
sep, top, per, chb, age = (rd(x) for x in ("step3k_isolation_separability.csv", "step3k_isolation_without_top_bat.csv", "step3k_isolation_per_bat.csv", "step3k_isolation_channel_blocked.csv", "step3k_isolation_age_matched.csv"))
V, SH = "all2_voiced_ref1", "segall_shape"
rows = []
if sep is not None:
    for an, lab in (("all files, leave-bats-out", "all files, folds blocked by bat"), ("pups only (Isolation vs their other contexts), leave-bats-out", "pups only, folds blocked by bat")):
        r = {s: sep[(sep.set == s) & (sep.analysis == an)].iloc[0] for s in (V, SH)}
        rows.append([lab] + [f"{f(r[s].AUC)} [{f(r[s].AUC_lo)}, {f(r[s].AUC_hi)}]" for s in (V, SH)])
    r = {s: sep[(sep.set == s) & sep.analysis.str.startswith("pups only, within channel")].iloc[0] for s in (V, SH)}
    rows.append(["pups only, within channel$\\times$month strata (weighted mean)", f(r[V].AUC), f(r[SH].AUC)])
if top is not None:
    t = top.set_index("set")
    rows.append(["without the unknown-emitter group (57\\% of Isolation files)"] + [f"{f(t.loc[s].AUC)} [{f(t.loc[s].AUC_lo)}, {f(t.loc[s].AUC_hi)}]" for s in (V, SH)])
if per is not None:
    g = per.groupby("set").AUC
    rows.append(["per identified pup (median; range; number of pups)"] + [f"{f(g.median()[s])}; {f(g.min()[s], 2)}--{f(g.max()[s], 2)}; {g.count()[s]}" for s in (V, SH)])
if chb is not None:
    for ho, lab in (("channel", "each channel held out (weighted mean; range; number)"), ("treatment_id", "each treatment held out (weighted mean; range; number)")):
        c = chb[(chb.held_out == ho) & chb.group.astype(str).str.startswith("POOLED")].set_index("set")
        rows.append([lab] + [f"{f(c.loc[s].AUC)}; {f(c.loc[s].AUC_min, 2)}--{f(c.loc[s].AUC_max, 2)}; {int(c.loc[s].n_groups)}" for s in (V, SH)])
if age is not None:
    a = age[age.age_window_days.astype(str).str.startswith("WEIGHTED")].set_index("set")
    rows.append(["pup age windows (file-weighted mean over windows with at least 15 recordings per class; per window in Table S4b)"] + [f(a.AUC.get(s, np.nan)) for s in (V, SH)])
write("S4", "Separability of pups' Isolation calls from all other contexts, as the area under the ROC curve (AUC; 0.5 = none) of out-of-fold scores of a balanced logistic regression on 50 principal components, for voiced file-level features (41,116 files) and syllable shapes (30,000 syllables). Validation unit: whole bats (five folds blocked by bat) unless the row states another held-out unit; brackets are 95\% intervals from resampling whole bats; ranges and counts refer to the groups listed in the row (Table S4b gives each group's sample size).", "tab:S4",
      ["analysis", "voiced file features", "syllable shape"], rows, "lcc")

# S4b: sample sizes and bat-level intervals of the age-window and channel/treatment-held-out AUCs
st = rd("step3k_isolation_strata_ci.csv")
if st is not None:
    rows, seen = [], set()
    for stratum in ("pup age (days)", "channel held out", "treatment held out"):
        for tag in (("all2_voiced_ref1", "segall_shape") if stratum.startswith("pup") else ("all2_voiced_ref1",)):
            d = st[(st.stratum == stratum) & (st.set == tag)]
            d = d.assign(_o=pd.to_numeric(d.group, errors="coerce")).sort_values("_o", kind="stable") if not stratum.startswith("pup") else d
            for _, r in d.iterrows():
                if np.isnan(r.AUC):
                    auc = "not evaluated (< 15 per class)" if stratum.startswith("pup") else "--"
                elif r.bats < 5:
                    auc = f"{f(r.AUC)} [interval not meaningful, {int(r.bats)} bat(s)]"
                else:
                    auc = f"{f(r.AUC)} [{f(r.AUC_lo)}, {f(r.AUC_hi)}]"
                lab_s = stratum if stratum not in seen else ""
                seen.add(stratum)
                rows.append([lab_s, "voiced file features" if tag == "all2_voiced_ref1" else "syllable shape", tex(str(r.group)),
                             f"{int(r.recordings):,}", f"{int(r.isolation):,}", f"{int(r.other):,}", str(int(r.bats)), auc])
    write("S4b", "Isolation versus other contexts within strata: recordings (files or syllables), Isolation (positive) and other-context (negative) counts, number of unique bats (pups for age windows) and AUC of the leave-bats-out score with a 95\\% percentile interval from 1,000 bootstrap resamples of whole bats. Age windows are days since birth for the 20 pups with known birth dates (pup age stratum), evaluated only with at least 15 recordings per class; the windows were fixed once at analysis time as an exploratory control after the Isolation group had been found, not pre-specified. Channel/treatment held out: model trained on all other channels (treatments), scored on the held-out one; evaluated only with at least 20 per class. Intervals reflect resampling of bats among scored recordings and do not include model-fitting variability. Youngest pups (0--40 days) contribute almost only Isolation recordings, so age and context cannot be separated there, and the lower AUC at 40--60 days should not be read as the same distinctness at every developmental stage.", "tab:S4b",
          ["stratum", "features", "group", "recordings", "Isolation", "other", "bats", "AUC [95\\% CI]"], rows, "llrrrrrl")

# S5b: data partitions
acc, ovl = rd("step7_sample_accounting.csv"), rd("step7_sample_overlap.csv")
if acc is not None and ovl is not None:
    role = {"discovery": "all development, exploratory discovery of the cluster, supervised tests blocked by bat/channel/treatment/age, choice of settings",
            "replication": "out-of-sample test of C1--C3 (random draw)", "completion": "second out-of-sample test of C1--C3 (all remaining files)",
            "reproduction only": "extracted only to reproduce the June table; not used for any test", "all annotated files": "every annotated file of the corpus (205,252 further recordings are unannotated and unused)"}
    rows = [[tex(r.partition), f"{int(r.files):,}", f"{int(r.isolation_files):,}", f"{int(r.positive_coded_files):,}", str(int(r.positive_coded_bats)), str(int(r.channels)), str(int(r.recording_days)), tex(role[r.partition])] for _, r in acc.iterrows()]
    write("S5b", "Data partitions and sample accounting. Partitions are disjoint sets of annotated files (one annotation row per file); the unit separating them is the recording (file). Bats, channels and recording days overlap: all 33 positive-coded bats of the replication sample and of the completion sample, and all 12 channels, also occur in the discovery sample (recording days: 329 of 330 and 338 of 338 shared), so the out-of-sample sets test over-fitting to the discovery recordings, not generalisation to new individuals, recording set-ups or colonies.", "tab:S5b",
          ["partition", "files", "Isolation files", "positive-coded files", "bats", "channels", "days", "role"], rows, "lrrrrrrp{5.2cm}")

# S9: syllable duration, one versus two Gaussians
dm, dg, db = rd("step6_duration_modes.csv"), rd("step6_duration_gmm.csv"), rd("step6_duration_bic_ci.csv")
if dm is not None and dg is not None and db is not None:
    rows = []
    for _, r in dm.iterrows():
        rows.append([tex(r.subset), f"{int(r.n_segments):,}", str(int(r.n_bats)), f"{r.dip:.4f} [{r.dip_bat_boot_q025:.4f}, {r.dip_bat_boot_q975:.4f}]", f(r.dip_p_iid, 2), f"{r.median_dur_ms:.0f} ({r.q05_ms:.0f}--{r.q95_ms:.0f})"])
    write("S9a", "Modality of log$_{10}$ syllable duration: number of syllables and bat groups, Hartigan dip statistic with a 95\\% interval from 100 bootstrap resamples of whole bats, the dip $p$ value (syllables treated as independent, so optimistic), and median (5th--95th percentile) duration in ms. All syllables of the 30,000-syllable analysis set that have a context label of at least 500 syllables are listed per context.", "tab:S9a",
          ["subset", "syllables", "bats", "dip [95\\% CI]", "dip $p$", "duration, ms"], rows, "lrrccc")
    g = dg.set_index("k")
    r = db.iloc[0]
    rows = [[str(int(k)), f"{g.loc[k].BIC:,.0f}", f"{g.loc[k].delta_BIC_vs_best:,.0f}"] for k in g.index]
    rows.append(["BIC(1) $-$ BIC(2) [95\\% CI]", f"{r.delta_BIC_1_minus_2:,.0f} [{r.ci_lo:,.0f}, {r.ci_hi:,.0f}]", f"{int(r.n_boot)} bat-level resamples"])
    write("S9b", f"One- versus two-component Gaussian mixtures for log$_{{10}}$ syllable duration ({int(r.n_syllables):,} syllables, {int(r.n_bat_groups)} bat groups): Bayesian information criterion (BIC; lower is better; 3 initialisations, seed 42) and its difference to the best number of components. A positive BIC(1) $-$ BIC(2) favours two components; the interval resamples whole bats. Two-component fit: means {tex(r.means_ms)} ms, weights {tex(r.weights)}. A lower BIC for more components is expected for a smooth, skewed distribution and is not evidence of discrete duration classes (dip test in Table S9a).", "tab:S9b",
          ["components", "BIC", "$\\Delta$BIC vs best"], rows, "lcc")

# ---------------------------------------------------------------- S5: replication
rows = []
for tag, lab in (("replication", "replication sample (12,000)"), ("rest", "completion sample (33,557)")):
    d = rd(f"step7_{tag}_summary.csv")
    if d is None:
        continue
    r = d.iloc[0]
    c2 = f"{f(r.C2_cluster_share, 3)} / {f(r.C2_isolation_share_within, 3)}" if bool(r.C2_pass) else "not met"
    rows.append([lab, f(r.C1_AUC, 3) + f" [{f(r.C1_AUC_lo, 3)}, {f(r.C1_AUC_hi, 3)}]", f(r.C1_d_prime, 2), f(r.C1_AUC_pups_only, 3), c2, f"{int(r.real_k_100)} vs {f(r.null_k_100_mean, 1)}; {f(r.real_noise_100, 2)} vs {f(r.null_noise_100_mean, 2)}"])
write("S5", "Out-of-sample evaluation of the Isolation finding with the pre-specified criteria. C1: AUC of the model trained on discovery files only (95\% interval resampling whole bats, 300 resamples), $d'$ = standardised separation of the Isolation and other-context scores, pups-only AUC; C2: share of files and share of Isolation-context files in the small UMAP--HDBSCAN cluster (fitted on the new files alone); C3: number of clusters and share of points labelled noise, real data versus the mean of six Gaussian-copula null datasets. Samples are described in Table S5b; they share bats and channels with the discovery sample.", "tab:S5",
      ["sample", "C1 AUC [95\\% CI]", "$d'$", "pups only AUC", "C2 share / purity", "C3 clusters; noise"], rows, "lccccc")

# ---------------------------------------------------------------- S6: PCA sensitivity
d = rd("step3n_pca_sensitivity.csv")
if d is not None:
    rows = [[tex(r.preprocessing), str(int(r.n_pc)), f(r.FPC2_excess), f(r.kmeans_silhouette_k2), f(r.prediction_strength_k2), f(r.dip_median_p), f(r.isolation_AUC_bat_blocked),
             f(r.get("AUC_label_vs_silence_fraction", np.nan), 2), f(r.get("AUC_label_vs_voiced_dur_s", np.nan), 2)] for _, r in d.iterrows()]
    write("S6", "Sensitivity of the voiced-feature conclusions to preprocessing and the number of principal components (PCs), 12,000 files. FPC$-$0.5 = fuzzy partition coefficient excess at $c=2$; $k$-means sil. = silhouette at $k=2$; PS = prediction strength at $k=2$ (halves split by bat); dip $p$ = median split-sample dip $p$ value; Isolation AUC from all 41,116 files with folds blocked by bat; the last two columns give the orientation-free AUC of the fuzzy $c=2$ label against silence fraction and voiced duration.", "tab:S6",
          ["preprocessing", "PCs", "FPC$-$0.5", "$k$-means sil.", "PS", "dip $p$ (median)", "Isolation AUC", "label vs silence", "label vs duration"], rows, "llccccccc")

# ---------------------------------------------------------------- S7: all clustering algorithms
S = rd("step3m_all_methods_summary.csv")
files = [x for x in glob.glob(str(RES / "step3m_all_methods_*.csv")) if "summary" not in x]
if S is not None and files:
    D = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
    labelled = set(D[D.iso_best_f1.notna()].set)
    rows = []
    for fam in list(D.family.drop_duplicates()):
        b, a = S[S.family == fam], D[D.family == fam]
        rows.append([tex(fam), str(b.set.nunique()), f(b.silhouette.median(), 2), str(int((b.silhouette >= 0.25).sum())), f(b.bat_transfer_ari.median(), 2),
                     str(int((b.iso_found == True).sum())), str(a[a.iso_found == True].set.nunique())])  # noqa: E712
    write("S7", f"Fifteen clustering algorithms (26 variants) on 19 representations ($n\\le6{{,}}000$ each): per variant, the number of representations with an admissible partition (at least two clusters holding $\\ge1\\%$ of the points), the median silhouette of its best configuration, the number of representations whose best configuration has silhouette $\\ge0.25$, the median bat-transfer ARI, and the number of the {len(labelled)} representations with context labels in which the best-silhouette configuration (best) or any configuration (any) contains a cluster with precision $\\ge0.7$ and recall $\\ge0.5$ for pups' Isolation calls. Silhouette: mean silhouette width of non-noise points (3,000-point subsample); ARI: adjusted Rand index between a half-bat fit transferred to the other half and a fit on that half.", "tab:S7",
          ["algorithm / variant", "admissible", "median sil.", "sil.$\\ge0.25$", "median transfer ARI", "Isolation (best)", "Isolation (any)"], rows, "lcccccc")
log_change("Step 9 supplementary tables written from result files: paper/tables/*.tex")
