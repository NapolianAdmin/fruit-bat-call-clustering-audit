# Expert review of the June analysis ("a fluid, continuous manifold ... acoustic modes")

Reviewed: `SUMMARY_all_tests.csv`, `call_level_data.csv` (16,000 rows), `t1/t3/t5/t6` tables, `robustness_sensitivity.csv`, `cluster_selection.png`, `t4_acoustic_individuality.png`, and the abstract you pasted. I also re-checked every claim against the audio-derived data in this project (15,992 files re-extracted from the archives; the June filenames were mapped to the dataset by testing which numbering reproduces the emitter labels: the integer in `N.wav` is the 0-based **row index of Annotations.csv**, for which the emitter column matches 100.0%). All numbers below are from the supplied files or from `results/`.

## 1. Summary of findings (what the files say, and what they support)

**What the files report.** Fuzzy c-means (m = 2) on 50 PCs gives two clusters (FPC 0.584, silhouette 0.226). The 10 bats use them in different proportions (chi-square V = 0.457), differ in membership entropy (Kruskal-Wallis, eta-squared 0.041), range from generalists (bats 111, 231: breadth 0.996-0.999) to specialists (bats 226, 228: 88-89% in cluster 1), and have smaller intra- than inter-bat distances (Mann-Whitney r = 0.185). Both clusters are used by every bat (T5) and 42/45 bat pairs differ in entropy (T6). The authors read this as a shared species-wide structure with individual signatures as non-random distributions over it.

**What the data actually support.**
- The two clusters are **an axis of recording length and silence**, not of call structure. Against the real audio the June cluster label has AUC 0.887 for silence fraction, 0.846 for span length and 0.852 for number of voiced segments (distinct rows). Cluster 1 = longer files with more voiced segments.
- The "individual specialists" are **bats recorded under different conditions**. Bats 226 and 228 (88-89% cluster 1) are both treatment 9 (channels 3/4, 3 months, median span 1.9-2.4 s, 15-16% Sleeping context); all other bats have median spans 1.3-1.6 s. Across the 10 bats the cluster-1 share correlates with median silence at Spearman rho = -0.90 (p = 0.0003) and with median span at rho = +0.94.
- Within channel x month strata the emitter x cluster V is 0.458 against a permutation null of 0.398 (97.5th percentile 0.404, 0/1,000): most of the association is bat <-> session structure; the excess is small.
- Biological takeaway that survives: bat identity is decodable from the audio **within** a channel on unseen days (pooled accuracy 0.65 vs 0.31 majority, +0.34, CI 0.31-0.37), but only **weakly across** channels (0.21 accuracy vs 0.57 with random splits, 36 bats; on the later 41,116-file sample channel-blocked balanced accuracy 0.125 against a within-stratum permutation null of 0.036, and per-channel normalisation did not help). Voice and the bat's roost position within a chamber cannot be separated with these data. "Universal repertoire vs individual identity" therefore cannot be concluded from this analysis.
- The abstract's *continuous manifold* statement is compatible with the data (no density valley; prediction strength across bats 0.70, below the 0.8 rule), but the continuum being described is mostly file duration/silence.

## 2. Statistical review

| item | verdict | pitfall | fix |
|---|---|---|---|
| T1 chi-square emitter x cluster, V = 0.457 | not appropriate as an inferential test | 16,000 rows = **9,588 distinct annotation rows** (each bat's 1,600 = ~950 distinct rows repeated; 39-41% repeats), calls nested in files/bats/sessions; chi2 3,339 vs 2,009 on distinct rows; `p = 0.0` | report V with a bat-level bootstrap CI (0.21-0.57), a channel x month permutation null (0.398) and a mixed logistic model with bat and session random effects; report p as < 1/(B+1) |
| T2 Kruskal-Wallis on entropy, eta-squared 0.041 | acceptable as description | FCM membership entropy is a deterministic function of distance to the cluster midpoint, depends on m and c, not an independent measurement of "ambiguity"; calls not independent; 33% of rows have entropy > 0.95 and 25% max membership < 0.6 | treat as a property of the fitted model; use bat-level resampling |
| T3 permutation test on breadth SD | not independent of T1 | built from the same bat x cluster table; permutation scheme not stated; `p = 0.0` | state exchangeable units (permute within session); drop as a separate "confirmation" |
| T4 Mann-Whitney intra vs inter distance, r = 0.185 | **not valid** | millions of pairs (12.8M intra-bat) treated as independent; p = 0 meaningless; intra < inter is expected from recording-condition differences alone; 6,412 identical-row pairs (0.05% of the intra-bat pairs) create the spike at 0 | PERMANOVA/ANOSIM or Mantel permuting bat labels within session; or classification on held-out sessions/channels |
| T5 "universality" (> 2% usage of both clusters, 2/2) | uninformative | with two clusters of ~52/48 mass and 1,600 calls per bat the criterion is almost guaranteed; no null | compare against a within-session shuffle; report usage distributions with CIs |
| T6 pairwise KS (45 tests), "42/45", "p_value = 0.9333" | multiplicity not handled; table mislabelled | no correction (Bonferroni 0.05/45: 39/45 on all rows, 35/45 on distinct rows); the "p_value" in the summary is a proportion; abstract-era PDF says 44/45, CSV says 42/45; KS D not reported | Holm/Bonferroni, report KS D with bat-bootstrap CIs, one number everywhere |
| "six-test battery" | not six independent confirmations | T1, T3, T5, T6 all derive from one bat x cluster usage table or one entropy variable | pre-specify one primary hypothesis; control the family-wise error |
| p-values of exactly 0 | reporting error | permutation p cannot be 0 | p < 1/(B+1) with the count of null replicates (e.g. 0/1,000) |

## 3. Robustness check (robustness_sensitivity.csv, cluster_selection.png)

| m | FPC | silhouette | ARI |
|---|---|---|---|
| 1.5 | 0.744 | 0.2260 | 0.983 |
| 2.0 | 0.584 | 0.2260 | 0.9995 |
| 2.5 | 0.524 | 0.2259 | 0.987 |

- **What it shows.** The hard partition is almost insensitive to the fuzzifier m (ARI >= 0.98) and the silhouette is constant at 0.226. That tests one thing only: m. It does not test the effect of different data, bats, sessions, features, silent frames, algorithms or sample size.
- **Interpretation caveats.** (i) FPC is not comparable across m (it depends on m and has lower bound 1/c); its drop from 0.744 to 0.524 is expected arithmetic, not instability. (ii) The ARI at m = 2.0 is 0.9995, not 1, so it compares two runs; the reference partition is not defined in the file. (iii) A silhouette of 0.226 is below the 0.25 threshold conventionally read as "no substantial structure". (iv) A near-perfect ARI is exactly what a split along one dominant axis produces; here that axis is file duration/silence.
- **cluster_selection.png.** FPC, silhouette and the gap statistic all decrease monotonically from the first point (c = 2). c = 2 is the smallest value tested, k = 1 is not evaluated, and FPC cannot pick k = 1 by construction (FPC >= 1/c; simulated single-population data give FPC = 0.500 at c = 2 and are also "optimal" at c = 2). The plots therefore cannot support "optimal c = 2" against a one-cluster alternative.
- **Stability that matters** (my tests, audio re-extracted, bats resampled): prediction strength across bats 0.70 (0.62-0.83) vs the 0.8 rule; bootstrap Jaccard 0.875; held-out-bat ARI of k=2 solutions 0.42-0.85 depending on method; on voiced frames only fuzzy c-means collapses to FPC = 0.500.

## 4. Actionable feedback (contradictions and missing steps)

1. **Contradictory conclusions across versions.** The abstract now says a "fluid, continuous manifold rather than discrete states". The 13 June PDF concluded the clusters "act as discrete behavioral states" because the flux test was non-significant (p_mean > 0.25), and the July manuscript says "two reproducible regimes". A non-significant test is not evidence for discreteness. Decide the claim, then test it with methods that allow k = 1.
2. **"Geodesic" flux.** A straight line between two centroids in Euclidean PCA space is not a geodesic on a manifold, and "length-matched null trajectories" are undefined. The July audit already found the related dip statistic had an endpoint artifact. Use graph geodesics (e.g. kNN/Isomap) and a stated null, or drop the term.
3. **"Avoids topological distortion".** 50-D PCA is itself a linear projection (91% of variance of 1,408 collinear summary features). It does not make the analysis manifold-faithful.
4. **Sample size.** N = 20,000 (abstract) vs 16,000 rows vs 9,588 distinct. Equalising by resampling with replacement created the duplicates; subsample without replacement (<= 948 per bat) or weight.
5. **Silence/duration/SNR control is missing** and it is the main result: the cluster label is predicted by silence and length (AUC about 0.9) and by recording variables on unseen bats (AUC 0.92-0.97). Rebuild features on voiced frames (FileInfo voiced segments, not the annotation span, which is ~87% unvoiced).
6. **Channel/session control and held-out validation are missing.** Bats are confounded with treatment, channel and month. Use bat-/session-blocked evaluation and mixed models.
7. **Identity claims.** "Unique, non-random distributions" per bat are not separable from recording conditions here; frame as "decodable within channel, only weakly across channels".
8. **Emitter codes.** Positive-only (as in June) is right; the July labels were attached by FileID to audio indexed by row, so the July emitter/context associations are not usable.
9. **Language.** "We prove", "novel framework" and p = 0 should be removed; report effect sizes with confidence intervals.
10. **Citation.** Reference [1] in the 13 June PDF ("Barking up the correct tree ...", PLOS Biology 2017) does not match the dataset paper cited elsewhere (Prat et al., Scientific Data 2017;4:170143). I could not verify it; fix before submission.
