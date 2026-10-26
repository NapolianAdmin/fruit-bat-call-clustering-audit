# Verification of the numbers in `paper/main.tex` (audit pass of 2026-10-08, after the paper compiled)

Every group of numbers in the paper was re-read against the file that produced it. "OK" = identical after rounding. Changes made because of the audit are listed in the last column.

| paper section / claim | source file(s) | result |
|---|---|---|
| Reproduction (Table 1: FPC 0.5786, XB 1.2496, silhouette 0.2195, sizes 1,860/2,037, ARI 1.000; other dB references ARI 0.72 / 0.73) | `step2_baseline_variants_hrowfull.csv`, `step2_baseline_report_hrowfull.md` | OK |
| Corpus counts (91,080 annotations: 39,147 positive / 44,075 negative / 7,858 zero; 293,238 files; start = 1 in 94%; median span 1.48 s; median voiced 12.6%; 41 positive bats) | `step1_summary.json` | OK |
| Sample (41,116 files: 18,450 positive / 19,170 negative / 3,496 zero; 12 channels, 14 months, 30 archives, 77 strata) | `set_all2_voiced_ref1_meta.csv` | OK (3,500 corrected to 3,496; "per bat / per context minimum" now says "wherever the corpus has that many") |
| Draft labels vs audio (0.2% / 22.6%; V 0.311 / 0.441; strata 0.299 vs null 0.235; Table 2 durations; r = -0.4223) | `step3f_draft_true_labels_*` | OK (plain V now paired with the bias-corrected value and its CI) |
| June table (9,588 distinct rows; 948-969 per bat; V 0.457 / 0.458, chi2 3,339 / 2,009; strata null 0.398 / 0.404; silence AUC 0.887; KS 42/45 (39) all rows, 40/45 (35) distinct) | `step3g_june_*` | OK |
| June specialists (226, 228: 88-89% cluster 1, treatment 9, channels 3/4, spans 1.9 / 2.4 s, 15-16% sleeping; rho -0.90, p 3e-4; rho +0.94 with span) | recomputed: `step3g_june_specialists*.csv` (new script `37_june_specialists.py`) | OK (these numbers had no saved script before) |
| Confound audit on 41,116 files (AUCs 0.887, 0.905, 0.844, 0.477, 0.914, 0.836, 0.585, 0.640, 0.968 with CIs; context V 0.302 vs null 0.130; FPC 0.583) | `step3a_all2_naive_ref1_*`, `step3b_methods_all2.csv` | OK |
| Simulated one-population control (FPC <= 0.503; picks c = 2: FPC 100%, silhouette 77-100%, XB 55-63%) | `step3d_control_summary.csv` | OK |
| Table 2 (k=1-allowed battery: original matrix, voiced 1,408, voiced MFCC; PS CIs) and the whole-span 15,992-file numbers | `step3c_old_matrix_summary.csv`, `step3c_all_*_summary.csv` | OK; "did not exceed" for the voiced dip statistic changed to "did not clearly exceed (p 0.16 / 0.06)" |
| Methods x representations (Table 3, times, stability range 0.46-0.85, BIC/ICL minimum at k = 6) | `step3b_methods_all2.csv`, `step3b_stability_all2.csv` | OK |
| HDBSCAN cluster composition (90-92% mass + 141-152-file micro-cluster; 7-10% noise; voiced-MFCC cluster 25% with 0.59 vs 0.14 s) | `step3b_hdbscan_sizes.csv` | OK (earlier "mostly noise" wording replaced) |
| Power analysis (detection rates by share and separation) | `step3h_power_all2_voiced_ref1_power.csv` | OK |
| Syllable level (85 ms, 32-207; dip 0.0012, p 0.18, bootstrap CI; contexts p 0.19-0.98; context decoding 0.066 / 0.198 / 0.203; shape FCM 0.500, silhouette 0.153, HDBSCAN 50% noise; scalar descriptors FPC 0.609; emitter 0.48 / 0.46 / 0.16; channel 0.575 vs 0.349; shape battery) | `step6_*`, `step3b_*_segall.csv`, `step3c_segall_shape_summary.csv` | OK; "BIC kept falling" corrected to "lowest at six components (not monotone)" |
| VAE (R2, silhouettes, HDBSCAN, stability, battery, context decoding) | `step6b_vae_reconstruction.csv`, `step3b_*_vae.csv`, `step3c_segall_vae8_b0.1_summary.csv`, `step3c_segall_pca8_summary.csv` | OK |
| Identity (random 0.593 / day 0.560 / channel 0.202; month 0.487; archive 0.504; whole-span 0.558 / 0.527 / 0.220), within-channel 0.648 vs 0.326, pair-aware (16,663 files) | `step4_emitter_id_*`, `step4c_*`, `step4b_pair_aware_all2_voiced_ref1.csv` | OK |
| Channel normalisation (balanced accuracies and permutation nulls, three sets) | `step4d_channel_norm_*.csv` | OK |
| Mixed models (LR chi2 1,643 / 315 / 1,485 / 366; ICC 0.075 / 0.132 / 0.102 / 0.209 with CIs; Fighting -0.45, Isolation +0.36) | `step4_mixed_context_lrt_all2_*`, `step4_repeatability_by_bat_all2_*`, `step4_mixed_context_effects_all2_*` | OK; effect sizes now labelled "among positive-coded bats" |
| Isolation: UMAP + HDBSCAN (voiced, whole-span, syllable shape, voiced MFCC), nulls, controls | `step3i_umap_*`, `step3i_umap_continuum_nulls.csv`, `step3j_umap_dissection_*` | OK |
| Isolation: supervised AUCs, pups only, within strata, leave-channel / treatment-out, per-pup, without unknown group, age windows, AVES | `step3k_isolation_*.csv` | OK |
| Replication (C1-C3; negative claims on unseen files) | `step7_replication_*.csv` | OK |
| Dataset B (FCM 0.629, AUCs, ARI, battery, piezo, event-length dip, decoding, detector threshold 10 dB / 87%; 20 dB F1 0.80) | `stepB_*`, `step3c_B_*` | OK; ARI range now stated for the 1,408-statistic features only (MFCC: -0.01 to 0.11 at k = 2) |
| AVES (two models x four variants; silhouettes, FPC, HDBSCAN, decoding, channel NMI, Isolation AUC) | `step3b_*_aves2bio.csv`, `step3b_*_aves2all.csv`, `step3k_isolation_aves.csv` | OK |
| Review of the June draft (robustness table; summary statistics; entropy > 0.95 in 33%; max membership < 0.6 in 25%; 12.8M intra-bat pairs; 6,412 identical-row pairs) | June files in `data/raw/june_outputs`, recomputed | OK (the 0.05% now says "of the intra-bat pairs") |

## Not independently verifiable here
* References: checked against bibliographic records found by web search, except the classical 1984-91 methods papers, librosa and scikit-learn (flagged in the paper). On 9 October 2026 the volume, pages and DOIs of Fischer 2017 (Anim Behav 134:229-237), Adolfsson 2019 (Pattern Recognit 88:13-26), Elie 2024 (Curr Biol 34(17):4062-4070.e7) and Hagiwara 2023 (ICASSP, pp. 1-5, doi 10.1109/ICASSP49357.2023.10095642) were confirmed against Crossref records and entered in the reference list.
* The statement that the earlier supervised study did not use blocked validation rests on a summary of its Methods; it is worded "as we read its Methods".
* The original extraction code of the draft is missing, so the cause of the label shift is unknown.

## Reproducibility re-runs (fresh processes, stored compact features, no network)
`27_isolation_separability.py`, `18_pair_aware.py`, `19_within_channel_id.py` and `06_confound_audit.py` were re-run on 2026-10-08 and every output table was compared with the stored version: identical (maximum absolute difference 0.0, text columns equal; bootstrap CIs and permutation nulls are seeded). The paper compiles with tectonic 0.17.0 (`paper/main.pdf`, 20 pages, no errors or overfull boxes).

## Version 3 additions (8 October 2026, afternoon)
| paper section / claim | source file(s) | result |
|---|---|---|
| Section 3.4: 15 algorithms / 26 variants / 1,558 configurations on 19 representations; admissibility counts (single linkage, OPTICS, DBSCAN p90 never; HDBSCAN, DBSCAN p50/p75, mean-shift, average linkage 4-7 of 19; others >= 11); median best silhouettes (0.13-0.18; spectral 0.25, 10 of 19 >= 0.25); median transfer ARI (spectral 0.67, k-means 0.58, FCM 0.53, others <= 0.51); Isolation recovery counts (best / any configuration) | `step3m_all_methods_*.csv`, `step3m_all_methods_summary.csv` (recomputed in the audit) | OK - three statements were corrected after checking (admissibility counts, ceiling of the other stabilities 0.47 -> 0.51, minority share of the spectral cluster 6% -> 7%) |
| Section 3.4: no balanced + well-separated + bat-stable partition on the four file-level Dataset-A representations; stable and balanced only for syllable scalar descriptors, Dataset-B microphone and one AVES HDBSCAN solution (62% noise); whole-span AVES split by channel (silhouette 0.41-0.42, ARI 0.86-0.95, NMI 0.15-0.17) | summary file, filter silhouette >= 0.25 and transfer ARI >= 0.6 and largest cluster <= 80% | OK |
| Discussion: k-means best-configuration silhouette 0.17 and NMI with silence 0.12 (file-level Dataset A); statement softened because the NMI of k-means / FCM (0.12) is only slightly above Ward (0.08) and spectral (0.10) | summary file | OK (claim reworded) |
| Syllable-level power (Supplement Table S2b; text in Section 3.3) | `step3h_power_segall_shape_power.csv` | OK |
| Sensitivity to PCs / scaling (Supplement Table S6) | `step3n_pca_sensitivity.csv` | OK |
| Cross-checks of the statistical helpers | ad hoc tests (see `logs/analysis_log.md`, 2026-10-08 15:40) | `auc_fast` equals scikit-learn `roc_auc_score` to 2e-16 over 200 tie-heavy random cases; bias-corrected Cramer's V equals an independent implementation of Bergsma's formula exactly (300 random tables) and plain V equals SciPy; FCM gives FPC 0.50 on unimodal Gaussian data and 0.97 on two separated blobs; `pyflakes` over all 47 scripts: no errors (one f-string without placeholder fixed) |
| Reproducibility re-runs | four scripts re-run from stored features | identical (maximum absolute difference 0.0) |

## Version 3 final additions (8 October 2026, evening): completion sample and abstract
| paper section / claim | source file(s) | result |
|---|---|---|
| Section 3.7 / Table S5: completion sample (33,557 files, 2,089 Isolation): AUC 0.991 [0.983, 0.995], d' 5.11, pups only 0.989; 12,000-file UMAP subsample: clusters 93.94% / 5.61%, noise 0.45%, small cluster 673 files, 26 emitter groups, 91.5% Isolation, recall 82.2%; min size 30: third cluster 54 files, 96% Isolation, 4 emitter groups | `step7_rest_summary.csv`, `step7_rest_clusters.csv` | OK |
| Section 3.7: copula nulls at min size 100 - replication 7-10 clusters, noise 68.5-72.5%; completion 7-8 clusters, noise 63.8-74.2%; at min size 30 five of six replication nulls give 2-3 clusters with < 1% noise (completion nulls 19-27 clusters, 69-77% noise) | `step7_replication_nulls.csv`, `step7_rest_nulls.csv` | OK after correction: the first draft of the sentence said "7-10 clusters, 64-76% noise" for both sets and "five of six nulls ... at min size 30" without saying that this applies to the replication sample only |
| Section 3.7: negative claims in the completion sample (FPC 0.5866; label vs silence 0.9035, voiced duration 0.9157, span 0.8419; voiced FPC 0.5000; PS 0.658 vs null mean 0.955 (min 0.945); Jaccard 0.883 vs 0.987; dip 0.0028 vs 0.0030, median p 0.9946) | `step7_rest_negative_claims.csv` | OK (null PS written as 0.955 / 0.945, not rounded to 0.96 / 0.95) |
| Abstract (250 words): all numbers traced to Sections 3.2-3.7 | main text | OK after correction: a first version quoted identity accuracy "0.65 against 0.33", a within-channel result that is not in the main text; replaced by day-blocked accuracy 0.56 (majority 0.145) and channel-blocked balanced accuracy 0.125 (shuffled 0.036) |
| Sample sizes: 41,116 discovery + 12,000 + 33,557 = 86,673 files; a further 1,313 annotated files were extracted only for the June-table reproduction; union of all extraction indices = 87,986 = all annotated files | `data/features/claude/*_index.csv` | OK after correction: the abstract said "the remaining 45,557 annotated files", which would have been 46,870; now "45,557 further files" and the Methods name the 1,313 |

## Supplement revision (25 October 2026)
| claim / item | source | result |
|---|---|---|
| Power statements in the main text (90-100% at 4 SD etc.), Fig. 2 and Tables S2a/S2b | `step3h_power_*_R200_*.csv`, `paper/supplementary_data/SD3` | updated to the 200-replicate run; first 20 replicates reproduce the earlier run exactly |
| Bat counts 41 / 40 / 36 / 33 / 18 (Table S0b) | `annotations_merged.csv.gz`, `step3b_decode_*.csv`, `step7_sample_accounting.csv` | counted from the data; the "36 bats" label of the emitter-identification figure was a stale label (the plotted analysis has 40) and was corrected |
| Age-window and channel intervals (Table S4b) | `step3k_isolation_strata_ci.csv` | AUCs equal the stored values exactly; intervals are new (bat-level bootstrap, 1,000 resamples) |
| Sample accounting and overlap (Table S5b, Fig. S6a) | `step7_sample_accounting.csv`, `step7_sample_overlap.csv` | 41,116 + 12,000 + 33,557 + 1,313 = 87,986 files; all 33 held-out bats and 12 channels occur in discovery |
| Duration BIC comparison (Table S9b) | `step6_duration_bic_ci.csv` | BIC(1) - BIC(2) = 2,077 [764, 4,237] (bat-level bootstrap) |
| Null positions (Table S1c) | `step3c_*_summary.csv` | counts of null replicates >= observed as stored |
