# Fruit-bat call clustering audit - are Egyptian fruit bat social calls discrete or graded once recording confounds are controlled?
n[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23662435.svg)](https://doi.org/10.5281/zenodo.23662435)

A confound-controlled re-audit of the July 2026 draft "Robust Two-Cluster Structure in Egyptian Fruit Bat Social Vocalizations"
by the same author. Everything here is new work; the earlier draft's files were only read. **The answer is in `results/MEMO.md`** (2 pages): graded, with one separable exception (pups' Isolation calls), not two discrete regimes. Paper draft: `paper/main.tex` (compile with `paper/compile.ps1` or any LaTeX engine; PDFs are included). Review of the June draft: `results/REVIEW_of_june_draft.md`.
Every analysis choice and every number that differs from the draft is in `logs/analysis_log.md`.

## Data
| | source | what we used |
|---|---|---|
| Dataset A | Prat et al. 2017, Figshare collection 10.6084/m9.figshare.c.3666502 (293,238 mono 16-bit 250 kHz WAVs in 30 zips) | `Annotations.csv`, `FileInfo.csv`, `Metadata.pdf` (downloaded, md5-checked); audio of 41,116 files **read by HTTP range request from the remote zips - no audio is stored** |
| Dataset B | Elie et al. 2024, Figshare 10.6084/m9.figshare.25886023 (10 bats, 5 deaf; each vocal event on an ambient mic AND a throat piezo) | `VocDataBase.zip` (1.81 GB, md5-checked), features extracted, **zip deleted afterwards** (no audio kept; re-fetch with `02b_datasetB.py extract`) |
| AVES | Earth Species Project public weights `aves-base-bio.torchaudio.pt` and `aves-base-all.torchaudio.pt` (loaded with `weights_only=True`) | 1,000-file subset (exploratory; plain 16 kHz downsample and 2x time-expanded variants, whole span and voiced); `aves-base-all` run: see `logs/queue_A.log` and `logs/analysis_log.md` for status |

Column meanings were checked in `Metadata.pdf` before use: emitter 0 = unknown; negative emitter = either of an interacting pair (role in doubt);
the annotation `Start/End sample` is ~the whole file (Start = 1 in 94% of rows) - the **voiced segments are the `FileInfo.csv` voice start/end pairs**.

## Environment
Python 3.13 virtual environment in `env/` (everything on D:; C: is nearly full). Versions: `logs/package_versions.txt` and the full `logs/pip_freeze.txt`. To keep C: untouched when installing packages, run `set PIP_CACHE_DIR=D:\AI\Claude\research\bat_audit\tools\pip_cache` (or use `pip install --no-cache-dir`) first; files that were once written to C: by this project were moved to `tools\moved_from_C\` and deleted from C: (pip cache entries, fontconfig cache). Global seed 42 (`code/common.py`).
Setup: `python -m venv env` then `env\Scripts\python -m pip install numpy pandas scipy scikit-learn statsmodels matplotlib pypdf requests tqdm librosa soundfile diptest torch torchaudio umap-learn`.

## Run order (from this folder; `env\Scripts\python.exe` = the venv python). Times are for an 8-core Windows laptop.
| step | command | what | time |
|---|---|---|---|
| 1 | `code\01_metadata_audit.py` ; `code\01b_datasetB_listing.py` | metadata audit of both datasets (needs the downloaded CSVs in `data/raw/prat2017_meta/`) | 2.5 min |
| 2a | `code\claude\02a_sample_design.py 15000` | stratified sample (channel x month, >= 100 per bat, >= 150 per context) | seconds |
| 2 | `code\claude\02_extract_features.py <manifest.csv> <prefix> all` | voiced + naive 1,408 features, MFCC(120), silence / SNR / noise floor / clipping from remote audio | 6 files/s |
| 2c | `code\claude\07_baseline_reproduction.py baseline` (FileID reading: fails) / `... hrowfull` (row-index reading: **reproduces the draft exactly**, \|r\|=1.000, FPC 0.5786) | the draft's integers are row indices of Annotations.csv; extract with manifest `hrow_manifest.csv`, merge retries with `merge_extractions.py hrowfull hrow hrow_retry` | 7 min |
| 2e | `code\claude\02a_sample_design.py 40000 extend`, then `02_extract_features.py data\features\claude\main2_manifest.csv main2 ref1 seg` and `... sample_all_manifest.csv segold ref1 segonly`, `merge_extractions.py segall segold main2`, `05_build_sets.py baseline+main+main2 ...` | extend to 41,116 files and add per-segment (syllable) features | ~2.5 h (network-bound) |
| 3f | `code\claude\14_draft_true_labels.py` ; `15_june_check.py` | the draft's associations with the labels that belong to its audio; audit of the June table | 1 min |
| 6 | `code\claude\16_build_segment_sets.py segall segall 30000` ; `17_segment_duration_modes.py segall segall` ; then 06/08/03c/12 on tags `segall_shape` etc. | syllable-level analysis (duration modality, shape clusterability, context decoding) | ~2 h |
| 4b/c | `code\claude\18_pair_aware.py <set>` ; `19_within_channel_id.py <set>` | pair-coded (negative) calls; identification within one channel | 2 min |
| queue | `powershell -File code\run_queue_night.ps1 A` / `B` | AVES x2 models and the heavy 03c batteries, unattended | hours |
| paper | `paper\main.tex` (+ `paper\sections`, `paper\tables`, `paper\figures`), compiled to `paper\main.pdf` (16 pages) and `paper\supplement.tex` / `supplement.pdf` (9 pages) ; `results\WHATS_NEW_v3.md` ; `results\REVIEW_of_june_draft.md` ; `results\VERIFICATION_of_paper_numbers.md` | IMRaD LaTeX draft (compile: `powershell -File paper\compile.ps1` - uses `tools\tectonic`, keeps its cache in `tools\tectonic_cache` and removes the 2 MB fontconfig cache that tectonic always writes to C:); expert review of the June analysis; claim-by-claim check of the paper's numbers against the result files | - |
| 2d | `code\claude\05_build_sets.py baseline+main ...` | standardise + PCA(50) analysis sets, identical files in every set | 2 min |
| 3a | `code\claude\06_confound_audit.py <set>` ; `03a_old_label_confounds.py` | does the cluster label follow silence / SNR / channel / date / context? | minutes |
| 3b | `code\claude\08_methods_and_representations.py <name> <sets...>` | FCM, k-means, GMM, HDBSCAN x representations: counts, time, held-out-bat stability, decodability | 3-10 min / set |
| 3c | `code\claude\03c_clusterability_old_matrix.py <B_null> <B_boot> <set>` | k=1-allowed tests resampled by bat: gap, prediction strength, bootstrap Jaccard, split-sample dip; Gaussian-copula and session-mean nulls | ~1 h / set |
| 3d | `code\claude\03d_simulated_control.py` | single-population controls with the draft's PCA spectrum | 6 min |
| 4 | `code\claude\04_stats_old_matrix.py <set>` | mixed models (bat, channel x month random effects), repeatability by bat, random vs blocked emitter ID | 10-60 min |
| 5 | `code\claude\11_aves_embeddings.py 1000 [aves-base-bio|aves-base-all]` then `05_build_sets.py sub:aves2bio` and `05_build_sets.py aves2bio`, then `08_methods_and_representations.py aves2bio <sets>` (prefix `aves2all` for the second model) | AVES embeddings on 1,000 files (4 variants; 16 kHz downsample and 2x time expansion) compared with handcrafted / MFCC on the same files | ~1-3 h per model (CPU) + 2 min |
| 3b+ | `code\claude\20_extra_methods.py <sets>` ; `code\claude\21_hdbscan_sizes.py <sets>` | Ward / spectral / BIRCH / DBSCAN on 6,000-point subsamples; sizes and covariates of the HDBSCAN clusters | ~5 min ; ~35 min |
| 3h | `code\claude\22_power_analysis.py all2_voiced_ref1 6000 20 40` | **power of the k=1 battery**: two-type mixtures (minority share 5/20/50%, separation 1-8 SD) spiked into a session-mean null; detection rate per statistic | ~25 min |
| 3i | `code\claude\23_umap_hdbscan.py <set> 12000 12 8` | the standard UMAP -> HDBSCAN pipeline with copula / session nulls, bat bootstraps and covariate association | ~30 min / set |
| 6b | `code\claude\24_vae_latent.py 60` then `08_methods_and_representations.py vae segall_vae8_b0.1 segall_vae8_b1.0 segall_pca8` and `03c ... <tag>` | VAE latent (trained on half the bats) vs 8-PC baseline on held-out bats | ~15 min |
| 4d | `code\claude\25_channel_normalisation.py <sets>` | does per-channel / per-session mean subtraction rescue cross-channel identification? (with a within-stratum permutation null) | ~20 min / set |
| 3j | `code\claude\26_umap_cluster_dissection.py <set> 12000 100` ; `34_umap_continuum_null.py` | what the small UMAP cluster is (context, pups, covariates, held-out-bat separation) and harder nulls (curved continuum, real minus the cluster) | ~5 min ; ~20 min |
| 3k | `code\claude\27_isolation_separability.py <sets>` ; `29_isolation_per_bat.py` ; `30_isolation_channel_blocked.py` ; `31_isolation_aves.py` ; `28_make_isolation_figure.py` | **Isolation group**: leave-bats-out / pups-only / within-stratum / leave-channel-out AUC, per-pup AUC, AVES comparison, Figure 7 | ~15 min |
| 7 | `code\claude\32_replication_sample.py 12000` ; `02_extract_features.py data\features\claude\repl_manifest.csv repl ref1 seg` ; `33_replication_analysis.py` | **out-of-sample replication** of the Isolation finding on 12,000 files never used before (criteria fixed in `logs/analysis_log.md` beforehand) | ~40 min + 10 min |
| 7+ | `code\claude\32_replication_sample.py 100000 rest` ; `02_extract_features.py data\features\claude\rest_manifest.csv rest ref1 seg` ; `33_replication_analysis.py rest` ; `35_replication_negative_claims.py rest` | **completion sample**: every annotated file not extracted before (33,557 files, 2,089 Isolation) - same pre-specified criteria; with the replication sample this covers all 87,986 annotated files | ~90 min extraction + 15 min |
| 3m | `code\claude\38_all_methods.py <representations>` then `38_all_methods.py summary` | **fifteen clustering algorithms (26 variants) x 19 representations**: silhouette, clusters, bat-transfer stability, Isolation-group recovery; heatmap | ~1 h with 3 workers |
| 3n | `code\claude\41_pca_sensitivity.py` | sensitivity to the number of PCs and to scaling | ~8 min |
| 3g+ | `code\claude\37_june_specialists.py` | the June table's "specialist" bats (treatment, span, silence; Spearman) | seconds |
| 9 | `code\claude\42_make_supplement_tables.py` | writes `paper\tables\*.tex` directly from the result files | seconds |
| 3k+ | `code\claude\35_replication_negative_claims.py` ; `36_isolation_age_matched.py <sets>` | negative results on the unseen files; Isolation within pup age windows (age and Isolation are confounded below 40 days) | ~10 min |
| 3e | `code\claude\12_pup_adult_check.py all_mfcc_voiced` | the Isolation-linked cluster: pups vs adults, age, strata permutation (forced k-means split; superseded by 3j/3k for the conclusion) | 1 min |
| fig | `code\claude\13_make_figures.py` | the four memo figures | seconds |
| B | `code\claude\02b_datasetB.py validate|thrtest|extract` ; `10_datasetB_analysis.py` | Dataset B pipeline + mic-vs-piezo agreement | 10 min |


Inputs outside this folder: only `OLD` in `code/common.py` (the draft's saved arrays under `D:\Downloadsalidation_suite_results...`, read-only); the June outputs were copied to `data/raw/june_outputs`.

## Outputs
`results/*.csv|json` (tables), `figures/*.png`, `logs/analysis_log.md` (decisions + changed numbers), `data/features/claude/` (compact features; no audio).

## Reporting conventions
Effect sizes with 95% intervals; bats (|emitter|) - never calls or files - are resampled for every stability / generalisation statement; permutation results are reported
as `p<1/(B+1) (0/B null replicates)`, never p=0; clusters are never called "call types" or "regimes" here (the Isolation group is described as "a separable group of calls that survives bat-, session- and channel-controlled checks").
Negative results are read together with `results/step3h_power_*` (what the battery could detect); candidate groups are confirmed with supervised tests blocked by bat, channel and session.

## Known limits (details in the memo)
* The draft's matrix IS reproduced from audio once its file numbers are read as Annotations.csv row indices (not FileIDs); its emitter/context labels were joined by FileID and belong to other files. The original extraction code is still missing, so the cause cannot be confirmed. Outputs computed under the wrong (FileID) reading are in `results/superseded_fileid_reading/`.
* Only 41 bats have positive emitter codes; 48% of annotations carry negative (pair) codes and 9% are unknown, so identity and context analyses use a minority of the corpus.
* Dataset B has no context labels; its sensors use different sample rates (192 kHz vs 50 kHz), and its voiced-frame detector threshold (10 dB) is a judgement call.
* AVES was trained for audible-range audio; at 16 kHz it sees only the < 8 kHz part of bat calls.
| 9+ | `code\claude#_figures_v4.py` ; `44_figures_v4_umap.py` ; `45_fig5_data.py` | version-4 figures at print size (7-8 pt text, vector PDF); 43 reads saved csv only, 44/45 recompute the seeded UMAP and out-of-fold scores after checking them against the stored values | ~3 min |
| 3h+ | `code\claude_power_analysis.py <tag> 6000 200 200 _R200` | power analysis with 200 replicates per cell and 200 null datasets (outputs suffixed `_R200`; the first 20-replicate run is kept) | ~1.5 h / set |
| 3k++ | `code\claude&_isolation_strata_ci.py` | sample sizes and bat-level bootstrap intervals for the age-window and held-out channel / treatment AUCs (reproduces the stored AUCs exactly) | ~10 min |
| 6b+ | `code\claude'_duration_bic_ci.py` | one- versus two-Gaussian BIC difference for syllable duration with a bat-level bootstrap | ~10 min |
| 7a+ | `code\claude8_sample_accounting.py` ; `49_flow_diagram.py` | exact partition accounting and overlap; the data-partition flow diagram | seconds |
| export | `code\claude(_export_supplementary_data.py` | writes `paper\supplementary_data\SD1-SD7` (machine-readable results) | seconds |
