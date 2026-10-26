# What is new compared with the previous drafts (version 3, 9 October 2026)

**Question:** are Egyptian fruit bat social calls discrete or graded once recording confounds are controlled?
**Short answer:** graded, with one separable exception (pups' Isolation calls). The earlier two-regime result came from silence and recording length.

Files: `paper/main.pdf` (16 pages), `paper/supplement.pdf` (9 pages), `results/MEMO.md` (findings on two pages), `results/VERIFICATION_of_paper_numbers.md` (every number traced to a result file), `logs/analysis_log.md` (every choice and every changed number).

## 1. The answer changed

| | June / July drafts | Version 3 |
|---|---|---|
| Structure of the calls | two reproducible acoustic regimes (FCM, c = 2, FPC 0.579) | graded; no balanced two-cluster partition survives on voiced frames or syllables |
| What the two clusters are | call types / regimes | a split between short, silent recordings and longer, voiced ones (label vs silence fraction AUC 0.89; recording variables alone predict it with AUC 0.97 on unseen bats) |
| Emitter and context associations (V = 0.145, 0.233) | evidence for structure | computed between cluster labels of one set of recordings and metadata of **other** recordings (labels agree with the audio's true annotation in 0.2% / 22.6% of rows); with the true labels V = 0.31 (context) and 0.44 (emitter), a modest excess over a channel-by-month null |
| Individual "acoustic modes" | present | emitter identity is decodable inside a microphone and on unseen days (accuracy 0.56, majority class 0.145), but across microphones only weakly (balanced accuracy 0.125 against 0.036 for shuffled labels); channel normalisation does not help |
| Anything discrete? | no | **one separable group**: pups' Isolation-context calls (6-7% of files), found blind by UMAP + HDBSCAN, separable on unseen bats (AUC 0.984), on held-out channels (0.983) and treatments (0.986), and in 12,000 and 33,557 files never used before (AUC 0.988 and 0.991, criteria written down beforehand) |

The original matrix was reproduced exactly (|r| = 1.000, FPC 0.5786, sizes 1,860 / 2,037, ARI = 1.000) once the integers in the file names were read as row indices of `Annotations.csv`; that is also how the label mismatch was found.

## 2. New analyses

* **Every annotated file processed**: 41,116 discovery files + 12,000 random never-used files + all 33,557 remaining ones = 86,673, plus 1,313 files extracted for the June reproduction = all 87,986 annotated files (205,252 unannotated recordings were not used).
* **Every clustering algorithm on every representation**: 15 algorithms (26 variants: k-means, fuzzy c-means, PAM, four hierarchical linkages, spectral, BIRCH, four GMMs, Dirichlet-process GMM, DBSCAN, OPTICS, HDBSCAN, mean-shift, affinity propagation, Louvain, Leiden, SOM) on 19 representations (1,558 configurations, 0 errors). No algorithm gave a balanced, well-separated and bat-stable partition on the file-level features; several (spectral, Leiden, Louvain, SOM, Ward, BIRCH, tied GMM) recover the Isolation group, fuzzy c-means never does.
* **Tests that allow one cluster, with power**: prediction strength, bootstrap Jaccard, split-sample dip (bats resampled; copula and session-mean nulls); spiking two-type mixtures into the null shows what they can detect (two equal types 4 SD apart; a 5% minority only at 8 SD), also at syllable level. A negative result is therefore stated as "excludes well-separated common types", not "no structure".
* **UMAP + HDBSCAN with controls**: copula, session-mean, curved-continuum and skewed nulls and "real minus the cluster". The earlier claim that real data differ from nulls by cluster count was **withdrawn**: the real data without the small cluster also fragment.
* **Isolation robustness**: bat-blocked supervised AUC, pups only, per pup, per channel / treatment, within pup-age windows (from day 40), without the unknown-emitter group, syllable level, AVES.
* **Syllables** (141,815), VAE latent (trained on 21 bats, tested on 21 unseen) and AVES embeddings: no added structure.
* **Sensitivity** to the number of PCs and scaling: the FCM collapse holds for 20-100 standardised PCs; with 5-10 PCs or unscaled data FCM instead splits on silence/duration.
* **Dataset B** (Elie et al. 2024): its two-way split follows event length and loudness; event length itself is multimodal.
* **Review of the June table**: 16,000 "balanced" calls are 9,588 distinct rows; its two "specialist" bats share one treatment and long files.

## 3. Manuscript format

The paper was written again from scratch in the standard article format used by current journals (Abstract, Introduction, Methods, Results, Discussion, References), not the older layout of the earlier drafts. The main text carries the argument; the detailed tables are in a separate supplement and are produced by script from the result files. The exact layout (for example a structured or a plain abstract) depends on the target journal and can be changed once the journal is chosen.

## 4. Errors found and fixed (research-related)

* The draft's file numbers were annotation-row indices, not file identifiers; this is the root cause of the label mismatch.
* In the all-algorithm comparison, single linkage "won" on silhouette by making singleton clusters; a partition now needs at least two clusters holding at least 1% of the points.
* "HDBSCAN returns mostly noise" was wrong (it returns one mass plus micro-clusters), and the UMAP-versus-null contrast was withdrawn after harder controls.
* Checking every number in the paper against the result files corrected eight wording or number errors in the first audit (for example the AVES silhouette range, plain vs bias-corrected Cramer's V, context effects limited to bats with positive codes) and five more in the final pass (null-cluster ranges, an identity figure in the abstract, a sample-size count).
* The claim "the answer depends on the algorithm" was rewritten from the data: it holds for the Isolation group, not for a balanced two-regime split.

## 5. Still open (research points only)

| Open point | Why it is not done | Can it be done? |
|---|---|---|
| Is the Isolation group a **call type**? | Isolation calls are recorded when a pup is alone, and pups under 40 days give almost only Isolation files, so call type, recording situation and age cannot be separated in this corpus. The checks (other bats, channels, treatments, sessions, age windows from day 40) show the group is real in the data, not what it is. | Not from these data. It needs new recordings (the same pups at the same age in different situations) or playback with live animals. Stop at "separable group". |
| Does it hold in **other colonies**? | A web search on 9 October 2026 found no other public annotated Egyptian fruit bat corpus. The only other one used here, Dataset B (Elie et al. 2024, a different lab, 10 adults), has no context labels, so it cannot test the Isolation finding. | Only if a group is found that will share annotated recordings. Otherwise stop. |
| The group was found **post hoc** | It appeared in an exploratory UMAP run, which cannot be undone. It is supported by bat-blocked supervised tests and by two replications whose criteria were written to the log before the files were analysed (12,000 and 33,557 never-used files, both passed). Those files come from the same corpus. | Cannot be removed; only another colony (previous row) would strengthen it. |
| Is identity in the **voice or in the position** of the bat? | A bat's files come from a median of 2 of the 12 channels (median 75% on its main one), so voice and roost position are confounded; channel normalisation did not remove the problem. | Needs the same bats recorded on several microphones: new recordings. Stop. |
| Is the multimodal event length in **Dataset B** real or from the authors' detector? | The public files are events already cut by the authors' detector, so the cutting rules cannot be tested without the continuous recordings or the detector settings. | Needs the Dataset B authors. The paper states it as unresolved. |
| How did the June draft's labels end up on other recordings? | The original extraction code is not available: the June archive (14 files) holds only tables and figures, and nothing bat-related is on this computer. We know what happened (row index read as file ID), not how the script did it. | Only the authors, if they find the original script. It does not affect the conclusions. |

Not listed here on purpose (they stay in the manuscript and will be completed before submission): author names and affiliations, declarations, repository DOI, the statement that the manuscript has not been peer reviewed, and journal-specific formatting.
