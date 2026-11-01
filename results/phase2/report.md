# Phase 2 implementation and evaluation report

Generated: 2026-10-28 (Asia/Kolkata)  
Status: **PASSED**

Phase 2 completed the spatial-feature comparison and froze the KoNViD research configuration for subsequent cross-dataset work. It reused the definitive DINO results, added the ResNet-50 control under the same frozen training protocol, and generated paired statistical comparisons and reproducibility artifacts.

**Selected research representation:** ResNet-50, chosen using inner-validation scores only. **Highest headline score on the chosen representation:** Ridge (Baseline 0) with mean per-fold SRCC 0.7801. This ranking does not establish a practically meaningful advantage over other heads. Deployment remains undecided until CPU eligibility is measured; no LIVE-VQC prediction is reported here.

## Scope and completed work

| Work item | Verified completion |
| --- | --- |
| Existing DINO neural CV fits reused | 100 |
| Existing deterministic DINO ridge folds reused | 5 |
| New ResNet neural CV fits | 100 |
| New deterministic ResNet ridge folds | 5 |
| New ResNet shuffled-control seed fits | 10 |
| Authoritative KoNViD videos | 1200 |
| ResNet out-of-fold prediction rows | 25200 |
| Reported feature/head configurations | 10 |
| Comparison pairs / metric results | 15 / 30 |

Each neural comparison uses the fixed 5 model seeds across 5 outer folds. Ridge is deterministic and runs once per fold. No additional tuning, feature extraction, motion combination, intermediate-layer search, YouTube-UGC evaluation or cloud execution was introduced. The existing `cache/v2` spatial cache, frozen split indices and Phase 1 selected hyperparameters were preserved.

## Implementation

| Component | Behavior |
| --- | --- |
| [`src/progress.py`](../../src/progress.py), [`src/training.py`](../../src/training.py) | Optional callback prints stage, run, epoch/batch, loss, validation SRCC, memory and completion counters, and writes atomic progress snapshots. Numerical-invariance tests cover identical weights, history and RNG state with reporting enabled/disabled. |
| [`src/phase2.py`](../../src/phase2.py), [`configs/phase2.yaml`](../../configs/phase2.yaml) | Verifies Phase 1 gates, source/config/checkpoint/prediction identities and authoritative OOF coverage. Derives ResNet configs by changing only representation and output root; persists immutable effective configs and an executed-source audit. |
| [`scripts/05_phase2.py`](../../scripts/05_phase2.py), [`scripts/05_train.py`](../../scripts/05_train.py) | Runs ResNet-specific default/selected controls before definitive CV. Reuses complete compatible fits; validates predictions before counting completion. Test-role prediction follows best-checkpoint restoration. |
| Validation-only selector in `src/phase2.py` | Consumes exactly one best inner-validation SRCC per neural head/fold/seed. Rejects duplicates, missing/nonfinite records and test-derived fields; excludes ridge from selection weighting. |
| [`src/metrics.py`](../../src/metrics.py), [`scripts/07_bootstrap.py`](../../scripts/07_bootstrap.py) | Aligns videos by authoritative ID, averages neural predictions across seeds, and shares one resample-index bank across all SRCC/KRCC pairs. Applies ordered CI classification and the primary Holm correction. |
| [`scripts/06_eval.py`](../../scripts/06_eval.py), [`scripts/13_make_tables.py`](../../scripts/13_make_tables.py) | Checks single-run artifacts and OOF coverage, generates separate Phase 2 summaries/tables, and publishes a source-bound KoNViD freeze with the passed gate last. Phase 1 summaries are not overwritten. |
| [`scripts/continue_phase2.sh`](../../scripts/continue_phase2.sh) | Requires tmux; sequences check → initial control → selected control → CV/ridge → selection → bootstrap → tables. Preserves failures with a failed gate and session log. |
| [`scripts/continue_phase1.sh`](../../scripts/continue_phase1.sh) | Removed automatic Git staging/commit operations; the Phase 2 runner also performs no Git mutations. |

The neural objective remains L1 on training-standardized MOS plus the frozen shared pairwise-rank term. AdamW, learning-rate scheduling, batch size, architecture tiers and per-head selected hyperparameters are inherited without retuning on ResNet. Ridge retains training-only feature/target standardization and inner-validation alpha selection. Checksum/config validation is retained on resumed runs; completed matches are reused rather than fitted again.

The user launched the ML queue in a separate `train` tmux session. Terminal progress is diagnostic; the artifact validators and generated phase gate determine research completion. Phone attachment was user-managed and was not independently observed.

## Feature selection and headline results

| Representation | Mean inner-validation SRCC | Selected |
| --- | --- | --- |
| dino | 0.7158 | False |
| resnet50 | 0.7636 | True |

Feature choice equally weights all 100 neural head/fold/seed validation scores per representation. The predeclared exact-tie preference is dino. Held-out test scores and ridge validation scores do not enter this calculation.

The headline values below are **mean ± sample SD of per-fold metrics**: neural rows summarize 25 fold × seed fits; ridge rows summarize 5 deterministic folds. They are not pooled-ensemble metrics or bootstrap intervals.

### Fixed DINO representation

| Head | SRCC mean ± SD | KRCC mean ± SD | Raw PLCC mean ± SD | Mapped PLCC mean ± SD |
| --- | --- | --- | --- | --- |
| Ridge (Baseline 0) | 0.7350 ± 0.0217 | 0.5471 ± 0.0181 | 0.7667 ± 0.0148 | 0.7697 ± 0.0140 |
| Mean-pool MLP | 0.6852 ± 0.0258 | 0.5024 ± 0.0218 | 0.7245 ± 0.0210 | 0.7320 ± 0.0195 |
| Bi-GRU | 0.6892 ± 0.0223 | 0.5057 ± 0.0174 | 0.7207 ± 0.0169 | 0.7254 ± 0.0144 |
| Transformer | 0.6930 ± 0.0229 | 0.5092 ± 0.0183 | 0.7298 ± 0.0181 | 0.7346 ± 0.0161 |
| Mamba (bidirectional) | 0.6895 ± 0.0329 | 0.5063 ± 0.0274 | 0.7287 ± 0.0267 | 0.7340 ± 0.0242 |

### ResNet-50 control (selected)

| Head | SRCC mean ± SD | KRCC mean ± SD | Raw PLCC mean ± SD | Mapped PLCC mean ± SD |
| --- | --- | --- | --- | --- |
| Ridge (Baseline 0) | 0.7801 ± 0.0405 | 0.5892 ± 0.0380 | 0.7863 ± 0.0370 | 0.7956 ± 0.0306 |
| Mean-pool MLP | 0.7487 ± 0.0286 | 0.5579 ± 0.0278 | 0.7614 ± 0.0267 | 0.7689 ± 0.0258 |
| Bi-GRU | 0.7526 ± 0.0277 | 0.5634 ± 0.0263 | 0.7659 ± 0.0220 | 0.7700 ± 0.0226 |
| Transformer | 0.7599 ± 0.0277 | 0.5699 ± 0.0256 | 0.7728 ± 0.0239 | 0.7775 ± 0.0219 |
| Mamba (bidirectional) | 0.7542 ± 0.0283 | 0.5643 ± 0.0268 | 0.7661 ± 0.0266 | 0.7727 ± 0.0232 |

Mapped PLCC uses a four-parameter logistic fit against each evaluated fold’s target labels, then averages the per-run mapped correlations. It is a target-label mapping, not a label-free transfer calibration. SRCC and KRCC use raw predictions; PLCC is not bootstrapped. Seed and fold SRCC variability are reported separately in [headline.csv](headline.csv) and the per-representation summaries.

## Paired statistical findings

All 10,000 seed-0 resamples use the same canonical video order and shared index bank for both metrics. Neural OOF predictions are averaged across seeds before resampling, and SRCC ranks are recomputed inside each resample. The primary KoNViD family contains the 6 chosen-feature neural SRCC pairs. KRCC, ridge comparisons and feature comparisons are secondary and unadjusted.

The approximate two-sided p-value uses the centered-bootstrap/add-one convention: `(1 + count(abs(bootstrap_delta - observed_delta) >= abs(observed_delta))) / (valid_draws + 1)`. It is not an exact permutation test. At least 99% of requested draws must be finite; undefined draws are not replaced.

Pointwise confidence intervals and Holm-adjusted p-values are separate quantities: CIs are not familywise-adjusted. The predeclared SRCC equivalence margin is ±0.01.

Classification is applied in order: a CI wholly inside ±0.01 establishes equivalence; a CI wholly beyond either margin establishes a practically meaningful advantage; otherwise a CI excluding zero is detectable but not shown to exceed the margin; otherwise the comparison is inconclusive. Endpoints are classified at full precision before display rounding.

| Primary pair (A − B) | Pooled SRCC difference | Pointwise 95% CI | Holm p | Holm rejection | Ordered CI interpretation |
| --- | --- | --- | --- | --- | --- |
| Mean-pool MLP − Bi-GRU | -0.0046 | [-0.0145, 0.0052] | 0.7249 | False | no detectable difference (inconclusive at this sample size) |
| Mean-pool MLP − Transformer | -0.0098 | [-0.0181, -0.0016] | 0.1020 | False | detectable difference, not shown to exceed the margin |
| Mean-pool MLP − Mamba (bidirectional) | 0.0021 | [-0.0058, 0.0100] | 0.7249 | False | no detectable difference (inconclusive at this sample size) |
| Bi-GRU − Transformer | -0.0052 | [-0.0135, 0.0028] | 0.6329 | False | no detectable difference (inconclusive at this sample size) |
| Bi-GRU − Mamba (bidirectional) | 0.0066 | [-0.0024, 0.0155] | 0.5687 | False | no detectable difference (inconclusive at this sample size) |
| Transformer − Mamba (bidirectional) | 0.0119 | [0.0039, 0.0201] | 0.0240 | True | detectable difference, not shown to exceed the margin |

No primary neural pair established a practically meaningful advantage beyond the declared margin (0 qualifying pairs). The ordered CI rule classified 4 pairs as inconclusive and 2 as detectable but not shown to exceed the margin. No primary pair established equivalence within the margin. 1 pair survived Holm adjustment: Transformer versus Mamba (bidirectional).

The Transformer is higher than Mamba in the pooled seed-averaged comparison, detectable but not shown to exceed the practical margin. Mamba does not show a demonstrated advantage in this experiment. These conclusions concern the cached-feature protocol, not all possible training regimes.

The secondary feature comparisons classify ResNet as practically better than the fixed DINO representation for all 5 evaluated heads/baselines; these comparisons remain unadjusted. Ridge is the highest headline row, but its pooled differences from Bi-GRU and Transformer are inconclusive. Its differences from mean pooling and Mamba are detectable but not shown to exceed the margin. The [ridge comparison table](head_comparison.csv) retain these distinctions; [feature_pairs.csv](feature_pairs.csv) contains the separate representation comparisons.

Headline differences and pooled-ensemble differences can differ in magnitude or ordering because they summarize different quantities. A higher headline mean alone is not a statistically demonstrated advantage.

## Leakage controls and verification

| ResNet control | Ensemble test-fold SRCC | 95% bootstrap CI | Status |
| --- | --- | --- | --- |
| initial | 0.0025 | [-0.1314, 0.1315] | passed |
| final | -0.0061 | [-0.1380, 0.1199] | passed |

Both controls independently permute training and validation targets within their original roles, using the fixed seed separately for each role, while preserving true held-out test labels and IDs. Each passed the unchanged conjunction: the SRCC interval includes zero and absolute SRCC is below the declared threshold. Saved permutations, policies, all seed artifacts and checkpoint/config bindings were validated.

The historical Phase 1 training-only final control remains failed: SRCC 0.1292, CI [0.0043, 0.2552]. The user-approved validation-permutation amendment was made after that failure, and its success does not erase the original outcome. Preserved evidence and this post-hoc disclosure remain in [summary.json](summary.json) and [DECISIONS.md](../../DECISIONS.md).

Verification completed with `228 passed in 20.60s` from `.venv/bin/python -m pytest -q`, plus shell syntax and actual Phase 1 prerequisite checks. Tests cover progress/RNG invariance, control gates, immutable reuse, selection isolation, bootstrap pairing/reranking, Holm/classification boundaries, reporting failure behavior and tmux runner ordering.

After execution, all 228 gate-listed published file hashes were verified. All 1811 files in the pretraining source/Phase 1 snapshot remained unchanged. Control artifacts, all CV/checkpoint/prediction identities, the complete OOF grid and comparison provenance were checked. The selection and freeze were independently reconstructed from saved inputs; all headline SRCC means and sample SDs were independently reproduced from per-video predictions. No training or bootstrap was repeated for this verification.

## Compute accounting and full-data refit handoff

| Saved job-wall accounting | Hours |
| --- | --- |
| Phase 2 GPU-job wall time | 0.4341 |
| Cumulative Phase 1 + Phase 2 GPU-job wall time | 11.3766 |
| Upper-bound remaining GPU planning allowance | 88.6234 |

Cumulative saved run/attempt wall time, not kernel utilization. Untimed interruptions and hands-on time unmeasured; remaining allowance is an upper bound. CPU statistics/report overhead is not measured.

| Chosen-feature neural head | Frozen median CV best epoch |
| --- | --- |
| Mean-pool MLP | 3 |
| Bi-GRU | 8 |
| Transformer | 8 |
| Mamba (bidirectional) | 5 |

The chosen-feature ridge full-data alpha is 1000, resolved by the saved `median_cv_selected_alpha` policy. Full source epoch and alpha records are retained for both representations; Phase 3 must not create a new validation split or alpha search. Full-KoNViD training has more steps per epoch than CV; median epochs are an approximation.

The deployment policy considers every reported representation/head candidate, including ridge, and selects the highest headline KoNViD CV SRCC within the measured CPU budget of 120 seconds per ten-second clip. Eligibility and the winner remain **unresolved**. Research feature selection is frozen, but CPU eligibility must be resolved using KoNViD/CPU evidence before LIVE-VQC predictions. Phase 2 completion does not certify deployment readiness.

## Limitations and known implementation issue

- Inner-validation selection/tuning uses videos that are test videos in other outer folds. This residual overlap limits an unbiased interpretation of the comparison.
- Bootstrap CIs describe test-video sampling noise for the seed-averaged ensemble; they exclude model-training and fold uncertainty. Seed/fold SDs are separate. The use of 95% intervals for equivalence is conservative relative to 90% TOST.
- Hyperparameters were tuned on the fixed DINO representation and reused on ResNet without retuning. Findings are conditional on this shared protocol.
- The active preprocessing and spatial-only cache are frozen. No claim is made about omitted motion, intermediate-layer search, YouTube-UGC or cloud experiments.
- Warning-only deterministic execution does not guarantee bitwise reproduction on every hardware/software stack.
- A deferred minor publication issue remains: `13_make_tables.py` updates the decision log before the final passed-gate write. If that final write fails, prose can claim completion while the machine-readable gate correctly fails. Actual final publication succeeded here; consumers must use the checked gate and source-bound artifacts.
- LIVE-VQC generalization, full-KoNViD refits, CPU deployment eligibility and serving validation are not completed by this phase.

## Reproduction and canonical artifacts

These commands assume the datasets, spatial caches and locked research environment are already prepared; they continue the pipeline rather than bootstrap a clean machine. The executed ML queue is:

```bash
tmux new -s train -c /home/joydeep/Projects/VQA
# After confirming the remote session, run inside tmux:
bash scripts/continue_phase2.sh
```

Completed compatible fits are checksum-validated and reused. A failed control stops the queue; never change a permutation, seed, threshold or selected config to obtain a passing result. No automatic Git commit/push is part of the runner.

Canonical outputs:

- [phase_gate.json](phase_gate.json): completion and published-file hashes.
- [effective_configs.json](effective_configs.json), [selection.json](selection.json): source-bound settings and validation-only choice.
- [headline.csv](headline.csv), [head_comparison.csv](head_comparison.csv), [feature_pairs.csv](feature_pairs.csv): headline and paired results.
- [comparisons.json](comparisons.json), [comparisons.meta.json](comparisons.meta.json): raw metrics, policy and shared-index provenance.
- [summary.json](summary.json), [time_accounting.json](time_accounting.json): variability, controls, limitations and accounting.
- [konvid_freeze.json](konvid_freeze.json): frozen research settings, full-data refit inputs and unresolved deployment eligibility.
- [phase1_before_training.json](phase1_before_training.json): pretraining source and Phase 1 checksum snapshot.
- [session.log](session.log), [progress.json](progress.json): diagnostic execution record and last training status.

This Markdown report is a derivative generated after verification; it is not a substitute for the canonical gate and JSON/CSV artifacts.

Execution commit recorded by the pretraining snapshot: `0dcf48b44a81caae21798d984ae30344fc85a91d`. Source identity: `5165c540f2f16997271b2974d80cf02814c5020129cc4733c690feaf503d9169`. Full CV artifact fingerprint: `c6514057359255c931dc67a01da741881d67161407938a896c5de2b863eba1c8`. Numerical tables in this report were formatted from saved outputs by the local report-generation script, not entered manually.
