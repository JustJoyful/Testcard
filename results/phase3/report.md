# Phase 3 LIVE-VQC generalization report

Generated: 2026-10-28 (Asia/Kolkata)
Status: **PASSED WITH QUALIFICATION**; fixed-epoch shuffled-label control: **FAILED**

Phase 3 evaluated the frozen KoNViD-1k configurations on all 585 LIVE-VQC videos. The selected research representation was ResNet-50, chosen in Phase 2 using KoNViD inner-validation scores. All four neural heads were refitted on all 1,200 KoNViD videos with five fixed seeds and their frozen median CV epochs; ridge used one deterministic frozen-alpha fit. The same ten configurations were evaluated on LIVE-VQC, with DINO results labeled secondary. The failed new control remains failed and is disclosed below; the qualified gate records the user-approved continuation, not a control pass.

## Protocol and result quantities

- Frozen spatial features: ResNet-50 control and DINOv2 last-layer CLS plus block-12 patch mean, reused from the checked cache. No LIVE-VQC result was used to change the KoNViD freeze, full-refit schedule, ridge alpha, CPU winner, or checkpoint selection.
- Neural headline values are mean ± sample SD across five per-seed LIVE metrics. The seed-0 result is shown separately. Ridge is deterministic and has one result. SRCC and KRCC use raw predictions; mapped PLCC uses a four-parameter logistic fit to LIVE-VQC target labels and is a descriptive target-domain metric, not label-free calibration.
- Paired comparisons use the per-video mean of five neural predictions, or the single ridge prediction, over the same 585 videos. Their 10,000 seed-0 video-bootstrap draws share one index bank. Pointwise CIs capture test-video sampling noise; they do not capture seed training variation. The six ResNet neural SRCC pairs alone receive Holm adjustment. Nine ridge/feature comparisons are secondary and unadjusted. The declared practical margin is ±0.01.

## LIVE-VQC headline results

### ResNet-50: selected representation

| Head | Runs | SRCC | Seed-0 SRCC | KRCC | Mapped PLCC |
| --- | --- | --- | --- | --- | --- |
| Ridge (Baseline 0) | 1 | 0.6439 | — | 0.4605 | 0.7065 |
| Mean-pool MLP | 5 | 0.6315 ± 0.0198 | 0.6230 | 0.4525 ± 0.0170 | 0.7043 ± 0.0143 |
| Bi-GRU | 5 | 0.6242 ± 0.0167 | 0.6108 | 0.4457 ± 0.0135 | 0.6874 ± 0.0089 |
| Transformer | 5 | 0.6224 ± 0.0244 | 0.6617 | 0.4437 ± 0.0191 | 0.6972 ± 0.0082 |
| Mamba (bidirectional) | 5 | 0.6249 ± 0.0234 | 0.6437 | 0.4464 ± 0.0193 | 0.6942 ± 0.0145 |

### DINOv2: secondary representation

| Head | Runs | SRCC | Seed-0 SRCC | KRCC | Mapped PLCC |
| --- | --- | --- | --- | --- | --- |
| Ridge (Baseline 0) | 1 | 0.6584 | — | 0.4764 | 0.7432 |
| Mean-pool MLP | 5 | 0.6092 ± 0.0065 | 0.6049 | 0.4364 ± 0.0051 | 0.7103 ± 0.0061 |
| Bi-GRU | 5 | 0.5793 ± 0.0373 | 0.5702 | 0.4100 ± 0.0300 | 0.7009 ± 0.0197 |
| Transformer | 5 | 0.6081 ± 0.0140 | 0.6123 | 0.4356 ± 0.0112 | 0.7164 ± 0.0066 |
| Mamba (bidirectional) | 5 | 0.5909 ± 0.0127 | 0.5848 | 0.4220 ± 0.0090 | 0.7009 ± 0.0043 |

The ResNet ridge deployment row has LIVE SRCC 0.6439, KRCC 0.4605, and mapped PLCC 0.7065. Its `seedNone` label denotes a deterministic ridge fit. DINO ridge has the highest numerical LIVE headline SRCC (0.6584), but this post-freeze observation does not change the deployment choice; the DINO-versus-ResNet ridge paired CI includes zero.

### KoNViD-to-LIVE context

| ResNet head | KoNViD CV SRCC | LIVE SRCC | LIVE − KoNViD |
| --- | --- | --- | --- |
| Ridge (Baseline 0) | 0.7801 | 0.6439 | -0.1362 |
| Mean-pool MLP | 0.7487 | 0.6315 | -0.1172 |
| Bi-GRU | 0.7526 | 0.6242 | -0.1285 |
| Transformer | 0.7599 | 0.6224 | -0.1375 |
| Mamba (bidirectional) | 0.7542 | 0.6249 | -0.1293 |

The KoNViD column comes from the [Phase 2 headline](../phase2/headline.csv). KoNViD values summarize five CV folds (25 neural fold × seed fits, five ridge folds); LIVE values summarize five full-data neural seeds or one ridge fit. These different estimands show lower numerical LIVE rank correlation, but their difference cannot isolate domain shift from the change in training set and fixed-epoch schedule. The cached sequence lengths also differ:

| Dataset | Videos | Median steps | Range |
| --- | --- | --- | --- |
| konvid | 1200 | 33 | 31–33 |
| livevqc | 585 | 41 | 39–41 |

## Paired LIVE-VQC comparisons

### Primary ResNet neural family (A − B)

| Pair | SRCC Δ | Pointwise 95% CI | Holm p | CI classification |
| --- | --- | --- | --- | --- |
| resnet50/Mean-pool MLP − resnet50/Bi-GRU | 0.0016 | [-0.0224, 0.0258] | 1.0000 | no detectable difference (inconclusive at this sample size) |
| resnet50/Mean-pool MLP − resnet50/Transformer | 0.0059 | [-0.0137, 0.0256] | 1.0000 | no detectable difference (inconclusive at this sample size) |
| resnet50/Mean-pool MLP − resnet50/Mamba (bidirectional) | 0.0050 | [-0.0107, 0.0207] | 1.0000 | no detectable difference (inconclusive at this sample size) |
| resnet50/Bi-GRU − resnet50/Transformer | 0.0043 | [-0.0138, 0.0214] | 1.0000 | no detectable difference (inconclusive at this sample size) |
| resnet50/Bi-GRU − resnet50/Mamba (bidirectional) | 0.0034 | [-0.0203, 0.0267] | 1.0000 | no detectable difference (inconclusive at this sample size) |
| resnet50/Transformer − resnet50/Mamba (bidirectional) | -0.0009 | [-0.0187, 0.0160] | 1.0000 | no detectable difference (inconclusive at this sample size) |

0 of 6 primary pairs survive Holm at 0.05; 6 pointwise SRCC CIs include zero. No primary pair establishes equivalence within ±0.01 or a practically meaningful advantage under the ordered CI rule. The full SRCC and KRCC intervals, raw p-values, draw counts, and classifications are in [comparisons.csv](comparisons.csv) and [comparisons.json](comparisons.json).

### Secondary ridge and representation comparisons (A − B)

| Pair | SRCC Δ | Pointwise 95% CI | Raw p | CI classification |
| --- | --- | --- | --- | --- |
| resnet50/Ridge (Baseline 0) − resnet50/Mean-pool MLP | -0.0041 | [-0.0262, 0.0176] | 0.7155 | no detectable difference (inconclusive at this sample size) |
| resnet50/Ridge (Baseline 0) − resnet50/Bi-GRU | -0.0025 | [-0.0304, 0.0244] | 0.8552 | no detectable difference (inconclusive at this sample size) |
| resnet50/Ridge (Baseline 0) − resnet50/Transformer | 0.0019 | [-0.0228, 0.0265] | 0.8821 | no detectable difference (inconclusive at this sample size) |
| resnet50/Ridge (Baseline 0) − resnet50/Mamba (bidirectional) | 0.0010 | [-0.0226, 0.0237] | 0.9343 | no detectable difference (inconclusive at this sample size) |
| dino/Mean-pool MLP − resnet50/Mean-pool MLP | -0.0181 | [-0.0613, 0.0249] | 0.4059 | no detectable difference (inconclusive at this sample size) |
| dino/Bi-GRU − resnet50/Bi-GRU | -0.0466 | [-0.0975, 0.0030] | 0.0678 | no detectable difference (inconclusive at this sample size) |
| dino/Transformer − resnet50/Transformer | -0.0151 | [-0.0595, 0.0279] | 0.5001 | no detectable difference (inconclusive at this sample size) |
| dino/Mamba (bidirectional) − resnet50/Mamba (bidirectional) | -0.0335 | [-0.0782, 0.0110] | 0.1410 | no detectable difference (inconclusive at this sample size) |
| dino/Ridge (Baseline 0) − resnet50/Ridge (Baseline 0) | 0.0145 | [-0.0276, 0.0565] | 0.5009 | no detectable difference (inconclusive at this sample size) |

All 9 secondary SRCC intervals include zero. These p-values are unadjusted and exploratory. Higher single-row headline scores do not establish paired superiority. All 15 comparison pairs also have KRCC intervals in the saved tables.

## CPU eligibility and deployment

A four-thread CPU benchmark measured 20 fixed KoNViD videos × 3 repeats × 10 candidates = 600 candidate records. Each result is normalized to a 10-second video. The eligibility statistic is each candidate's **maximum** normalized total over 60 records; one-time model loading is excluded. Extraction combines decode, preprocessing, and backbone inference. The extraction/head columns below come from the row producing that candidate's maximum total. The frozen budget is 120.0000 seconds per 10 seconds of video.

| Candidate | Frozen CV SRCC | Max-row extraction s/10s | Max-row head s/10s | Max total s/10s | Eligible |
| --- | --- | --- | --- | --- | --- |
| resnet50/ridge | 0.7801 | 8.3276 | 0.0001 | 8.3278 | yes |
| resnet50/mean_pool | 0.7487 | 8.3276 | 0.0007 | 8.3283 | yes |
| resnet50/bigru | 0.7526 | 8.3276 | 0.0020 | 8.3296 | yes |
| resnet50/transformer | 0.7599 | 8.3276 | 0.0015 | 8.3291 | yes |
| resnet50/mamba | 0.7542 | 8.3276 | 0.0051 | 8.3327 | yes |
| dino/ridge | 0.7350 | 28.7209 | 0.0001 | 28.7210 | yes |
| dino/mean_pool | 0.6852 | 28.7209 | 0.0006 | 28.7215 | yes |
| dino/bigru | 0.6892 | 28.7209 | 0.0021 | 28.7229 | yes |
| dino/transformer | 0.6930 | 28.7209 | 0.0013 | 28.7222 | yes |
| dino/mamba | 0.6895 | 28.7209 | 0.0048 | 28.7257 | yes |

All 10 candidates meet the declared budget. The predeclared winner rule selects the highest frozen KoNViD CV SRCC among eligible candidates, resolving to **resnet50/ridge**. Its measured maximum is **8.3278 s/10s**. The earlier source-bound CPU grid also chose the same candidate at 9.4925 s/10s and remains saved as [cpu_eligibility_before_qualification.json](cpu_eligibility_before_qualification.json); the new grid is the operative evidence after the label-path code change. These are raw-video-to-score CPU measurements on KoNViD, not Phase 4 API request latency. Serving must remeasure end-to-end cost with four PyTorch threads.

**Deployment artifact:** [model.npz](refit/resnet50/ridge/seedNone/model.npz) (deterministic full-KoNViD ridge), SHA256 `4c9782aefca4b81da4d15da31681f11ca8ded4e890d36d67fb478663583327a3`. Its identity is pinned by [deployment.json](deployment.json) and the [Phase 3 gate](phase_gate.json).

## Failed shuffled-label control and qualification

The selected ResNet mean-pool fixed-epoch control used the frozen three-epoch regime and separate, fixed train and validation label permutations on KoNViD fold 0. True fold-0 test MOS was retained only for evaluation. Under the unchanged rule, each seed must have |SRCC| < 0.15 **and** a 95% bootstrap CI containing zero. The original run stopped at seed 3 and remains recorded as failed; seed 4 was completed once separately and also failed. No seed, permutation, configuration, threshold, or winner was changed to seek a pass.

| Seed | Rule result | Trained SRCC | Trained 95% CI | Untrained SRCC | Trained–init prediction r |
| --- | --- | --- | --- | --- | --- |
| 0 | passed | -0.1011 | [-0.2280, 0.0277] | -0.3124 | -0.0135 |
| 1 | passed | -0.0214 | [-0.1466, 0.1028] | 0.2151 | 0.0714 |
| 2 | passed | -0.0656 | [-0.1901, 0.0586] | 0.0240 | 0.0515 |
| 3 | failed | -0.2322 | [-0.3466, -0.1136] | -0.1724 | 0.0955 |
| 4 | failed | -0.2361 | [-0.3540, -0.1106] | -0.1843 | 0.4849 |

The **Phase 1/2-comparable five-seed ensemble**, based on mean trained predictions per held-out video, has SRCC **-0.1563**, CI **[-0.2779, -0.0306]**. It also fails the original rule and is not a substitute pass. The post-failure untrained-initialization ensemble has SRCC **-0.2027**, CI **[-0.3230, -0.0747]**; it is diagnostic only.

Trained and untrained test-SRCC signs agree for seeds 0, 3, and 4 and disagree for seeds 1 and 2. Their per-video prediction correlations are near zero for seeds 0–3 and materially higher for seed 4; the three-epoch trained predictions are **not** generally near-identical to initialization. Initialization correlation therefore does not explain away the trained failure. The untrained ensemble failure shows that the fixed absolute-SRCC rule has poor specificity on this fold. The negative trained ensemble association remains unresolved. The saved [investigation](control_fixed_epoch/investigation.json), [untrained baseline](control_fixed_epoch/untrained_baseline.json), [original failed diagnostic](control_fixed_epoch/diagnostic.json), and [seed-4 result](control_fixed_epoch/seed4_result.json) preserve the exact values and per-video evidence.

A label-path audit found no concrete route from held-out LIVE MOS into full refit updates, CPU winner selection, or LIVE score computation. Full refits intentionally train on **all** KoNViD MOS values; the qualified-control validator reads fold-0 KoNViD test MOS to audit the historical failure. LIVE prediction loads feature-only items and writes label-free prediction rows. Only reporting reads LIVE MOS for metrics and mapped PLCC. This audit does not prove every possible leak absent.

The user approved continuation with the control explicitly failed. This is the second post-hoc control amendment, after the Phase 1 v2 correction. The [dated decision](../../DECISIONS.md) records the qualification and recommends predeclaring an untrained-initialization null baseline for future controls instead of relying on a fixed |SRCC| threshold. The current threshold and failed classification were not changed.

## Limits and interpretation

- KoNViD feature selection used inner-validation scores from videos also present in head comparisons.
- Fixed-epoch shuffled-label control failed at seeds 3 and 4; this qualified continuation does not reclassify that result.
- Five-seed trained ensemble also failed the original rule; untrained baseline failed it, while trained predictions were mostly unlike initialization.
- Full-data median CV epochs approximate a different steps-per-epoch schedule.
- KoNViD and LIVE-VQC differ in domain and sequence lengths.
- Center crop loses edge content, especially for portrait videos.
- UGC and separate length-effect claims were cut from scope.
- The Phase 3 report contains no matched published SOTA comparison. External results require citation to original papers and exact evaluation protocols; no such numerical claim is inferred from these saved outputs.
- The CPU benchmark excludes one-time loading and does not measure the future FastAPI request path. Phase 4 must verify serving parity and actual request latency before deployment.

All Phase 1–3 model checkpoint `.pt` and `.npz` binaries required by the Phase 3 gate are present locally and staged through Git LFS. Git stores small pointers; the real checkpoint bytes upload to LFS when the commit is pushed. The frozen backbone weights and feature cache remain separate, ignored inputs needed for full gate revalidation; the deployment model binary is among the staged checkpoints.

## Evidence and verification

The [Phase 3 gate](phase_gate.json) has status `passed with qualification`, `control_status` `failed`, and SHA256 entries for 198 canonical outputs. All entries and the pinned decision, freeze, source-code, deployment-checkpoint, and CPU-evidence hashes were verified before this report was written. The test suite `.venv/bin/python -m pytest -q` returned **253 passed**. The report is a derived narrative written after the gate; it does not alter the gate-listed metrics or artifacts.

Pinned identities: Phase 2 gate `c7b0262d3dba5fe7f0da651e51de9052dcd90dca8b98e51bd6e9417bf386d8d7`; KoNViD freeze `4a19edfb683fcf3b08265de32822eb22cebe34f8aa8e0fd42e4e08783aeceb8c`; Phase 3 source `68a001f55650d758a69b09e789d6be55a6a0646b23baea2808bd8e9770841c56`; deployment checkpoint `4c9782aefca4b81da4d15da31681f11ca8ded4e890d36d67fb478663583327a3`.

Source tables: [per-run LIVE metrics](metrics.csv), [headline](headline.csv), [paired comparisons](comparisons.csv), [CPU grid](cpu_eligibility.json), [sequence lengths](lengths.json), [deployment](deployment.json), and [summary](summary.json).
