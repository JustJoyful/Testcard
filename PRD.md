# PRD: Where Does the Quality Signal Live? Features vs. Temporal Heads in Cached-Feature Video Quality Assessment

## 1. Problem

No-reference video quality assessment (VQA) predicts how a human would rate a video's quality, without a pristine reference. Recent state-of-the-art models (FAST-VQA, DOVER) reach roughly 0.88+ SRCC on KoNViD-1k, but they are end-to-end and expensive. A lighter pipeline built from frozen, cached features is useful only if we know which parts of it actually matter: the feature representation or the temporal model on top of it.

Claims such as "Mamba is better for video quality" are not established. Mamba is a linear-time state space model, but at the sequence lengths used in VQA (roughly 32 to 40 steps for KoNViD-1k and LIVE-VQC at 4 fps) its efficiency advantage is mostly irrelevant. Whether it helps is an empirical question.

## 2. Goal

Primary question: in a cheap, cached-feature VQA pipeline, how much of the quality signal comes from the frozen feature representation, and how much from the temporal head?

Within that, measure with proper statistics whether State Space Models (Mamba) give any detectable advantage over simpler temporal aggregators (mean pooling, Bi-GRU, Transformer encoder) when all heads see identical cached features. Report the result honestly, whichever way it goes. A tie between heads is a valid and expected headline.

## 3. Research Questions

RQ1. Does temporal ordering matter at all? Mean pooling vs. the sequence models, on the same features.

RQ2. Does Mamba beat Bi-GRU and Transformer on in-dataset performance (KoNViD-1k, 5-fold CV)?

RQ3. How much does the feature representation matter when comparing ResNet-50 (control) with the fixed DINOv2 last-layer CLS plus block-12 patch mean? The separate layer comparison and motion features are cut; other planned DINO statistics are cached without being evaluated as separate variants.

RQ4. Do the heads generalize across datasets (train KoNViD-1k, test LIVE-VQC)?

RQ5 (length extrapolation) is cut from the 10-day scope together with YouTube-UGC; no length-generalization claim is made.

## 4. Non-Goals

- Beating SOTA (FAST-VQA, DOVER). We report the gap and explain it; we do not try to close it.
- End-to-end training of a video backbone. Features are frozen and cached.
- Production-grade or real-time serving. The backend exists to demonstrate the pipeline, not to handle traffic.
- Training on raw pixels or on more than one consumer GPU.
- Claiming frame-level quality labels. None exist in these datasets.

## 5. Constraints

- Compute location: all feature extraction, training and evaluation run locally on the RTX 3050 6GB laptop. SlowFast, YouTube-UGC and all cloud work are cut.
- Serving: CPU only, fp32.
- Storage: limited local disk. Raw videos may be deleted after feature caching. Cached features are the artifacts that are kept.
- Time: deadline 2026-11-04, with approximately 8 hands-on hours/day and 10 unattended GPU hours/night for the 10-day plan. Equal tuning budgets are 10 trials per head. Actual training runtime must be measured in Phase 1 before committing the full grid.
- Licensing: dataset terms must be checked before publishing results. No UGC or cloud processing is planned.

## 6. Datasets

| Dataset | Role | Size | Label scale | Where processed |
|---|---|---|---|---|
| KoNViD-1k | Core training and 5-fold CV | About 1,200 videos, about 8 s, 960x540 | MOS, 1 to 5 | Local |
| LIVE-VQC | Cross-dataset generalization | 585 videos, about 10 s, mobile capture, includes portrait | MOS, 0 to 100 | Local |

Clip lengths, frame rates, and resolutions above are to be verified from the data in Phase 0 and recorded in the manifest.

## 7. Metrics

- SRCC (Spearman) and KRCC: scale-invariant, no mapping needed.
- PLCC (Pearson): computed after fitting a 4-parameter logistic between predictions and MOS on the target dataset. Stated explicitly in the README. PLCC is reported as a point estimate per run; it is not part of the bootstrap comparisons.
- Headline number: mean and standard deviation of per-fold SRCC across the 25 fold x seed runs. This is the number in the main tables.
- Comparison number: paired bootstrap over videos on pooled out-of-fold predictions, averaged across seeds (one prediction per video per head). This is a seed-averaged ensemble and is not the same quantity as the headline mean; both are labeled in the README.
- Equivalence margin: declared before any results exist. A pairwise SRCC difference within +/-0.01 is treated as practically equivalent. The margin is a design choice, not a measured quantity. Actual CI widths are reported as measured.
- Decision table for a pairwise difference A minus B, using its two-sided 95% bootstrap CI (checked in this order):
  1. CI entirely inside [-0.01, +0.01]: "equivalent within margin".
  2. CI entirely above +0.01 (or below -0.01): "A better than B (practically meaningful)".
  3. CI excludes zero but overlaps the margin: "detectable difference, not shown to exceed the margin".
  4. CI includes zero and extends beyond the margin: "no detectable difference (inconclusive at this sample size)".
  A 95% two-sided CI is used throughout; this is more conservative for equivalence than the 90% CI of a TOST procedure, and the README says so.
- Multiple comparisons: the primary family is the 6 pairwise head comparisons on the chosen feature set, on KoNViD-1k; LIVE-VQC is a second family of 6. Holm-adjusted bootstrap p-values are reported for these families alongside the unadjusted CIs. Spatial feature-control comparisons are secondary and unadjusted; no length comparison is planned.
- Interval scope: the bootstrap captures test-set sampling noise only. It does not capture training or fold variance, so intervals are somewhat optimistic. Seed and fold standard deviations are always reported alongside, and the CI is described as applying to the seed-averaged ensemble.
- Expected power: with highly correlated heads, CI widths on KoNViD may be close to or wider than the margin, and wider still on LIVE-VQC (585 videos) with the smaller held-out dataset. "Inconclusive" is an expected and acceptable outcome.

## 8. Success Criteria

The project succeeds if all of the following hold, regardless of which head wins:

1. All four heads are trained and evaluated under identical protocol and matched head-only parameter budgets.
2. Every number in the README is generated by a script from saved per-video predictions.
3. Pairwise head differences are reported with confidence intervals and classified with the decision table in section 7.
4. Cross-dataset results on LIVE-VQC are reported, including where they fall short.
5. The deployed model is the exact checkpoint whose numbers are reported: the Phase 3 model trained on all of KoNViD-1k with the chosen configuration, seed 0 (declared now, not chosen after seeing results). Its LIVE-VQC numbers are reported for that seed as well as for the 5-seed summary.
6. CPU latency per video is measured and reported for every reported feature configuration, with extraction and head inference timed separately.
7. The offline-to-serving parity check passes and its numbers are in the README.

## 9. Realistic End State

Planning expectations, not results (Rule 22 applies to any number that reaches the README):

- The system predicts a single 1-to-5 quality score for an uploaded short video (best at about 8 to 10 s) without a reference, on CPU, in tens of seconds to a few minutes per video.
- Accuracy is expected to sit below published end-to-end SOTA on KoNViD-1k, with a further drop on LIVE-VQC due to domain shift. Both gaps are reported and explained.
- The most likely research outcome is that the feature representation matters much more than the temporal head, and that most head comparisons are equivalent or inconclusive.
- The system does not give trustworthy per-second scores, does not run in real time, and is not expected to generalize to long videos or very different content (gaming, screen recordings) without retraining.

## 10. Deliverables

- Reproducible training and evaluation code, organized per stage, with a locked local environment and the shared preprocessing/features modules used for serving.
- Cached feature manifests (not videos) and fold indices committed to the repo.
- Results tables and figures generated from saved outputs.
- README: question, feature ablation, head comparison with CIs, LIVE-VQC cross-dataset results, gap to published SOTA with reasons, limitations, what was not done, then the demo.
- Minimal serving stack: FastAPI, a same-origin HTML upload page and one Dockerfile. Process a single CPU job at a time and display its score/error. No Celery, Redis, separate worker, React build, Compose or per-timestep graph.

## 11. Risks

| Risk | Mitigation |
|---|---|
| All heads tie (likely with strong frozen features) | That is a result. Feature ablation is designed to show where the signal is. |
| Ten-day scope exceeds available time | Use spatial features only, bounded equal tuning and minimal serving; time the first training run before the full grid. |
| Disk space runs out | Retain source archives; delete extracted raw copies only after feature checksums pass. No UGC data is processed. |
| Test-set leakage through tuning | Tuning on fold 0 with an inner validation split only. Residual overlap with other folds' test sets is disclosed. Rules in RULES.md. |
| Resume claims ahead of data | Resume bullets written only after results exist. |
| Bootstrap CIs are optimistic (no training variance) | Report seed and fold SDs alongside. Describe CIs as applying to the seed-averaged ensemble. |
| Feature-layer choices multiply the ablation matrix | Cache layers anyway but evaluate only the declared CLS plus block-12 mean DINO representation and the ResNet control. |
| Portrait videos decoded sideways (rotation metadata ignored) | Decoder must apply rotation metadata; verified in Phase 0 on LIVE-VQC portrait clips. |
| Offline (GPU fp16) and serving (CPU fp32) features drift | Parity check on 20 videos; numbers reported. |
