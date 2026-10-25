# Phases

Each phase has an exit criterion. Do not start the next phase until the earlier criterion passes. The active 10-day scope is recorded in `configs/scope.yaml` and the dated decisions log.

Deadline: **2026-11-04**. Availability: approximately 8 hands-on hours/day and 10 unattended GPU hours/night for 10 planning days. All computation is local.

## Phase 0: Preflight

Goal: verify data access, preprocessing, pretrained execution, extraction cost and disk capacity before research runs.

- Inventory KoNViD-1k and LIVE-VQC, join MOS one-to-one, record provenance and licenses, preserve the source archives.
- Verify the locked environment in a fresh environment.
- Time the fixed 20-video sample on GPU fp16 and CPU fp32 with per-stage measurements and three repeats.
- Verify shared FFmpeg timestamps, ties, boundary handling and rotation; visually review three LIVE-VQC portraits.
- Confirm pinned DINOv2 ViT-B and ResNet-50 weights/transforms. Completed SlowFast checks remain historical preflight evidence; SlowFast is excluded from the active research scope.
- Record the CPU budget, deadline, working hours, unattended compute allowance and disk reserve.
- Skip all YouTube-UGC work, including metadata selection and cloud checks.
- Generate the readiness report, then freeze the active spatial policy in `configs/features_v1.yaml` and ARCHITECTURE.md.

Exit criterion: complete valid preflight measurements, confirmed active backbones, human portrait review, feasible extraction/disk projections and recorded schedule/CPU decisions. `results/phase0/feasibility.json` checks the conservative historical three-backbone measurement envelope; `scoped_feasibility.json` reports the smaller active spatial-only projection. Freeze evidence is recorded separately after readiness passes.

## Phase 1: Core Slice (KoNViD-1k, DINOv2)

Goal: finish the definitive DINO head-comparison runs by **2026-10-30**, with all four heads and the required controls.

- In Task 1 cache DINOv2 CLS and patch mean/std from blocks 6, 9 and 12 at 448×448, plus ResNet-50, for KoNViD and then LIVE-VQC in the same sequential overnight queue. Use `configs/features_v1.yaml`, atomic writes and checksum manifests; retain archives. LIVE-VQC feature extraction is allowed now, but prediction/model selection is forbidden until the KoNViD configuration freeze.
- Use the declared primary feature set, CLS plus block-12 patch mean. Keep the other cached statistics without running a layer comparison or claiming they improve quality.
- Generate and commit the fixed five folds and 15% stratified inner-validation splits once, seed 0.
- Implement mean-pool MLP, Bi-GRU, Transformer and Mamba with matched head-only size tiers. Test padding invariance, naive Mamba recurrence and causal-variant causality.
- Add ridge Baseline 0 and shuffled-label control with the binding pass threshold.
- Benchmark one training run per head before selecting search ranges, maximum epochs and patience. Record actual runtime; the full project has no measured training-time guarantee yet.
- Tune the shared rank-loss weight first with mean pooling on fold 0; freeze it before head tuning.
- Run **10 tuning candidates per head**, all on fold 0 inner validation and the fixed DINO feature set. Tuning defaults to five seeds; on measured runtime no-go only, record a common `[0,1]` tuning-seed fallback before rank/head tuning. Tuning trials are not reported results. Reported CV and controls remain five seeds. Freeze `configs/protocol.yaml` and reuse settings across feature sets.
- Run the final DINO head comparison once: all four heads with five folds × five seeds. Save predictions, inner-validation SRCC, epochs and run metadata. These are the real research fits; Phase 2 must reuse the same 100 completed runs, not run a second DINO grid.
- Generate metrics/tables from saved outputs and run CPU fp32 versus GPU fp16 feature parity on the fixed 20 videos.

Exit criterion: all four heads and ridge run under the frozen protocol; required tests, shuffled-label control and parity pass; tables are generated. These are the definitive DINO comparison runs. Phase 2 adds the ResNet control and final joint statistical reporting using the saved DINO outputs; it never repeats unchanged DINO fits.

## Phase 2: Spatial Feature and Head Comparison

Goal: compare the fixed DINO representation with the ResNet-50 control, then compare heads fairly.

- Reuse Phase 1 DINO results; run all four heads and ridge on ResNet-50 with the same frozen protocol and no additional tuning.
- Run no SlowFast extraction, motion combinations or separate DINO layer comparison.
- Choose between the two reported spatial feature sets using mean inner-validation SRCC across folds/seeds, never held-out-fold results. Disclose the residual tuning/selection overlap.
- Produce a spatial feature-control table and the primary four-head comparison table.
- Use paired bootstrap on pooled out-of-fold predictions averaged across seeds; classify CIs using the declared ±0.01 margin and apply Holm adjustment to the six primary head pairs. Report seed/fold SDs and interval limitations.

Exit criterion: generated tables and confidence intervals exist; feature/protocol choices are frozen without using held-out folds. The KoNViD configuration is now frozen for LIVE-VQC predictions.

## Phase 3: LIVE-VQC Generalization

- Verify/reuse the LIVE-VQC spatial cache produced in Phase 1 Task 1; repair only missing/corrupt entries using the same frozen policy. Do not re-extract a complete valid cache. Predictions remain forbidden before the KoNViD configuration freeze.
- Train each head and ridge on all KoNViD with the frozen chosen feature set. Use five seeds and each head's median CV early-stopping epoch; seed 0 is the deployment checkpoint.
- Predict LIVE-VQC only after the KoNViD freeze. Report SRCC/KRCC and mapped PLCC, with the mapping disclosed.
- Bootstrap cross-dataset head comparisons as the second primary Holm family and report sequence-length/domain-shift limitations.
- Run no YouTube-UGC subset, cloud work, length conditions or UGC predictions. State the cut in the final README.

Exit criterion: generated LIVE-VQC tables/CIs exist, and the selected seed-0 deployment checkpoint is identified by hash.

## Phase 4: Minimal Serving and Reporting

- Deploy the reported seed-0 full-KoNViD checkpoint for the best eligible spatial configuration within the accepted CPU budget.
- Build a FastAPI service with an HTML upload page served by the same process. A single uploaded clip is processed at a time, using shared `src/preprocess.py` and `src/features.py` on CPU fp32, then the page displays its score or a useful failure.
- Use a single Dockerfile with CPU-only PyTorch wheels, pinned weights and the reported checkpoint. No Celery, Redis, separate worker, React build, SQLite job queue or Docker Compose.
- Bound uploads and work admission, clean temporary uploads on success/failure, and document the request/retention limits. Blocking inference must not allow unbounded concurrent jobs.
- Measure extraction and head inference separately, and the actual end-to-end request cost. Re-run serving parity against the deployed checkpoint.
- Write the honest README with generated results, CIs, LIVE-VQC performance, limitations, all cuts, reproducibility and clean Docker run instructions.

Exit criterion: a clean Docker build and end-to-end upload work, the serving parity threshold passes, and README numbers match saved outputs.

## Ten-Day Schedule and Compute Limits

The following is a planning allocation, not a runtime result or a promise:

| Calendar checkpoint (Asia/Kolkata) | Main work |
|---|---|
| Oct 25–26 | Task 1 and sequential KoNViD → LIVE-VQC extraction; implement Tasks 2–6 while GPU runs |
| Oct 28, end of day | Task 7 pilot complete; **if not, stop and cut scope before tuning** |
| Oct 30, end of day | Phase 1 complete: definitive DINO CV, ridge, controls and parity |
| Oct 31 | ResNet control and final comparison tables; reuse DINO fits |
| Nov 1–2 | Full-KoNViD models and LIVE-VQC evaluation from cached features |
| Nov 3–4 | Minimal serving, parity, clean Docker run and README |


Parallelize unattended extraction/training with independent implementation only after the owning phase permits it. Measure training early; do not reduce reported-run seeds, folds, controls, confidence intervals, LIVE-VQC or parity. The only preauthorized seed cut is the documented tuning-only fallback after a runtime no-go. The October 28 calendar checkpoint also applies even if GPU hours look sufficient.

## Scope Cuts

- SlowFast and all motion feature experiments: cut.
- YouTube-UGC, metadata subset selection and all cloud/length work: cut.
- Separate DINO layer comparison: cut; cache all planned layers anyway.
- Tuning budget: 10 trials per head, same budget for all four heads.
- Serving: FastAPI + same-origin upload page + single Dockerfile.

Never cut: shuffled-label control, ridge baseline, required correctness tests, confidence intervals, LIVE-VQC, applicable parity checks or honest reporting. Further changes require dated decisions.

## Measured Spatial Extraction Estimate

Generated from `results/phase0/scoped_feasibility.json` and the completed timing CSV.

DINOv2 + ResNet-50 extraction for KoNViD and LIVE-VQC, with the 2× planning allowance: **10.37 GPU hours** against **100 available hours**.

Predicted fp16 cache, including every planned DINO statistic: **0.87 GiB**. Additional disk including atomic-write headroom, one transient raw video and reserve: **6.78 GiB**.

Phase 0 readiness is go and the spatial v1 policy is frozen. See `results/phase0/scope_report.md` and `preprocessing_freeze.json`. Training runtime remains unmeasured; benchmark it early in Phase 1.

## Phase 1 execution status (2026-10-25)

Implementation is present on `phase1-core-slice`; fixed indices are committed. Sequential GPU extraction has started. Phase 1 remains **in progress**, with a draft training protocol until the real leakage diagnostic, pilots and measured runtime gate pass. The authoritative completion status is generated by `scripts/06_eval.py` in `results/phase1/phase_gate.json`; passing offline tests alone does not satisfy this phase. No LIVE-VQC prediction or later-phase work has started.
