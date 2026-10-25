# Architecture

## 1. Overview

The system has two halves that never run together.

Offline half (local GPU, research): raw video goes through timestamp sampling and frozen DINOv2/ResNet-50 backbones, producing cached feature arrays. All extraction, training and evaluation run locally. SlowFast, YouTube-UGC and cloud work are excluded from the active 10-day scope.

Online half (CPU, serving): a FastAPI process accepts an upload through its own HTML page, runs the same spatial preprocessing/backbone and chosen head, and returns a MOS score. One Dockerfile packages this process.

Offline and serving must import the same `src/preprocess.py` and `src/features.py`. The active cache policy is `configs/features_v1.yaml`; research/schedule limits are `configs/scope.yaml`. Completed three-backbone Phase 0 measurements remain historical evidence under `results/phase0/`; their original measurement settings remain in `configs/phase0.yaml`.

## 2. Spatial Preprocessing Policy (v1, frozen 2026-10-25)

The active spatial settings below are frozen in `configs/features_v1.yaml` after passing Phase 0 readiness. Changes require a new cache version and affected evidence reruns. The freeze record is `results/phase0/preprocessing_freeze.json`.

Shared decoding:
- Decoding: the verified FFmpeg/ffprobe path; never mix decoders. Decode sequentially and take the frame nearest each target timestamp. Do not seek, since seeking is unreliable in some containers.
- Rotation: the decoder must apply container rotation metadata, so portrait phone videos are upright before resize and crop. The shared decoder disables FFmpeg autorotation and explicitly applies the ffprobe rotation before resize/crop. Verified in Phase 0 on LIVE-VQC portrait clips. The applied rotation is recorded per video in the manifest.
- Frame sampling is by timestamp, not by frame index. Variable frame rate makes "every Nth frame" incorrect.

Spatial branch (DINOv2 and ResNet-50):
- Sampling rate: 4 fps for all experiments. 8 fps is not planned (see the cut order in PHASES.md).
- Resize: short side to 448 pixels, preserving aspect ratio.
- Crop: center crop to 448x448. Edge content is lost for non-square videos. This is documented as a limitation, especially for portrait LIVE-VQC videos.
- Normalization: ImageNet mean and standard deviation.

Motion branch: excluded from active extraction and serving. The completed SlowFast transform/weight checks remain historical Phase 0 evidence, including 32 boundary-replicated frames, reference alpha-4 pathway indices, bilinear 256 resize/crop and Kinetics normalization. No motion features are included in the active cache or feature comparison.

Precision:
- Offline extraction runs on GPU under fp16 autocast. Cached arrays are stored as fp16.
- Serving runs on CPU in fp32, since fp16 inference on CPU is slow or unsupported for these backbones.
- The resulting drift is measured by the serving parity check in section 4, not assumed to be zero.

## 3. Feature Extraction

Spatial features:
- DINOv2 ViT-B/14, selected and measured in Phase 0, pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Use the variant without registers. If a registers variant is ever used, register tokens are excluded from all patch statistics. Load through torch.hub pinned to a specific repository commit, recorded in meta.json.
- Outputs per frame from a single forward pass: last-layer CLS token, plus mean of patch tokens and standard deviation across patch tokens from several blocks (6, 9, 12 on ViT-B, 1-indexed, which is 0-indexed 5, 8, 11; adjusted for ViT-L). Patch statistics exclude the CLS token.
- Intermediate tokens come from get_intermediate_layers with norm=True, so every block passes through the final LayerNorm and all blocks share one scale. The pinned implementation and real pretrained outputs were checked in Phase 0; record these settings in meta.json.
- Intermediate blocks are hypothesized to keep more low-level texture and blur information than the last layer, and the std-patch vector keeps spatial-variation information that a mean discards. The separate layer comparison is cut. Cache these outputs anyway, but make no claim that intermediate/std statistics improve quality. The research DINO representation is fixed to last-layer CLS plus block-12 patch mean before results exist.
- 448 is a multiple of the patch size 14 (32x32 patches).
- ResNet-50 (ImageNet), run on the same 448x448 crops so the comparison with DINOv2 does not confound backbone with input resolution. Global average pool output. Control only; never concatenated with other features.

Active feature sets: DINOv2 last-layer CLS plus block-12 patch mean, and ResNet-50 as a separate control. ResNet is never concatenated with DINO. All cached DINO mean/std branches remain available without being independently selected or tuned. SlowFast features are excluded. Actual dimensions and weight hashes are recorded in `configs/features_v1.yaml` and the Phase 0 backbone evidence.

## 4. Cache Layout and Parity Checks

    cache/
      v1/
        konvid/
          dino_cls/              {video_id}.npy  shape [T, D] fp16
          dino_patchmean_L{n}/   {video_id}.npy  shape [T, D] fp16   (one directory per chosen block)
          dino_patchstd_L{n}/    {video_id}.npy  shape [T, D] fp16   (one directory per chosen block)
          resnet50/              {video_id}.npy  shape [T, D] fp16
        livevqc/
          (same structure)
        manifest.csv
        meta.json

manifest.csv has one row per video and feature set, with columns:
video_id, dataset, feature_set, T, mos, mos_scale, source_size_bytes, source_sha256, duration_s, fps, width, height, rotation_applied, feature_sha256, extraction_env (local or cloud), status (ok or failed), failure_reason.

meta.json records: backbone names, weight sources and hub commit, get_intermediate_layers settings, fps, resolutions, crops, normalization per branch, decoder and version, locked environment hash, GPU model per environment, and date.

Rules for the cache:
- Extraction is resumable. Skip a video if its file exists and its feature_sha256 matches the manifest.
- Every array is written atomically (write to a temp file, then rename), so a crash never leaves a partial file.
- Frames are decoded and processed in memory. Frames are not written to disk.
- Active cache storage projection is generated in `results/phase0/scoped_feasibility.json` from actual dimensions and inventoried timestep counts, retaining all planned DINO statistics and the ResNet control. No motion/UGC arrays are allocated.
- The cache is versioned. Changing any preprocessing parameter creates a new version directory.
- The DINOv2 blocks/statistics to cache were frozen in Phase 0. Adding a layer later means a new cache version and re-extraction.
- LIVE-VQC features may be extracted at any time, but no LIVE-VQC prediction is made until the KoNViD config is frozen (RULES.md Rule 2).
- YouTube-UGC is cut; no subset, cloud notebook or UGC cache is produced.

Parity checks (script 04_parity_check.py):
- Serving parity: run 20 KoNViD videos through the serving path (CPU, fp32) and compare against the cached GPU fp16 features. Report the minimum per-timestep cosine similarity per feature set and the per-video absolute prediction difference of the deployed head. Pass threshold: cosine at least 0.999 on every timestep. If it fails, investigate before Phase 4.
- Cloud parity is inactive while UGC is cut. Reinstatement requires a dated decision and the binding cloud gate before any UGC extraction.

## 5. Data Splits

- KoNViD-1k: 5 stratified folds by MOS quantile, generated once with a fixed seed. Index stored at data/splits/konvid_folds.csv and never regenerated.
- Within each training fold, an inner validation split of 15 percent, stratified by MOS quantile, is drawn once with a fixed seed and reused by every seed, head, and feature set. Training seeds change only initialization and batch order. Stored in the same index file.
- Known limitation: the fold-0 inner validation set (used for tuning) and every fold's inner validation set (used for feature selection) are drawn from videos that are test videos in other folds. This is disclosed in the README.
- LIVE-VQC: no folds. Used only as a held-out test set after the frozen config is selected.
- YouTube-UGC subset selection is excluded from the active plan.

## 6. Temporal Heads

All heads share one interface:

    input:  features [B, T, D], lengths [B]
    output: pooled MOS prediction [B]

Per-timestep scores are not part of the core interface. The mean-pool MLP has no per-step output, and the datasets have no frame-level labels. A per-timestep graph in the demo is an optional extra that needs a separate per-step readout.

Shared front end, identical across heads:
- LayerNorm, then linear projection D to H, with H fixed at 256 for all heads and feature sets. The DINO representation uses CLS plus block-12 patch mean; ResNet is a separate control. Input dropout and weight decay are tuned inside the same fixed budget. The projection is the only place feature dimension is reduced.

Heads:
- Mean pool MLP: masked mean over valid timesteps, then a 2-layer MLP to one scalar.
- Bi-GRU: bidirectional GRU on packed sequences (pack_padded_sequence), so the backward direction starts at each sequence's last valid step. Masked mean of outputs, then a 2-layer MLP.
- Transformer encoder: sinusoidal positional encoding (not learned, so length extrapolation is not blocked), key padding mask for variable length. Masked mean of outputs, then a 2-layer MLP.
- Mamba: pure-PyTorch selective scan (no custom CUDA kernels), run bidirectionally: a forward scan and a scan over each sequence reversed within its valid length (not a flip of the padded tensor), outputs re-aligned and combined. Masked mean, then a 2-layer MLP. Unit-tested against a naive recurrence. Causality tested separately for the causal variant used in a side experiment.

Fairness constraints:
- Head-only trainable parameters (everything after the shared front end) matched within about 5 percent across the GRU, Transformer, and Mamba heads, for each feature set. Matching is enforced inside the tuning search space: architecture size is tuned as a size tier (for example small and medium), and each tier has predefined per-head widths and depths that meet the match. Total parameters are reported too. The mean-pool MLP is matched where feasible by hidden width, otherwise its count is reported.
- Same loss: L1 on z-scored MOS (z-scored with training-fold statistics only) plus a pairwise rank term over in-batch pairs, with one rank weight shared by all heads. The rank weight is tuned first, with the mean-pool head on fold 0, then frozen before any head is tuned. Pairs with tied MOS are excluded from the rank term. No loss ablation unless time remains.
- Same optimizer, learning-rate schedule, batch size, and early-stopping rule.
- Same tuning budget (same number of trials, same fold used for tuning, same search method).

Training protocol (frozen in configs/protocol.yaml before tuning; values below are defaults to confirm after timing one run):
- Optimizer AdamW, cosine schedule with linear warmup.
- Batch size 64.
- Early stopping on inner-validation SRCC, patience 15 epochs, maximum 200 epochs. The best-epoch checkpoint is kept and its epoch number logged.
- Tuning: random search with a fixed seed, the same number of trials per head (10 trials per head under the 10-day scope, the same for every head). Search ranges are written in the config before tuning. Tuning defaults to seeds 0–4; a measured pilot no-go may activate seeds [0,1] for rank/head selection only, dated before tuning. Reported CV/generalization and shuffled-label controls always use all five seeds.
- Ridge (Baseline 0): mean-pooled, standardized features; alpha chosen from a fixed log-spaced grid on the inner validation split.

Length handling: every head masks padding. The mean pool uses the true length, not the padded length. A unit test checks that each head's prediction for a sequence is unchanged when extra padding is appended.

## 7. Training and Evaluation Pipeline

Phase 1 produces the definitive fixed-DINO CV runs. Phase 2 reuses their saved checkpoints, metrics and OOF predictions and adds ResNet control runs/statistical tables; it does not rerun unchanged DINO fits. LIVE-VQC features are extracted in Phase 1 Task 1, while LIVE-VQC prediction remains gated on the KoNViD configuration freeze.

For each head, feature set, fold, and seed:
1. Load cached features and MOS labels for the training fold.
2. Train with inner validation for early stopping. Log the final inner-validation SRCC (used for feature selection) and the early-stopping epoch.
3. Predict on the held-out fold. Save per-video predictions, config, seed, fold, and the checkpoint hash.
4. Compute SRCC, PLCC, and KRCC on the held-out fold.

Shuffled-label control: labels are shuffled within the training set of fold 0, the cheapest head is trained, and SRCC is computed on the held-out fold. It passes when the 95% bootstrap CI of its SRCC includes zero and its absolute SRCC is below 0.15.

Aggregation (done by script, never by hand):
- Headline: per-fold SRCC (and KRCC, PLCC) summarized as mean and standard deviation over the 25 fold x seed runs.
- For bootstrap comparisons, each video's out-of-fold prediction is averaged across seeds, giving one prediction per video per head, pooled across folds.
- Paired bootstrap over videos (10,000 resamples) gives 95% CIs on SRCC and KRCC differences between heads. PLCC is not bootstrapped.
- Each difference is classified with the decision table in PRD.md section 7. Holm-adjusted p-values are computed for the primary families.
- The bootstrap captures test-set sampling noise only, not training or fold variance, so intervals describe the seed-averaged ensemble and are somewhat optimistic. Seed and fold standard deviations are reported next to them.

Cross-dataset (Phase 3):
- Train each head (and ridge) on all of KoNViD-1k with the frozen config and the chosen feature set, 5 seeds. There is no validation split at this stage, so each head's epoch count is that head's median early-stopping epoch from CV. Training on the full set gives more steps per epoch than CV, which is a known imprecision and is noted.
- Predict on LIVE-VQC. Record the T distribution per dataset in the manifest and report it, since LIVE-VQC clips are somewhat longer than KoNViD.
- For PLCC: fit a 4-parameter logistic on LIVE-VQC predictions vs MOS, then compute PLCC on the fitted values. SRCC and KRCC use raw predictions.

## 8. Repository Layout

    vqa-ablation/
      configs/           protocol.yaml, frozen head and feature configs, one file per experiment
      data/
        splits/          fold and inner-validation indices (committed)
        manifests/       dataset manifests (committed)
        raw/             videos (NOT committed, may be deleted after caching)
      cache/             feature arrays (NOT committed)
      env/
        requirements.lock  locked local research environment
      scripts/
        00_timing.py
        01_make_splits.py          folds and inner-validation splits
        02_extract_spatial.py      DINOv2 and ResNet-50
        04_parity_check.py         serving feature/prediction parity
        05_train.py                heads, ridge, shuffled-label control (by config)
        06_eval.py
        07_bootstrap.py
        08_cross_dataset.py
        12_latency.py
        13_make_tables.py
      src/
        preprocess.py    shared with serving
        features.py      shared with serving
        heads/           mean_pool.py, bigru.py, transformer.py, mamba.py
        metrics.py
        logistic_map.py
      results/           per-video predictions, metrics, generated tables
      serving/
        api/             FastAPI app and same-origin HTML upload page
        Dockerfile       single CPU-only application image
      README.md

Scripts are numbered by stage and run in order. Each script takes a config file as input and writes to results/ or cache/. No script depends on state left by a previous script beyond its declared inputs.

## 9. Serving Architecture (Phase 4)

    browser: same-origin HTML upload page
        | GET /                  POST /upload
        v                            |
    one FastAPI process (one in-flight CPU job)
        1. validate bounded upload and decode limits
        2. shared preprocess.py and features.py, CPU fp32
        3. reported seed-0 full-KoNViD head checkpoint
        4. return score or useful error; delete temporary upload

One Dockerfile packages the API, page, CPU-only dependencies, pinned spatial weights and selected checkpoint. No Celery, Redis, separate worker, React build, SQLite queue or Docker Compose is part of this scope.

Rules for serving:
- Deploy the Phase 3 full-KoNViD seed-0 model for the reported spatial configuration with the highest headline CV SRCC meeting the accepted CPU budget. No separately quantized or unevaluated variant.
- Verify checkpoint/feature hashes at startup; use the same shared modules and active spatial preprocessing as offline.
- Admit at most one inference job at a time; reject busy requests rather than accumulating an unbounded queue. Keep the event loop responsive and bound upload size/duration and temporary storage.
- Clean uploads on both success and failure and document any retention policy.
- Test a clean Docker build/run. Measure extraction, head and request latency and pass the fixed-video serving parity check.

## 10. Reproducibility

- Every run records: config hash, seed, fold, code commit, cache version, locked environment hash, and checkpoint hash.
- Fix Python, NumPy and PyTorch seeds for all heads. Use `torch.use_deterministic_algorithms(True, warn_only=True)` for every head and retain warnings; unsupported deterministic cuDNN/RNN backward must not stop training solely for this reason. Document observed nondeterministic operations and the absence of a bitwise-reproducibility guarantee in the README.
- Research uses env/requirements.lock. Serving pins the corresponding CPU-only stack and must pass parity; there is no cloud notebook in this scope.
- A single command per phase regenerates the tables from saved outputs.

## Verified Phase 0 spatial implementation (2026-10-25)

The shared FFmpeg path decodes RGB24 sequentially with autorotation disabled and applies the ffprobe display-matrix rotation explicitly with counter-clockwise NumPy rotation. Unsupported non-right-angle rotations fail. Presentation timestamps are relative to the first presented frame; nearest-frame ties choose the earlier frame, including duplicate timestamps. Spatial targets exclude the duration endpoint. SlowFast uses `(arange(32)-15.5)*2/30` offsets and replicates boundary frames. The rolling buffer retains overlapping windows; FFmpeg errors and probe/decode frame-count mismatches fail the clip. Frames are never written to disk.

Spatial tensors resize their short side to 448 using float-tensor bicubic interpolation with antialias=True, then center crop and apply ImageNet normalization. The ResNet control shares these exact spatial tensors. Motion tensors normalize by 0.45/0.225, resize the short side to 256 using bilinear interpolation with align_corners=False and no antialiasing, center crop, and pack the reference slow/fast indices. A synthetic tensor test matches installed PyTorchVideo 0.1.5 reference scaling exactly. DINO population patch std uses correction=0; intermediate outputs request norm=True and class tokens separately.

See [official SlowFast transform](https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/) and [pinned DINO intermediate API](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py). Real portrait review, pretrained CPU/GPU smoke and fixed-sample timing have passed. The Phase 0 readiness report passed, and `configs/features_v1.yaml` is now frozen; `results/phase0/preprocessing_freeze.json` records the source/config/weight identities. The original readiness report describes the historical pre-freeze measurement settings, while this separate record establishes the active v1 freeze. Historical SlowFast details above do not enable motion in the active scope.

## Phase 1 matched implementation (2026-10-25)

The single `matched` tier in `configs/head_tiers.yaml` fixes width/depth before tuning. Actual front/head/total counts for both feature dimensions are generated in `results/phase1/head_parameters.json`; `tests/test_phase1_budget.py` enforces the head-only 5% bound, including the mean MLP.

Mamba uses two independent pre-LayerNorm residual blocks, aligned forward/backward within each valid prefix and averaged before masked pooling. Each uses a causal depthwise convolution, input-dependent B/C, positive softplus delta and the real recurrence `h=exp(delta*A)*h+delta*B*u`, `y=sum(C*h)+D*u`. Padding is sanitized before projection/convolution; inactive states freeze and outputs are zero. Float64 values/gradients are checked against independent scalar loops. A_log starts at log(1..16), D at one; dt weights are uniform ±rank^(-1/2), with inverse-softplus bias for log-uniform dt in [0.001,0.1]. The local implementation uses no extension or einops.

Inspected reference identity: state-spaces/mamba `e9594ce1c732d97440f0332fdc43170a2294dbfa`, [Mamba-1 block](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba_simple.py) and [selective scan](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/selective_scan_interface.py). The declared project optimizer applies the same weight decay policy to every head, including A_log/D.
