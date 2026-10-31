# Decisions Log

Every change to the plan is recorded here with a date, what changed, and why (RULES.md, Rule 35).

## Current decisions (supersede historical entries below)

- Deadline: 2026-11-04 (10-day scope, Asia/Kolkata).
- Working allowance: ~8 h/day, equivalent to ~56 h/week and ~80 hands-on hours over the 10-day plan. Unattended GPU: ~10 h/night, approximately 100 hours over 10 planning nights.
- CPU latency budget: 120 seconds per 10-second video (accepted after Phase 0 measurements; head/API overhead still unmeasured).
- Decoder: FFmpeg/ffprobe; shared timestamp/rotation implementation verified.
- DINOv2: ViT-B/14 without registers, commit 7764ea0f912e53c92e82eb78a2a1631e92725fc8; actual pretrained CPU/GPU smoke passed.
- Active backbones: DINOv2 ViT-B and ResNet-50. SlowFast is cut; its successful preflight measurements are retained as historical evidence.
- YouTube-UGC: cut entirely, including metadata subset selection, notebook, cloud smoke/parity work and length experiments.

## 2026-10-25: plan revision after review

1. Shared loss changed from L1 on z-scored MOS to L1 plus a pairwise rank term, same weight for all heads.
   Why: training closer to the evaluation metrics (SRCC, KRCC) is standard in VQA work. Applied identically to every head, so the comparison stays fair. The earlier argument that L1 shrinks variance mainly affects PLCC, which the logistic mapping already handles, so no separate loss ablation is planned.

2. DINOv2 cache expanded: last-layer CLS, plus mean-patch and std-patch from several blocks (e.g. 6, 9, 12 on ViT-B), all from one forward pass.
   Why: intermediate layers and spatial variation may keep more low-level quality information than the last-layer mean. This is a hypothesis, tested by one dedicated layer comparison in Phase 2 with a fixed head. One winner is carried forward. The cache spec is decided before extraction because changing it means re-extracting.

3. Equivalence margin of +/-0.01 SRCC declared in PRD.md before any results.
   Why: lets a null result be stated precisely (equivalent, different, or inconclusive) instead of vaguely. The margin is a design choice. Actual CI widths are reported as measured. Intervals are described as covering test-set noise only, with seed and fold SDs reported alongside.

4. Rule 30 amended: deploy the best reported configuration that meets a stated CPU latency budget; a spatial-only configuration is acceptable.
   Why: SlowFast latency on CPU is unmeasured but likely large. Phase 0 timing sets the budget. No SlowFast toggle or quantized variant is built, since neither adds research value.

5. YouTube-UGC: one-clip-at-a-time download, verify, extract, delete. No URL streaming. Subset of 300+ clips if time allows, chosen from metadata only.
   Why: a dropped connection during streaming can silently produce truncated frame sequences. 150 clips would be too few to separate heads, so the target is 300+ with the CI width reported either way. Bucket URL and metadata columns must be verified manually in Phase 0.

## 2026-10-25: feasibility audit

6. Core head interface now outputs pooled MOS only. Per-timestep scores removed from the interface.
   Why: the mean-pool MLP has no per-step output, and the datasets have no frame-level labels. The demo graph is now an optional extra that needs its own per-step readout, built last.

7. Feature selection moved from held-out fold results to mean inner-validation SRCC across folds and seeds. Layer comparison uses the mean-pool MLP as the declared fixed head.
   Why: selecting on held-out folds and then reporting on them leaks. The mean-pool head compares features without a temporal confound.

8. Shared front-end width fixed at H=256. Parameter matching is on total trainable parameters across GRU, Transformer, and Mamba; the mean-pool MLP is matched where feasible, otherwise reported, along with head-only counts. (Superseded by item 20.)
   Why: matching all four heads within 5 percent was not realistic for a mean-pool MLP, and the front end dominates the total parameter count.

9. ResNet-50 control runs at the same 448x448 crops as DINOv2.
   Why: otherwise the feature ablation confounds backbone with input resolution.

10. Hyperparameters tuned on one fixed feature set (DINOv2 last-layer CLS plus mean-patch) and reused on others without re-tuning. Stated as a limitation.
    Why: keeps the tuning budget bounded. Re-tuning per feature set would multiply tuning runs several times over.

11. SlowFast spec made concrete (32-frame clip, about 2 s, one per timestep, boundary replicate) with a Phase 0 check that the weights load. Fallback is a torchvision video backbone, renamed honestly in tables.
    Why: the common SlowFast library can fail to install on recent PyTorch, and the earlier spec was ambiguous about window length.

12. Added Phase 0 tasks (real deadline and weekly hours, decoder choice, ViT-B vs ViT-L), an Effort and Compute Estimates section, and docker compose to Phase 4.
    Why: "fits the schedule" had no schedule, and Docker was in the original scope but missing from the task list. Estimates are guesses to be replaced with measurements.

13. Combined vqa-ablation-plan.md removed. The five separate files are the single source of truth.
    Why: two copies drift apart.

## 2026-10-25: second review and compute location

14. Compute location fixed: everything runs locally on the RTX 3050 laptop except YouTube-UGC feature extraction, which runs in a free cloud notebook (Kaggle primary, Colab fallback). Only UGC features (about 0.5 GB) come back; UGC training and analysis run locally. Rules 37 and 38 added.
    Why: the raw UGC clips do not fit local disk or data quota, while KoNViD-1k and LIVE-VQC fit locally and the cached-feature design makes local training cheap. A cloud bucket download is fast from a datacenter.

15. Research question reframed: the primary question is where the quality signal lives (features vs temporal head); the Mamba comparison is one part of it. RQ5 (UGC length) marked exploratory and underpowered. A "Realistic End State" section added to PRD.md.
    Why: the likely outcome is that features dominate and heads tie. The reframed question has a useful answer either way, and RQ5 at about 300 clips cannot resolve differences near the margin.

16. Motion branch given its own frozen preprocessing (dense ~15 fps grid, 256 resize and crop, Kinetics normalization), to be confirmed against the weights' reference transform in Phase 0.
    Why: the spatial-branch policy (448 crop, ImageNet normalization, 4 fps) does not match what SlowFast was trained on.

17. Precision policy split: fp16 autocast offline on GPU with fp16 storage; fp32 on CPU for serving. Serving parity check added (Rule 39).
    Why: fp16 inference on CPU is slow or unsupported, so serving features cannot be bit-identical to the cache. The drift is measured instead of assumed away.

18. Decoder must apply rotation metadata; verified in Phase 0 on LIVE-VQC portrait clips (Rule 40).
    Why: some decoders ignore the rotation flag, which would center-crop portrait videos sideways with no error.

19. DINOv2 extraction details fixed: non-register ViT, get_intermediate_layers with norm=True, CLS excluded from patch statistics, torch.hub pinned to a commit.
    Why: each of these silently changes the cached features if left unspecified.

20. Parameter matching moved from total to head-only parameters, per feature set, enforced through matched size tiers in the search space. Supersedes item 8.
    Why: with a ~2M-parameter front end, matching totals within 5 percent is almost automatic and does not constrain the heads; tuned depths and widths would otherwise break any matching.

21. Bidirectional heads reverse each sequence within its valid length (Bi-GRU on packed sequences); padding-invariance unit test required for every head.
    Why: flipping a padded tensor puts padding at the start of the reverse scan, and KoNViD's near-constant lengths would hide the bug until LIVE-VQC or UGC.

22. Training protocol made concrete in configs/protocol.yaml (15 percent stratified inner validation, early stopping on inner-validation SRCC, patience and max epochs, AdamW with cosine schedule, batch 64, random search with a fixed trial count per head, ridge alpha grid, z-scoring with training-fold statistics). The rank weight is tuned first with the mean-pool head and frozen before head tuning.
    Why: these were unspecified, and the shared rank weight had no defined owner when each head is tuned separately.

23. Shuffled-label control given a pass threshold: 95% CI includes zero and absolute SRCC below 0.15.
    Why: "materially above zero" had no number, and at about 240 test videos the SRCC standard error is around 0.06.

24. Statistical decision rules unified into one four-category table (PRD.md section 7; Rules 16 and 17 rewritten). 95% two-sided CIs kept and stated as conservative relative to TOST. Holm-adjusted p-values added for the primary families (Rule 41). PLCC removed from the bootstrap. Headline (mean per-fold SRCC) and comparison (seed-averaged pooled ensemble) numbers both defined and labeled.
    Why: the old Rules 16 and 17 could give opposite verdicts on the same CI, and the number of comparisons was unaccounted for.

25. Tuning-overlap limitation disclosed: inner validation sets contain videos that are test videos in other folds.
    Why: this was disclosed for feature selection but not for hyperparameter tuning.

26. Phase 3 now explicitly extracts LIVE-VQC features (allowed before the config freeze; no predictions until then) and uses the chosen feature set for all four heads plus ridge. Phase 0 extraction projection covers LIVE-VQC too.
    Why: no phase previously owned LIVE-VQC extraction.

27. UGC subset restricted to 720p and 1080p originals.
    Why: the README note assumed 720p or 1080p, but the old selection rule only excluded 2160p.

28. Deployed checkpoint defined as the Phase 3 full-KoNViD model, seed 0 (declared now), for the reported configuration with the highest headline KoNViD CV SRCC that meets the CPU latency budget. Rules 9, 28, and 30 updated.
    Why: CV headline numbers are means over 25 runs, so no single CV checkpoint "has" them; the full-KoNViD model has its own reported LIVE-VQC numbers. Declaring the seed in advance avoids picking the best seed.

29. Manifest extended with source and feature checksums, rotation, fps, resolution, extraction environment, and failure status. Frames are processed in memory, never written to disk.
    Why: resumable extraction and Rule 6 depended on checksums the manifest did not store.

30. Serving details: API and worker share one Docker volume for uploads and SQLite; CPU-only PyTorch wheels; weights from the pinned hub commit.
    Why: unspecified before, and each would break a clean build or a working demo.

31. Repository scripts renumbered and completed (splits, parity check, UGC subset and cloud fetch, length test, latency), plus a cloud notebook wrapper and env/requirements.lock. Storage and input-dimension figures corrected (largest model input about 7,700, since ResNet-50 is never concatenated). Run estimate raised to about 680 to include tuning. 8 fps variant added to the cut order as not planned.
    Why: several planned steps had no script, and the estimates left out tuning runs.

## 2026-10-25: Phase 0 execution attempt — blocked preflight

- Phase 0 remains incomplete; Phase 1 must not begin. Task 1 could not establish a verified locked environment. No preprocessing policy is frozen and no timing or feasibility result is claimed.
- Git inspection found an empty, read-only `.git/`; `git status` fails. Preserve it without initializing over unknown metadata. Worktree setup, task commits and commit-range review are unavailable; files and the plan ledger remain in the current workspace.
- Created an isolated `.venv/` targeting the already installed Python 3.11.15 as a conservative PyTorchVideo/notebook candidate. Package installation failed because the execution shell cannot resolve `pypi.org`. `env/requirements.in` is an unresolved direct dependency list; `env/requirements.lock` is deliberately absent until actual compatible packages can be resolved and installed. SciPy is required to read LIVE-VQC's MATLAB label source.
- FFmpeg/ffprobe are installed and proposed as the shared decoder; timestamp, rotation and transform verification remain pending. Installed executable versions and hardware evidence are in `results/phase0/environment.json`.
- `nvidia-smi` cannot communicate with the NVIDIA driver. GPU memory, CUDA runtime availability and GPU inference cannot be verified. CPU execution cannot stand in for mandatory GPU measurements.
- Deadline is October 31, 2026 (Asia/Kolkata). The user can leave the laptop open for unattended runs. Numeric weekly labor hours, unattended compute hours, disk reserve and user-accepted CPU budget remain unset. Optional UGC remains pending an access/schedule decision; no cloud tooling or extraction was started.
- KoNViD's video archive contains no MOS metadata; obtain the separate official metadata source and verify the one-to-one join before sample selection. LIVE-VQC's archive includes `data.mat`; labels have not yet been parsed or joined.
- Consulted official references: [torchvision compatibility](https://github.com/pytorch/vision#installation), [PyTorch installation](https://pytorch.org/get-started/locally/), [SlowFast reference](https://github.com/pytorch/hub/blob/master/facebookresearch_pytorchvideo_slowfast.md), [KoNViD source](https://database.mmsp-kn.de/konvid-1k-database.html), and [LIVE-VQC source and terms](https://live.ece.utexas.edu/research/LIVEVQC/index.html). These references do not establish local package or pretrained-weight compatibility. The KoNViD page describes research access and Creative Commons source videos; exact per-source license terms remain to be recorded. LIVE-VQC requires preserving its full copyright notice and acknowledging/citing its source; retain the archived readme.
- Resume at Task 1 after package access and CUDA are restored. Complete lock installation/import checks before proceeding to the inventory, preprocessing, model and timing tasks. Retain existing archives and do not delete raw data.

### Host verification supplied after the initial attempt

The user's shell successfully ran `nvidia-smi`: RTX 3050, 6,144 MiB, driver 615.78.08, CUDA UMD 13.4. The driver is available on the host; the earlier failure describes agent execution access, not an absent host driver. The user also successfully ran `git init`. Package connectivity and PyTorch CUDA runtime access still require verification. KoNViD's existing video ZIP is present and verified; the missing item is its separate MOS metadata, not the videos.

### Metadata join and installed environment verification

KoNViD_1k_metadata.zip is now present. Verified its CRC and exact one-to-one match of video filename stems to `flickr_id` in `KoNViD_1k_mos.csv`: no duplicates, missing labels, extra label rows or out-of-range/nonfinite MOS. The attributes table also covers the same IDs, but its MOS values are rounded; use the dedicated full-precision MOS table. Evidence and source checksums: `results/phase0/konvid_label_join.json`. This supersedes the earlier missing-MOS blocker; full clip metadata inventory remains pending.

The user's dependency install succeeded. Agent-side `uv pip check` passes, and torch, torchvision, NumPy, YAML, pytest and SciPy import successfully. Observed torch 2.14.1+cu130 and torchvision 0.29.1+cu130. `torch.cuda.is_available()` remains false inside the agent environment despite working host `nvidia-smi`. Host-terminal execution is a viable candidate for GPU measurements; it must import the same shared code and use the same lock/config. No training or GPU workload has started, and package consistency does not confirm SlowFast or pretrained weights work.

## 2026-10-25: Phase 0 implementation and scope updates

- The installed 49-package environment is now pinned in `env/requirements.lock`; a fresh `.venv-check` installed those pins offline using a writable copy of the host uv cache, passed `uv pip check`, and imported the required libraries. `uv pip install/check` substitutes for pip because uv environments do not include pip. Notebook compatibility remains unverified.
- Git initialization succeeded on the host but agent writes to `.git/` remain blocked; an actual worktree attempt failed. Keep changes in place and retain the plan ledger; do not claim commits or branch isolation.
- Inventory uses metadata-only ffprobe to avoid decoding every frame merely to obtain duration/fps/resolution. Extraction still probes and samples full presentation timestamps. Inventory success is not full-decode verification. The initial full-PTS inventory run was interrupted and rerun without changing originals.
- SlowFast's official PackPathway uses `torch.linspace(0,31,8).long()` (indices 0,4,8,13,17,22,26,31), rather than literal every-fourth sampling. Follow the official reference, with bilinear interpolation, align_corners=False and no antialiasing; synthetic tensor equivalence is tested against the installed reference functional implementation. Population patch standard deviation uses correction=0.
- DINOv2 ViT-B remains the candidate. Pin official inspected commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; inspected its intermediate-layer API, norm=True behavior and separation of patch/class/register tokens. Real pretrained loading is still required before confirmation.
- Add `scripts/00_backbone_check.py --config configs/phase0.yaml` as the host-terminal entry point for downloading pretrained weights, recording hashes/dimensions, and running one real sample on both GPU fp16 and CPU fp32. The timing CLI refuses to benchmark unconfirmed backbones.
- The user describes hands-on availability as “whenever I'm free”; record this qualitatively and leave numeric schedule inputs unset. Do not claim the schedule gate passes.
- At the user's request, cloud access/parity/smoke checks move to the start of Phase 3 UGC, after LIVE-VQC results. Phase 0 may lock a metadata-only subset; if that blocks core work it may be skipped. The official metadata and attribution URLs could not be retrieved in this environment, so metadata schema/subset are not fabricated. No UGC raw clip is downloaded, and Phase 3 parity remains mandatory before any UGC extraction.

### Review fixes and verified portrait handling

- Independent review identified that hashing decision-only config fields would invalidate expensive measurements when recording the CPU budget afterward. Preserve the full config hash for audit, but validate measurement reuse with an explicit execution fingerprint (seed, samples, datasets, cache version, decoder settings, preprocessing, backbone settings and timing policy), plus code/lock/weight/manifest hashes. Only confirmation flags, schedule/feasibility, UGC decision and CPU budget are excluded. Runtime-input changes still invalidate evidence. This refines the plan's stale-evidence policy without permitting changed extraction settings to reuse timings.
- Publish completed inventory CSVs atomically, and bind inventory summaries to exact CSV hashes. Reports reject interrupted/stale manifests instead of using partial timestep totals to underestimate disk requirements. Regression tests for both review findings failed before the fixes and pass afterward; the full offline suite passes.
- Reserve 5 GiB of free disk beyond projected incremental artifacts as the initial engineering allowance. Retain the source archives and the small fixed preflight sample; do not delete raw clips merely because timing succeeded.
- Full decodes succeeded for six real selected clips at differing frame rates. The user visually confirmed decoded LIVE-VQC E001, G001 and G002 previews are all upright; source hashes, rotations, inspected timestamps and human verdicts are saved in `results/phase0/rotation_review.json`. No decoded frames were written to disk.

## 2026-10-25: Real pretrained checks and timing verified

Host `00_backbone_check.py` completed successfully for pretrained DINOv2 ViT-B, ResNet-50 and SlowFast R50 on both CUDA fp16 autocast and CPU fp32. Actual weight hashes and output dimensions are saved in `results/phase0/backbones.json`. No fallback is required.

`results/phase0/run.json` is complete; the fixed-sample timing CSV has complete successful coverage and current execution/code/lock/manifest/weight hashes. Measurement-derived estimates are saved in `results/phase0/feasibility.json` and `measurement_summary.md`; extraction/storage estimates in PHASES.md were generated from those outputs. These do not establish later training or API latency.

The user accepted a CPU budget of 120 seconds per 10-second video to leave room for motion features. This is a deployment selection constraint; Phase 4 must still measure actual extraction plus head/API overhead. Schedule allowances remain pending; do not freeze preprocessing or start Phase 1 while the report is no-go.

The remaining Phase 0 manual review/documentation is assigned a one-hour engineering planning allowance, explicitly excluding later phase development/training. Numeric user availability remains pending.

## 2026-10-25: User-approved 10-day scope and deadline

Each cut below supersedes conflicting historical scope above and is reflected in PHASES.md, ARCHITECTURE.md, PRD.md, RULES.md and the active configs.

1. **Schedule:** deadline changed to 2026-11-04. User availability is approximately 8 hands-on hours/day and 10 unattended GPU hours/night for the 10-day plan. Record 56 hands-on hours/week, 80 total hands-on hours and approximately 100 unattended GPU hours. These are planning allowances, not measured training capacity or a guarantee that every later phase fits.
2. **Cut SlowFast:** exclude motion feature extraction, DINO+motion combinations and motion serving. Keep completed preflight checks/timings/weight identities as historical evidence; do not delete existing weights or archives. Active extraction uses only DINOv2 and ResNet-50. Cost: this study cannot claim a motion-feature comparison.
3. **Cut YouTube-UGC:** exclude metadata subset selection, cloud setup/smoke/parity work, UGC extraction and length experiments from all phases. This supersedes the earlier Phase 3 deferral. Cost: no length-extrapolation result or claim. Serving parity remains mandatory.
4. **Skip the separate layer comparison:** cache DINO CLS plus block-6/9/12 patch means and population standard deviations anyway, but evaluate the predeclared DINO representation (CLS plus block-12 mean) against ResNet-50. No layer winner is selected after seeing results. Cost: the cached intermediate/std hypotheses are not tested in this scope.
5. **Reduce tuning to 10 trials per head:** keep equal search budgets, fixed fold-0 inner validation and the common rank-loss protocol for all four heads. Freeze search ranges after first-run timing in Phase 1. Cost: the smaller search may under-tune a head; disclose this limitation. Five folds, five seeds, ridge, shuffled-label controls, required correctness tests and confidence intervals remain required.
6. **Simplify serving:** FastAPI serves an HTML upload/result page and performs one CPU inference job at a time, packaged by one Dockerfile. Remove the planned Celery/Redis worker architecture, React build, SQLite job queue and Compose. Bound uploads/admission, preserve cleanup, use shared preprocessing/features and enforce serving parity. Cost: the demo has no durable asynchronous job queue and supports one in-flight job.

Active policy is in `configs/scope.yaml` and `configs/features_v1.yaml`. Keep `configs/phase0.yaml` as the settings for the completed historical measurement envelope; update only schedule/decision inputs there. Its conservative three-backbone report can pass using the new allowance without rerunning or relabeling the 360 saved measurements. The smaller active extraction/cache projection is generated separately from those same saved outputs. Do not change source/measurement hashes or remove the SlowFast rows to make the reduced scope appear to have been measured originally.

### Phase 0 closure under the revised scope

The conservative historical readiness report passes with the new schedule allowances. The spatial-only projection is generated in `results/phase0/scoped_feasibility.json`; `scope_report.md` summarizes it. Active spatial preprocessing is frozen for cache v1 in `configs/features_v1.yaml`, with config/code/lock/weight identities in `results/phase0/preprocessing_freeze.json`. The immutable readiness snapshot records eligibility before this freeze, rather than relabeling the original benchmark configuration as cache v1. No feature arrays have yet been extracted into v1.

The first Phase 1 training run per head remains a separate feasibility gate for the full research grid. These extraction measurements do not guarantee the remaining development, tuning, training, evaluation and serving fit the deadline. Existing installed SlowFast dependencies/weights are retained to preserve locked preflight reproducibility; no new SlowFast work is scheduled. No training, fold generation or LIVE-VQC predictions were started while applying these scope changes.

## 2026-10-25: Phase 1 plan revision after schedule review

These are requested revisions to the implementation plan; execution still awaits approval of that revised plan.

1. **Reuse definitive DINO runs:** Phase 1's five-fold/five-seed DINO grid is the real head-comparison training, not disposable diagnostics. Phase 2 loads these same completed runs/OOF predictions, adds ResNet control fits and completes the statistical tables. No second unchanged DINO grid is scheduled.
2. **Move LIVE-VQC extraction into Task 1:** queue its DINO/ResNet feature extraction immediately after KoNViD in the overnight slot. This changes only extraction timing. LIVE-VQC labels/results cannot guide tuning or feature/model selection, and no LIVE-VQC predictions are allowed until the KoNViD configuration freeze.
3. **Remove unnecessary infrastructure:** one manually scheduled extraction writer, no writer lock; ordinary pytest verification, no tests.json/JUnit-hash exit gate; config/split/cache hashes per run, no multi-layer compatibility-fingerprint system. Keep atomic writes, checksum manifests, immutable splits, model/recurrence/padding tests, leakage controls and test-fold isolation. Required lock/weight/checkpoint identities and code commit remain audit fields.
4. **Warning-only determinism:** use torch.use_deterministic_algorithms(True, warn_only=True) uniformly across heads, retain warnings and disclose nondeterministic operations in the README. A cuDNN GRU backward determinism limitation must not become an avoidable training blocker. No bitwise reproducibility claim.
5. **Conditional tuning-only seed cut:** default remains five tuning seeds. If Task 7's measured runtime gate is no-go, the preauthorized fallback is seeds [0,1] for both rank-loss and head selection, identical across heads and recorded in protocol/DECISIONS.md before tuning. This reduces selection work only; five-seed CV, generalization and shuffled-label controls are unchanged. The fallback is **not activated by this planning revision**.
6. **Calendar checkpoint:** Phase 1 targets October 30 end of day. If Task 7 (pilot) is not complete by October 28 end of day (Asia/Kolkata), stop and cut scope before tuning; a GPU-hour projection cannot waive this checkpoint. Reserve the remaining project days for ResNet, LIVE-VQC, the demo and honest reporting.

Frozen feature configs, recorded measurements and Phase 0 hash artifacts are unchanged. No Phase 1 extraction, splits or training have been run while revising the plan.

## 2026-10-25: Phase 1 execution started

- The user authorized execution of the revised core-slice plan. Git and CUDA access now work in this environment. Preserved the verified Phase 0 checkout in a baseline commit and created `phase1-core-slice`; source archives, weights, raw clips and cache arrays stay ignored. Work remains in the original directory on this named branch so existing ignored inputs are available.
- Created and committed immutable outer/inner splits; provenance and generated role counts are in `data/splits/konvid_folds.meta.json`. Repeated split invocation verifies the same files without rewriting.
- Implemented the planned cache, data, heads, trainer, ridge, selection, CV and parity interfaces. Generated parameter counts are in `results/phase1/head_parameters.json`. The frozen preprocessing/features modules are unchanged.
- Real fixed-sample GPU extraction passed for both spatial backbones. Full KoNViD → LIVE-VQC extraction is running sequentially. Hardware completion, controls, training pilot, runtime approval, tuning, CV/ridge and parity remain pending; Phase 1 is not complete and the protocol remains draft. No tuning-seed fallback has been activated. Consult generated extraction evidence and the execution ledger rather than treating offline tests as hardware evidence.
- Code commits are audit fields rather than numerical compatibility keys: unrelated additions do not invalidate already verified caches/runs. A fitting correction requires a dated affected-run invalidation and a new shuffled-label control.

### Phase 1 implementation review

An independent whole-branch review identified restart and evidence-binding issues. These were reproduced and fixed before any real head training: retain immutable pilot/selection snapshots across audit-only commits and the protocol freeze, validate authoritative control roles/MOS/permutation, bind the final control to the selected mean-pool config, and replace stale success evidence on Git prerequisite failure. Preselection CPU ridge timing and dated stop decisions are implemented. See `docs/superpowers/reviews/2026-10-25-phase-1-review.md` for evidence. Frozen feature production is unchanged; no cached arrays are invalidated by these pre-training consumer fixes.

### Phase 1 real feature parity verified

The fixed-sample CPU fp32 versus cached GPU fp16 parity artifact passed and was revalidated against current source/reference checksums. Evidence is in `results/phase1/parity.json`. This does not complete Phase 1: full cache extraction, controls, pilots, tuning and definitive CV/ridge remain pending in the live sequential queue. No LIVE-VQC predictions were made.

## 2026-10-26: Repair variable-frame-rate decoder output timing

The six LIVE-VQC failures (A078, A080, A106, B001, B004, B054) are rawvideo output-muxer timestamp collisions, not source timestamp regressions or corrupt decodes. The original command exits zero but logs non-monotonic output DTS after rounding distinct source timestamps to its default encoder time base. Matching the demuxer time base removes this error without dropping frames. Per-source RGB stream hashes/frame counts from both commands are saved in `results/phase1/decoder_diagnosis.json`; a generated VFR fixture reproduces the failure independently.

Before changing shared preprocessing, apply Rule 24: activate a new v2 feature/cache policy and regenerate all feature arrays and fixed-sample parity. Preserve v1 config/freeze/evidence and cached arrays; do not rewrite the original freeze hashes or silently relabel v1 arrays. No head fitting has started, so no accuracy experiment exists to invalidate. Frozen splits, source archives, sampling/rotation/normalization, pretrained weights, head sizes and selection protocol stay the same. Training remains gated on complete v2 caches and regenerated parity. Previous extraction attempts remain in the compute accounting.

The minimal decoder change is `-enc_time_base demux` on the raw RGB output. Retain strict FFmpeg diagnostics, `-xerror`, decode-length and frame-count validation. Never suppress the DTS text or ignore unrelated errors. The active feature config declares its versioned freeze record; cache validation must bind config and freeze versions. This supersedes the plan's v1 path references for active commands only; historical Phase 0 evidence remains unchanged.

Reference: [official FFmpeg encoder time-base documentation](https://www.ffmpeg.org/ffmpeg.html#Advanced-options). FFmpeg documents the default video encoder time base as reciprocal frame rate and `demux` as the input demuxer time base.

### Decoder repair verified and v2 queue resumed

All six failed IDs pass the shared decoder and both real GPU backbones. Checksum-bound per-branch evidence is generated in `results/phase1/decoder_repair/verification.json`; the independent focused review found no Important/Critical issues. Fresh v2 feature and parity runs precede the existing leakage/pilot/runtime/selection/CV gates. The original failed v1 evidence remains under `results/phase1/decoder_repair/`; no clips, branches, folds or seeds were dropped, and no old arrays were relabeled.

## 2026-10-27: Final control failed; definitive CV remains stopped

All shared-rank and equal-budget head tuning fits completed, with frozen selected configs. The selected mean-pool final shuffled-label control completed all five seeds but failed: SRCC 0.12918040273838305, 95% CI [0.004278085064662253, 0.25522434250352]. The absolute-SRCC condition passes; the interval excludes zero.

Under Rule 10, no CV/ridge fits or LIVE-VQC predictions are launched. The saved permutation, seeds, split indices, selected configs and pass criterion are unchanged. Do not reroll the permutation, pick another config from the held-out control result, or relabel this result as passing.

Read-only main and independent audits verify role separation, authoritative IDs/MOS, fixed training permutation, training-only target statistics, checkpoint/config bindings, best-epoch selection and exact bootstrap reproduction. Independent CPU inference agrees with saved GPU predictions and validation scores. No concrete implementation defect was detected; this does not prove chance caused the failure or establish absence of every possible leakage mechanism. No fitting correction or post-hoc threshold change is justified by these checks.

The existing gate remains failed. Further experimental progress requires a resolved cause or an explicit, transparently documented research-protocol decision; a generic request to continue is not authorization to bypass it. Machine-readable and human-readable audits are saved under results/phase1/control_final/. Offline verification: `.venv/bin/python -m pytest -q` → 147 passed.

## 2026-10-27: User-approved negative-control v2 amendment

The user explicitly approved “Approve the correction and complete Phase 1” for the written design `docs/superpowers/specs/2026-10-27-negative-control-repair-design.md`. The original training-only final control remains failed and unchanged; audits found no concrete code defect. True validation MOS was an allowed checkpoint-selection supervision path, but its contribution to the particular failure is not proven.

Active negative controls now independently permute training and validation targets within their original roles, seed 0 separately for each role, and use true held-out test labels. The original training permutation is reused exactly; the validation permutation is persisted once. Five model seeds, optimizer, rank weight, fixed head configs, bootstrap and numerical pass criteria remain unchanged. New result directories `control_initial_v2/` and `control_final_v2/` bind policy and both permutation hashes, and cannot resume legacy controls as complete. Original controls stay in their original directories with explicit historical disclosure.

Only control construction and evidence checks change. Shared numerical fitting, feature caches/freeze, immutable splits and all selected real-label configs/rank/tuning artifacts remain unchanged and reusable. The valid measured budget is verified/reused on continuation rather than rewriting projections for already completed selection work. Both amended controls must pass once before definitive CV/ridge; any failure remains a stop. Reports disclose this post-hoc amendment and the original failure, even if v2 passes.

### Cache verification lookup optimization

During the amended real controls, repeated manifest scans were identified as an integrity-check bottleneck. Coverage and dataset loading now index the already-read manifest once by dataset/video/branch. Each key keeps every matching row, preserving duplicate rejection and all existing status/source/checksum/shape/dtype/finite checks. The list-based path remains supported. Dataset ID/feature/label order and all numerical fitting behavior are unchanged; existing controls/selection/real-fit artifacts remain reusable. A regression failed before the helper existed, the full suite passes, and a read-only comparison on actual arrays is bitwise equal. The focused independent review is saved in `docs/superpowers/reviews/2026-10-27-manifest-lookup-review.md`.

## 2026-10-27: Phase 1 exit verified

The generated phase gate passes. Definitive DINO/ridge artifacts are complete and must be reused in Phase 2. Saved GPU job wall hours: 10.942467243436393; upper-bound remaining planning allowance from saved attempts: 89.0575327565636 hours. These are job wall measurements, not kernel utilization; hands-on time remains unmeasured. See results/phase1/summary.json, time_accounting.json and phase_gate.json. No LIVE-VQC predictions were made.

## 2026-10-27: Shuffled-label control evidence closed; v2 standard for later feature sets

This entry closes the control-design decision before Phase 2. Values below are generated directly from the preserved saved outputs; no model was retrained, permutation rerolled or pass threshold changed. The original failed result remains failed.

**Original selected-config control (training targets shuffled; true validation targets).** The five-seed averaged prediction on the 240-video outer test fold has SRCC 0.12918040273838305, with 95% video-bootstrap CI [0.004278085064662253, 0.25522434250352]. The CI excludes zero, so the control failed despite absolute SRCC below 0.15. The diagnostic records 10000 valid bootstrap draws and 0 invalid draws. Source: [original diagnostic](results/phase1/control_final/diagnostic.json).

The inner-validation scores below are measured against **true validation MOS at each seed’s selected epoch**, not shuffled-validation scores or held-out test scores. Each value matches the selected history entry and the maximum saved validation SRCC; each selected checkpoint checksum was verified.

| Seed | Best epoch | Inner-validation SRCC at selected epoch | Saved run |
| ---: | ---: | ---: | --- |
| 0 | 6 | 0.25354790580931924 | [seed 0](results/phase1/control_final/seed0/run.json) |
| 1 | 5 | 0.17477905790631865 | [seed 1](results/phase1/control_final/seed1/run.json) |
| 2 | 19 | 0.1666256360575032 | [seed 2](results/phase1/control_final/seed2/run.json) |
| 3 | 7 | 0.12358955632604288 | [seed 3](results/phase1/control_final/seed3/run.json) |
| 4 | 18 | 0.18718504091660654 | [seed 4](results/phase1/control_final/seed4/run.json) |

Their arithmetic mean is 0.1811454394031581. All five selected validation scores are positive. This is the positive-validation pattern, rather than near-zero selected validation scores accompanying a positive test result.

**Mechanism and interpretation.** Although gradient fitting used permuted training targets, checkpoint selection maximized SRCC against real validation labels. This left a real-label supervision path: early stopping could select a checkpoint whose feature-to-quality association aligned with true validation MOS, and that association could also carry to the test videos. The positive selected validation scores support this explanation. They are selected maxima, not unbiased estimates or a formal significance test; these observations do not prove this mechanism uniquely caused the original test association or prove absence of every possible leak. Existing role/permutation/checkpoint/CPU-inference/bootstrap audits found no concrete implementation defect; see [the independent audit](results/phase1/control_final/independent_audit.md).

**Post-hoc amendment and observed result.** The user approved policy v2 on 2026-10-27 **after the original control failed**. It independently permutes training and validation targets within their own roles, with a separate fixed seed-0 RNG for each permutation, while retaining true test labels/IDs. The original training permutation, frozen model configurations, five model seeds and numerical thresholds were retained. Both amended controls were run once; their saved test-fold results are:

| Amended control | Test-fold ensemble SRCC | 95% bootstrap CI | Gate |
| --- | ---: | --- | --- |
| [Initial/default](results/phase1/control_initial_v2/diagnostic.json) | -0.10300331599991715 | [-0.2220614423993075, 0.021783227062860476] | passed |
| [Final/selected](results/phase1/control_final_v2/diagnostic.json) | -0.1189693161394431 | [-0.23995964728281755, 0.0066654817834086826] | passed |

**Settled standard for later feature sets.** Under Rule 10, policy v2 is now the standard shuffled-label control for **every later feature set, including ResNet in Phase 2**, and every new pipeline change. Use fold 0 and the cheapest mean-pool head, independently shuffle targets within training and validation, preserve true held-out test targets, retain all five model seeds and fixed permutation provenance, and require both default and selected-config controls to pass. The unchanged conjunction is a 95% bootstrap CI including zero and absolute SRCC below 0.15, using 10,000 seed-0 video-bootstrap draws. Do not reuse a DINO control as evidence for ResNet; any new failure stops further work for investigation without searching permutations, seeds, configs or thresholds for a pass. Preserve and disclose the original failure and this post-hoc amendment in later reports. The control-design decision is closed on this basis; this entry does not run or certify the future ResNet control.


## 2026-10-27: Deployment selection rule fixed before LIVE-VQC evaluation

The user fixed the deployment rule during Phase 2 plan review, before ResNet comparison fits or LIVE-VQC predictions: **highest headline KoNViD CV SRCC among reported configurations, ridge included, within the 120 s CPU budget per 10-second video.** The candidate pool includes both reported spatial representations and ridge alongside all four neural heads; it is not restricted to the validation-chosen representation. Headline SRCC is the saved mean of per-fold results, not the seed-averaged pooled ensemble score. The research feature-selection rule remains based only on inner validation.

Rank candidates by that saved CV headline score. For exact score ties, prefer lower measured CPU latency, then a stable candidate identifier. Eligibility requires measured end-to-end CPU latency within the budget; missing measurements are unresolved, not evidence of eligibility. Resolve eligibility and the deployment winner before LIVE-VQC evaluation, using KoNViD/CPU evidence only. LIVE-VQC labels, predictions or metrics must not influence the rule, candidate ordering or winner.

The Phase 2 KoNViD freeze must preserve this rule, budget, candidate configurations/ranking and provenance, plus Phase 3 refit inputs: each head's median early-stopping epoch on the chosen features and ridge's alpha selection/full-refit policy. It must also retain refit inputs for a deployment candidate from either reported representation. Deployment uses the corresponding full-KoNViD seed-0 artifact, with five-seed reporting; a CV checkpoint is not a deployment replacement. A candidate outside the validation-chosen representation needs its own reported full-KoNViD refit before deployment.

This records the policy now; it does not claim an eligible winner, create a completed freeze, or report results that do not yet exist. The corrected Phase 2 implementation plan remains local and awaits approval. Actual epoch/alpha values and candidate scores will be generated from validated saved outputs. The plan must pause after Task 6 offline checks and review pass, allowing the user to commit before any Task 7 execution; no automatic commit or training launch is authorized by this entry.

## 2026-10-27: Phase 2 aggregation and inference predeclared before ResNet fitting

Phase 2 equally weights the four neural heads, five folds and five seeds: the mean of all 100 best inner-validation SRCCs for each spatial representation. Ridge validation scores are reported separately and do not affect feature choice. Exact feature-selection ties choose DINO. Held-out predictions/metrics are excluded from the selector; the residual overlap between selection/tuning videos and other folds' test videos remains disclosed.

All paired comparisons share one canonical video-ID order and one seed-0 bank of 10,000 video-bootstrap draws. Predictions are averaged across the five seeds before resampling, and SRCC is reranked within each resample. The approximate two-sided p-value is (1 + count(abs(bootstrap_delta - observed_delta) >= abs(observed_delta))) / (valid_draws + 1), labeled centered-bootstrap/add-one, not an exact permutation test. At least 99% of the original draws must be finite; undefined draws are not replaced. Only the six chosen-feature neural SRCC pairs form the primary KoNViD Holm family. KRCC, ridge-minus-neural and feature comparisons remain secondary/unadjusted. CIs are pointwise 95% intervals on the seed-averaged ensemble's test-sampling noise, separate from Holm-adjusted p-values and training/fold variability.

Phase 3 refit inputs will be generated for both reported representations: each neural head's integer median best epoch over its 25 CV fits, and ridge's median of its five CV-selected alphas. Ridge retains the fixed grid, inner-validation SRCC selection with ties to smaller alpha, and training-only feature/target standardization; no new full-data alpha search or validation split. The existing all-ten-candidate deployment rule and measured CPU eligibility requirement remain binding. No experimental outcome or completed freeze is claimed by this predeclaration.

## 2026-10-28: Phase 2 exit verified

Generated validation-only selection: resnet50. Mean inner-validation SRCC by representation: {"dino": 0.7157960881764193, "resnet50": 0.7636316431083497}. Source-bound tables, paired CIs/Holm, full-refit inputs and the KoNViD freeze are saved in results/phase2/. Deployment CPU eligibility remains unresolved and must be resolved before LIVE-VQC predictions. Original failed control and post-hoc v2 amendment are disclosed in summary.json. No LIVE-VQC predictions were made.

## 2026-10-28: Phase 3 LIVE-VQC secondary comparisons fixed before prediction

Run full-KoNViD fixed-epoch refits for all four heads on DINO as well as the selected ResNet representation, with five seeds each, plus one frozen-alpha ridge fit per representation. Report DINO LIVE-VQC metrics as secondary regardless of the CPU deployment winner. Compare DINO with ResNet for the four same-head pairs and ridge, and compare ResNet ridge with each of its four neural heads. Use the shared 10,000-draw video index bank for SRCC and KRCC intervals and centered-bootstrap p-values; label all nine comparisons secondary and unadjusted. The six ResNet neural SRCC pairs alone form the Phase 3 primary Holm family. LIVE-VQC predictions and labels do not affect refit settings or CPU winner selection.

The KoNViD CPU eligibility benchmark uses four PyTorch CPU threads. Save `cpu_threads: 4` in both CPU evidence and `deployment.json`; Phase 4's single FastAPI inference process must call `torch.set_num_threads(4)` before model work and measure serving latency again. The active Phase 4 scope has no Celery worker. This decision updates the Phase 3 plan before any LIVE-VQC prediction; it does not claim an eligible winner or measured LIVE-VQC result.

## 2026-10-28: Phase 3 fixed-epoch control stop

The KoNViD-only CPU benchmark completed and resolved `resnet50/ridge` as the eligible deployment candidate; the measured grid and winner are in [`cpu_eligibility.json`](results/phase3/cpu_eligibility.json). The new fixed-epoch shuffled-label control stopped at seed 3 because it failed both unchanged pass conditions. Its per-seed scores and intervals are in [`diagnostic.json`](results/phase3/control_fixed_epoch/diagnostic.json); seed 4 was not run.

An audit of the saved artifacts confirmed separate train and validation permutations, untouched fold-0 test labels, the frozen configuration and epoch, matching checkpoint and code hashes, and an independently reproduced seed-3 bootstrap interval. These checks found no concrete implementation defect and do not prove the absence of a leak. The Phase 3 gate remains failed. Full-KoNViD refits and LIVE-VQC predictions have not run; no seed, permutation, configuration, threshold, or winner was changed to seek a pass.

## 2026-10-28: Phase 3 qualified continuation after fixed-epoch control failure

The user authorized a **qualified continuation** after the fixed-epoch control failed. The original [`diagnostic.json`](results/phase3/control_fixed_epoch/diagnostic.json) remains **failed**, with seed 3 failed. Seed 4 was finished once using the same frozen three-epoch configuration, fixed train/validation permutations, seed, bootstrap and numerical rule. Its independent [`seed4_result.json`](results/phase3/control_fixed_epoch/seed4_result.json) is also **failed**. No threshold, seed, permutation, configuration or CPU winner was changed. This is the **second post-hoc control amendment** in the project, after the Phase 1 v2 amendment; it authorizes continuation and does not turn this control into a pass.

| Seed | Fixed-rule result | Trained test SRCC | Trained 95% CI | Untrained test SRCC | Trained vs. untrained prediction Pearson r |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | passed | -0.1011 | [-0.2280, 0.0277] | -0.3124 | -0.0135 |
| 1 | passed | -0.0214 | [-0.1466, 0.1028] | 0.2151 | 0.0714 |
| 2 | passed | -0.0656 | [-0.1901, 0.0586] | 0.0240 | 0.0515 |
| 3 | **failed** | -0.2322 | [-0.3466, -0.1136] | -0.1724 | 0.0955 |
| 4 | **failed** | -0.2361 | [-0.3540, -0.1106] | -0.1843 | 0.4849 |

The **Phase 1/2-comparable figure**, computed from the mean of the five trained predictions per held-out video using the same 10,000 seed-0 video-bootstrap draws, is SRCC **-0.1563**, 95% CI **[-0.2779, -0.0306]**. It also fails the original conjunction and is **not a substitute pass**. The five-seed untrained-initialization ensemble is SRCC **-0.2027**, 95% CI **[-0.3230, -0.0747]**; untrained seeds 0, 1, 3 and 4 individually fail the fixed rule. The untrained result demonstrates poor specificity of a fixed absolute-SRCC cutoff on this fold.

The proposed explanation that three-epoch trained predictions are dominated by their untrained initialization is **not supported by the direct prediction correlations**. The trained versus untrained Pearson correlations above imply less than 1% shared prediction variance for seeds 0–3 and about 24% for seed 4. Their test-SRCC signs agree for seeds 0, 3 and 4 but disagree for seeds 1 and 2. Removing each seed's linear initialization component leaves trained SRCCs -0.1056, -0.0376, -0.0679, -0.2145 and -0.1851. Thus initialization correlation cannot explain away the trained failure; the negative trained ensemble association remains an unresolved control limitation. The complete per-video trained and untrained rows, hashes, bootstrap diagnostics, sign check and ensemble scores are in [`investigation.json`](results/phase3/control_fixed_epoch/investigation.json) and [`untrained_baseline.json`](results/phase3/control_fixed_epoch/untrained_baseline.json).

The label-path audit found no mechanism carrying validation or held-out LIVE labels into model updates, frozen hyperparameter selection, CPU winner selection or LIVE score computation. Fixed-epoch full refits use all 1,200 KoNViD MOS values by design; the control verifier reads fold-0 test MOS only to evaluate the preserved failed control. The refit and LIVE prediction entry points now omit LIVE MOS from their in-memory context, and the LIVE feature dataset and prediction rows omit MOS. `predict` batches features without labels. Only the reporting stage attaches LIVE MOS and fits the reported PLCC mapping. This is a data-flow finding, not proof that every possible leak is absent.

Phase 3 may proceed to full refits, LIVE predictions and reporting only with a gate whose status is **`passed with qualification`**, whose `control_status` remains **`failed`**, and whose hashes include this decision and all control evidence. The reported results must disclose the failed per-seed and ensemble control. For future studies, predeclare an untrained-initialization null baseline and compare trained controls against that distribution instead of relying on a fixed |SRCC| threshold; do not change the current project's rule retroactively.

## 2026-10-30: Approved Phase 4a static decoder packaging amendment

The exact host/cache FFmpeg `n9.0.2` static archive was unavailable in the checked distributions. Before any serving parity measurement, the user approved BtbN `n9.0.2-37-gae19868753` from the same release branch plus 37 maintenance commits. Use only the immutable dated archive URL:

https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-29-19-17/ffmpeg-n9.0.2-37-gae19868753-linux64-gpl-9.0.tar.xz

Full archive SHA256: `83537b0f6a97905f325080f6f5916e70ec252d1ef3e4ce4746f755b88720cbf5`. The verified binaries report `n9.0.2-37-gae19868753-20261029`. This replaces only the exact-version packaging prerequisite; the model, weights, frozen preprocessing, cached features and research results are unchanged. Record complete host and image `ffmpeg -version` and `ffprobe -version` output side by side, with archive and image identities.

All parity thresholds remain exactly predeclared: minimum cosine at least 0.999 for every aligned timestep, absolute cached-reference/raw score difference at most 0.01 MOS, and CLI/API absolute score difference at most 1e-6. If parity fails, stop and report the failure. The fallback is building tag `n9.0.2` from source in a Docker builder stage; thresholds must not be loosened. The request-latency budget is unchanged. No fallback or result is claimed before measurement.

Historical gates retain their original decision-log identity. The original bytes are preserved in `results/phase4a/decisions_research.md`, and `results/phase4a/packaging_amendment.json` binds that exact historical prefix, this approved appended entry and the current complete log. Phase 3 remains passed with qualification; its shuffled-label control remains failed.


## 2026-10-31: Approved Phase 4b local demo and in-process polling

The user selected the local Docker demo first, preserving Phase 4a media/resource limits and adding in-process polling. Extend the existing single-process FastAPI service with one admitted `/upload` job and `/status/{task_id}`; retain `/predict` and the exact shared prediction JSON. No hosting, paid resources, Celery, Redis, Compose or Gradio. Model, weights, sampling, CPU fp32 and four torch threads remain unchanged.

Accept checked MP4/MOV/WebM, at most 104857600 file bytes, 105906176 body bytes, 60 seconds receipt, 30 seconds duration, 2073600 source pixels, 1920 longest side and 60 fps. Additional submissions get 503 before body reads. Accepted jobs own admission and temporary files through completion, including caller cancellation. Delete video bytes on completion/error; retain at most 32 terminal metadata records for 900 seconds in process memory, with no durable jobs or server-cancellation promise.

Preserve Phase 4a evidence/image as historical proof; new evidence belongs to results/phase4b/. Per-timestep cosine >=0.999, cached/raw score difference <=0.01 MOS, front-door difference <=1e-6, and maximum normalized warm time <=120 seconds per ten seconds are unchanged. If parity fails, stop and report; only then consider the approved tag-n9.0.2 Docker-builder fallback. Phase 3 remains passed with qualification, its original control remains failed, and supplementary post-hoc diagnostics do not replace that failure.

Mandatory execution checkpoint: finish Tasks 1-2 and present backend verification plus current DESIGN.md, then STOP before Task 3. Do not create serving/web/, install frontend packages/fonts, scaffold, mock up or edit a page until the user explicitly resumes frontend work. The implementation request does not waive this checkpoint. No commits, pushes or publication are authorized by this execution.

## 2026-10-31: Post-hoc exploratory pre-declaration — why DINOv2 lost to ResNet-50 on KoNViD

**Post-hoc and exploratory.** This analysis is declared after the Phase 1–3 results were seen. It is diagnostic only and changes no frozen configuration, deployed model, phase gate, headline result or existing artifact. All outputs go to `results/posthoc_dino/`. No variant, metric or comparison may be added after any result below is computed. All comparisons are unadjusted.

Question: is quality information present in DINOv2's earlier layers and lost at the last layer, or is the DINO deficit relative to ResNet-50 model-wide?

### Part A: DINO layer comparison (ridge only, cached `cache/v2` features, CPU)

Exactly five variants. Branches are concatenated in the listed order, per frame, then mean-pooled over frames:

- **A0**: `dino_cls` + `dino_patchmean_L12` (the Phase 1/2 DINO ridge feature set).
- **L6**: `dino_patchmean_L6` + `dino_patchstd_L6`.
- **L9**: `dino_patchmean_L9` + `dino_patchstd_L9`.
- **L12**: `dino_patchmean_L12` + `dino_patchstd_L12`.
- **ALL**: `dino_cls`, `dino_patchmean_L6`, `dino_patchstd_L6`, `dino_patchmean_L9`, `dino_patchstd_L9`, `dino_patchmean_L12`, `dino_patchstd_L12` (7 branches).

Protocol, identical to the Phase 1/2 ridge:
- Same immutable folds and inner-validation splits (`data/splits/konvid_folds.csv`).
- Frames are cast to float32 and mean-pooled in float64.
- `src/baselines.fit_ridge` is used, with training-only feature and target standardization.
- Alpha is chosen from the grid {0.001, 0.01, 0.1, 1, 10, 100, 1000} by inner-validation SRCC, smaller alpha winning ties.
- One deterministic fit per fold, on the fold's train role; the test role gives out-of-fold predictions.

Each fold's selected alpha is recorded. A selection of 0.001 or 1000 is flagged as a grid-edge hit and reported as a limitation. The grid is not extended.

**A0 reproduction check.** A0's per-fold test SRCC must match the per-fold SRCC recomputed from the saved Phase 1/2 DINO ridge predictions (`results/phase1/ridge/fold*/predictions.json`) within an absolute difference of 1e-9 in every fold. If it does not, stop and report before any other variant is reported.

Reported:
1. **Per-fold results:** mean ± SD of per-fold SRCC and KRCC for each variant, next to the saved Phase 2 ResNet-50 ridge and DINO ridge rows.
2. **Paired bootstrap** of pooled out-of-fold predictions. It uses 10,000 draws from one shared seed-0 index bank (`src.metrics.make_bootstrap_indices`) over the KoNViD inventory order, with SRCC and KRCC as A minus B, a two-sided 95% CI, unadjusted, and classification by the PRD §7 table with margin 0.01. The ResNet-50 ridge OOF predictions are the saved Phase 2 ones. The comparisons are exactly these nine:
   - L6, L9, L12 and ALL vs A0.
   - A0, L6, L9, L12 and ALL vs ResNet-50 ridge.
3. **LIVE-VQC (exploratory):** for each variant, one full-KoNViD ridge fit (`fit_ridge_fixed_alpha`) at the median of its five CV-selected alphas, the Phase 3 rule. It is evaluated on all 585 LIVE-VQC videos, reporting SRCC, KRCC and logistic-mapped PLCC fitted on LIVE labels, as in Phase 3. LIVE results feed back into nothing.

Interpretation rule, fixed now: the KoNViD gap is the ResNet-50 ridge mean per-fold SRCC minus A0's mean per-fold SRCC. A variant "closes most of the gap" if its mean per-fold SRCC recovers more than 50% of that gap. If L6, L9 or ALL does so, the earlier-layer hypothesis is supported; otherwise the deficit is called model-wide. The bootstrap classifications are reported alongside and qualify this, and mixed evidence is stated as mixed.

### Part B: distortion sensitivity (optional, GPU, run by the user)

Frames:
- Decode the 20 fixed Phase 0 KoNViD IDs (`data/splits/phase0_konvid_ids.txt`) with the shared `src/preprocess.py` sampling path.
- Pool the sampled frames and choose 50 without replacement using `numpy.random.default_rng(0)`.

Distortions, applied to the decoded uint8 RGB frame before `spatial_tensor`:
- Gaussian blur σ ∈ {1, 2, 4}, via `scipy.ndimage.gaussian_filter` over the spatial axes only.
- Additive Gaussian noise σ ∈ {5, 15, 30} on the 0–255 scale, from `default_rng(0)`, clipped and rounded to uint8.
- JPEG re-encode at Pillow quality ∈ {50, 20, 5}.
- The clean frame.

Branches: ResNet-50 pooled, DINO CLS L12, and DINO patch-mean L6, L9 and L12, extracted with the shared `src/features.py` backbones.

Metric: for each branch and level, the mean (1 − cosine) between each clean frame and its distorted version, divided by the mean (1 − cosine) over all pairs of different clean frames. Higher means the branch reacts more to degradation relative to content.

Output: a CSV, plus one figure with the ratio against distortion level, one line per branch and one panel per distortion type. Part B can support or contradict Part A; mixed evidence is stated as mixed.

### Deliverables and checks

- CSV/JSON for every number, a short `report.md` and figures, all generated from saved outputs, with no hand-typed numbers.
- After the analysis, all Phase 1–4a gate-listed hashes are verified unchanged.
- `NOTES.md` records the findings and limitations: post-hoc, multiple variants, alpha-grid edges and the A0 reproduction check.

This entry is the only post-hoc append. Its prefix and bytes are pinned in `results/posthoc_dino/decision_append.json`, which `src/serving_bundle.verify_snapshot` checks.
