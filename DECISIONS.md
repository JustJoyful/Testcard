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
