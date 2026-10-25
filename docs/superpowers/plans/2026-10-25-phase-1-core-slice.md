# Phase 1 Core Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phase 1 by 2026-10-30: cache both datasets, produce the final DINO head-comparison CV runs, and verify ridge, leakage control and parity within the approved spatial-only scope.

**Architecture:** Keep the frozen spatial decoder/backbones unchanged. Add a small atomic cache, immutable split/data utilities and independent head modules; a shared training function handles one run, while numbered scripts orchestrate extraction, tuning, evaluation and parity. Expensive jobs run locally from Kitty using the same checkout, environment and configs; code editing continues only in files those jobs do not consume.

**Tech Stack:** Existing Python 3.11, PyTorch/torchvision, NumPy, SciPy, PyYAML and pytest; FFmpeg/ffprobe. No scikit-learn, Mamba CUDA extensions, tracking service, cloud service or new dependency is needed.

**Spec:** `PHASES.md` Phase 1; `ARCHITECTURE.md` §§2–7; `RULES.md`; `PRD.md`; current decisions in `DECISIONS.md`; `configs/scope.yaml`; frozen `configs/features_v1.yaml`; `results/phase0/preprocessing_freeze.json`. These together are the spec; this plan proposes the implementation details they leave open.

## Global Constraints

- “The test fold is never used for tuning, early stopping, feature selection, or any other decision.”
- “LIVE-VQC is not used for predictions until the KoNViD-1k config is frozen.” Task 1 extracts LIVE-VQC features after KoNViD; LIVE-VQC prediction and model selection remain forbidden throughout Phase 1.
- “Fold and inner-validation indices are generated once and committed. They are never regenerated.”
- “Every reported accuracy experiment runs with 5 seeds.” Use `[0,1,2,3,4]` for reported CV/generalization results and shuffled-label controls. Tuning defaults to those five seeds; if Task 7 is no-go, the approved fallback is `[0,1]` for rank/head tuning only, recorded before tuning in DECISIONS.md. Tuning trials and seed-0 timing pilots are not reported accuracy results.
- “The declared DINO representation is last-layer CLS plus block-12 patch mean”; retain all seven planned DINO cache branches, without a layer comparison.
- “The mean-pool MLP is matched where feasible, and otherwise its count is reported.” GRU/Transformer/Mamba head-only counts must satisfy `max/min <= 1.05`, separately from the common front end.
- “The rank weight is tuned first, with the mean-pool head on fold 0, then frozen before head tuning.” Head tuning is exactly 10 candidate configurations per head, same search space/budget and fixed inner-validation indices.
- “Offline extraction runs on GPU under fp16 autocast. Cached arrays are stored as fp16.” CPU feature parity uses fp32; head training uses fp32 for all heads.
- “Frames are decoded and processed in memory. Frames are not written to disk.” Preserve source archives and user-owned raw files.
- “Offline, serving, and cloud-notebook code import the same preprocess.py and features.py. Duplicating these files is not allowed.” Cloud work is cut; retain the shared local/serving implementation.
- “No head may see padded timesteps as signal.” Reverse each sequence within its valid length, not across its padded extent.
- Shuffled-label gate: two-sided 95% SRCC bootstrap CI includes zero **and** absolute SRCC `<0.15`; failure stops further experiments until investigated.
- Project deadline `2026-11-04`; Phase 1 target `2026-10-30` (Asia/Kolkata); planning allowances 80 hands-on hours and 100 unattended GPU hours. Measure training before promising the full grid fits. Do not reduce reported-run seeds, folds, controls, parity or CIs. Any two-seed tuning fallback must be dated and applied equally to all heads.
- SlowFast, all YouTube-UGC work, separate layer comparison and serving implementation are excluded. Phase 2 adds ResNet training and completes the statistical tables by reusing Phase 1 DINO outputs; it must not rerun the same 100 DINO fits. Deployment remains later work.
- Keep config, split and cache hashes per run, plus the required lock/weight/checkpoint hashes, code commit, seed, fold and label mode for audit. No multi-layer compatibility-fingerprint system. No manually entered result numbers.

## Review Focus

1. Interrupted extraction and orphan arrays: no incomplete video is reusable, valid cache entries are preserved, and failure of one dataset cannot certify the other. Tests belong to Task 1. Extraction uses one manually scheduled writer; no lock implementation.
2. Equal MOS, label-file reorder and stale split inputs: stratification is deterministic; train/validation/test never overlap; zero target variance cannot silently become a valid experiment. Tests belong to Tasks 2 and 5.
3. Unequal sequence lengths and hostile padding: every head agrees with per-sequence unpadded inference, including reverse scan and convolution boundaries; causal sequence embeddings cannot depend on a future suffix. Tests belong to Tasks 3–4.
4. Leakage through normalization, early stopping, tuning or resume: held-out labels cannot affect trained weights/config choice, and incompatible or partial results cannot be resumed as complete. Tests belong to Tasks 5, 8 and 9.
5. Undefined correlations, constant features and boundary parity: diagnostics retain failures by ID rather than hiding NaNs, accepting zero-norm vectors or averaging away a bad timestep. Tests belong to Tasks 6 and 10.

---

## Starting State, Decisions and Execution Order

Phase 0 passed: source inventories cover 1,200 KoNViD and 585 LIVE-VQC clips; real spatial pretrained CPU/GPU checks and the original 360 timing measurements passed. `configs/features_v1.yaml` freezes the active spatial policy; no full feature cache, folds, heads, training scripts or protocol config exists yet. The current tests are `tests/test_phase0_*.py`.

Use Tasks 1–10 in order, with GPU work overlapping independent implementation. Task 1 sequentially extracts KoNViD DINO/ResNet and then LIVE-VQC DINO/ResNet in the same overnight queue while Tasks 2–6 implement independent files. Keep Task 1 hardware work marked pending in the ledger while Tasks 2–6 use temporary synthetic caches for code/tests; return to verify both real caches before any real training. No two extraction writers run at once; no writer lock is built. Never edit `src/cache.py`, `src/preprocess.py`, `src/features.py`, the extraction script, its config or lock while that extraction is running. Task 7 applies both the runtime gate and the calendar checkpoint below. A no-go may activate only the documented two-seed tuning fallback; reported CV remains five seeds. If that still cannot fit, stop and cut scope explicitly rather than letting implementation consume the week.

Implementation choices proposed for approval with this plan:

- One matched head-size tier, approximately 0.41M head-only parameters, avoids selecting incompatible size tiers independently. Tune optimizer/dropout settings, not an unconstrained parameter count.
- Default tuning: ten configurations/head × five seeds = 200 head fits total; four rank-weight candidates × five seeds = 20. On measured no-go only, use tuning seeds `[0,1]` for 80 head fits and eight rank fits, log the cut, then recheck runtime. Final DINO CV remains 100 fits across five seeds; controls retain all five seeds.
- Default training cap/patience remain the architecture's 200/15, batch 64, with a common five-epoch LR warmup; change those only after the pilot and a dated, explicit global protocol decision.
- Reserve 30 GPU hours for Phases 2–4 and contingency when checking the Phase 1 budget. This is a planning allowance, not measured future runtime.
- Use an interpretable rank loss: mean L1 plus lambda times mean pairwise logistic loss over unequal-target pairs. Cache and output scores remain video-level.

### Calendar Checkpoints (Asia/Kolkata)

- **October 25–26:** implement Task 1, launch both datasets' extraction sequentially overnight, and build Tasks 2–6 while the GPU works.
- **Hard checkpoint — 2026-10-28, end of day:** “If Task 7 (pilot) isn’t done by Oct 28, stop and cut scope.” Record the missed checkpoint and a concrete cut/revised schedule in DECISIONS.md **before** starting tuning. Unfinished infrastructure is not a reason to silently consume the remaining week. The optional two-seed tuning cut cannot fix a missing implementation by itself.
- **Phase 1 target — 2026-10-30, end of day:** complete the definitive DINO CV/ridge runs, controls, parity and normal pytest check. If the target slips, explicitly revise downstream scope/time; do not call partial CV complete.
- **October 31:** Phase 2 adds ResNet control fits and final comparisons, reusing all valid Phase 1 DINO artifacts.
- **November 1–2:** full-KoNViD models and LIVE-VQC evaluation from the already extracted features, only after the KoNViD configuration freeze.
- **November 3–4:** minimal demo, serving parity, clean Docker check and README.

These are calendar checkpoints checked in the execution ledger, not a scheduler, lock service or test-attestation system. The runtime gate cannot override the October 28 stop-and-cut checkpoint.

### Provenance and Git Preflight

- Validate the immutable Phase 0 readiness snapshot, freeze record, active config, lock, the frozen shared-module checksums and actual weight hashes. Do not rerun `00_report.py` as a Phase 1 gate after adding modules: its historical broad `src/*.py` fingerprint will change even when the frozen preprocessing does not. Preserve historical evidence and verify the existing freeze record rather than adding a new fingerprint system.
- Do not change frozen preprocessing/features to remove the read-only-array warning or optimize away unused motion-window construction in this phase. Consume the spatial frame from the existing iterator and ignore the third item; never invoke a motion backbone. A real preprocessing change requires a versioned policy and affected verification, not a rewritten old hash.
- Git currently has no commits and the agent cannot write `.git`. At execution, use the worktree skill and existing isolation if available. If Git is still blocked, retain code/tests and ledger in place, and give host-terminal Git commands. Do not run `git init` over metadata or claim commits happened.
- Before any real training, the host must have committed the baseline and split artifacts. Initial baseline staging must explicitly name `.gitignore`, planning docs, `configs`, `env`, `src`, `scripts`, `tests`, `data/manifests`, `docs` and small `results/phase0` evidence; verify archives/raw/cache/weights are ignored. Create a named Phase 1 branch or isolated workspace using the worktree skill. Resolve actual working-directory paths before giving Kitty commands. In a new worktree, `.venv`, source ZIPs, weights and uncommitted raw/cache data are not checked out: link the existing ignored inputs explicitly (or configure their absolute paths), keep manifests/configs/source in the worktree, and schedule only one writer for the cache.
- Commit steps below apply when Git writes work; otherwise record the exact pending paths/subject, and perform the commit from Kitty before consuming immutable artifacts. No merge, push or publishing is part of this plan.

## File Map

| Files | Responsibility |
|---|---|
| `src/cache.py`, `scripts/02_extract_spatial.py` | Feature artifact schema, atomic writes, validation/resume and spatial extraction |
| `src/splits.py`, `scripts/01_make_splits.py` | Immutable outer folds and inner-validation indices |
| `src/data.py` | Checked feature dataset and variable-length batching |
| `src/heads/common.py`, `mean_pool.py`, `gru.py`, `transformer.py`, `mamba.py`, `__init__.py` | Shared front end, masking, four independent heads and factory |
| `src/losses.py`, `src/training.py` | One shared loss and one fit/early-stop/checkpoint loop |
| `src/metrics.py`, `src/baselines.py` | Correlations, disclosed logistic PLCC mapping, bootstrap diagnostic and deterministic ridge |
| `src/tuning.py`, `scripts/05_train.py` | Bounded pilot/rank/tuning/control/CV orchestration |
| `scripts/04_parity_check.py`, `scripts/06_eval.py` | Feature parity, final reusable DINO result summaries and exit report |
| `configs/protocol.yaml`, `head_tiers.yaml`, `rank_loss.yaml`, `tuning_trials.yaml`, `heads/*.yaml` | Declared, then frozen training/selection inputs; no measured results typed into configs |
| `data/splits/konvid_folds.csv`, `konvid_folds.meta.json`, `phase0_konvid_ids.txt` | Immutable folds/provenance and existing 20-video parity IDs |
| `cache/v1/{konvid,livevqc,manifest.csv,meta.json}` | Ignored feature arrays and live manifest; small verified snapshots under `data/manifests/` are commit-eligible |
| `results/phase1/` | Run outputs, checkpoints (ignored), timings, controls, ridge, metrics, parity and report |
| `tests/test_phase1_*.py` | Offline tests using temporary synthetic features/model doubles; host hardware gates remain separate |

No service framework, generic experiment scheduler or new dependency. Add `*.pt`/`*.pth`/`*.npz` under `results/` to `.gitignore`; keep run JSON, CSV and small manifests eligible for commit.

### Configuration and Artifact Contracts

Task 5 creates `configs/protocol.yaml` with these exact top-level fields: `schema_version: 1`, `status: draft`, `feature_set: dino`, `cache_root: cache/v1`, `inventory: data/manifests/konvid.csv`, `splits: data/splits/konvid_folds.csv`, `head_tiers: configs/head_tiers.yaml`, `device: cuda`, `cpu_threads: 4`, `seeds: [0,1,2,3,4]`, `batch_size: 64`, `max_epochs: 200`, `patience: 15`, `warmup_epochs: 5`, `learning_rate: 0.0003`, `weight_decay: 0.001`, `input_dropout: 0.1`, `head_dropout: 0.1`, `rank_weight: 0.1`, `rank_candidates: [0,0.1,0.3,1]`, `gradient_clip_norm: 1.0`, `results_root: results/phase1`, `trials_per_head: 10`, `tuning_seed: 0`, `tuning_seeds: [0,1,2,3,4]`, `deterministic_warn_only: true`. Task 7 changes status to `frozen` after the runtime gate; selected shared lambda lives in Task 8's generated rank config. Any changed default is dated and frozen before tuning.

`head_tiers.yaml` has one `matched` mapping: `front_width:256`, `mean_pool_hidden:1598`, `gru_hidden:156`, `transformer_layers:1`, `transformer_nhead:4`, `transformer_ff:256`, `mamba_layers:1`, `mamba_inner:230`, `mamba_state:16`, `mamba_dt_rank:16`, `mamba_conv:4`, `readout_hidden:64`. Task 3 initially supplies the conventional-head fields; Task 4 completes/validates the mapping. `build_head` receives a config with the resolved `tier` mapping plus `input_dropout` and `head_dropout`; selected head configs embed these resolved values so inference does not depend on a mutable search file.

CLI modes share `--config` and `--features`; `single` additionally requires head/fold/seed/output. Other modes use `results_root` with the directories named in their tasks. Failed mode prerequisites exit nonzero with named missing gates. Trial-result rows use `candidate_id,seed,validation_srcc,status` (status `complete|failed`) and provenance; CV prediction rows include `run_kind: cv` as well as `label_mode: real`. Control rows have `run_kind: control_initial|control_final` and `label_mode: shuffled`.

`fit_one` returns `status,checkpoint_path,run_json_path,best_epoch,best_validation_srcc,target_stats`; it loads the selected weights back into the supplied model. Save plain state dictionaries/primitive metadata, not pickled model objects; validate checkpoint checksums before `torch.load(..., weights_only=True)`. Ridge array artifacts use ignored `.npz` files (no pickle); small JSON metadata records their hashes.

All numbered script invocations below run from repository root as `.venv/bin/python scripts/<script-name> ...` (the absolute active worktree root must be used in Kitty). They are future interfaces to implement, not commands available at planning time.

Keep run reuse simple: a saved run must be complete, its checkpoint checksum must verify, and its recorded immutable run-config, split and cache hashes must match the requested inputs. Save the resolved run config once per run; do not edit it after completion. Record code commit, lock and weight identities for audit. A real bug fix that changes fitting requires an explicit decision about which runs to rerun and a new leakage control, not a compatibility-migration framework. Pilot configs remain their own snapshots and are not resumed as CV runs. No module-by-module numerical/selection fingerprints, test-attestation files or writer-lock infrastructure are added.

At every task's GREEN step, run its named test files and `.venv/bin/python -m pytest -q`; expected exit 0 with all existing tests still passing. Hardware commands are separate steps/evidence and cannot be satisfied by offline doubles. Tests below show a minimal concrete assertion per owner; implement the additional named failure cases in that owner's test step as well.

### Task 1: Atomic Spatial Cache and Extractor

**Files:** Create `src/cache.py`, `scripts/02_extract_spatial.py`, `tests/test_phase1_cache.py`, `tests/test_phase1_extract.py`, `data/splits/phase0_konvid_ids.txt`; modify `.gitignore`. Read existing `src/preprocess.py`, `src/features.py`, `scripts/00_inventory.py`; leave frozen modules unchanged.

**Interfaces:**
- Consumes `iter_samples(path: Path, config: dict)`, `spatial_tensor(frame)`, `load_backbone(name, device, config)`, `extract_features(model,samples,name,device)` and existing `sha256(path)`.
- Produces `validate_cache(root: Path, feature_config: Path, freeze_record: Path) -> dict` (verified metadata, or ValueError for incompatible/incomplete requested inputs); `write_video_features(root: Path, dataset: str, video_id: str, arrays: dict[str,np.ndarray], source_row: dict, meta: dict) -> list[dict]`; `load_cached_video(root: Path, dataset: str, video_id: str, feature_set: str) -> np.ndarray`.
- `feature_set='dino'` means `[dino_cls,dino_patchmean_L12]` concatenated in that order (`[T,1536]`); `'resnet50'` means `[T,2048]`. DINO cache branches are all seven outputs, each `[T,768]`; ResNet directory is `resnet50/`, not `resnet50_pooled/`.
- CLI `02_extract_spatial.py --config PATH --dataset {konvid,livevqc} --backbone {dino,resnet50} [--sample-ids PATH]`. A subset invocation cannot certify full-dataset coverage.
- `cache/v1/manifest.csv` uses ARCHITECTURE.md's columns, one row per video/branch. Meta records the frozen feature config, lock, weight/decoder identities, code commit, expected IDs/branches and separate dataset coverage. Existing checksum manifests validate source/array integrity; adding training code does not invalidate features. Do not build a separate producer-fingerprint layer.

- [ ] **Write failing cache tests:** `test_interrupted_multibranch_write_is_not_reusable`, `test_manifest_publication_is_atomic`, `test_checksum_shape_dtype_or_source_change_rejects_resume`, `test_subset_is_not_complete_dataset`, `test_konvid_completion_does_not_certify_livevqc`, `test_unsafe_video_id_is_rejected`. Assert an exception between the third and fourth DINO branch writes leaves no reusable complete record; a restart repairs the whole video; untouched valid files/rows remain byte-identical.
- [ ] **Write failing extraction tests:** `test_spatial_extraction_caches_all_seven_dino_branches`, `test_resnet_uses_same_spatial_tensor`, `test_failure_retains_id_and_cannot_shrink_expected_ids`, `test_existing_video_archive_unchanged`. Model doubles emit independently known features for two frames; assert eight expected branch mappings, fp16 storage, exact T, and no motion-loader call. A missing CUDA device fails clearly; doubles are test-only, with no production fake-weight option.

```python
def test_partial_cache_cannot_be_loaded(tmp_path):
    import numpy as np
    import pytest
    from src.cache import load_cached_video
    folder = tmp_path / "konvid/dino_cls"
    folder.mkdir(parents=True)
    np.save(folder / "v000.npy", np.ones((2,768), dtype=np.float16))
    with pytest.raises(ValueError, match="missing|incomplete"):
        load_cached_video(tmp_path, "konvid", "v000", "dino")
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_cache.py tests/test_phase1_extract.py -q`. Expected: failing tests for missing implementation.
- [ ] Implement the interfaces. Use file-handle `np.save(..., allow_pickle=False)` to a same-directory temporary file, flush/fsync and `os.replace`; verify finite arrays, dimensions, T and SHA-256 before atomically publishing manifest rows. Run one extraction command at a time; do not implement a writer lock. Read/modify/write CSV atomically; no database. A video is reusable only when all requested branches, the frozen feature config and source/array checksums verify. A crash/orphan `.npy` is not a successful video.
- [ ] Implement sequential archive-member materialization with source-size/SHA checks and safe member names; reuse verified user raw clips without deleting them. Only process-owned temporary copies may be cleaned; original archives are always retained. Record failures by ID and exit nonzero if any expected video fails; never drop it or regenerate the sample/folds. Materialize `phase0_konvid_ids.txt` from the existing config, rejecting a changed preexisting list.
- [ ] Run the two test files and the full suite; expected all pass. From Kitty run each backbone first with `--sample-ids data/splits/phase0_konvid_ids.txt`, validate features/checksums, then repeat without that flag for full KoNViD. Queue `--dataset livevqc --backbone dino` and then `--dataset livevqc --backbone resnet50` immediately afterward, with identical frozen preprocessing and no training/prediction call. Expected: all 1,200 KoNViD IDs and 585 LIVE-VQC IDs × eight branches valid, dataset coverage checked separately, no motion/UGC outputs, unchanged archives. The LIVE-VQC arrays are sealed for later Phase 3 prediction; only feature integrity/finite-value checks are allowed now. Save extraction timing and coverage to `results/phase1/extraction.json`; snapshot verified manifest/meta under `data/manifests/phase1_cache.csv` and `data/manifests/phase1_cache.meta.json`. Do not commit arrays.
- [ ] Commit the task's source/tests, `.gitignore`, fixed sample list and small manifest/evidence snapshots: `feat: cache frozen spatial features atomically`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add .gitignore src/cache.py scripts/02_extract_spatial.py tests/test_phase1_cache.py tests/test_phase1_extract.py data/splits/phase0_konvid_ids.txt data/manifests/phase1_cache.csv data/manifests/phase1_cache.meta.json results/phase1/extraction.json
git commit -m "feat: cache frozen spatial features atomically"
```

### Task 2: Immutable Splits and Checked Training Data

**Files:** Create `src/splits.py`, `src/data.py`, `scripts/01_make_splits.py`, `tests/test_phase1_splits.py`, `tests/test_phase1_data.py`, `data/splits/konvid_folds.csv`, `data/splits/konvid_folds.meta.json`.

**Interfaces:**
- Consumes Task 1 `load_cached_video(...)` and source `data/manifests/konvid.csv`.
- Produces `make_splits(rows: list[dict], seed: int = 0) -> list[dict]`; each output has `outer_fold,video_id,role`, with role `train|val|test`. Each video appears once per outer fold and is test once across five folds.
- `CachedFeatureDataset(root: Path, manifest: Path, splits: Path, fold: int, role: str, feature_set: str)` yields `{'video_id':str,'features':Tensor[T,D] fp32,'mos':float}`. `collate_videos(items: list[dict]) -> dict` returns IDs, zero-padded `features[B,T,D]`, CPU int64 `lengths[B]` and raw `mos[B]`.
- CLI `01_make_splits.py --manifest data/manifests/konvid.csv --output data/splits/konvid_folds.csv --seed 0`. Existing output is verified, never overwritten; metadata binds exact source content and algorithm identity.

- [ ] Write failing `test_split_membership_disjoint_and_each_video_test_once`, `test_reordered_rows_give_identical_splits`, `test_equal_mos_ties_are_deterministic`, `test_existing_or_stale_split_cannot_be_regenerated`. On 100 fixture IDs spanning five strata, assert each fold has 20 test, 12 validation and 68 train IDs, all roles disjoint and their union complete.
- [ ] Write failing `test_collate_unequal_lengths`, `test_missing_or_nonfinite_cached_features_fail_by_id`, `test_feature_order_is_cls_then_layer12_mean`. For two feature sequences of lengths 2/4, assert `lengths == [2,4]`, padding zeros and original order preserved; bad dtype/dimensions/T fail before training. No cache failure silently removes an ID.

```python
def test_every_fixture_video_is_test_once():
    from collections import Counter
    from src.splits import make_splits
    rows = [{"video_id":f"v{i:03}", "mos":float(1+i%5), "status":"ok"} for i in range(100)]
    actual = make_splits(rows, seed=0)
    assert len(actual) == 500
    assert Counter(r["video_id"] for r in actual if r["role"] == "test") == {r["video_id"]:1 for r in rows}
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_splits.py tests/test_phase1_data.py -q`; expected missing-implementation failures.
- [ ] Implement five deterministic rank-quantile MOS strata by sorting `(mos,video_id)` and dividing into five nearly equal groups. Shuffle each stratum with seed 0, distribute round-robin across outer test folds; within each outer training pool select 15% validation with proportional stratum allocation and deterministic largest-remainder rounding (ties by stratum index), RNG derived from `(0,outer_fold)`. This defines MOS-tie behavior without label-dependent retries.
- [ ] Implement checked feature loading, raw-label lookup and batching. Validate all split IDs and exact coverage against the authoritative inventory/cache before fitting; use `np.load(allow_pickle=False)`, promote to fp32 and never return writable views of immutable caches. Keep normalization out of this dataset; Task 5 owns fit-role statistics.
- [ ] Rerun tests/full suite; expected all pass. Run the split CLI once on real data. Expected: 6,000 role rows; each fold 240 test, 144 val and 816 train IDs; each source ID test exactly once. Run the CLI again; expected successful verification with identical CSV SHA, no rewrite. Save split validation evidence and **commit indices before any real training**.
- [ ] Commit source/tests and both immutable split artifacts: `feat: freeze stratified folds and feature batches`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add src/splits.py src/data.py scripts/01_make_splits.py tests/test_phase1_splits.py tests/test_phase1_data.py data/splits/konvid_folds.csv data/splits/konvid_folds.meta.json
git commit -m "feat: freeze stratified folds and feature batches"
```

### Task 3: Shared Front End and Three Conventional Heads

**Files:** Create `src/heads/__init__.py`, `src/heads/common.py`, `src/heads/mean_pool.py`, `src/heads/gru.py`, `src/heads/transformer.py`, `tests/test_phase1_heads.py`, initial `configs/head_tiers.yaml`; write `results/phase1/head_parameters_initial.json`.

**Interfaces:**
- `FeatureFrontEnd(input_dim: int, dropout: float)` maps frames to H=256 with LayerNorm(D), input Dropout and Linear(D,256).
- `valid_mask(lengths: Tensor, T: int) -> Tensor[B,T]`; `reverse_valid(x: Tensor[B,T,H], lengths: Tensor) -> Tensor[B,T,H]`; `masked_mean(x, lengths) -> Tensor[B,H]`.
- `build_head(name: str, input_dim: int, config: dict) -> nn.Module`; `.forward(features[B,T,D], lengths[B]) -> Tensor[B]`. Initially names are `mean_pool|bigru|transformer`; Task 4 adds `mamba` without changing the signature.
- Each model exposes `.front_end` and `.head`; head-only counts are all trainable parameters under `.head`, common-front counts are separate.

- [ ] Write parametrized failing `test_padding_invariance`, `test_batch_equals_individual_unpadded_eval`, `test_invalid_lengths_fail`, `test_output_shape_and_finite_gradients` for all three names. Test lengths `[1,3]`, right padding with large values **and NaNs**, and added padded timesteps; expected eval predictions agree within `atol=1e-6,rtol=1e-5`. Reject lengths 0, >T or noninteger. Include positional encoding beyond the usual training length.
- [ ] Write `test_reverse_only_valid_prefix`: for values `[1,2,3,999,999]` with length 3, assert the valid reversed prefix `[3,2,1]` and no padded value enters it.

```python
def test_reverse_only_valid_prefix():
    import torch
    from src.heads.common import reverse_valid
    x = torch.tensor([[[1.],[2.],[3.],[999.],[999.]]])
    actual = reverse_valid(x, torch.tensor([3]))
    torch.testing.assert_close(actual[0,:3,0], torch.tensor([3.,2.,1.]))
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_heads.py -q`; expected missing-implementation failures.
- [ ] Implement mask sanitization with `torch.where` **before** normalization/projection, so NaN padding cannot leak through `NaN*0`. Use packed Bi-GRU (`enforce_sorted=False`, CPU lengths), masked Transformer keys and masked pooling. Mean-pool performs masked average then its MLP. Dropout is off for invariance checks (`eval()`), not silently disabled in training.
- [ ] Declare one matched size tier: mean MLP hidden 1598; one-layer bidirectional GRU hidden 156/direction; one Transformer encoder layer, H=256, four attention heads, FF width 256, GELU, `norm_first=True`, sinusoidal positions and no extra final encoder norm. GRU/Transformer use a `Linear(input,64) → GELU → Dropout → Linear(64,1)` readout; mean uses the same pattern with hidden 1598. GRU dropout is after its sequence encoder/readout, not unsupported recurrent dropout on a single layer. All Linear/GRU biases stay enabled. Runtime tests, not estimated counts in prose, certify the final budget after Task 4.
- [ ] Rerun this file/full suite; expected all pass on CPU. Save actual common/head/total parameter counts from instantiated modules, without claiming model quality.
- [ ] Commit: `feat: add masked mean gru and transformer heads`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add src/heads tests/test_phase1_heads.py configs/head_tiers.yaml results/phase1/head_parameters_initial.json
git commit -m "feat: add masked mean gru and transformer heads"
```

### Task 4: Pure-PyTorch Bidirectional Mamba and Budget Gate

**Files:** Create `src/heads/mamba.py`, `tests/test_phase1_mamba.py`, `tests/test_phase1_budget.py`; modify `src/heads/__init__.py`, `src/heads/common.py`, `configs/head_tiers.yaml`.

**Interfaces:**
- `selective_scan(u: Tensor[B,T,I], delta: Tensor[B,T,I], A: Tensor[I,N], B: Tensor[B,T,N], C: Tensor[B,T,N], D: Tensor[I], lengths: Tensor[B]) -> Tensor[B,T,I]` (delta already positive, A negative).
- `SelectiveBlock.forward_sequence(x: Tensor[B,T,256], lengths: Tensor) -> Tensor[B,T,256]` is causal; use it for prefix-causality tests, not the final video-level MOS output.
- Factory adds `name='mamba'`, primary bidirectional only. `parameter_counts(model: nn.Module) -> dict[str,int]` in `common.py` returns `front_end,head,total`.

- [ ] Write failing `test_scan_known_recurrence` with I=N=B=1, `u=[1,2]`, `delta=[log(2),log(2)]`, `A=-1`, B=C=1, D=0: outputs `[log(2),2.5*log(2)]`. Add `test_scan_matches_independent_naive_values_and_gradients` on tiny float64 tensors (`atol=1e-9,rtol=1e-7`) and `torch.autograd.gradcheck`; the independent scalar-loop reference lives in tests only.
- [ ] Write failing `test_causal_sequence_prefix_ignores_future_suffix`, `test_mamba_padding_and_batch_invariance`, `test_reverse_direction_respects_each_valid_length`. Compare causal **sequence embeddings** before the changed suffix; a pooled video prediction is not a prefix score. Add all-four-head budget assertion `max(counts)/min(counts) <= 1.05` for input D=1536 and D=2048; front-end parameters cannot enter that ratio.

```python
def test_scan_known_recurrence():
    import math
    import torch
    from src.heads.mamba import selective_scan
    one = torch.ones(1,2,1,dtype=torch.float64)
    u = torch.tensor([[[1.],[2.]]],dtype=torch.float64)
    out = selective_scan(u, one*math.log(2), -torch.ones(1,1,dtype=torch.float64),
                         one, one, torch.zeros(1,dtype=torch.float64), torch.tensor([2]))
    expected = torch.tensor([[[math.log(2)],[2.5*math.log(2)]]],dtype=torch.float64)
    torch.testing.assert_close(out, expected, atol=1e-9, rtol=1e-7)
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_mamba.py tests/test_phase1_budget.py -q`; expected failures before implementation.
- [ ] Implement a Mamba-1-style selective block in ordinary PyTorch: pre-LayerNorm(256); bias-free input projection to `2I`; causal depthwise Conv1d(kernel 4, groups I, bias enabled), SiLU; bias-free x projection to `r+2N`; dt projection with bias and softplus; `A=-exp(A_log)`, learned D skip, SiLU gate and bias-free output projection to 256, with residual. Set I=230, N=16, r=16. Initialize A_log from log(1..16), D=1 and softplus(dt_bias) log-uniform in `[0.001,0.1]`; document the reference initialization. Do not install/import compiled Mamba or einops.
- [ ] The recurrence is `h_t=exp(delta_t*A)*h_(t-1)+delta_t*B_t*u_t`, `y_t=sum_N(C_t*h_t)+D*u_t`; freeze state and zero outputs after valid length. This follows the real-valued variable-B/C reference, rather than substituting an ordinary fixed SSM. Preserve float64 in numerical tests. Use separate forward/backward blocks, reverse within valid lengths and average the aligned directional residual outputs. Mask-pool, then the common 64-wide readout. No per-step MOS or causal accuracy experiment is added.
- [ ] Rerun tests/full suite and regenerate exact parameter counts. Expected one fixed tier satisfies the 5% gate for all three sequence heads and also the mean MLP; if literal counts differ, adjust integer hidden/inner widths **before training**, regenerate counts and record the choice. Never match totals by counting the front end or accepting independent unmatched tiers.
- [ ] Record algorithm/initialization and measured counts in ARCHITECTURE.md and small `results/phase1/head_parameters.json`; commit: `feat: add verified bidirectional selective scan head`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add src/heads tests/test_phase1_mamba.py tests/test_phase1_budget.py configs/head_tiers.yaml results/phase1/head_parameters.json ARCHITECTURE.md
git commit -m "feat: add verified bidirectional selective scan head"
```

Reference reviewed while planning: [official selective-scan reference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py) and [Mamba-1 block](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py). Implement the equations above locally; do not import the optional-extension package. Record the inspected upstream revision/source identity during execution.

### Task 5: Shared Loss, Metrics and One-Run Trainer

**Files:** Create `src/losses.py`, `src/metrics.py`, `src/training.py`, `scripts/05_train.py`, `configs/protocol.yaml`, `tests/test_phase1_loss.py`, `tests/test_phase1_training.py`, `tests/test_phase1_metrics.py`.

**Interfaces:**
- `quality_loss(pred: Tensor[B], target_z: Tensor[B], rank_weight: float) -> Tensor`.
- `fit_target_stats(mos: np.ndarray) -> dict[str,float]` returns `mean,std` (population std); `correlations(y_true: np.ndarray, y_pred: np.ndarray) -> dict` returns `srcc,krcc,plcc_raw,status,reason`, with explicit nulls for undefined values.
- `fit_one(model: nn.Module, train_data: Dataset, val_data: Dataset, config: dict, output: Path, seed: int, provenance: dict) -> dict` returns completed artifact paths and best epoch/validation SRCC. It receives **no test dataset/labels**.
- `predict(model: nn.Module, dataset: Dataset, target_stats: dict, device: str) -> list[dict]` returns raw-MOS `video_id,prediction` in input-ID order. No clipping to [1,5]; no per-step predictions.
- Initial CLI: `05_train.py --mode single --config configs/protocol.yaml --features configs/features_v1.yaml --head NAME --fold N --seed N --output PATH`; later tasks add modes. Single real runs are not public accuracy results.

- [ ] Write failing `test_rank_loss_ignores_ties`, `test_correct_order_has_lower_rank_loss`, `test_rank_zero_equals_mean_l1`, `test_constant_training_mos_fails`. Assert targets `[0,2]`, predictions `[0,1]`, lambda 0 yield L1 `0.5`; all tied targets yield a finite zero rank term. Reject nonfinite inputs and negative lambda.
- [ ] Write failing `test_normalization_uses_fit_rows_only`, `test_validation_selects_best_checkpoint_not_last`, `test_test_label_perturbation_cannot_change_fit`, `test_failed_or_incompatible_run_is_not_resumed`. Tiny test-only PyTorch models/datasets make fitting fast. Hold train/val fixed, perturb only held-out labels in the orchestration fixture, and compare trained tensor values/best epoch, not checkpoint-file hashes containing provenance. Assert val/test labels cannot change train mean/std. Invalid validation SRCC cannot win selection.

```python
def test_rank_zero_equals_mean_l1():
    import torch
    from src.losses import quality_loss
    value = quality_loss(torch.tensor([0.,1.]), torch.tensor([0.,2.]), rank_weight=0.)
    torch.testing.assert_close(value, torch.tensor(0.5))
```

- [ ] Write metric tests: identical ranks → SRCC/KRCC 1; reversed ranks → -1; ties use SciPy's standard handling; constant vectors/nonfinite values produce explicit failed diagnostics rather than an invented zero. Run all three files; expected failures for missing implementation.
- [ ] Implement mean L1 plus lambda times mean `softplus(-sign(y_i-y_j)*(pred_i-pred_j))` over `i<j` unequal-target pairs. Normalize targets only using the fit-role MOS mean/std; reject std=0. Never recompute stats from validation/test rows. Cache arrays remain immutable; no dataset-level neural feature standardization beyond the common LayerNorm.
- [ ] Implement fp32 AdamW training for every head, batch 64, max epochs 200, patience 15, linear LR warmup five epochs then cosine decay, gradient clipping norm 1.0. Apply the same optimizer policy to all trainable parameters; exclude frozen backbones entirely. Pilot defaults: LR `3e-4`, weight decay `1e-3`, input/head dropout `0.1`, rank weight `0.1`. Early stopping maximizes raw inner-validation SRCC; ties retain the earlier epoch. Save the best checkpoint atomically, including target stats, model config, optimizer/protocol settings and epoch; load it before prediction. Set Python/NumPy/Torch seeds, disable cuDNN benchmarking and request `torch.use_deterministic_algorithms(True, warn_only=True)` identically for all heads, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` established before CUDA use. Preserve warnings in run logs; unsupported deterministic cuDNN/RNN operations are a reproducibility limitation, not a training failure. Record the warning-only policy and observed nondeterministic operations in the final README; do not claim bitwise reproducibility.
- [ ] Write `run.json` atomically with statuses running/failed/complete and config, split and cache hashes; record the feature policy, code commit, lock/weight/checkpoint hashes, seed/fold/head and label mode as audit metadata. A completed matching run may be verified/reused; interrupted runs restart from scratch (epoch-resume machinery is not needed). Changing test labels may change audit metadata, never fit statistics/weights/selection. Test-role loading occurs only after checkpoint selection for CV evaluation.
- [ ] Run all task tests/full suite; expected all pass. Commit: `feat: train cached feature heads with shared loss`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add src/losses.py src/metrics.py src/training.py scripts/05_train.py configs/protocol.yaml tests/test_phase1_loss.py tests/test_phase1_training.py tests/test_phase1_metrics.py
git commit -m "feat: train cached feature heads with shared loss"
```

### Task 6: Ridge Baseline and Diagnostic Statistics

**Files:** Create `src/baselines.py`, initial `scripts/06_eval.py`, `tests/test_phase1_ridge.py`; extend `src/metrics.py`, `tests/test_phase1_metrics.py`.

**Interfaces:**
- `fit_ridge(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, alphas: list[float]) -> dict`; `predict_ridge(artifact: dict, X: np.ndarray) -> np.ndarray`.
- `fit_logistic_plcc(y_true: np.ndarray, y_pred: np.ndarray) -> dict` returns mapped PLCC, four coefficients, status/reason and fitting-domain disclosure.
- `bootstrap_srcc(y_true: np.ndarray, predictions: np.ndarray, seed: int = 0, draws: int = 10000) -> dict` returns point SRCC, percentile 2.5/97.5 CI, valid/invalid draw counts and status. This is the control diagnostic, not the Phase 2 paired head comparison.

- [ ] Write failing `test_ridge_standardizes_from_training_only`, `test_constant_feature_is_safe`, `test_alpha_selected_by_validation_not_test`, `test_ridge_deterministic`. Assert prediction invariance when validation extremes change without refitting train statistics; zero-variance feature scales become 1; ties select the earliest ascending alpha.
- [ ] Write failing `test_logistic_mapping_is_disclosed_and_finite`, `test_constant_prediction_mapping_fails_explicitly`, `test_bootstrap_is_seeded_and_retains_invalid_draw_counts`. Known affine/monotone fixtures have correct rank signs; all-constant data cannot yield a passing control. Require at least 99% finite bootstrap draws, otherwise a failed diagnostic; do not silently discard a material fraction.

```python
def test_reversed_ranks_remain_negative():
    import numpy as np
    import pytest
    from src.metrics import correlations
    result = correlations(np.arange(5,dtype=float), -np.arange(5,dtype=float))
    assert result["srcc"] == pytest.approx(-1.)
    assert result["krcc"] == pytest.approx(-1.)
```

- [ ] Run ridge/metric tests; expected failures for missing functions.
- [ ] Mean-pool valid cached frames for ridge; fit feature means/stds and target stats using the same train-role IDs as neural fitting. Use dual ridge with NumPy/SciPy linear solves, an intercept via centering, and alpha grid `[0.001,0.01,0.1,1,10,100,1000]`. Select by inner-validation SRCC; keep the fit-role model without refitting on validation. Ridge runs once per outer fold (seed null), not five identical seed copies.
- [ ] Implement target-domain four-parameter logistic `b2+(b1-b2)/(1+exp(-(x-b3)/s))`, s>0, using bounded SciPy least squares and stable expit; save coefficients/convergence and clearly label that PLCC mapping uses target labels. SRCC/KRCC stay unmapped. Mapping failure is reported, not hidden. Implement seeded video-level bootstrap with explicit undefined-draw accounting.
- [ ] Run tests/full suite; expected all pass. CLI `06_eval.py --config configs/protocol.yaml --results results/phase1 --features configs/features_v1.yaml` initially supports validated diagnostic/ridge summaries; Task 9 adds completeness checks and final Phase 1 report.
- [ ] Commit: `feat: add deterministic ridge and diagnostic metrics`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add src/baselines.py src/metrics.py scripts/06_eval.py tests/test_phase1_ridge.py tests/test_phase1_metrics.py
git commit -m "feat: add deterministic ridge and diagnostic metrics"
```

### Task 7: Early Leakage Check, Training Pilot and Runtime Gate

**Files:** Extend `scripts/05_train.py`; create `src/tuning.py`, `tests/test_phase1_pilot.py`; finalize measured defaults in `configs/protocol.yaml`; generate `results/phase1/control_initial/`, `training_pilot.json`, `runtime_budget.json`; update DECISIONS.md/PHASES.md from outputs.

**Interfaces:**
- Modes `--mode control-initial` and `--mode pilot` use Task 5 fitting and Task 6 bootstrap. `--mode pilot` fits one complete early-stopped fold-0 run/head, seed 0, without outer-test predictions.
- `src/tuning.py` produces `planned_fit_counts(scope: dict, protocol: dict) -> dict` and `project_runtime(pilot_rows: list[dict], scope: dict, protocol: dict, spent: dict) -> dict`. Use `protocol.tuning_seeds` (default `scope.seeds`) for rank/head selection counts, and always `scope.seeds` for reported CV/controls. Count keys are `pilot,control_initial,rank,tune,control_final,cv,ridge_experiments`; return `status: go|no-go` and explicit assumptions/totals from the runtime function.
- Pilot JSON records synchronized epoch/fit wall time, actual completed epochs, best epoch, peak CUDA allocation, batch/precision/threads and the run config/split/cache hashes. Runtime-budget JSON records exact proposed fit counts, measured-rate assumptions, optimistic/2×/max-epoch scenarios, remaining GPU allowance, reserved 30 GPU hours and go/no-go.

- [ ] Write failing `test_control_failure_blocks_pilot_and_tuning`, `test_pilot_never_reads_test_role`, `test_runtime_budget_counts_all_five_seed_trials`, `test_two_seed_tuning_fallback_preserves_five_seed_cv`. Assert default tuning gives 200 head fits/20 rank fits, while explicit `tuning_seeds=[0,1]` gives 80/eight; CV stays 100 and each control stays five seeds in both cases. Include both five-seed initial/final controls, four pilots and ridge work; do not price just 40 tuning fits.

```python
def test_runtime_budget_counts_all_five_seed_trials():
    from src.tuning import planned_fit_counts
    scope = {"heads":["mean_pool","bigru","transformer","mamba"], "folds":5,
             "seeds":[0,1,2,3,4], "tuning":{"trials_per_head":10}}
    counts = planned_fit_counts(scope, {"rank_candidates":[0,0.1,0.3,1]})
    assert counts["tune"] == 200 and counts["rank"] == 20 and counts["cv"] == 100
    assert counts["control_initial"] == counts["control_final"] == 5
    smaller = planned_fit_counts(scope, {"rank_candidates":[0,0.1,0.3,1], "tuning_seeds":[0,1]})
    assert smaller["tune"] == 80 and smaller["rank"] == 8
    assert smaller["cv"] == 100 and smaller["control_final"] == 5
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_pilot.py -q`; expected failures.
- [ ] Implement initial shuffled-label control with default mean-pool settings: permute **training-role labels only**, permutation seed 0 fixed for all model seeds, preserve true validation/test labels and original split memberships. Persist permutation IDs/hash. Train five seeds and average held-out fold-0 predictions once per video; evaluate against untouched test MOS. Require CI including zero and `abs(srcc)<0.15`; never reroll a failing permutation. Keep control outputs outside real-CV results. This is a diagnostic held-out evaluation, never a tuning input.
- [ ] Implement pilot modes and synchronization. Head timing runs after cache completion, on the real fold-0 train/val set and fixed size tier. Save measurements separately from publication metrics. Re-evaluate cache/split integrity and ensure indices are committed before launching any real fit.
- [ ] Run task tests/full suite; expected all pass. In Kitty run `05_train.py --mode control-initial --config configs/protocol.yaml --features configs/features_v1.yaml`, then the analogous `--mode pilot` only after a passing diagnostic. Expected: five control seeds with a passing diagnostic, four successful real pilots, measured wall/epoch/memory figures and no test predictions from the pilot.
- [ ] Generate runtime projections from those outputs with the budget anchored at Phase 1 start. Let spent be actual Phase 1 GPU work to date, including extraction; remaining extraction allowance is `max(0,full_spatial_planning_hours-actual_extraction_hours_already_spent)`. Remaining head budget is `100-spent-remaining_extraction_allowance-30`. Project only fits not yet completed, so extraction/pilots/initial controls are never charged twice. Ridge is separately timed CPU work: five ridge experiments contain seven alpha fits each; do not charge them as GPU hours. Use 2× observed fit costs as the planning gate, plus a separately labeled max-epoch upper bound. If the five-seed tuning projection is no-go, record the measured reason and activate `tuning_seeds=[0,1]` for both rank-weight and head selection in DECISIONS.md/protocol before tuning starts, identically for all four heads. Recompute the projection using 80 head fits and eight rank fits. Keep ten candidates/head, five-seed CV/controls, all folds and mandatory checks. If still no-go, stop and record a further scope proposal. Changes to global epoch cap/patience are proposals supported by timing and must be dated before any tuning result exists.
- [ ] Freeze common measured protocol defaults/search bounds and record their SHA; commit: `feat: gate phase one grid on measured training cost`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add src/tuning.py scripts/05_train.py tests/test_phase1_pilot.py configs/protocol.yaml results/phase1 DECISIONS.md PHASES.md
git commit -m "feat: gate phase one grid on measured training cost"
```

### Task 8: Rank Weight Then Equal-Budget Head Tuning

**Files:** Modify `src/tuning.py`; create `tests/test_phase1_tuning.py`, `configs/rank_loss.yaml`, `configs/tuning_trials.yaml`, `configs/heads/{mean_pool,bigru,transformer,mamba}.yaml`; extend `scripts/05_train.py`.

**Interfaces:**
- `make_trials(protocol: dict, count: int = 10, seed: int = 0) -> list[dict]` returns a persisted ordered candidate list reused by each head.
- `select_trial(rows: list[dict], expected_seeds: list[int]) -> dict` rejects any incomplete/failed candidate with ValueError and otherwise selects by mean inner-validation SRCC over the frozen tuning seed list (five by default, two only after the documented Task 7 fallback), ties by ascending candidate ID; it cannot consume test metrics.
- Modes `--mode rank` and `--mode tune`; tune refuses absent/invalid rank freeze, failed initial control, runtime no-go, split/cache mismatch or uncommitted split artifacts.

- [ ] Write failing `test_rank_frozen_before_head_tuning`, `test_trial_requires_every_declared_tuning_seed`, `test_tuning_never_reads_outer_test`, `test_ten_same_candidates_per_head`, `test_changed_config_split_or_cache_rejects_resume`. Changing held-out labels/metrics cannot change selection. A missing/failed member of the frozen tuning seed list is an incomplete candidate, not an opportunity to drop another seed; partial artifacts cannot be winners.

```python
def test_trial_requires_every_declared_tuning_seed():
    import pytest
    from src.tuning import select_trial
    rows = [{"candidate_id":0, "seed":0, "validation_srcc":0.5, "status":"complete"}]
    for seeds in ([0,1,2,3,4], [0,1]):
        with pytest.raises(ValueError, match="incomplete|seed"):
            select_trial(rows, expected_seeds=seeds)
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_tuning.py -q`; expected failures.
- [ ] Select shared lambda from `[0,0.1,0.3,1]` with mean pooling, fixed pilot-default optimizer/dropout settings and the frozen tuning seed list on fold-0 train/val only. Choose highest mean validation SRCC, ties by list order. Save selection scores/config/split/cache hashes and generated `rank_loss.yaml` before any head tuning. No held-out prediction is needed for rank selection.
- [ ] Persist exactly ten seed-0 candidate configurations: LR log-uniform `[1e-4,3e-3]`, AdamW weight decay log-uniform `[1e-6,1e-2]`, input/head dropout independently from `{0,0.1,0.2}`. The architecture tier/batch/epochs/patience and shared lambda remain fixed. Use the identical ordered candidate list for all four heads; fit each candidate with every declared tuning seed. Candidate failure pauses/debugs that candidate; it does not grant a replacement or hide a seed.
- [ ] Store validation-only selection JSON and generate each head config from the winning candidate with config/split/cache hashes, the selected shared rank weight and input feature set. Those frozen configs are reused on the ResNet control in Phase 2 without tuning. Never select width/depth independently outside the matched tier.
- [ ] Run tests/full suite; expected all pass. Kitty: run `05_train.py --mode rank --config configs/protocol.yaml --features configs/features_v1.yaml`, then the same command with `--mode tune`. Expected: 20 rank/200 head fits by default, or eight/80 only with the predeclared two-seed fallback; every declared tuning seed/candidate accounted for, four generated frozen head configs and zero outer-test predictions from selection. Do not present these tuning scores as reported accuracy results.
- [ ] Commit configs, source/tests and small selection evidence: `feat: freeze shared rank weight and head tuning`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add src/tuning.py scripts/05_train.py tests/test_phase1_tuning.py configs/rank_loss.yaml configs/tuning_trials.yaml configs/heads results/phase1
git commit -m "feat: freeze shared rank weight and head tuning"
```

### Task 9: Final Leakage Gate and Definitive DINO Head-Comparison Runs

**Files:** Extend `scripts/05_train.py`, `scripts/06_eval.py`; create `tests/test_phase1_cv.py`; write `results/phase1/control_final/`, `cv/`, `ridge/`, `metrics.csv`, `summary.json`, `report.md`.

**Interfaces:**
- Mode `--mode control` repeats Task 7's fixed-permutation diagnostic with the **selected** mean-pool config and shared lambda; it does not retune against control results.
- Mode `--mode cv` runs four heads × five folds × five seeds and deterministic ridge once/fold, using the frozen configs. A prediction row contains `video_id,dataset,feature_set,head,fold,seed,mos,prediction,checkpoint_sha256,run_id,run_kind,label_mode`.
- `scripts/06_eval.py` adds `validate_oof(rows: list[dict], fold_by_id: dict[str,int], heads: list[str], seeds: list[int]) -> dict` with `status,missing_keys,duplicate_keys,wrong_fold_ids`; wrong label/run kinds fail validation.
- `06_eval.py` requires exactly one real out-of-fold prediction per video/head/seed and five ridge fold outputs. It emits per-run SRCC/KRCC/raw and mapped PLCC, per-fold/seed summaries, best-epoch distributions and readiness diagnostics.

- [ ] Write failing `test_failed_control_blocks_real_cv`, `test_cv_complete_coverage_and_no_duplicates`, `test_control_or_pilot_rows_cannot_enter_headline`, `test_predictions_use_best_checkpoint_and_inverse_target_scale`, `test_mismatched_complete_run_cannot_be_skipped`, `test_reusing_complete_dino_cv_never_calls_fit`. Assert 100 neural runs, five ridge fits, 1,200 OOF rows/head/seed and 1,200 total ridge OOF rows. Missing one fold/seed/ID or duplicated predictions must make report exit nonzero.

```python
def test_missing_oof_predictions_fail():
    import importlib
    validate_oof = importlib.import_module("scripts.06_eval").validate_oof
    result = validate_oof([], {"v000":0}, ["mean_pool"], [0])
    assert result["status"] == "failed"
    assert result["missing_keys"]
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_cv.py -q`; expected failures.
- [ ] Implement final control before real CV; save the five individual outputs and seed-averaged video bootstrap diagnostic. Reuse the same declared training permutation, not a new “easier” control. Failure stops CV and requires investigation recorded by ID/run, not suppressing the diagnostic.
- [ ] For each real run, fit only train-role rows, select epoch only on validation, load the best checkpoint and then predict the test role. Model seeds change initialization/batch order only; splits, normalization ownership and candidate selection stay fixed. Save raw-MOS predictions/checkpoint hashes atomically; never overwrite incompatible successful outputs. Completed matching CV artifacts can be verified and skipped; interrupted runs restart from scratch. Ridge uses the same roles and its validation-selected alpha.
- [ ] Implement completeness-first summaries. Report mean/std over the 25 fold×seed metrics with seed/fold variability separately; use sample std (`ddof=1`) for run summaries. Preserve undefined metric reasons and fail the phase gate when mandatory metrics are undefined; don't replace with zero or average fewer runs. Save seed-averaged OOF predictions for later Phase 2 paired bootstrap, labeled as an ensemble rather than the headline quantity. These are real DINO head-comparison results, reusable without training again. Do not choose hyperparameters or feature sets from held-out metrics; Phase 2 chooses between fixed DINO and ResNet using inner-validation scores, then completes the final inferential tables from saved predictions.
- [ ] Run tests/full suite, then Kitty `05_train.py --mode control --config configs/protocol.yaml --features configs/features_v1.yaml`, the same command with `--mode cv`, then `06_eval.py --config configs/protocol.yaml --results results/phase1 --features configs/features_v1.yaml`. Expected: passing final control; complete verified DINO/ridge artifacts and generated metrics/report. The report labels these as the final research runs for the fixed DINO head comparison, not disposable pipeline diagnostics. Phase 2 must load the existing `results/phase1/cv/`, ridge outputs and seed-averaged OOF predictions for its joint tables and paired-bootstrap/Holm reporting. Repeating the CV command with matching config/split/cache hashes must perform zero new fits; test that behavior. Only an explicit research/protocol/code correction with affected-run invalidation permits a DINO rerun.
- [ ] Commit source/tests and small prediction/metric/evidence files, not checkpoints: `feat: complete controlled five fold five seed core runs`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add scripts/05_train.py scripts/06_eval.py tests/test_phase1_cv.py results/phase1
git commit -m "feat: complete controlled five fold five seed core runs"
```

### Task 10: Feature Parity and Phase 1 Exit Gate

**Files:** Create `scripts/04_parity_check.py`, `tests/test_phase1_parity.py`, `tests/test_phase1_report.py`; extend `scripts/06_eval.py`; generate `results/phase1/parity.json`, `phase_gate.json`; update PHASES.md/DECISIONS.md with generated results and measured remaining budget.

**Interfaces:**
- `minimum_cosine(reference: np.ndarray, candidate: np.ndarray) -> float` in `scripts/04_parity_check.py`; compare matching `[T,D]` branches in float64 arithmetic, reject zero norms/nonfinite/mismatched arrays.
- CLI `04_parity_check.py --features configs/features_v1.yaml --sample-ids data/splits/phase0_konvid_ids.txt --cache cache/v1` regenerates CPU fp32 branches using shared modules and compares to actual GPU fp16 cached arrays. No MOS prediction is performed.
- Final `06_eval.py` exits zero only if cache/split/config consistency, parameter budget, initial/final controls, complete CV/ridge, actual pilot/budget and feature parity all pass. `phase_gate.json` lists absent/failed requirements rather than using missing files as success.

- [ ] Write failing `test_cosine_boundary`, `test_missing_timestep_or_branch_fails`, `test_zero_nan_wrong_source_or_partial_cache_fails`, `test_one_bad_timestep_cannot_be_averaged_away`. Assert identical nonzero vectors score 1, constructed cosine 0.998 fails and 0.999 passes, with threshold `>=0.999` at every timestep in every required branch.
- [ ] Write failing report tests for a missing CV seed, failed control, wrong run config/split/cache hash or absent hardware parity. A Phase 0 smoke or fabricated offline fixture cannot satisfy Phase 1 hardware parity.

```python
def test_cosine_boundary():
    import importlib
    import numpy as np
    cosine = importlib.import_module("scripts.04_parity_check").minimum_cosine
    reference = np.array([[1.,0.]])
    good = np.array([[0.999,np.sqrt(1-0.999**2)]])
    bad = np.array([[0.998,np.sqrt(1-0.998**2)]])
    assert cosine(reference, good) >= 0.999
    assert cosine(reference, bad) < 0.999
```

- [ ] Run `.venv/bin/python -m pytest tests/test_phase1_parity.py tests/test_phase1_report.py -q`; expected failures.
- [ ] Implement parity over the original fixed 20 IDs, all seven DINO branches and ResNet-50. Save each video/branch minimum, source/reference checksums, precision/device, feature config, code commit, lock and weight identities for audit. If a branch fails, diagnose before continuing; do not relax the threshold, discard a video or reuse old metrics after a real preprocessing change. Phase 4 still adds deployed-head prediction parity later.
- [ ] Run normal `.venv/bin/python -m pytest -q`; expected all Phase 0 and Phase 1 offline tests pass. Record the ordinary command/result in the execution ledger. Pytest passing and the data/result exit report are separate acceptance checks: no test JSON, JUnit XML or test-hash attestation is produced or consumed by the report. From Kitty run `04_parity_check.py --features configs/features_v1.yaml --sample-ids data/splits/phase0_konvid_ids.txt --cache cache/v1`, then `06_eval.py --config configs/protocol.yaml --results results/phase1 --features configs/features_v1.yaml`. Expected: complete real parity evidence, passing controls/CV/ridge, no missing gate and nonzero status for any intentionally withheld required artifact in the offline tests.
- [ ] Generate updated training/extraction/remaining-time estimates from saved outputs, explicitly distinguishing wall time, GPU allocation and hands-on estimates. Update the dated decisions and phase status. Exit Phase 1 only on a passing report; preserve honest no-go/failure evidence otherwise. No LIVE-VQC prediction, ResNet head training or serving app is started by completing this plan. Phase 2 reuses the definitive DINO runs; only the ResNet control fits and final statistical reporting are added.
- [ ] Commit small outputs/config snapshots/source/tests/docs: `docs: verify phase one core slice and parity`. Run from the active workspace (or its host terminal when `.git` is blocked):

```bash
git add scripts/04_parity_check.py scripts/06_eval.py tests/test_phase1_parity.py tests/test_phase1_report.py results/phase1 PHASES.md DECISIONS.md
git commit -m "docs: verify phase one core slice and parity"
```

## Self-Review and Handoff

Self-review completed against the current spec: no Phase 1 requirement is omitted. The measured runtime gate deliberately resolves feasibility before expensive tuning; it is not a missing default. No source change, split creation or workload was executed while planning.

Coverage checked: all Phase 1 bullets in PHASES.md map to Tasks 1–10; controls/budgets/test metrics use the binding thresholds; required tests and five-seed reported results are retained; the two-seed tuning-only fallback is explicit and conditional; frozen feature branches remain cached; all scope cuts remain excluded. File and interface names above are shared unchanged by their consumers. The five Review Focus classes each have named tests in their owner tasks. No implementation bodies are prescribed except the selective recurrence and numerical equations that distinguish the intended model/loss/mapping.

Dependency scan for execution: Task 1 cache schema → Tasks 2/6/9/10; Task 2 split/role dataset → Tasks 5–9; Task 3 common interface → Task 4 and the trainer; Tasks 5–6 run/metric schemas → Tasks 7–10; Task 7 budget/protocol freeze → Task 8; Task 8 selected configs → Task 9; complete cache/CV/control artifacts → Task 10. Review these joins in the execution ledger before Task 1. No model/data API is supplied by an unstated task.

Recommended execution: **Native**, continuing the inline method used for Phase 0, with one independent whole-branch review. The tasks share cache/config/split inputs and use the same measured machine; repeated implementer/reviewer contexts would cost more than they help. Keep a per-task ledger and RED→GREEN evidence. GPU jobs run from Kitty if the agent sandbox still cannot access CUDA/Git, with concrete commands produced only after their scripts are implemented and verified.

This is a planning deliverable. Review and approve it before implementation; no Phase 1 extraction, folds, tuning or training is started by this document.
