# Phase 0 Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish measured evidence that local extraction and eventual CPU inference are feasible, then freeze preprocessing and record the Phase 0 go/no-go decisions.

**Architecture:** Build the shared preprocessing and frozen-feature modules first, with small command-line tools for inventory, timing, and feature parity. Keep evidence under `results/phase0/`; derive the feasibility report from saved measurements. Optional cloud work imports those same modules and cannot extract UGC before parity passes.

**Tech Stack:** Python, PyTorch/torchvision, NumPy, PyYAML, pytest, FFmpeg/ffprobe as the proposed decoder; PyTorchVideo if SlowFast works. Pin actual compatible versions during execution rather than guessing them here.

**Spec:** `PHASES.md` (Phase 0 and cut order), `ARCHITECTURE.md` (§§2–5, 8, 10), `RULES.md`, `PRD.md`, `DECISIONS.md`. Read these together with `AGENTS.md` before execution.

## Execution amendment: 10-day scope (2026-10-25)

The user set deadline 2026-11-04, ~8 hands-on h/day and ~10 unattended GPU h/night over 10 days. SlowFast and all YouTube-UGC work are cut from active research; completed preflight measurements remain historical evidence. Separate layer comparison is skipped while layers are cached; tuning is 10 trials/head; serving is FastAPI + upload page + one Dockerfile. See DECISIONS.md and configs/scope.yaml. This amendment supersedes conflicting future work in the original plan without discarding collected evidence.

## Global Constraints

- “All extraction, training, and evaluation run locally, except YouTube-UGC feature extraction, which runs in a free cloud notebook.”
- “Frames are decoded and processed in memory. Frames are not written to disk.”
- “Sampling rate: 4 fps for all experiments. 8 fps is not planned.”
- “Crop: center crop to 448x448.” Spatial normalization is ImageNet normalization.
- “Offline extraction runs on GPU under fp16 autocast. Cached arrays are stored as fp16.”
- “Serving runs on CPU in fp32”.
- “Clip: 32 frames on a dense timestamp grid of about 15 fps”; SlowFast alpha 4, short-side resize and center crop 256, mean 0.45/std 0.225 per channel, subject to reference-transform confirmation.
- “No UGC clip is extracted until the cloud parity check (10 KoNViD videos, cosine at least 0.999 per timestep against the local cache) passes.”
- “Videos are never committed to the repo.” Do not delete existing archives or raw clips during preflight.
- No LIVE-VQC predictions, training, fold generation, head tuning, demo, or full-dataset feature extraction in Phase 0. Probe LIVE-VQC metadata and portrait orientation only.
- Record plan changes with a date in `DECISIONS.md`. Do not change `AGENTS.md`.
- Sample seed proposed here: 0. Timing is an engineering measurement, not a one-seed accuracy experiment. Record nonapplicable fold/head-checkpoint fields explicitly.

## Review Focus

1. Variable frame rate and timestamp ties: choose the nearest presentation timestamp deterministically (earlier frame wins ties); test in Task 3.
2. Rotation metadata, including unsupported orientation: return upright frames or a recorded failure, never silently crop; test and visually inspect in Task 3.
3. Short clips and motion windows beyond endpoints: replicate boundary frames without changing window length; test in Task 3.
4. Missing weights, CUDA OOM, or CPU-only execution: fail with useful evidence, never substitute random weights or fabricate GPU timings; test in Tasks 4–5.
5. Corrupt archives/clips and interrupted cache writes: preserve originals, identify failures by ID, and reject incomplete parity inputs; test in Tasks 2 and 6.

## Starting State and Execution Order

The checkout contains five planning documents, `AGENTS.md`, and two dataset ZIP archives. No source, lockfile, tests, or runnable pipeline exists. `.git/` is present but Git commands currently report no repository; investigate before commits. Do not initialize over unknown metadata. Worktree setup and commits require a functioning repository; otherwise retain files and report that limitation.

Order: 1 → 2 → 3 → 4 → 5 → optional 6 → 7. Task 6 is conditional and may be cut first. This is one preflight pipeline, not separate implementation projects.

### File Map

| Files | Responsibility |
|---|---|
| `.gitignore`, `env/requirements.in`, `env/requirements.lock` | Keep data/secrets out of Git; reproducible Python dependencies |
| `configs/phase0.yaml` | Paths, sample IDs, seed, pinned weights, transforms, timing and feasibility settings |
| `scripts/00_inventory.py` | Archive validation and dataset metadata inventory |
| `src/preprocess.py` | Shared probe, timestamp sampling, rotation, spatial/motion preparation |
| `src/features.py` | Shared frozen backbone loading and extraction |
| `scripts/00_timing.py` | Per-video GPU and CPU measurements |
| `scripts/04_parity_check.py` | Minimal feature-level cloud parity check |
| `scripts/09_ugc_subset.py`, `scripts/10_ugc_fetch_extract.py` | Conditional metadata selection and one-clip cloud smoke test |
| `notebooks/ugc_cloud.ipynb` | Thin wrapper installing the lock and invoking shared code |
| `scripts/00_report.py` | Measurement-derived feasibility report and gate |
| `tests/test_phase0_*.py` | Small offline tests; separate hardware smoke checks |
| `data/manifests/`, `data/splits/` | Dataset provenance, metadata and immutable sample IDs |
| `results/phase0/` | Environment record, timings, parity evidence, failure logs, generated report |
| `ARCHITECTURE.md`, `DECISIONS.md`, `PHASES.md` | Confirmed transforms, dated decisions, measured estimates |

All CLI commands below are interfaces to implement, not commands available today. Run from repository root; scripts import `src/` without copying modules. Keep helpers in their owning files; no service layers or general pipeline framework.

### Task 1: Establish the Environment and Decision Inputs

**Files:** Create `.gitignore`, `env/requirements.in`, `env/requirements.lock`, `configs/phase0.yaml`, `results/phase0/environment.json`; modify `DECISIONS.md` pending fields.

**Interfaces:** Produces the shared YAML config: `seed`, `datasets`, `samples`, `decoder`, `backbones`, `timing` (`warmup=1`, `repeats=3`, `cpu_threads`), and `feasibility` (`deadline`, `weekly_hours`, `available_compute_hours`, `reserve_disk_bytes`). Unset user decisions are null and prevent the final gate from passing.

- [ ] Record the user's actual deadline and weekly hours; separately record available unattended compute hours through the deadline. Inspect OS, Python, CUDA/driver, GPU memory, CPU, RAM, free disk, archive byte sizes, and current Git status. Save evidence without credentials.
- [ ] Propose FFmpeg/ffprobe for sequential timestamp-aware decoding and inspect their installed versions. Record the choice and executable versions; if unsuitable, use OpenCV only with equivalent timestamp/rotation tests and a dated decision.
- [ ] Ignore `.venv/`, raw data, root dataset ZIPs, `cache/`, downloaded weights, Python caches, and credential directories (`.aws/`, `.codex/`, `.agents/`). Keep manifests, configs, tests, and small result artifacts eligible for commit.
- [ ] Create an isolated Python environment and minimal direct dependency list. Verify official compatibility instructions during execution, including SlowFast and notebook compatibility; resolve exact transitive pins into `env/requirements.lock`. Pin backbone code/weights independently of Python packages. If SlowFast cannot load, Task 4 owns the fallback.
- [ ] Verify a fresh environment with `python -m pip install -r env/requirements.lock` and `python -m pip check`; import torch, torchvision, NumPy, YAML and pytest; record CUDA availability. Record system decoder dependencies alongside the lock hash. A missing GPU blocks GPU measurements, not a silently substituted CPU result.
- [ ] Commit the environment/config/ignore changes when Git is functional: `chore: establish phase zero environment`. Never stage the whole checkout blindly.

### Task 2: Verify Dataset Access, Provenance, and Disk Plan

**Files:** Create `src/__init__.py`, the metadata-probing portion of `src/preprocess.py`, `scripts/00_inventory.py`, `tests/test_phase0_inventory.py`, `data/manifests/konvid.csv`, `data/manifests/livevqc.csv`, `data/manifests/datasets.json`; update config sample IDs and `DECISIONS.md`.

**Interfaces:** `inspect_archive(path: Path) -> dict` returns checksum, compressed/uncompressed sizes, member count, and validation status. CLI: `python scripts/00_inventory.py --config configs/phase0.yaml`. Manifests contain `video_id,dataset,relative_path,mos,mos_scale,source_size_bytes,source_sha256,duration_s,fps,width,height,rotation_applied,status,failure_reason`; failed probes leave unknown values empty. Dataset provenance JSON holds official source/license URLs, access date, release/version or archive identity, label source/hash, license terms, and redistribution restrictions.

`src/preprocess.py` initially produces `probe_video(path: Path) -> dict` using ffprobe JSON, returning the manifest's duration/fps/dimensions plus presentation timestamps and rotation metadata; failed probes raise an identified error that inventory records. Task 3 extends this same file. Metadata probing does not claim that rotation has been visually verified.

- [ ] Write failing tests: unsafe `../` or absolute archive member is rejected; CRC-corrupt member is reported; duplicate IDs and missing MOS are failures; input archives remain byte-identical. A failed video must retain its ID and reason in inventory output.
- [ ] Run `python -m pytest tests/test_phase0_inventory.py -q`; expect failure for missing implementation.
- [ ] Implement ZIP inspection without bulk extraction. Verify access and license terms against original dataset sources during execution; do not infer version or permission from filenames. Locate MOS metadata, validate one-to-one video/label joins, and extract only needed clips into ignored `data/raw/` after calculating capacity. No destructive cleanup command belongs in this script.
- [ ] Inventory all video metadata by sequentially materializing one member at a time if necessary; remove only temporary copies the inventory process itself created. Preserve user archives. Record duration/fps/resolution distributions, failures, and compressed/uncompressed/temporary/cache/weight disk costs.
- [ ] Select and save 20 KoNViD IDs with seed 0, independent of MOS quality or outcomes; select at least three LIVE-VQC portrait IDs for visual checks. Save IDs in config. If fewer valid candidates exist, stop and resolve access rather than shrinking the requirement.
- [ ] Run the tests and inventory CLI; require explicit successful joins and a disk plan that fits free bytes with the configured reserve. Decide whether later extraction retains raw clips; deletion later requires verified feature checksums, not merely a successful timing run.
- [ ] Commit manifests, code, tests and dated decisions: `feat: inventory phase zero datasets`.

### Task 3: Implement and Verify Shared Preprocessing

**Files:** Modify `src/preprocess.py`; create `tests/test_phase0_preprocess.py`; update `ARCHITECTURE.md` §2 and config.

**Interfaces:** `probe_video(path: Path) -> dict` supplies Task 2 metadata. `nearest_indices(pts: np.ndarray, targets: np.ndarray) -> np.ndarray` selects nearest monotonic presentation timestamps. `iter_samples(path: Path, config: dict) -> Iterator[tuple[float, np.ndarray, np.ndarray]]` yields timestamp, upright spatial RGB frame, and boundary-replicated motion RGB window; decoding uses an in-memory rolling buffer. `spatial_tensor(frame: np.ndarray) -> torch.Tensor` and `motion_tensors(frames: np.ndarray, config: dict) -> list[torch.Tensor]` implement confirmed transforms. Spatial samples use `np.arange(0, duration_s, 0.25)`; SlowFast uses 32 offsets `(np.arange(32)-15.5)*2/30` seconds and alpha 4.

- [ ] Write failing tests asserting `nearest_indices([0,.1,.4], [.05,.3]) == [0,2]`; repeated nearest frames are allowed, absent timestamps fail. Assert targets never include the duration endpoint.

```python
def test_nearest_timestamp_and_tie():
    from src.preprocess import nearest_indices
    actual = nearest_indices(np.array([0.0, 0.1, 0.4]), np.array([0.05, 0.3]))
    np.testing.assert_array_equal(actual, [0, 2])
```
- [ ] Add tests for metadata rotations 0/90/180/270, unsupported rotation failure, spatial output `[3,448,448]`, aspect preservation before cropping, and normalization of known RGB values. Use generated temporary test videos for decoder integration; never repository frame dumps.
- [ ] Add short-window tests: 32 frames always produced, out-of-range targets select first/last frame, SlowFast slow/fast lengths are 8/32, output spatial dimensions are 256×256. Verify decoding errors become identified failures, not truncated successful clips.
- [ ] Run `python -m pytest tests/test_phase0_preprocess.py -q`; verify failure, implement the interfaces, then require all tests to pass. Use presentation timestamps rather than frame-number/fps estimates; do not seek between samples.
- [ ] Decode the selected real clips at different frame rates. Visually inspect at least three LIVE-VQC portrait clips in memory and save IDs, applied rotations, inspected timestamps and human verdicts in `results/phase0/rotation_review.json`. Do not mark visual review complete from metadata alone.
- [ ] Update the confirmed decoder details in the architecture; commit: `feat: add shared timestamp preprocessing`.

### Task 4: Load Frozen Backbones and Confirm Their Transforms

**Files:** Create `src/features.py`, `tests/test_phase0_features.py`; modify config and `ARCHITECTURE.md` §§2–3.

**Interfaces:** `load_backbone(name: str, device: str, config: dict) -> torch.nn.Module`; `extract_features(model: torch.nn.Module, samples: torch.Tensor | list[torch.Tensor], name: str, device: str) -> dict[str, torch.Tensor]`. Returns finite detached features; DINO keys are `cls`, `patchmean_L6/L9/L12`, `patchstd_L6/L9/L12` for ViT-B; other backbones return `pooled`. No head or MOS prediction is introduced.

- [ ] Write failing tests using tiny model doubles: evaluation mode and gradients disabled, CPU fp32 output, patches exclude CLS/register tokens, `get_intermediate_layers` receives `norm=True`, missing weights fail rather than using random initialization. Require the documented patch-standard-deviation convention to be recorded in config (proposal: population std, `correction=0`).
- [ ] Run `python -m pytest tests/test_phase0_features.py -q`; implement interfaces and require passing tests. DINO uses the non-register architecture and an exact inspected torch.hub commit; hash downloaded weights. ResNet-50 is a separate control using identical 448 spatial crops.
- [ ] Load actual pretrained DINOv2 ViT-B, ResNet-50 and SlowFast R50 locally; run one real sample on GPU fp16 autocast and CPU fp32. Record output dimensions, finite checks, weight identities and exact transforms. Verify SlowFast sampling, alpha, resize/crop and normalization against its pinned reference implementation; test tensor equivalence on the same synthetic input.
- [ ] If SlowFast cannot load/run, save the failure and choose torchvision R3D-18 pretrained weights as the proposed minimal fallback. Adopt that weight enum's verified transform and temporal requirements, label it `r3d18` everywhere, and update config/architecture/decisions before timing. Never call fallback features SlowFast.
- [ ] Keep ViT-B as default. Benchmark ViT-L only if ViT-B leaves credible CPU/GPU headroom and the user wants the extra candidate; if attempted, pin its intermediate blocks and report its own timings. OOM is a recorded candidate failure, not permission to alter 448 crops or precision silently.
- [ ] Commit confirmed loaders and transform evidence: `feat: verify frozen feature backbones`.

### Task 5: Measure the Local Pipeline on 20 Videos

**Files:** Create `scripts/00_timing.py`, `tests/test_phase0_timing.py`, `results/phase0/timings.csv`, `results/phase0/run.json`.

**Interfaces:** CLI: `python scripts/00_timing.py --config configs/phase0.yaml`. Uses Tasks 3–4 directly. Output rows contain `video_id,backbone,device,precision,repeat,duration_s,T,decode_s,preprocess_s,inference_s,total_s,peak_memory_bytes,status,failure_reason`; run metadata records config/lock/code/weights hashes, cache version, seed, fold=null, head_checkpoint=null, hardware, decoder version, CPU threads, and warmup policy.

- [ ] Write failing tests with a fake clock/extractor: all 20 IDs appear on both devices for every chosen backbone; failures retain rows and prevent a complete status; CUDA synchronization brackets GPU inference; warmup/model download/load time cannot enter steady-state inference columns. CPU-only runs cannot claim CUDA results.
- [ ] Run `python -m pytest tests/test_phase0_timing.py -q`, implement timing, rerun to pass. Use `time.perf_counter`, synchronized GPU measurements, one warmup and three measured repeats. Fix/log CPU thread count. Measure model loading separately; record end-to-end extraction wall time so overlapping stage times are not incorrectly summed.
- [ ] Run the CLI over the fixed 20 KoNViD clips for DINO, ResNet and the confirmed motion backbone on GPU fp16 and CPU fp32. DINO measures all planned intermediate statistics in the same forward pass; motion runs one window per spatial timestep. Process backbones sequentially to fit memory.
- [ ] Validate finite positive times, expected feature shapes and sample counts, and complete successful coverage. Record OOM/decode failures explicitly and rerun only after identifying the cause. Do not replace difficult clips silently.
- [ ] Record a user-accepted CPU budget in seconds per 10-second video based on measurements. Report normalized extraction time as an estimate with head inference and API overhead still unmeasured; reserve headroom rather than asserting Phase 4 compliance.
- [ ] Commit tooling and small measurements: `feat: measure phase zero extraction costs`.

### Task 6: Decide Optional UGC Feasibility With a Parity Gate

**Files:** Conditionally create `scripts/04_parity_check.py`, `scripts/09_ugc_subset.py`, `scripts/10_ugc_fetch_extract.py`, `notebooks/ugc_cloud.ipynb`, `tests/test_phase0_cloud.py`, `data/splits/ugc_subset.txt`, `data/manifests/ugc_metadata.json`, `results/phase0/cloud.json`; update config and decisions.

**Interfaces:** `select_subset(rows: list[dict], seed: int, count: int) -> list[str]`; `minimum_cosine(reference: np.ndarray, candidate: np.ndarray) -> float`. CLIs use `--config configs/phase0.yaml`; parity additionally takes `--reference PATH --candidate PATH`, returning nonzero if any feature/timestep is below 0.999, missing, nonfinite, or shape-mismatched. Cloud fetch additionally takes `--video-id ID` and refuses IDs outside the committed subset or absent passing parity for the current hashes.

- [ ] First assess remaining time/storage and user access to Kaggle (phone verification, GPU, internet, persistent storage); fall back to Colab if unavailable. If neither is usable or UGC threatens the core schedule, record `ugc=skipped` with reason and finish this task without creating unused tooling.
- [ ] If continuing, verify the official UGC bucket pattern, metadata columns, license and notebook terms during execution. Download metadata only locally. Test deterministic selection, unique IDs, category/MOS strata, 720p/1080p-only eligibility, and failure when fewer than 300 eligible clips exist. Use five equal-width bins over the documented MOS range and proportional category×bin allocation with deterministic largest-remainder rounding; propose count=300, seed=0. Save sorted IDs and metadata hash; commit before extracting any UGC video.
- [ ] Add failing parity tests: identical nonzero arrays score 1; 0.998 fails; 0.999 passes; missing timesteps, NaNs, zero-norm vectors and incomplete writes fail. Add smoke-flow tests proving parity failure prevents download/extraction, wrong ID fails, and raw-file deletion occurs only after atomic features, checksums and persistent manifest sync succeed. Run `python -m pytest tests/test_phase0_cloud.py -q` before and after implementation.
- [ ] Generate reference features for 10 fixed KoNViD IDs locally using shared modules and current config; write arrays atomically plus checksums and `ARCHITECTURE.md` cache manifest/meta fields. These are small parity fixtures in ignored cache, not the full Phase 1 extractor. Notebook installs the identical lock, imports identical `src/`, probes system decoder parity, and extracts those 10 clips on cloud GPU. Require cosine ≥0.999 at every timestep for every feature branch intended for UGC.
- [ ] Only after passing parity, download one ID from the saved subset in the notebook, validate size and full decode/frame count, extract and sync features/manifest/checksums, then delete its raw file. Do not stream a video URL or store raw UGC locally. Preserve a failed clip's ID/reason; an interrupted sync cannot be called success.
- [ ] Record measured transfer/extraction times, current quota/session/disk limits with observation date, persistent-storage checks and projected 300-clip cost. Decide go/no-go; log failure and skip UGC if parity or access cannot be resolved within the available schedule.
- [ ] Commit conditional tooling and evidence only: `feat: verify optional cloud extraction feasibility`. Production resumability and the full UGC loop remain Phase 3 work.

### Task 7: Generate the Feasibility Report and Freeze Preprocessing

**Files:** Create `scripts/00_report.py`, `tests/test_phase0_report.py`, `results/phase0/feasibility.json`, `results/phase0/report.md`; modify `ARCHITECTURE.md`, `DECISIONS.md`, `PHASES.md`.

**Interfaces:** `build_report(config: dict, inventory: dict, timings: list[dict], evidence: dict) -> dict`. CLI: `python scripts/00_report.py --config configs/phase0.yaml`; exits zero only when all Phase 0 gates pass, otherwise emits missing/failed gate names and nonzero status. Reports reference saved evidence hashes.

- [ ] Write failing tests: missing deadline/hours/CPU budget/rotation review or incomplete 20-video device coverage blocks completion; failed mandatory backbone checks block; explicitly skipped UGC is allowed; active UGC requires passing parity and smoke evidence; changed config/weights/lock hashes invalidate stale timing/parity evidence.
- [ ] Test projection arithmetic with fabricated measurements: compute per-backbone seconds per decoded second from measured totals and apply to each dataset's inventoried total duration. Show KoNViD and LIVE-VQC separately and combined. Label extrapolation from KoNViD to LIVE-VQC, report repeat variation and an explicit 2× planning allowance; compare with available compute hours. Compare active labor with actual weekly hours and deadline separately. Test insufficient disk or time returns no-go.
- [ ] Run `python -m pytest tests/test_phase0_report.py -q`; implement and rerun to pass. Disk accounting includes archive retention, unpacked or transient raw data, environment, weights, feature arrays, temporary atomic writes and reserve. Predict feature bytes from actual dimensions/timestep counts/dtype; avoid reusing unverified planning estimates.
- [ ] Run `python -m pytest tests/test_phase0_*.py -q` and the report CLI. Require saved measurements for every stage on 20 videos on both devices, confirmed weights/transforms, real portrait review, feasible disk/time, and recorded deadline/hours, decoder, DINO size/commit, motion choice, CPU budget, and UGC decision. Hardware/notebook checks must be explicitly reported separately from offline unit tests.
- [ ] Freeze confirmed spatial and motion preprocessing in `ARCHITECTURE.md`, including timestamp grid, rotation, resize/crop, normalization, precision, interpolation, patch-std convention and backbone transforms. Record the config hash/cache version and all decisions with date. Update extraction/storage estimates in `PHASES.md` from generated report; leave later training estimates clearly unmeasured. If anything changes, rerun affected evidence before claiming completion.
- [ ] Commit report tooling and documentation: `docs: freeze phase zero feasibility decisions`. Hand off to Phase 1 only on a passing report; a no-go means adjust the plan in writing, never bypass the exit criterion.

## Self-Review and Handoff

Coverage: environment/schedule (Task 1), data/license/disk (Task 2), timestamp/rotation policy (Task 3), actual backbone loading/transform confirmation (Task 4), GPU/CPU timing and latency budget (Task 5), UGC selection/cloud smoke with prior parity (Task 6), projections/freeze/exit criterion (Task 7). Review Focus failure classes each have named assertions in their owning task. Shared interfaces and output paths are defined above; later tasks consume them unchanged.

Planning assumptions requiring confirmation during execution: deadline/hours and unattended compute availability; CPU budget after measurements; optional cloud-account access. No runtime results or package compatibility claims are asserted by this plan. FFmpeg, seed 0, repeat policy, UGC binning, 2× projection allowance and R3D-18 fallback are proposed implementation decisions, not previously approved experimental results.

Recommended execution: **Native**, because these tasks share preprocessing/configuration interfaces and real-hardware evidence; implement sequentially with one final independent review. No implementation is authorized by the planning request alone.
