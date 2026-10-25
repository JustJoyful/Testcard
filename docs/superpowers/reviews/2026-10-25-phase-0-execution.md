# SDD ledger — plan: docs/superpowers/plans/2026-10-25-phase-0-preflight.md

Spec: RULES.md, PHASES.md, ARCHITECTURE.md, PRD.md, DECISIONS.md (read).

Pre-flight shared interfaces:
- Tasks 1→2–7: shared config and exact dependency lock; missing decisions remain null and block the exit gate.
- Tasks 2→3/5/7: probe schema, labels, 20 seeded KoNViD IDs and 3 portrait LIVE IDs; no labels are present in the KoNViD video ZIP.
- Tasks 3→4/5/6: shared timestamp decoding and confirmed transforms; motion reference transform must be checked before timing.
- Tasks 4→5/6/7: actual pretrained weight hashes and dimensions; no random-weight substitution.
- Tasks 5/6→7: complete measurements and current evidence hashes; absent evidence must prevent passing.

Ruling: Preserve the empty, read-only .git directory and work in the current workspace without commits — the plan explicitly permits retained files when Git is not functional — no branch isolation or Git recovery is available.
Ruling: Maintain this plan's ledger manually at the prescribed workspace path — sdd-workspace fails because git rev-parse cannot resolve a repository — automatic task-start/task-done and commit-range review cannot run.
Task 1: in progress; no tasks complete.
Task 1: Ruling: Use the installed Python 3.11.15 for the isolated environment — it is a more conservative target for PyTorchVideo and notebook compatibility than system Python 3.14; compatibility remains unverified until installation — notebook compatibility may still require another environment.
Task 1: Blocked: dependency installation cannot resolve pypi.org; NVIDIA driver is unavailable. Do not fabricate a requirements.lock or claim GPU measurements.

Task 1: User supplied deadline 2026-10-31; laptop can remain open for unattended runs. Numeric working/compute hours remain unspecified; do not infer approval of a numerical feasibility allowance.
Task 2 preliminary read-only evidence: both archive CRC scans passed; unsafe members and duplicate video IDs absent; SHA-256 before/after identical. No bulk extraction. Full inventory/labels/probes not implemented or claimed.

Task 1: User host verification supersedes driver-absence interpretation: nvidia-smi passes on host; git init succeeded in user shell. Agent access and dependency connectivity still require checks.

Task 1: Dependency installation now succeeds in user shell; agent uv pip check and core imports pass. Agent torch.cuda.is_available() is false. Fresh lock reinstall and real backbone checks remain pending.
Task 2 preliminary join: KoNViD metadata CRC passes; dedicated MOS table matches all video IDs one-to-one. Config label source set; source hash and detailed checks in results/phase0/konvid_label_join.json.
Task 1: Ruling: Use uv pip install/check for fresh-lock verification — the uv-created environment has no pip, and uv installed all 49 exact pins offline from a private writable copy of the host cache — package resolution is checked, not network reproducibility or cloud compatibility.
Task 1: Verified env/requirements.lock in .venv-check; 49 packages install and pip check passes; torch/torchvision/NumPy/YAML/pytest/SciPy imports pass. CUDA remains false only inside agent sandbox. SlowFast architecture imports/constructs; actual pretrained weights still pending.
Task 1: Ruling: Continue independent implementation with missing numerical schedule fields as null — the shared config explicitly defines these as final-gate blockers, not a ban on offline code — phase completion remains blocked until supplied.
Task 3: Ruling: Follow the official SlowFast reference's PackPathway linspace(0,T-1,T//4).long() rather than literal every-fourth indexing — the spec requires confirmation against the weight reference and the official implementation includes the final fast frame — changes slow-path indices, so must be frozen and recorded before caching.
Task 2/3: Ruling: Inventory calls probe_video(timestamps=False), extraction defaults to full PTS probing — full ffprobe frame scans unnecessarily decode every dataset video during inventory; metadata-only probe has a real generated-video test — frame-level decode corruption remains an extraction-time failure, not an inventory success claim.
Task 3: Rotation fixture initially failed because FFmpeg 9 no longer writes a display matrix from legacy rotate metadata. Inspected the generated stream and FFmpeg help; switched fixture creation to -display_rotation. Generated rotation now passes without changing production orientation policy.
Task 4: Ruling: Add scripts/00_backbone_check.py for the mandatory real-weight smoke — user will execute GPU jobs from Kitty, so actual six-device checks need a concrete runnable entry point — adds one small script; successful tests never stand in for real hardware evidence.
Task 1: Worktree attempt failed creating .git/worktrees/phase0 (read-only filesystem). Continue in current workspace using skill sandbox fallback; commits remain unavailable. No source branch exists yet.
Task 6: User explicitly defers all cloud work to the start of Phase 3 UGC after LIVE-VQC results; retain metadata-only subset selection if practical. Attempted official webpage and bucket metadata through web tools; CSV/attribution could not be retrieved and schema not verified. No guessed schema or subset is committed.
Task 2/3: Metadata-only probe regression found that ffprobe -show_entries frame=... implicitly enables frame scans even without -show_frames. Added a failing subprocess-boundary characterization test, removed the frame section in metadata-only mode, and reran successfully. Full inventory was restarted; no samples had been frozen by interrupted runs.
Task 2: Deflated-member corruption exposed zlib.error escaping ZipFile.testzip. Added test RED→GREEN; inspect_archive now reads each member under a named error boundary and reports the corrupt member.
Task 3: Offline preprocessing suite passes including exact installed-reference motion transform comparison. Real human portrait verdict remains pending.
Task 4: Offline feature suite passes including failed-weight/no-CUDA handling; no pretrained or CUDA execution is claimed.
Task 5: Timing tests pass for complete failed-row coverage, no fabricated CUDA rows, and synchronized inference timing. Real 360-row measurement remains pending host GPU execution.
Task 7: Report tests pass for missing/stale evidence, projection arithmetic, disk/compute/labor limits and optional UGC treatment. No freeze or go result is claimed.
Verification: OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q → 46 passed (before final review).
Task 2: All 1200 KoNViD and 585 LIVE-VQC metadata probes/label joins pass; archive SHA-256 values match preflight originals. Fixed 20 KoNViD IDs selected by seed 0 and 3 portrait IDs E001/G001/G002 recorded. Selected raw clips materialized atomically and checksums verified. Cache/weight disk projection remains pending real dimensions.
Task 3: Full real decode checks pass on six selected clips across differing frame rates; sample T/window lengths validated and evidence saved. User confirmed E001/G001/G002 in-memory previews are all upright; human rotation review passed and decoder marked confirmed.
Task 2: Ruling: Reserve 5 GiB of additional disk capacity — conservative local preflight allowance in available storage, not a measured model requirement — could reject a feasible but tighter disk plan; adjust explicitly if necessary.
Final review: independent read-only gpt-6-astra review; two Important findings, no Critical findings or deferred minors.
Final: fixed decision-only config updates invalidating measurements — test_decision_only_config_hash_change_keeps_existing_measurements and test_execution_fingerprint_ignores_decisions_but_tracks_runtime_inputs RED→GREEN, suite 50/50.
Final: Ruling: Retain full config hash for audit but validate measurements against extraction-affecting config, code, lock, weight and manifest hashes — post-measurement CPU budget/schedule decisions cannot change already measured features — a future runtime input must be added to the explicit execution fingerprint or stale evidence could be accepted.
Final: fixed interrupted inventory publishing inconsistent manifests — test_interrupted_inventory_preserves_published_manifest and test_partial_manifest_with_stale_success_summary_is_no_go RED→GREEN, suite 50/50. CSV publication is atomic; report requires exact summary/manifest hash agreement. Existing complete manifests were revalidated against archive IDs before binding hashes.
Final: Ruling: Reviewer deferred actual pretrained/CUDA/OOM/latency judgment — host smoke and timing remain mandatory before go — real hardware failure may still require a motion fallback or scope cut.
Final: Ruling: Reviewer deferred portrait judgment — user has now confirmed all three decoded previews are upright and verdicts are saved — uninspected timestamps may still contain content ambiguity, but metadata/synthetic rotation paths and full selected-clip decoding are checked.
Final: Ruling: Reviewer deferred cloud parity/cache-write/smoke checks — user explicitly moved cloud work to Phase 3; no UGC extraction or feature cache writer exists here — Phase 3 must implement and pass its gate before UGC extraction.
Final: Ruling: Reviewer deferred schedule/freeze judgment — numeric labor/compute limits and real benchmark evidence remain unset, so report remains no-go and preprocessing unfrozen — Phase 1 cannot begin on the current evidence.
Final: Ruling: Reviewer deferred later training/serving/predictive judgment — outside Phase 0, preserved in later phase criteria — no prediction-quality or serving-parity claim follows from these tests.
Final: Ruling: Reviewer deferred network/notebook compatibility judgment — fresh local offline-lock installation passed, while notebook checks move with cloud work — a free notebook may require a compatible environment adjustment and dependent parity rerun.
Final verification after review fixes: OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q → 50 passed in 3.78s. Backbone-check/timing CLI help smoke passed. Report CLI exits 1 with named missing gates, as expected without real weights/timings/budgets; results/phase0/feasibility.json and report.md generated from saved inputs.
Handoff: resume hardware-dependent Task 4 in host Kitty with 00_backbone_check.py, then Task 5 with 00_timing.py. Both take --config configs/phase0.yaml. No task with pending hardware/schedule/commit requirements is marked complete. Keep workspace/ledger for continuation; final branch finishing is not applicable yet.

Task 4: Actual host pretrained smoke passed on all three backbones and both devices; evidence hashes verified current.
Task 5: All 360 saved timing rows pass complete coverage, finite/positive stage time validation and current hashes; run status complete. User accepted 120-second budget per 10-second video.
Task 7: Measurement-derived extraction/storage estimates generated and PHASES.md updated. Only schedule/labor allowances remain pending; preprocessing remains unfrozen and Phase 1 is not authorized by the current no-go report.

Task 7: Ruling: Use a one-hour engineering allowance for remaining Phase 0 review/docs — implementation, independent review and hardware measurement are done — manual wrap-up may exceed this; later phase effort is explicitly excluded and unmeasured.

User scope amendment: deadline 2026-11-04, ~8 hands-on h/day and ~10 unattended GPU h/night for 10 days. Cut SlowFast, all UGC/cloud work, separate layer comparison; cache layers anyway. Equal tuning budget 10/head; serving FastAPI + upload page + one Dockerfile. Each cut logged individually in DECISIONS.md and propagated across active planning docs/configs.
Ruling: Preserve the completed three-backbone benchmark config/rows as historical evidence and add a spatial-only active v1 policy — deleting/relabeling measurements would lose provenance, while decision-only schedule changes leave execution hashes valid — the readiness envelope is conservative and consumers must use configs/scope.yaml + features_v1.yaml for active research.
Ruling: Record v1 freeze in a separate source/config/lock/weight identity artifact after readiness passes — original measurements used the historical phase0-unfrozen cache label and no production cache exists — future extractors must validate the v1 policy rather than treating the benchmark cache label as the active cache version.
Phase 0 complete under user-amended scope: readiness report exits 0; complete historical measurements retain valid hashes; active spatial projection fits available compute/disk; spatial v1 policy frozen; 50 unit tests pass. Training/serving claims remain pending later phases.
Integration limitation: no commits exist and .git is read-only to this agent. Keep all files and this workspace/ledger; do not delete the only execution record or present a merge/push choice for a nonexistent development branch.

Final scope verification: 50 tests passed; report CLI exits 0; original run/backbone execution hashes remain valid; scoped projection and v1 freeze hashes verified. No Phase 1 folds/training started. All changes retained uncommitted because agent .git writes are blocked.
