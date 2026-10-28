# Control v2 and Phase 1 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the user-approved control correction and complete Phase 1 only if its unchanged numerical gates pass.

**Architecture:** Keep the numerical trainer, feature caches, splits, protocol and selected real-label configs unchanged. Put amended controls in `control_initial_v2/` and `control_final_v2/`, with a dedicated immutable YAML control policy and independently persisted validation permutation. Reuse original training permutation. Bind the new policy/permutations in control run identities and report validation; retain v1 outcomes in place as historical artifacts.

**Tech Stack:** Existing Python/NumPy/PyTorch/YAML/pytest; no dependency additions.

**Spec:** `docs/superpowers/specs/2026-10-27-negative-control-repair-design.md`, explicitly approved by the user in this conversation. Original Phase 1 plan applies except its training-only shuffle requirement is superseded for negative controls.

## Global Constraints

- Five model seeds; fixed seed-0 training permutation, independent seed-0 validation permutation within its own IDs. True test IDs/labels unchanged.
- The pass conjunction remains CI including zero and absolute SRCC below 0.15; use the existing 10,000 seed-0 draws.
- No alternative permutation, winner, threshold or method is selected from held-out results. If either new control fails, stop and preserve evidence.
- Do not modify caches/preprocessing, immutable splits, numerical training behavior, selected configs, rank/head tuning outputs or original controls.
- Existing measured protocol/budget is verified and reused after the completed grid selection; do not reproject already-completed selection work as new fits.
- Final report explicitly discloses the post-hoc amendment and original failure. No LIVE-VQC prediction or later phase work.

## Review Focus

- Stale legacy controls must not unlock CV; policy and both permutation hashes must verify.
- Permuted labels remain attached to the original role's feature IDs; no target leakage across roles.
- Checkpoint selection uses shuffled validation targets, while held-out reporting uses true targets.
- Real-label fit/tuning identities stay reusable, and a valid frozen budget is not rewritten during resume.
- Missing/failed amended controls or any absent CV/ridge/metric/parity prerequisite must fail the report.

### Task 1: Versioned control contract and tests

Files: `configs/control_v2.yaml`, `src/tuning.py`, `scripts/05_train.py`, `scripts/06_eval.py`, `scripts/continue_phase1.sh`, relevant control/pilot/report tests; dated plan/rules/docs amendment.

- [x] Add failing tests for deterministic independent role permutation, shuffled validation passed to fitting, unchanged held-out targets, policy/permutation provenance, legacy rejection and unchanged real-fit budget reuse.
- [x] Run focused tests and observe missing-interface/old-behavior failures.
- [x] Implement `control_policy()`, `control_directory(root, final=False)` and `permute_control_role(data,role)` in existing tuning utilities; keep all current checksum, role/MOS/seed/CI validation.
- [x] Update every caller and resume commit path. Initial/final controls share immutable train/validation permutations, but use separate original default/selected model configs. Preserve original control directories unchanged.
- [x] Update binding documents with the approved post-hoc amendment before any real amended fit. Explain that validation supervision is a demonstrated path, not a proven cause of the old failure.
- [x] Focused/full tests pass; independent focused review; fix demonstrated Important issues with RED→GREEN tests. Commit source/config/docs.

### Task 2: Real controls, CV/ridge and exit

- [x] Validate complete cache, unchanged committed splits and frozen configs/protocol; run five-seed initial v2 control, then five-seed final v2 control. Never reroll a failure.
- [x] If both pass, execute 100 definitive CV fits and five ridge fits with existing frozen configs. Save predictions and hashes; reuse complete fits, restart partial fits only.
- [x] Run ordinary pytest and the final completeness-first report; require zero missing/failed gates and all mandatory metrics finite.
- [x] Generate tables/ensembles/time accounting and updated phase documents from saved outputs, including explicit original-control failure/amendment disclosure. Commit small artifacts, never arrays/checkpoints.

## Verified execution outcome (2026-10-27)

Completed under the explicitly approved v2 protocol. The single completion worker exited zero after both five-seed controls, 100 definitive neural CV fits, five ridge fits, `.venv/bin/python -m pytest -q` (159 passed) and `scripts/06_eval.py --config configs/protocol.yaml --results results/phase1 --features configs/features_v2.yaml` (passed, no missing/failed gates). The report contains generated tables, 25,200 validated OOF predictions and 6,000 seed-averaged ensemble rows. All 70 snapshotted original/frozen files match their preserved checksums. See `results/phase1/report.md`, `phase_gate.json`, and the three focused reviews under `docs/superpowers/reviews/2026-10-27-*review.md`. No later-phase experiments, merge or push were performed.
