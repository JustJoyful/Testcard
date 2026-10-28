# Proposed negative-control protocol correction

Status: explicitly approved by the user on 2026-10-27 (“Approve the correction and complete Phase 1”). This is a post-hoc protocol amendment following a failed original control, not a claim that the original result passed.

## Goal and evidence boundary

Complete Phase 1 with an honest, reproducible negative control, the unchanged matched heads/selection protocol and definitive five-fold/five-seed CV/ridge outputs. The original final control failed its CI requirement. Main and independent audits found no concrete implementation defect; they did verify that checkpoint selection uses true validation MOS. That real-label selection path is intentional in the existing plan and could contribute to a nonzero control association. Its contribution to this particular result is not established.

## Approaches

1. Retain the original protocol exactly: keep the failed gate and stop CV. This preserves the predeclared rule but cannot complete Phase 1 on the existing evidence.
2. Recommended proposal: version the negative control so both gradient fitting and checkpoint selection use permuted targets. Keep the same fixed seed-0 training permutation; independently permute labels within validation only, using a predeclared seed 0 and deterministic validation-ID order. Keep held-out test labels/IDs unchanged. Preserve the same five model seeds, selected model config, optimizer, rank weight, early stopping, bootstrap draws/seed, and conjunction of CI including zero plus absolute SRCC below 0.15. Rerun both initial and selected-config controls once under this declared policy. A new failure remains a stop; do not search permutations or methods for a pass.

## Contract and scope

- Persist a control-only versioned policy and training/validation permutation provenance. Never mix labels or IDs across roles.
- Archive the original initial/final control artifacts, including the failure, unchanged. Active controls use new paths/identities and cannot resume old results as complete.
- Real-label fitting/tuning behavior, numerical trainer, feature caches, frozen splits, rank selection and all selected head configs remain unchanged. Existing real-label selection work is reusable because only the negative-control construction changes.
- The artifact validator and exit report must require the approved control policy, exact role memberships, independently reproducible permutations, checksum-verified checkpoints, all five seeds and the original numerical pass thresholds.
- Preserve previous outcomes and explicitly disclose the post-hoc amendment in DECISIONS, the plan and final report. Passing the amended diagnostic does not erase the original failure or establish absence of every possible leak.
- After both amended controls pass, run the definitive CV/ridge grid, generate tables/ensembles, run the ordinary test suite and require the complete Phase 1 exit report to pass. If either diagnostic fails, stop with evidence; no further automatic changes.

## Validation and rollback

Test target/feature association, role isolation, deterministic independent permutations, validation-target perturbation before checkpoint choice, rejection of legacy/stale policy artifacts, all-five-seed/CI gates, and reuse of unchanged real fits. Exercise the real controls with current cached data. Original artifacts/configs remain recoverable; rollback restores the original active policy and its failed gate, without deleting results or relabeling them.

## Approval boundary

The execution plan explicitly says to permute training-role labels only and preserve true validation/test labels (Tasks 7 and 9). This proposal changes that research procedure. Generic requests to fix code do not silently authorize reclassifying the observed failed experiment. No pass threshold, held-out labels, hyperparameter winner or split may be changed to obtain a pass.
