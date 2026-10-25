# Phase 1 whole-branch review

Reviewed baseline `83a30f0` through `f9815ab`, the implementation plan (including its five Review Focus items), RULES.md, PHASES.md, ARCHITECTURE.md, frozen feature config, and ledger rulings. Source, tests, CLI orchestration and artifact consumers were inspected. The existing 128-test suite was not rerun. Isolated reproductions used temporary directories and CPU only; no source, cache, protocol, split or Git state was changed. This review file is the sole requested checkout write.

## Strengths

- Cache publication removes old manifest references before a group repair and only publishes a complete group; readers validate checksums, shape, dtype, finite values and requested source identities. Dataset coverage is separate and authoritative inventory IDs are retained.
- All heads sanitize hostile padding before shared projection. GRU packing, valid-prefix reversal, causal convolution and the independent scalar scan/gradient tests address the consequential numerical boundaries.
- Training and selection are structurally separate from test prediction. Fit-role MOS normalization, restored best checkpoints, shared rank loss, fixed candidate ordering and complete tuning seed sets are implemented clearly.
- CV predictions have checksum-bound run/checkpoint identities; OOF coverage and wrong kinds/labels are checked, and undefined metrics remain failures. Parity uses minimum cosine over timesteps and rejects zero norms.
- The ledger correctly leaves hardware execution and phase exit pending; there is no unsupported completion claim.

## Issues

### Critical

None found.

### Important — fix before the expensive selection/CV workflow

1. **An audit-only commit change blocks rank/head-selection resume.** `scripts/05_train.py:167` (exact comparison at 171), callers at 187–193 and 212–220.

   `fit_one` deliberately excludes `code_commit` from compatibility, in agreement with the ledger ruling. However, rank/tune reconstruct each selection result and its rows with the *current* provenance, then compare the entire result against the immutable YAML. A documentation or reporting commit between invocations therefore rejects a compatible frozen result. In a partially completed `--mode tune` run, the next invocation hits the already-finished mean-pool head and stops before it can continue later heads. It also rewrites `selection.json` with the new audit provenance before discovering that YAML is incompatible.

   **Confirmed reproduction:** In a temporary cwd, patch only selection prerequisites, role loading and fitting with deterministic completed validation results. Call `rank_runs(protocol, {'code_commit':'a','split_sha256':'same'})`, then repeat with only `code_commit:'b'`. Actual result: `ValueError: incompatible immutable config: configs/rank_loss.yaml`. Both data/config identity and validation results are unchanged. Tune uses the same persistence function and reconstruction pattern.

   **Fix:** Preserve already-frozen selection artifacts when substantive identity and scores match, preserving their original audit commit; do not overwrite selection JSON before validating compatibility. Apply consistently to rank and each selected head. Add a partial-head-tuning restart test across an unrelated commit, asserting completed fits are skipped and remaining heads continue.

2. **Freezing protocol metadata makes its completed pilots incompatible with their own resume command.** `scripts/05_train.py:123`, `scripts/05_train.py:157` (config written at 159, budget at 161), together with `src/training.py:79`.

   Pilots hash the entire draft protocol into each fitting config. The same command then changes `status` to `frozen`, and possibly `tuning_seeds` to `[0,1]`, in the protocol on disk. Repeating `--mode pilot` now constructs a different fit identity even though no pilot training setting changed. A failure between the protocol write and the budget write leaves a frozen protocol with missing/stale budget; the documented CLI cannot repair it because pilot reuse fails first. The allowed two-seed no-go fallback also changes metadata before a later retry.

   **Confirmed reproduction:** Fit a tiny CPU mean-pool pilot through the real `fit_run` using draft protocol (two epochs, eight synthetic videos). Invoke `pilot_runs` with the same protocol except `status='frozen'`, mocking only control acceptance and role loading. Actual result: `ValueError: .../pilot/mean_pool: incompatible completed run`. No training-relevant field changed.

   **Fix:** Separate the immutable pilot fitting snapshot from mutable gate/selection metadata, or reuse a validated saved pilot snapshot when completing the freeze transaction. Preserve the original actual pilot completion time on reuse rather than stamping a later invocation time. Add fault injection after protocol publication/before budget publication and verify the CLI completes without new fitting; also cover the two-seed transition.

3. **The standalone phase report does not require the final control to use the selected mean-pool config.** `scripts/06_eval.py:175`–177.

   `cv_runs` correctly compares the validated final diagnostic's `config_sha256` with the selected mean-pool config (`scripts/05_train.py:303`–305). `phase_report` validates that a final control passes on the current data but discards this relationship. An otherwise internally valid final-control directory for another config can certify the phase alongside the selected CV outputs. This is precisely the standalone report's responsibility when consuming saved artifacts, even if the normal producer initially wrote correct outputs.

   **Confirmed isolated gate reproduction:** Supply successful prerequisites and selected config B; return a valid final-control diagnostic whose config hash names A; provide complete neural/ridge OOF rows and successful independent gates. The real `phase_report` returns `status: passed`. The actual OOF checker runs in this reproduction; unrelated gate consumers and summary computation are isolated doubles. No check in this report compares A with B.

   **Fix:** Apply the same selected-mean-pool config comparison as `cv_runs` in the final-control report gate. Add a passing report fixture and then replace only its final-control evidence with valid evidence for a different config; require nonzero exit and an explicit final-control failure.

4. **Reusable control evidence is not bound to authoritative held-out IDs/MOS or the actual training permutation.** `src/tuning.py:35`–59, especially 48–59.

   The checker proves that all five output files agree with their own diagnostic and recomputes the bootstrap. It never verifies that those IDs are exactly fold-0 test IDs, that MOS equals the source inventory, or that the saved permutation covers the fold-0 training role with the declared seed-0 mapping. The provenance hashes do not establish these relationships: they merely repeat the expected input identities. Therefore self-consistent wrong-role or relabeled outputs can pass the mandatory leakage gate and authorize selection/CV. CV evaluation already performs the corresponding authoritative label/role checks; controls need them too.

   **Confirmed reproduction using the real `require_control`, with no checker mocks:** Create five checksum-bound temporary control artifacts using the first 240 **fold-0 training** IDs from the committed split, MOS `arange(240)` (not the inventory MOS), seeded independent predictions, and an empty permutation JSON with its correct hash. Recompute the normal 10,000-draw diagnostic. It has SRCC `-0.10372141877463148`, CI `[-0.23178309322946428, 0.022590002180218362]`; `require_control` returns `passed`. The reproduction used test-only checkpoint bytes because the checker only hashes these files; the point is the missing role/MOS/permutation relationship, not checkpoint authenticity.

   **Fix:** Give the checker the declared inventory/split inputs (or resolved expected role/label maps), require exact fold-0 test coverage and authoritative MOS, and validate the complete fixed training permutation. Add tests that retain valid internal checksums/diagnostics while substituting one wrong-role ID, one MOS, an omitted video, and an invalid permutation.

5. **An expected Git prerequisite failure aborts before replacing a stale passing phase report.** `scripts/06_eval.py:167`–174; source at `scripts/05_train.py:28`–30.

   `prerequisites` uses `subprocess.run(..., check=True)` to reject uncommitted/changed split artifacts. Its `CalledProcessError` is not caught by `phase_report`'s gate exception list. Consequently the process exits nonzero but never writes the required failed `phase_gate.json`/report; a previous passed report remains on disk, misleading consumers that inspect saved evidence. This is an expected prerequisite failure, not an unexpected programmer exception.

   **Confirmed reproduction:** Write a passing report in a temporary results directory, then rerun `phase_report` with its prerequisite raising `CalledProcessError(1, ['git','diff','--exit-code','HEAD'])`. Actual: the exception escapes and the old passed JSON remains byte-identical.

   **Fix:** Translate Git prerequisite failures to a descriptive `ValueError`, or catch `subprocess.CalledProcessError` in the gate wrapper. Add an integration-level dirty/untracked-split failure case that asserts a failed report is published and replaces an existing passing artifact.

### Minor / plan completion items

- **Ridge CPU budget evidence is incomplete before selection.** `src/tuning.py:94` records only the 35 alpha-fit count; `scripts/05_train.py:271`–291 measures ridge later during final CV. Task 7 calls for separately timed CPU ridge work. Add the small timing/projection evidence before declaring Task 7 fully complete; do not charge it as GPU time.
- **No-go/calendar decisions need durable evidence and a correctly dated follow-up.** `scripts/05_train.py:139` throws before publishing a calendar failure; 154 hardcodes `2026-10-25` for a fallback that may execute on another day; 163 requests a scope proposal but does not record one. Keep the automatic stop, save the failed gate with the actual date, and complete the required concrete scope proposal in the execution ledger/DECISIONS before restarting. Choosing the human scope cut is an execution decision rather than something to fabricate automatically.

## Meaningful missing tests

The unit suite is strongest around numerical helpers and weakest at the state transitions that join them. A small temporary-artifact orchestration fixture should cover draft control/pilot → freeze/fallback → rank → partial head tuning → commit-only resume → final control → CV/report. It can use cheap fitting doubles while retaining actual identity/persistence/gate code. Existing `test_test_label_perturbation_cannot_change_fit` changes only an audit string, not test labels through orchestration; existing CV reuse coverage calls only one `fit_run`; existing phase-report coverage checks missing inputs but never starts from a complete passing artifact set. Extend these to the real joins above. Full hardware reruns or a duplicated broad suite are unnecessary for these fixes.

## Declined to judge

- Real extraction completion, CUDA head convergence/runtime, five-seed controls, full tuning/CV/ridge and final CPU/GPU parity success: these are explicitly pending live evidence, and review did not compete with the running extraction/parity jobs.
- Multi-writer cache races: the approved plan explicitly schedules one writer and excludes lock implementation.
- Invalidation on unrelated code commits: the ledger explicitly rules code commit audit-only; reviewed consistency with that ruling instead of demanding a new source-fingerprint system.
- Phase 2 statistical comparisons/Holm reporting, LIVE-VQC prediction, cloud and deployed-head serving parity: outside this Phase 1 implementation and explicitly later/cut scope.
- Automatic selection of a further research scope cut after a final no-go: requires the planned recorded execution decision; the review checks that failure remains a stop rather than inventing an unapproved cut.
- Cryptographic proof that saved hardware evidence was genuinely measured, beyond the planned artifact checks: no test-attestation system is part of the approved design. Findings above concern concrete input/config relationships the consumers can and should validate.

## Assessment

**Ready to merge? With fixes.** The core numerical and data separation work is solid, but ordinary restart transitions can block expensive work and the standalone exit/control consumers do not yet enforce all promised relationships. Fix the five Important findings and their focused orchestration regressions before launching rank/head tuning. Hardware milestones remain pending independently of this review.

## Implementer fix evidence

All five Important findings were reproduced before fixes and now have passing regressions. Coverage includes partial head tuning across a commit, four-head pilot reuse after protocol freeze/two-seed metadata transition, fault injection between protocol and budget publication followed by CLI repair, a selected-config mismatch in an otherwise passing report fixture, self-consistent wrong-role/MOS/missing-ID/permutation controls, and replacement of a stale passing report on Git failure.

Both plan-completion items were addressed in the same pass: ridge CPU timing now runs before selection; calendar/no-go decisions persist with the actual local date and a concrete proposal that does not activate another cut. Generated exit reporting includes job wall time, separate CPU costs and unmeasured hands-on time.

Verification: `.venv/bin/python -m pytest -q` → **143 passed**. This is implementer verification, not a second independent review. Real extraction, parity, controls, pilot, tuning and CV/ridge remain separately gated; this review does not certify pending hardware outcomes.
