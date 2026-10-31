# Rulings I made

Chronological record for this plan. Earlier pending-check rulings are historical: the completed image, parity, latency and final gate now supersede those pending states. Each entry preserves the reason and its cost.

- Ruling: Execute in the current checkout and do not commit — the approved plan/spec explicitly require both, overriding generic worktree/commit workflow — changes remain local and require later review/integration.
- Task 1: Ruling: Add src/serving_bundle.py to share validation — setup and production need the same fail-closed checks without shipping research scripts — one extra small module must be included in the image.
- Task 1: Ruling: Use uv pip check and explicit index arguments for installation — research interpreter has no pip; uv provides the equivalent environment compatibility check without modifying it — deployment instructions must retain both index URLs.
- Task 4: Ruling: Review local working changes directly — plan forbids commits, so review-package's required nonempty commit range cannot represent this work — reviewer must include the listed untracked files explicitly.
- Final: Ruling: Docker build/decoder/resource-cap readiness remains unjudged — exact static archive prerequisite has not passed, so no image readiness is claimed — cost if wrong: clean-container failures remain undiscovered until Task 4.
- Final: Ruling: Fixed-video full-reference/timestep parity and latency remain unjudged — Task 5 cannot run before the clean-image prerequisite — cost if wrong: numerical drift or request-budget failures remain unknown.
- Final: Ruling: Final report/gate-last completion remains unjudged — Task 6 is pending and no passed Phase 4a gate is written — cost if wrong: phase closure needs further implementation/verification.
- Final: Ruling: Presentation/hosting/final README/publication remain outside this execution — explicitly excluded by the approved scope — cost if wrong: no public demonstration/release is provided.
- Final: Ruling: Public-internet hardening and hostile local actors rewriting code/manifests are not claimed — review covers the authorized local serving boundary — cost if wrong: production exposure needs a separate security assessment.
- Final: Ruling: Forcible process termination cleanup is not claimed — cooperative cancellation retains files/slot correctly; process death cannot execute finally blocks — cost if wrong: a killed local process may leave temporary upload files.
- Final: Ruling: Exhaustive codecs/pathological-media stress coverage remains unjudged — bounded rotation/VFR and failure fixtures cover the declared local path, not every codec implementation — cost if wrong: unusual media may fail or occupy the slot until decoder exit.
- Final: Ruling: Full historical compatibility retains the recorded calendar-test failure — old research source is frozen and the failed test writes real DECISIONS — cost if wrong: that historical test still fails after its deadline; remaining research cases are checked separately.
- Task 4: Ruling: Verify the original decision-log prefix plus exactly the approved append — user requires a dated entry while historical gates pin the original bytes — future unapproved changes still fail; evidence readers must distinguish historical/full-current hashes.
- Task 4: Ruling: Prepare the bundle on the host after full historical verification; builder verifies that portable bundle — preparation requires every gate-bound neural artifact, which conflicts with the specified minimal image context — build depends on a verified pre-build bundle rather than rerunning research preparation inside Docker.
- Task 4/5: Ruling: Prepare evidence tooling/tests during the clean rebuild — this avoids idle time; actual reference/parity/latency commands remain gated on verified Task 4 — progress bookkeeping must retain the pending build verification.
- Task 6: Ruling: Review the complete stable implementation while warm measurements collect — avoids serial reviewer delay; reviewer must not run tests/load while CPU timing is in flight — final acceptance still waits for all saved measurements, tests and any fixes.
- Final: Ruling: Capture an independent source proof from the actual immutable measured image after timing — preserves existing build/parity/latency hashes and verifies identical runtime bytes — provenance requires retaining this extra proof alongside measured artifacts.
- Final: Ruling: Final numerical/publication outcome is executor-verified after static review — reviewer did not introduce test/inference load during timing — no independent rerun of numerical measurements is claimed.
- Final: Ruling: Broader public authentication/abuse/network hardening stays outside local Phase 4a — specified bounded local service is the deliverable — public exposure needs separate assessment.
- Final: Ruling: Forced process termination recovery remains unclaimed — cooperative cleanup is verified, Python finalizers cannot cover process death — hard kills may leave local temp files.
- Final: Ruling: Exhaustive pathological codec/aspect-ratio stress remains unclaimed — declared limits and fixed fixtures/corpus are verified — unusual media may fail or occupy the slot until decoder exit.
- Final: Ruling: Hostile local rewriting of manifests/proofs/amendment records remains outside trust boundary — approved local artifacts are checksum-bound, not signed against the operator — a hostile local operator is not defended against.
- Final: Ruling: Historical calendar-test failure and legacy motion dependency stay separately disclosed — original research source/environment remain frozen — calendar test still fails after cutoff; motion tests require research environment.
- Final: Ruling: Presentation/hosting/README/publication/integration remain excluded — user authorized serving core without commits — local result is not a published/merged release.
- Final: Ruling: Packaging is Linux x86_64 only — pinned CPU wheels and static archive target that architecture — other platforms require separate packaging/parity work.
- Task 6: Ruling: Use report mode as final completion verification — required unit/regression commands already passed and are hash-bound, final publication must run after notes/evidence — preserved test logs remain the detailed test record.

# Deferred minors

- Final: minor (deferred): unusual decoder process-wait TimeoutExpired can yield generic API 500 / CLI traceback; failure returns no score and cleanup still runs. Documented in review.md and operations; no shared decoder modification.
