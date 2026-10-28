# Final shuffled-label control investigation

Phase 1 remains blocked before CV/ridge under RULES.md Rule 10. No configurations, model seeds, training permutation, split memberships, thresholds or bootstrap seeds were changed. No additional fitting was launched to obtain a passing result.

## Observed failure

Final control SRCC: 0.12918040273838305; two-sided 95% percentile CI: [0.004278085064662253, 0.25522434250352]. The absolute-SRCC condition passes, but the interval excludes zero. There were 10000 valid and 0 invalid draws. Initial control SRCC: 0.07427258946968268, CI: [-0.05189994539084714, 0.20111890816556402].

## Checks performed

- Authoritative role membership: {'train': 816, 'val': 144, 'test': 240}; roles are disjoint and cover all KoNViD IDs. Split metadata/checksum verifies.
- Saved permutation exactly matches the declared seed-0 permutation of training-role IDs; unchanged between initial/final controls. Self-label positions: 0.
- Target mean/std match training-role MOS only. Permuted versus original training-label SRCC: 0.04506344208470326 (diagnostic only; no permutation was reselected).
- All five initial and final checkpoints/config hashes verify, prediction IDs equal the fold-0 test role in declared order, and test MOS equals the authoritative inventory.
- No cross-role duplicate source checksums or byte-identical CLS feature arrays were found.
- Seed-averaged predictions and all bootstrap draw accounting reproduce from individual saved outputs. An independent scalar rank/bootstrap implementation using the same predeclared draws yields the same CI.
- Source inspection: `fit_one` receives train/validation datasets only; test-role loading/prediction occurs after selecting/restoring each checkpoint. Control training labels are replaced inside the training role only. Validation/test labels remain true as the plan requires. Rank/head tuning use fold-0 train/validation only.
- Ordinary offline verification: `.venv/bin/python -m pytest -q` → 147 passed. Unit tests do not override a failed real control.

## Conclusion and limits

The failure is reproducible. No implementation defect has been identified by these checks. This is not proof that the association is chance, nor proof that every possible leakage mechanism is absent. The mandatory gate therefore remains failed. Changing the permutation, dropping a model seed, choosing a different winning config from held-out control outcomes, or relaxing the criterion to obtain a pass would change the experiment after seeing its result and is not performed.

The four selected head configs and all tuning results remain frozen and saved. No definitive CV/ridge run has started, and no LIVE-VQC MOS prediction was made. An explicit research-protocol decision is needed before bypassing or replacing the declared gate; a generic request to continue does not establish that decision.

Machine-readable evidence: `audit.json`, `diagnostic.json` and the five seed subdirectories in this folder. Independent review is stored alongside this report when it completes.
