# Final negative-control investigation

**Finding: no concrete implementation or saved-artifact defect detected in the reviewed paths. The final gate remains failed.** This is not proof that the failure is chance, nor evidence sufficient to waive the gate.

## Saved result, independently checked

- Initial control: status `passed`, SRCC `0.07427258946968268`, 95% CI `[-0.05189994539084714, 0.20111890816556402]`.
- Final control: status `failed`, SRCC `0.12918040273838305`, 95% CI `[0.004278085064662253, 0.25522434250352]`.
- Final bootstrap: `10000` valid draws, `0` invalid draws. The absolute-SRCC condition is satisfied, but the CI excludes zero, so the implemented conjunction correctly fails.

## Investigation performed

- Inspected `scripts/05_train.py` control construction, fresh model creation, resume identity, selected-config loading and test prediction; `src/training.py` normalization, training, early stopping, checkpoint restoration and inference; `src/data.py` role selection/order; `src/metrics.py` bootstrap; `src/losses.py` and the mean-pool/shared head.
- Authoritative fold-0 roles contain `816` training, `144` validation and `240` test IDs. Their sets are disjoint and their union matches the inventory. The saved permutation exactly equals the existing seed-0 NumPy permutation in the dataset's training-ID order, using training labels only. Both controls bind permutation SHA `0694ac8a59669426cb1e8e602a32ccb1346ea7055d2192e5dca18e2e10cbd13d`.
- Every initial/final seed output contains exactly the authoritative test IDs in the expected order, with original inventory MOS. Ensemble predictions were reconstructed from the saved seed outputs and match the diagnostic exactly. No cross-seed ordering mismatch or relabeling was detected.
- Loaded all saved control checkpoints on CPU with `weights_only=True`. Checkpoint hashes, per-run and top-level configs, embedded checkpoint configs, seed/run-kind/fold identities, target statistics and selected epochs agree. The final config equals the frozen selected mean-pool config. All control checkpoint hashes are distinct; no initial checkpoint was reused as a final checkpoint.
- Saved target mean/std equal independently computed statistics of the shuffled training-role labels. Because permutation preserves aggregate statistics, this statistics check alone cannot prove that fitting used shuffled labels; the actual code's explicit shuffled list and data flow were also inspected. Validation/test targets are not involved in fitting statistics.
- Each saved best epoch is the earliest maximum validation SRCC in its history, and the checkpoint embeds that same epoch. Training only consumes the shuffled fit list; validation is used under inference mode for epoch selection. Test loading/prediction occurs after fit completion and restoration. Models/optimizers are created separately for each run, and no tuning checkpoint is loaded into control training.
- Re-ran **inference only**, on CPU, from all saved control checkpoints using checksum/source-verified cached test/validation feature branches. Test outputs agree with saved GPU predictions within floating-point tolerance; each reconstructed validation SRCC exactly matches the saved best-validation score. This finds no wrong-checkpoint, inverse-scale or stale-prediction discrepancy.
- Independently recomputed the existing bootstrap using individual SciPy `spearmanr` calls for the same saved-prediction pairs and the original seed-0 resampling sequence. Final point estimate and CI endpoints match exactly; initial endpoints match to floating-point rounding. This used no alternative random seed, permutation, selection criterion or resampling count.
- Compared the actual initial fitting commit against the final fitting commit for the reviewed source files: no differences. The initial diagnostic's later audit commit reflects artifact reuse; numerical provenance remains consistent. Initial-to-final fitting changes are the validation-selected learning rate, weight decay and shared rank weight, plus protocol status. No source change explains the observed shift.

## Interpretation and limits

No corrective code change is supported by these checks. The prescribed control retains true validation labels and selects the best epoch against them; consequently the negative control includes supervised validation-based selection even though gradient fitting uses shuffled training targets. That is the documented protocol, not a newly identified implementation violation. This observation does not establish the cause of this particular failed test and does not authorize changing the control after viewing its result.

The checks establish consistency of the saved artifacts with the inspected implementation; they do not mathematically prove absence of every possible leak, hidden data dependence, source near-duplicates or chance fluctuation. No training was repeated, no permutation/model seed was rerolled, no held-out result was used to select a replacement config, and no thresholds/artifacts were modified. The full test suite and full cache revalidation were not repeated. No CV was launched.

**Disposition:** preserve the failed diagnostic and keep CV blocked. Any next action must respect the binding failure rule; this investigation supplies no basis for labeling the existing result a pass or silently changing the experiment.
