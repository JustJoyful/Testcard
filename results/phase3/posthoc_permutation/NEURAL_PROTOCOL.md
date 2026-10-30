# Matched neural permutation null: predeclared protocol

Written 2026-10-29 before running the matched neural diagnostic.

Diagnostic only. The historical fixed-epoch control stays failed. The Phase 3 gate, its output hashes, source digest, frozen settings and pinned decision are unchanged. No permutation is used to replace the failed control or choose a model.

Use exactly permutation seeds **1 through 200 inclusive**, excluding historical seed 0. For each permutation, independently permute training and validation labels within their existing fold-0 roles by restarting `numpy.random.default_rng(permutation_seed)` for each role, matching the original control's seed convention. Concatenate shuffled training rows followed by shuffled validation rows, as the historical control does. Test labels remain true and are used only for evaluation.

For each permutation, train model seeds **0, 1, 2, 3, 4** with the frozen selected ResNet-50 mean-pool configuration and exactly **3 epochs**. Reuse `seed_everything`, `build_head`, `fit_fixed_epochs` and `predict`, including the original CUDA device, optimizer, loss, warmup, dropout, standardization, batch generator and fixed final checkpoint. Do not use early stopping, validation selection, retuning, or replacement seeds. On failure, stop and resume the same predetermined work; do not discard a permutation. Retain checkpoints, per-model runs and test predictions, and per-permutation ensemble predictions and SRCC.

Compute each permutation's test SRCC from the arithmetic mean of the five models' raw predictions per test video. Compute the central 95% null interval as NumPy's default linear quantiles at 0.025 and 0.975 over all 200 ensemble SRCCs. Read the historical ensemble's full precision value from the preserved investigation, rather than using its rounded display value.

**Predeclared interpretation:** if the historical ensemble SRCC lies within that interval, endpoints included, report **“consistent with chance under the control’s own procedure”**; otherwise report **“not explained by chance; investigate.”** This finite Monte Carlo diagnostic does not prove the absence or presence of leakage. Its interval is a permutation interval, not a confidence interval for accuracy. Do not apply this interpretation to a partial run.

The script lives beside the ridge diagnostic to preserve the frozen source digest. It writes new evidence under `neural/` only. Input, script and protocol identities are pinned in its execution record and checked on resume. The original gate's output, source and decision hashes are checked before training and before final reporting. An exclusive process lock prevents concurrent writers.

Preflight without training:

```bash
.venv/bin/python results/phase3/posthoc_permutation/neural_null.py --check-only
```

Run or resume:

```bash
.venv/bin/python results/phase3/posthoc_permutation/neural_null.py
```
