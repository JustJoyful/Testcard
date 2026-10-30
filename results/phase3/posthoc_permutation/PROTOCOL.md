# Post-hoc diagnostic protocol

Recorded before executing these diagnostics on 2026-10-29.

Hypothesis: a fixed shuffled-label realization can induce correlated predictions across model seeds; seed averaging need not remove such a shared direction. Test whether a surrogate ridge procedure yields test associations of comparable magnitude under random permutations. This cannot establish the neural failure's cause or rule out leakage.

Use existing fold-0 ResNet features and label ordering. Verify the original seed-0 train and validation permutations against the saved mapping, then compute their permuted-versus-true label SRCCs. For the ridge surrogate, use exactly permutation seeds 0 through 999, restarting NumPy's generator separately within each role as in the original control. Combine train and validation rows for fitting; preserve true test labels for evaluation only. Standardize features on fitting rows and use the frozen Phase 3 ResNet ridge alpha, without tuning. No permutation is selected for deployment or gate replacement.

Save each test SRCC, the central 95% permutation interval, the historical-permutation ridge predictions, descriptive tail fractions for the failed neural ensemble statistic, input hashes and a generated report. The interval is not a bootstrap CI and the cross-model tail fractions are not neural-control p-values. Check the batched solve against the shared ridge implementation. Preserve the gate, control classification, thresholds, original outputs, source code and pinned decision text.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python results/phase3/posthoc_permutation/diagnose.py
```

The script is kept with the supplementary evidence so it does not change the frozen Phase 3 source tree digest. This is diagnostic work after held-out results were seen and must remain labeled post-hoc.
