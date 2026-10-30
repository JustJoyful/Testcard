# Post-hoc matched neural permutation null

All 200 predetermined permutation seeds (1–200) completed with five model seeds (0–4) each. Each model used the historical frozen ResNet mean-pool config and three fixed epochs on fold-0 train plus validation rows, with within-role shuffled labels. Test MOS was used only for evaluation.

The historical five-seed ensemble SRCC is -0.156308. The central 95% null interval over per-permutation ensemble SRCCs is [-0.233694, 0.158288] (linear 2.5th and 97.5th percentiles).

Predeclared interpretation: **consistent with chance under the control’s own procedure**.

This is a finite post-hoc Monte Carlo diagnostic, not proof of the absence or presence of leakage. The interval is a permutation interval, not an accuracy confidence interval. The original control remains **failed**, and the Phase 3 gate remains **passed with qualification**. All original output, source and pinned decision hashes were verified unchanged.

See [protocol](../NEURAL_PROTOCOL.md), [per-permutation results](per_permutation.csv), and [summary and hashes](summary.json). Per-model checkpoints, run records and label-free test predictions are retained in each permutation directory.
