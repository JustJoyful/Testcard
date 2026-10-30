# Post-hoc permutation diagnostics

Date: 2026-10-29. Diagnostic only; the original control remains failed and Phase 3 remains passed with qualification.

The historical permuted-versus-true label SRCC is 0.045063 for training and -0.072328 for validation. These correlations alone cannot measure alignment with a quality-related feature direction.

The ridge null uses 1000 predetermined permutation seeds, independently within train and validation, and fits their combined rows as in the fixed-epoch control. The frozen ridge alpha is 1000; no alpha selection uses permuted validation or test scores. Feature standardization uses fitting rows only. Test labels are used only for evaluation.

The surrogate null has mean 0.001002, SD 0.091531, and central 95% interval [-0.183023, 0.168492]. This is a permutation interval, not a bootstrap confidence interval. The historical permutation produces ridge test SRCC -0.050479.

The neural ensemble reference SRCC is -0.156308; it falls inside this surrogate interval. The fraction of ridge permutations at or below it is 0.042; the fraction with at least its absolute magnitude is 0.088. These are descriptive cross-model comparisons, not neural permutation p-values. The fraction with |SRCC| at least 0.15 is 0.096; bootstrap-CI inclusion of zero was not tested here, so this is not the full original-rule failure rate.

Ridge is a surrogate, not the three-epoch neural mean-pool procedure. Tail fractions are descriptive, not calibrated neural-control p-values. This does not establish the cause or rule out leakage. A matched neural permutation null would be needed to assess the failed neural statistic directly. Consistency of signs across model seeds sharing one permutation does not establish a systematic defect across permutations.

All protected gate outputs, decision text and source-code hashes remained unchanged. The batched calculation matched the shared ridge implementation on the first and last permutation. See [summary.json](summary.json), [per-permutation scores](ridge_null.csv), and [historical-permutation test predictions](historical_permutation_ridge_predictions.csv).
