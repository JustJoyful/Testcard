# Phase 1 core slice

Status: passed

Definitive fixed-DINO head comparison; Phase 2 must reuse completed CV and ridge artifacts.

Headline: mean per-fold SRCC across 25 fold × seed runs. Ensemble OOF is a separate quantity.
Original training-only control failed. User approved independent validation-label permutation on 2026-10-27; the unchanged numerical gates apply to the new controls. This post-hoc amendment does not erase the original failure.

PLCC logistic mapping uses target labels. Warning-only determinism; no bitwise reproducibility claim.

| Head | Runs | SRCC mean ± SD | KRCC mean ± SD | Raw PLCC mean ± SD | Mapped PLCC mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| ridge | 5 | 0.7350 ± 0.0217 | 0.5471 ± 0.0181 | 0.7667 ± 0.0148 | 0.7697 ± 0.0140 |
| mean_pool | 25 | 0.6852 ± 0.0258 | 0.5024 ± 0.0218 | 0.7245 ± 0.0210 | 0.7320 ± 0.0195 |
| bigru | 25 | 0.6892 ± 0.0223 | 0.5057 ± 0.0174 | 0.7207 ± 0.0169 | 0.7254 ± 0.0144 |
| transformer | 25 | 0.6930 ± 0.0229 | 0.5092 ± 0.0183 | 0.7298 ± 0.0181 | 0.7346 ± 0.0161 |
| mamba | 25 | 0.6895 ± 0.0329 | 0.5063 ± 0.0274 | 0.7287 ± 0.0267 | 0.7340 ± 0.0242 |

Mapped PLCC is the mean of per-run target-label logistic mappings. Ridge has five deterministic fold fits; neural rows summarize 25 fold × seed fits. Paired comparison CIs/Holm reporting follow in Phase 2.
