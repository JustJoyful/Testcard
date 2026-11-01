# Dataset licenses and terms

**Verified against each dataset's own page on 2026-11-01.** Both are public research datasets released for academic use.

This repository distributes **no videos and no cached features**. It holds only manifests (IDs, MOS, checksums and metadata), fold indices, model artifacts and generated results. Raw videos (`data/raw/`) and feature caches (`cache/`) are excluded by `.gitignore`.

| Dataset | Used | Source | License / terms as stated | Accessed | Redistribute videos | Redistribute derived features |
|---|---|---|---|---|---|---|
| KoNViD-1k | yes (training, 5-fold CV) | <https://database.mmsp-kn.de/konvid-1k-database.html> | No single license is named. "KoNViD-1k is freely available to the research community"; the 1,200 clips were drawn from "793436 Creative Commons (CC) video sequences" of YFCC100m, so each clip keeps its own CC license; cite both references. | 2026-11-01 | governed by each clip's CC license (not checked per clip; not needed, none are distributed here) | not addressed; research use only, none distributed here |
| LIVE-VQC | yes (cross-dataset test only) | <https://live.ece.utexas.edu/research/LIVEVQC/index.html> | No license is named. "Copyright (c) 2018 The University of Texas at Austin"; "Permission is hereby granted, without written agreement and without license or royalty fees," to use, copy, modify and distribute the database, its results and documentation for any purpose, provided the copyright notice appears in all copies and publications acknowledge LIVE at UT Austin. Provided "AS IS". Download requires a form. | 2026-11-01 | permitted with the full copyright notice (none distributed here) | permitted under the same grant ("results" are covered); none distributed here |
| YouTube-UGC | no (cut from scope; see DECISIONS.md) | n/a | n/a | n/a | n/a | n/a |

## Citations the dataset owners request

- KoNViD-1k: V. Hosu, F. Hahn, M. Jenadeleh, H. Lin, H. Men, T. Szirányi, S. Li, D. Saupe, "The Konstanz natural video database (KoNViD-1k)," QoMEX 2017; and "The Konstanz natural video database," http://database.mmsp-kn.de, 2017.
- LIVE-VQC: Z. Sinno and A. C. Bovik, "Large-Scale Study of Perceptual Video Quality," IEEE Transactions on Image Processing, 28(2):612–627, 2019; "Large Scale Subjective Video Quality Study," ICIP 2018; and "LIVE Video Quality Challenge Database," 2018.

## Notes

- No videos or cached features are distributed, so the per-clip Creative Commons terms of KoNViD-1k do not restrict this repository.
- If videos or features are ever published, keep the LIVE-VQC copyright notice and check each KoNViD-1k clip's CC license (some CC variants forbid commercial use or derivatives).
