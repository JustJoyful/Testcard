# Dataset licenses and terms

**Pending user verification.** Each entry was read from the dataset's own page on the access date shown. Confirm the full terms before publishing.

This repository distributes **no videos and no cached features**. It holds only manifests (IDs, MOS, checksums and metadata), fold indices, model artifacts and generated results. Raw videos (`data/raw/`) and feature caches (`cache/`) are excluded by `.gitignore`.

| Dataset | Used | Source | License / terms as stated | Accessed | Redistribute videos | Redistribute derived features |
|---|---|---|---|---|---|---|
| KoNViD-1k | yes (training, 5-fold CV) | <https://database.mmsp-kn.de/konvid-1k-database.html> | No license is named. The page says the database "is freely available to the research community" and asks users to cite two references. | 2026-11-01 | unclear (not addressed) | unclear (not addressed) |
| LIVE-VQC | yes (cross-dataset test only) | <https://live.ece.utexas.edu/research/LIVEVQC/index.html> | No license is named. "Copyright (c) 2018 The University of Texas at Austin"; "Permission is hereby granted, without written agreement and without license or royalty fees", provided "the copyright notice in its entirety appear in all copies of this database" and the database "is acknowledged in any publication that reports research using this database". Download requires a form. | 2026-11-01 | unclear: the quoted grant covers copies that keep the notice, but the full permission sentence was not captured; verify it | unclear (not addressed) |
| YouTube-UGC | no (cut from scope; see DECISIONS.md) | n/a | n/a | n/a | n/a | n/a |

## Citations the dataset owners request

- KoNViD-1k: V. Hosu, F. Hahn, M. Jenadeleh, H. Lin, H. Men, T. Szirányi, S. Li, D. Saupe, "The Konstanz natural video database (KoNViD-1k)," QoMEX 2017; and "The Konstanz natural video database," http://database.mmsp-kn.de, 2017.
- LIVE-VQC: Z. Sinno and A. C. Bovik, "Large-Scale Study of Perceptual Video Quality," IEEE Transactions on Image Processing, 28(2):612–627, 2019; "Large Scale Subjective Video Quality Study," ICIP 2018; and "LIVE Video Quality Challenge Database," 2018.

## To verify

1. LIVE-VQC: read the complete permission sentence on the page or in the download package, and record whether it allows redistributing the videos and derived data.
2. KoNViD-1k: check whether the download package or its README carries terms beyond "freely available to the research community".
3. Decide whether any derived artifact you publish (the deployed ridge model, per-video predictions in `results/`) needs a notice or acknowledgement under either set of terms.
