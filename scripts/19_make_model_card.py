"""Stage the Hugging Face model folder hf_release/testcard-r50-ridge/ (model.npz + README.md); uploads nothing.

    .venv-serving/bin/python -m scripts.19_make_model_card

Every number comes from results/ (Rule 15). The Limitations and Citation sections are copied from the generated
README.md so both stay identical. Fails if the checkpoint hash does not match results/phase3/deployment.json.
"""
import csv
import json
import shutil
from pathlib import Path

from src.features import sha256

OUT = Path('hf_release/testcard-r50-ridge')
CHECKPOINT = Path('results/phase3/refit/resnet50/ridge/seedNone/model.npz')
REPO = 'https://github.com/JustJoyful/Testcard'


def section(text, title):
    start = text.index(f'## {title}\n')
    end = text.index('\n## ', start + 1)
    return text[start + len(f'## {title}\n'):end].strip()


def row(path, **match):
    return next(r for r in csv.DictReader(open(path)) if all(r[k] == v for k, v in match.items()))


def main():
    deployment = json.loads(Path('results/phase3/deployment.json').read_text())
    if sha256(CHECKPOINT) != deployment['checkpoint_sha256']:
        raise SystemExit('checkpoint does not match results/phase3/deployment.json')
    clean = json.loads(Path('results/phase4/clean_build/steps.json').read_text())['result']['reported']
    alpha = json.loads(Path('results/phase3/refit/resnet50/ridge/seedNone/run.json').read_text())['alpha']
    konvid = row('results/phase2/headline.csv', feature_set='resnet50', head='ridge')
    live = row('results/phase3/headline.csv', feature_set='resnet50', head='ridge')
    n_live = sum(1 for r in csv.DictReader(open('data/manifests/livevqc.csv')) if r['status'] == 'ok')
    readme = Path('README.md').read_text()
    f = lambda v: f'{float(v):.3f}'
    pm = lambda m, s: f'{f(m)} ± {f(s)}'
    card = f"""---
license: cc-by-nc-4.0
tags:
- video-quality-assessment
- no-reference-vqa
- ridge-regression
- resnet50
datasets:
- KoNViD-1k
- LIVE-VQC
metrics:
- srcc
- plcc
- krcc
---

# testcard-r50-ridge

A no-reference video quality model from [Testcard]({REPO}). It predicts a mean opinion score on the KoNViD-1k 1–5 scale.

## What it is

Ridge regression (alpha {alpha:g}) on mean-pooled, frozen ResNet-50 features (torchvision `IMAGENET1K_V2`, global-pooled, 2048-d), fitted once on all of KoNViD-1k. It is a single deterministic fit, with no seed and no fold. `model.npz` holds the feature standardization, the coefficients and the target scaling.

The checkpoint does not run on raw video by itself. It needs Testcard's shared preprocessing and feature code (`src/preprocess.py`, `src/features.py`), which decodes the video, samples frames and extracts the ResNet-50 features the model was trained on.

## How to use

Clone the [GitHub repository]({REPO}) and follow its Quickstart to create the serving environment and the verified bundle. Then score a clip:

```bash
.venv-serving/bin/python -m src.predict clip.mp4 --json
```

The output contains the unclamped score, the extraction and head timings, and the verified model, weights and preprocessing hashes.

## Results

The table shows this checkpoint's accuracy within KoNViD-1k and on transfer to LIVE-VQC.

| Evaluation | SRCC | PLCC (logistic-mapped) | KRCC | Fits |
|---|---|---|---|---|
| KoNViD-1k, 5-fold cross-validation | {pm(konvid['srcc_mean'], konvid['srcc_std'])} | {pm(konvid['plcc_mapped_mean'], konvid['plcc_mapped_std'])} | {pm(konvid['krcc_mean'], konvid['krcc_std'])} | mean ± SD across {konvid['run_count']} folds, one deterministic fit each |
| LIVE-VQC: trained on all KoNViD-1k, tested on all {n_live} videos | {f(live['srcc_mean'])} | {f(live['plcc_mapped_mean'])} | {f(live['krcc_mean'])} | single deterministic fit |

- **Metrics:** SRCC and KRCC are computed on raw predictions. PLCC is computed after a 4-parameter logistic fit to the target dataset's labels, and is not bootstrapped.
- **Different quantities:** the KoNViD-1k row is a cross-validation mean, while the LIVE-VQC row is a single transfer evaluation.
- **Against the neural heads:** in the project's paired bootstrap comparisons, this ridge baseline was numerically above every neural temporal head on the same features, but not by a practically meaningful margin.

The [repository README]({REPO}#results-head-comparison) has the full comparison and intervals. It also has a post-hoc, exploratory, unadjusted layer analysis, which did not change this model.

## Limitations

{section(readme, 'Limitations').replace('as described above', f'see the [validity section]({REPO}#validity-and-audit-trail)')}

## Integrity

| Artifact | SHA256 |
|---|---|
| `model.npz` (this checkpoint) | `{deployment['checkpoint_sha256']}` |
| ResNet-50 weights (`IMAGENET1K_V2`) | `{clean['weights_sha256']}` |
| Preprocessing freeze | `{clean['preprocessing_sha256']}` |

The serving code checks all three at startup and refuses to run on a mismatch.

## License and dataset terms

The checkpoint is released under CC BY-NC 4.0, because its training clips (KoNViD-1k, drawn from Creative Commons videos) carry their own Creative Commons terms. The Testcard code is MIT-licensed. Dataset terms are recorded in [docs/dataset_licenses.md]({REPO}/blob/main/docs/dataset_licenses.md). This repository contains no videos and no extracted features, only the ridge checkpoint and this card.

## Citation

{section(readme, 'Citation')}
"""
    OUT.mkdir(parents=True, exist_ok=True)
    extra = {p.name for p in OUT.iterdir()} - {'model.npz', 'README.md'}
    if extra:
        raise SystemExit(f'unexpected files in {OUT}: {sorted(extra)}')
    shutil.copyfile(CHECKPOINT, OUT / 'model.npz')
    if sha256(OUT / 'model.npz') != deployment['checkpoint_sha256']:
        raise SystemExit('copied checkpoint hash mismatch')
    (OUT / 'README.md').write_text(card)
    print(f'staged {OUT}: model.npz ({deployment["checkpoint_sha256"][:12]}…) and README.md')


if __name__ == '__main__':
    main()
