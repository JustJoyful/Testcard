"""Post-hoc exploratory Part B: distortion-to-content sensitivity per backbone branch (GPU).

Predeclared in DECISIONS.md (2026-10-31, commit 448efe2). Run from the repository root:
    .venv/bin/python -m results.posthoc_dino.part_b
Writes only under results/posthoc_dino/.
"""
import csv
import importlib
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter

from src.cache import atomic_json, atomic_write, csv_bytes
from src.features import extract_features, load_backbone, sha256
from src.preprocess import iter_samples, spatial_tensor

OUT = Path('results/posthoc_dino')
CONFIG = Path('configs/features_v2.yaml')
IDS = Path('data/splits/phase0_konvid_ids.txt')
FRAMES = 50
LEVELS = {'clean': [None], 'blur': [1, 2, 4], 'noise': [5, 15, 30], 'jpeg': [50, 20, 5]}
BRANCHES = {'resnet50': ('resnet50', 'pooled'), 'dino_cls_L12': ('dino', 'cls'),
            'dino_patchmean_L6': ('dino', 'patchmean_L6'), 'dino_patchmean_L9': ('dino', 'patchmean_L9'),
            'dino_patchmean_L12': ('dino', 'patchmean_L12')}


def distort(frame, kind, level, rng):
    """Applied to the decoded uint8 RGB frame; the result stays uint8 like a real decoded frame."""
    if kind == 'clean':
        return frame
    if kind == 'jpeg':
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, 'JPEG', quality=level)
        return np.asarray(Image.open(buffer).convert('RGB'))
    if kind == 'blur':
        out = gaussian_filter(frame.astype(np.float32), sigma=(level, level, 0))
    else:
        out = frame.astype(np.float32) + rng.normal(0, level, frame.shape)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return 1 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Part B needs CUDA (the same fp16 extraction path as the cache)')
    config = yaml.safe_load(CONFIG.read_text())
    source_video = importlib.import_module('scripts.02_extract_spatial').source_video
    rows = {r['video_id']: r for r in csv.DictReader(Path('data/manifests/konvid.csv').open())}
    settings = config['datasets']['konvid']
    ids = IDS.read_text().split()

    # Pass 1 counts sampled frames per clip; pass 2 keeps only the 50 chosen ones.
    counts = {}
    for key in ids:
        with source_video(rows[key], settings) as video:
            counts[key] = sum(1 for _ in iter_samples(video, config))
    pool = [(key, index) for key in ids for index in range(counts[key])]
    wanted = {pool[i] for i in np.random.default_rng(0).choice(len(pool), FRAMES, replace=False).tolist()}
    frames = []
    for key in ids:
        if not any(k == key for k, _ in wanted):
            continue
        with source_video(rows[key], settings) as video:
            frames += [(key, index, frame) for index, (_, frame, _) in enumerate(iter_samples(video, config))
                       if (key, index) in wanted]
    if len(frames) != FRAMES:
        raise ValueError('frame selection is not reproducible across decode passes')

    conditions = [(kind, level) for kind, levels in LEVELS.items() for level in levels]
    rng = np.random.default_rng(0)
    models = {name: load_backbone(name, 'cuda', config) for name in ('resnet50', 'dino')}
    collected = {branch: [[None] * FRAMES for _ in conditions] for branch in BRANCHES}
    for f, (_, _, frame) in enumerate(frames):
        for c, (kind, level) in enumerate(conditions):
            tensor = spatial_tensor(distort(frame, kind, level, rng)).unsqueeze(0)
            values = {name: extract_features(model, tensor, name, 'cuda') for name, model in models.items()}
            for branch, (backbone, key) in BRANCHES.items():
                collected[branch][c][f] = values[backbone][key][0].cpu().numpy()
    features = {branch: np.array(collected[branch]) for branch in BRANCHES}  # [condition, frame, dim]

    results = []
    for branch, array in features.items():
        clean = array[0]
        content = float(np.mean([distance(clean[i], clean[j]) for i, j in itertools.combinations(range(FRAMES), 2)]))
        for c, (kind, level) in enumerate(conditions[1:], start=1):
            shift = float(distance(clean, array[c]).mean())
            results.append(dict(branch=branch, distortion=kind, level=level, distortion_distance=shift,
                                content_distance=content, ratio=shift / content))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **features)
    atomic_write(OUT / 'part_b_features.npz', buffer.getvalue())
    atomic_write(OUT / 'part_b.csv', csv_bytes(results, list(results[0])))
    atomic_json(OUT / 'part_b.json', dict(status='complete', label='post-hoc exploratory', predeclaration_commit='448efe2',
        frames=[dict(video_id=k, sample_index=i) for k, i, _ in frames], conditions=conditions,
        device=torch.cuda.get_device_name(0), script_sha256=sha256(__file__), config_sha256=sha256(CONFIG),
        outputs_sha256={f: sha256(OUT / f) for f in ('part_b.csv', 'part_b_features.npz')}, results=results))
    print(json.dumps(dict(status='complete', frames=len(frames), rows=len(results))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
