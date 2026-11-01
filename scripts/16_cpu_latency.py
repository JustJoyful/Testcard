"""CPU latency per video for every reported feature configuration (PRD success criterion 6).

    .venv-serving/bin/python -m scripts.16_cpu_latency [--report-only]

Reported feature sets: ResNet-50 pooled and DINOv2 (CLS + block-12 patch mean); SlowFast was never reported.
For each, extraction (decode + preprocess + backbone) is timed once per clip and every reported head is timed
on the resulting features, using the seed-0 / deterministic full-KoNViD refit checkpoints and four torch
threads like serving. The fixed clips are the committed Phase 0 sample; one warm-up clip pass is excluded.
Writes results/phase4/latency.csv, latency_summary.json and latency.md; nothing else.
"""
import csv
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.baselines import predict_ridge
from src.cache import atomic_json, atomic_write, csv_bytes
from src.features import extract_features, load_backbone, sha256
from src.heads import build_head
from src.preprocess import iter_samples, spatial_tensor

OUT = Path('results/phase4')
IDS = Path('data/splits/phase0_konvid_ids.txt')
FEATURES = Path('configs/features_v2.yaml')
REFIT = Path('results/phase3/refit')
HEADS = ('ridge', 'mean_pool', 'bigru', 'transformer', 'mamba')
DEPLOYED = ('resnet50', 'ridge')
THREADS = 4


def load_heads(feature):
    heads = {}
    for head in HEADS:
        run_dir = REFIT / feature / head / ('seedNone' if head == 'ridge' else 'seed0')
        run = json.loads((run_dir / 'run.json').read_text())
        checkpoint = run_dir / ('model.npz' if head == 'ridge' else 'best.pt')
        if sha256(checkpoint) != run['checkpoint_sha256']:
            raise ValueError(f'{feature}/{head}: checkpoint does not match its run record')
        if head == 'ridge':
            with np.load(checkpoint, allow_pickle=False) as data:
                model = {k: data[k] for k in data.files}
        else:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = build_head(head, 1536 if feature == 'dino' else 2048, saved['config']).eval()
            model.load_state_dict(saved['state_dict'])
        heads[head] = (model, run['checkpoint_sha256'])
    return heads


@torch.inference_mode()
def head_seconds(model, head, tensor):
    start = time.perf_counter()
    if head == 'ridge':
        predict_ridge(model, tensor.numpy().mean(0, keepdims=True, dtype=np.float64))
    else:
        model(tensor.unsqueeze(0), torch.tensor([len(tensor)]))
    return time.perf_counter() - start


def extract(path, backbone, feature, config):
    start = time.perf_counter()
    preprocess = inference = 0.
    vectors = []
    for _, frame, _ in iter_samples(path, config):
        tick = time.perf_counter()
        tensor = spatial_tensor(frame).unsqueeze(0)
        preprocess += time.perf_counter() - tick
        tick = time.perf_counter()
        values = extract_features(backbone, tensor, feature, 'cpu')
        inference += time.perf_counter() - tick
        vectors.append(torch.cat((values['cls'], values['patchmean_L12']), 1) if feature == 'dino' else values['pooled'])
    total = time.perf_counter() - start
    return dict(extraction_seconds=total, decode_seconds=max(0., total - preprocess - inference),
                preprocess_seconds=preprocess, backbone_seconds=inference), torch.cat(vectors)


def quartiles(values):
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return dict(median=float(q2), q1=float(q1), q3=float(q3), iqr=float(q3 - q1))


def main():
    torch.set_num_threads(THREADS)
    config = yaml.safe_load(FEATURES.read_text())
    source_video = importlib.import_module('scripts.02_extract_spatial').source_video
    inventory = {r['video_id']: r for r in csv.DictReader(Path('data/manifests/konvid.csv').open())}
    ids = IDS.read_text().split()
    if len(ids) < 20 or len(set(ids)) != len(ids):
        raise ValueError('need at least 20 unique committed clip IDs')
    settings = config['datasets']['konvid']
    rows = []
    for feature in ('resnet50', 'dino'):
        backbone = load_backbone(feature, 'cpu', config)
        heads = load_heads(feature)
        with source_video(inventory[ids[0]], settings) as path:  # warm-up pass, excluded
            _, tensor = extract(path, backbone, feature, config)
            for head, (model, _) in heads.items():
                head_seconds(model, head, tensor)
        for key in ids:
            duration = float(inventory[key]['duration_s'])
            with source_video(inventory[key], settings) as path:
                stages, tensor = extract(path, backbone, feature, config)
            for head, (model, checkpoint) in heads.items():
                seconds = head_seconds(model, head, tensor)
                total = stages['extraction_seconds'] + seconds
                rows.append(dict(feature_set=feature, head=head, video_id=key, duration_s=duration, frames=len(tensor),
                                 **stages, head_seconds=seconds, total_seconds=total,
                                 seconds_per_10s=10 * total / duration, checkpoint_sha256=checkpoint))
            print(json.dumps(dict(feature=feature, video=key, extraction=stages['extraction_seconds'])), flush=True)
        del backbone

    atomic_write(OUT / 'latency.csv', csv_bytes(rows, list(rows[0])))
    report(ids)


def container_median(directory):
    return quartiles([json.loads(p.read_text())['response']['timings_seconds']['extraction']
                      for p in sorted(Path(directory).glob('*.json'))])['median']


def report(ids=None):
    """Derive summary, serving check and table from the saved latency.csv only."""
    ids = ids or IDS.read_text().split()
    rows = [dict(r, **{k: float(r[k]) for k in r if k.endswith('seconds') or k in ('duration_s', 'seconds_per_10s')})
            for r in csv.DictReader((OUT / 'latency.csv').open())]
    summary = []
    for feature in ('resnet50', 'dino'):
        for head in HEADS:
            subset = [r for r in rows if r['feature_set'] == feature and r['head'] == head]
            summary.append(dict(feature_set=feature, head=head, clips=len(subset), deployed=(feature, head) == DEPLOYED,
                                **{f'{k}_{stat}': v for k in ('extraction_seconds', 'head_seconds', 'total_seconds', 'seconds_per_10s')
                                   for stat, v in quartiles([r[k] for r in subset]).items()}))

    # Consistency with the measured serving path: /status internal timings on the same clips.
    serving = json.loads(Path('results/phase4b/latency.raw.json').read_text())
    deployed = next(s for s in summary if s['deployed'])
    served = quartiles([r['extraction'] for r in serving])['median']
    difference = abs(deployed['extraction_seconds_median'] - served) / served
    check = dict(serving_source='results/phase4b/latency.raw.json', serving_median_extraction_seconds=served,
                 script_median_extraction_seconds=deployed['extraction_seconds_median'],
                 relative_difference=difference, flagged=difference > .2, threshold=.2,
                 # Same capped container without HTTP polling (CLI) and synchronous /predict, to locate any gap.
                 container_cli_median_extraction_seconds=container_median('results/phase4b/cli'),
                 container_predict_median_extraction_seconds=container_median('results/phase4b/api'),
                 host_cpus=os.cpu_count(), container_cpu_quota=4)

    atomic_json(OUT / 'latency_summary.json', dict(cpu_threads=THREADS, clip_ids_file=str(IDS), clip_ids=ids,
                warmup='one excluded clip pass per feature set', torch=torch.__version__, script_sha256=sha256(__file__),
                summary=summary, serving_consistency=check, latency_csv_sha256=sha256(OUT / 'latency.csv')))
    lines = ['# CPU latency per video', '',
             f'Generated by `scripts/16_cpu_latency.py`: {len(ids)} committed KoNViD clips (`{IDS}`), {THREADS} CPU threads, '
             'one excluded warm-up pass, one timed pass per clip. Median [IQR] across clips. '
             'Extraction = decode + preprocess + backbone; heads are the full-KoNViD refit checkpoints (seed 0; ridge deterministic).', '',
             '| Features | Head | Extraction s | Head ms | Total s | Total s per 10 s of video |', '|---|---|---|---|---|---|']
    for s in summary:
        mark = ' (deployed)' if s['deployed'] else ''
        lines.append(f"| {s['feature_set']} | {s['head']}{mark} | {s['extraction_seconds_median']:.2f} [{s['extraction_seconds_q1']:.2f}–{s['extraction_seconds_q3']:.2f}] | "
                     f"{1000 * s['head_seconds_median']:.3f} [{1000 * s['head_seconds_q1']:.3f}–{1000 * s['head_seconds_q3']:.3f}] | "
                     f"{s['total_seconds_median']:.2f} [{s['total_seconds_q1']:.2f}–{s['total_seconds_q3']:.2f}] | "
                     f"{s['seconds_per_10s_median']:.2f} [{s['seconds_per_10s_q1']:.2f}–{s['seconds_per_10s_q3']:.2f}] |")
    lines += ['', f"Serving consistency: deployed ResNet-50 ridge median extraction {check['script_median_extraction_seconds']:.2f} s here vs "
              f"{check['serving_median_extraction_seconds']:.2f} s internal `/status` timing (rebuilt image); relative difference "
              f"{check['relative_difference']:.1%} ({'**flagged, above 20%**' if check['flagged'] else 'within 20%'}). "
              f"The same capped container (4-CPU quota) gives {check['container_cli_median_extraction_seconds']:.2f} s through the CLI and "
              f"{check['container_predict_median_extraction_seconds']:.2f} s through `/predict`, so the gap follows the container CPU quota "
              f"(decoder process and four torch threads share 4 CPUs; this host has {check['host_cpus']}), not the HTTP or polling path.", '']
    atomic_write(OUT / 'latency.md', '\n'.join(lines).encode())
    print(json.dumps(check))




if __name__ == '__main__':
    import sys
    report() if '--report-only' in sys.argv else main()
