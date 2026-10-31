"""Run the fixed-epoch shuffled control and full-KoNViD refits."""
import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cache import atomic_json
from src.cache import atomic_write
from src.baselines import fit_ridge_fixed_alpha
from src.data import CachedFeatureDataset, InventoryFeatureDataset
from src.features import sha256
from src.heads import build_head
from src.metrics import bootstrap_srcc
from src.phase3 import load_phase3_context, require_cpu_winner
from src.training import config_hash, fit_fixed_epochs, predict, seed_everything
from src.tuning import control_directory, permute_control_role, require_control

HEADS = ('mean_pool', 'bigru', 'transformer', 'mamba')


def _validate_refit_sources(spec):
    sources = spec['cv_selected_alphas'] if 'cv_selected_alphas' in spec else spec['cv_epochs']
    if len(sources) != (5 if 'cv_selected_alphas' in spec else 25):
        raise ValueError('incomplete frozen CV source roster')
    for source in sources:
        path = Path(source['run_path'])
        artifact = path.parent / ('model.npz' if 'cv_selected_alphas' in spec else 'best.pt')
        if sha256(path) != source['run_sha256'] or sha256(artifact) != source['checkpoint_sha256']:
            raise ValueError(f'changed CV source: {path}')
    return [source['run_sha256'] for source in sources]


def _provenance(context, feature, head, spec):
    return {**context['provenance'], 'phase2_gate_sha256': context['phase2_gate_sha256'],
            'freeze_sha256': context['freeze_sha256'], 'cv_run_sha256': _validate_refit_sources(spec),
            'code_sha256': context.get('code_sha256'),
            'head': head, 'feature_set': feature, 'fold': None,
            'inventory_sha256': sha256(context['config']['konvid_inventory']),
            'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}


def run_refits(context: dict, *, neural_only=False) -> None:
    config = context['config']
    root = Path(config['results_root']) / 'refit'
    for feature in ('resnet50', 'dino'):
        data = InventoryFeatureDataset(Path(config['cache_root']), Path(config['konvid_inventory']), 'konvid', feature)
        for head in HEADS:
            spec = context['freeze']['refit_by_feature'][feature][head]
            provenance = _provenance(context, feature, head, spec)
            frozen = spec['config']
            epochs = spec['median_best_epoch']
            for seed in range(5):
                seed_everything(seed, frozen['cpu_threads'])
                model = build_head(head, 1536 if feature == 'dino' else 2048, frozen)
                output = root / feature / head / f'seed{seed}'
                run = fit_fixed_epochs(model, data, frozen, output, seed, provenance, epochs)
                print(f'full refit {feature}/{head}/seed{seed}: {run["status"]}', flush=True)
    if not neural_only:
        run_ridge_refits(context)


def run_ridge_refits(context):
    config = context['config']
    candidates = {c['candidate_id']: c for c in context['freeze']['deployment']['candidates']}
    for feature in ('resnet50', 'dino'):
        spec = context['freeze']['refit_by_feature'][feature]['ridge']
        provenance = _provenance(context, feature, 'ridge', spec)
        alpha = spec['phase3_alpha']
        dataset = InventoryFeatureDataset(Path(config['cache_root']), Path(config['konvid_inventory']), 'konvid', feature)
        x = np.array([dataset[i]['features'].numpy().mean(0, dtype=np.float64) for i in range(len(dataset))])
        y = np.array([dataset[i]['mos'] for i in range(len(dataset))])
        output = Path(config['results_root']) / 'refit' / feature / 'ridge' / 'seedNone'
        output.mkdir(parents=True, exist_ok=True)
        identity = {'config_sha256': config_hash(candidates[f'{feature}/ridge']['config']),
                    'alpha': alpha, 'provenance': provenance,
                    'inventory_sha256': sha256(config['konvid_inventory']),
                    'video_ids_sha256': config_hash([dataset[i]['video_id'] for i in range(len(dataset))])}
        metadata = output / 'run.json'
        checkpoint = output / 'model.npz'
        if metadata.exists():
            run = json.loads(metadata.read_text())
            if run.get('status') != 'complete' or run.get('identity') != identity or sha256(checkpoint) != run.get('checkpoint_sha256'):
                raise ValueError(f'incompatible ridge refit: {output}')
            continue
        artifact = fit_ridge_fixed_alpha(x, y, alpha)
        buffer = io.BytesIO()
        np.savez(buffer, **artifact)
        atomic_write(checkpoint, buffer.getvalue())
        atomic_json(metadata, {'status': 'complete', 'identity': identity, 'alpha': alpha,
                               'feature_set': feature, 'head': 'ridge', 'seed': None,
                               'selection': 'fixed_alpha', 'provenance': provenance,
                               'checkpoint_sha256': sha256(checkpoint)})
        print(f'full ridge refit {feature}: complete', flush=True)


def run_fixed_control(context):
    from src.phase3 import require_fixed_control
    root = Path(context['config']['results_root']) / 'control_fixed_epoch'
    if (root / 'diagnostic.json').exists():
        return require_fixed_control(context)
    for phase_root in ('results/phase1', 'results/phase2/resnet50'):
        for final in (False, True):
            require_control(control_directory(phase_root, final), context['provenance'])
    feature = context['selected_feature_set']
    spec = context['freeze']['refit_by_feature'][feature]['mean_pool']
    config = spec['config']
    provenance = _provenance(context, feature, 'mean_pool', spec)
    train = CachedFeatureDataset(Path(config['cache_root']), Path(config['inventory']), Path(config['splits']), 0, 'train', feature)
    val = CachedFeatureDataset(Path(config['cache_root']), Path(config['inventory']), Path(config['splits']), 0, 'val', feature)
    test = CachedFeatureDataset(Path(config['cache_root']), Path(config['inventory']), Path(config['splits']), 0, 'test', feature)
    shuffled_train, train_map = permute_control_role(train, 'train')
    shuffled_val, val_map = permute_control_role(val, 'val')
    atomic_json(root / 'permutations.json', {'train': train_map, 'val': val_map})
    shuffled = shuffled_train + shuffled_val
    results = []
    for seed in range(5):
        seed_everything(seed, config['cpu_threads'])
        model = build_head('mean_pool', 1536 if feature == 'dino' else 2048, config)
        run = fit_fixed_epochs(model, shuffled, config, root / f'seed{seed}', seed,
                               {**provenance, 'control_permutations_sha256': sha256(root / 'permutations.json'),
                                'label_mode': 'shuffled'}, spec['median_best_epoch'])
        predictions = predict(model, test, run['target_stats'], config['device'])
        y = np.array([test[i]['mos'] for i in range(len(test))])
        p = np.array([row['prediction'] for row in predictions])
        score = bootstrap_srcc(y, p)
        passed = score['status'] == 'ok' and score['ci95'][0] <= 0 <= score['ci95'][1] and abs(score['srcc']) < .15
        rows = [{**row, 'mos': float(label), 'checkpoint_sha256': run['checkpoint_sha256']}
                for row, label in zip(predictions, y, strict=True)]
        atomic_json(root / f'seed{seed}' / 'predictions.json', rows)
        results.append({'seed': seed, 'status': 'passed' if passed else 'failed', 'bootstrap': score,
                        'run_sha256': sha256(root / f'seed{seed}' / 'run.json'),
                        'checkpoint_sha256': run['checkpoint_sha256'],
                        'predictions_sha256': sha256(root / f'seed{seed}' / 'predictions.json')})
        if not passed:
            atomic_json(root / 'diagnostic.json', {'status': 'failed', 'seeds': results,
                        'freeze_sha256': context['freeze_sha256'],
                        'code_sha256': context.get('code_sha256'),
                        'phase2_gate_sha256': context['phase2_gate_sha256']})
            raise ValueError(f'seed {seed} fixed-epoch shuffled control failed')
    diagnostic = {'status': 'passed', 'feature_set': feature, 'seeds': results,
                  'permutations_sha256': sha256(root / 'permutations.json'),
                  'code_sha256': context.get('code_sha256'),
                  'freeze_sha256': context['freeze_sha256'],
                  'phase2_gate_sha256': context['phase2_gate_sha256']}
    atomic_json(root / 'diagnostic.json', diagnostic)
    return require_fixed_control(context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['control', 'qualification', 'fit'], required=True)
    args = parser.parse_args()
    context = load_phase3_context(args.config, include_live_labels=False)
    require_cpu_winner(context)
    if args.mode == 'control':
        run_fixed_control(context)
    elif args.mode == 'qualification':
        from src.phase3 import require_qualified_control
        require_qualified_control(context)
    else:
        from src.phase3 import require_qualified_control
        require_qualified_control(context)
        run_refits(context)


if __name__ == '__main__':
    main()
