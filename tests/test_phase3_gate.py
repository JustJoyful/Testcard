import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.features import sha256
from src.training import config_hash


def source_fixture(tmp_path):
    root = tmp_path
    phase2 = root / 'phase2'
    phase2.mkdir()
    (phase2 / 'selection.json').write_text(json.dumps({'selected_feature_set': 'resnet50'}))
    freeze = {'status': 'research_configuration_frozen', 'selected_feature_set': 'resnet50',
              'selection_sha256': config_hash(json.loads((phase2 / 'selection.json').read_text())),
              'deployment': {'candidates': [{'candidate_id': f'c{i}'} for i in range(10)]}}
    (phase2 / 'konvid_freeze.json').write_text(json.dumps(freeze))
    gate = {'status': 'passed', 'outputs_sha256': {name: sha256(phase2 / name)
            for name in ('selection.json', 'konvid_freeze.json')}}
    (phase2 / 'phase_gate.json').write_text(json.dumps(gate))
    for dataset, ids in [('konvid', ['k1', 'k2']), ('livevqc', ['l1', 'l2'])]:
        path = root / f'{dataset}.csv'
        with path.open('w') as out:
            writer = csv.DictWriter(out, fieldnames=['video_id', 'dataset', 'mos', 'source_sha256', 'source_size_bytes'])
            writer.writeheader()
            writer.writerows({'video_id': key, 'dataset': dataset, 'mos': 50,
                              'source_sha256': 'a' * 64, 'source_size_bytes': 1} for key in ids)
    config = {'phase2_gate': str(phase2 / 'phase_gate.json'), 'phase2_freeze': str(phase2 / 'konvid_freeze.json'),
              'konvid_inventory': str(root / 'konvid.csv'), 'live_inventory': str(root / 'livevqc.csv'),
              'cache_root': str(root / 'cache'), 'feature_config': str(root / 'features.yaml'),
              'cpu_sample_ids': str(root / 'ids.txt'), 'results_root': str(root / 'results'),
              'cpu_repeats': 3, 'cpu_statistic': 'maximum_normalized_total'}
    (root / 'ids.txt').write_text('k1\nk2\n')
    (root / 'features.yaml').write_text('cache_version: v2\n')
    path = root / 'phase3.yaml'
    path.write_text(yaml.safe_dump(config))
    return path, phase2


def test_phase2_gate_pins_all_outputs_and_selection(tmp_path):
    from src.phase3 import load_phase3_context
    config, phase2 = source_fixture(tmp_path)
    assert load_phase3_context(config)['selected_feature_set'] == 'resnet50'
    gate_path = phase2 / 'phase_gate.json'
    gate = json.loads(gate_path.read_text())
    gate['status'] = 'failed'
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        load_phase3_context(config)
    gate_path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_phase3_context(config)
    gate['status'] = 'passed'
    gate_path.write_text(json.dumps(gate))
    (phase2 / 'selection.json').write_text(json.dumps({'selected_feature_set': 'dino'}))
    with pytest.raises(ValueError):
        load_phase3_context(config)


def test_configured_freeze_must_be_gate_pinned(tmp_path):
    from src.phase3 import load_phase3_context
    config, phase2 = source_fixture(tmp_path)
    copied = tmp_path / 'alternate_freeze.json'
    freeze = json.loads((phase2 / 'konvid_freeze.json').read_text())
    freeze['deployment']['candidates'][0]['candidate_id'] = 'altered'
    copied.write_text(json.dumps(freeze))
    policy = yaml.safe_load(config.read_text())
    policy['phase2_freeze'] = str(copied)
    config.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError):
        load_phase3_context(config)


def test_source_hash_tracks_uncommitted_code(tmp_path):
    from src.phase3 import source_code_sha256
    for folder in ('src', 'scripts'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'stage.py').write_text('value = 1\n')
    first = source_code_sha256(tmp_path)
    (tmp_path / 'src/stage.py').write_text('value = 2\n')
    assert source_code_sha256(tmp_path) != first


def test_direct_check_cli_imports_project_package():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, 'scripts/08_phase3.py', '--help'],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_live_cache_gaps_are_branch_specific(tmp_path, monkeypatch):
    from src.phase3 import live_cache_gaps, load_phase3_context
    import src.phase3 as phase3
    config, _ = source_fixture(tmp_path)
    context = load_phase3_context(config)
    monkeypatch.setattr(phase3, 'reusable', lambda root, dataset, key, branches, row:
                        not ((key == 'l1' and branches[0] == 'dino_cls') or
                             (key == 'l2' and branches == ['resnet50'])))
    assert live_cache_gaps(context) == {'dino': ['l1'], 'resnet50': ['l2']}


def test_inventory_dataset_rejects_bad_ids_and_mos(tmp_path, monkeypatch):
    from src.data import InventoryFeatureDataset
    import src.data as data
    config, _ = source_fixture(tmp_path)
    monkeypatch.setattr(data, 'read_manifest', lambda root: [])
    monkeypatch.setattr(data, 'read_branch', lambda root, dataset, key, branch, records, row:
                        np.ones((2, 2048 if branch == 'resnet50' else 768), dtype=np.float16))
    path = tmp_path / 'livevqc.csv'
    items = InventoryFeatureDataset(tmp_path / 'cache', path, 'livevqc', 'resnet50')
    assert [items[i]['video_id'] for i in range(len(items))] == ['l1', 'l2']
    with pytest.raises(ValueError):
        InventoryFeatureDataset(tmp_path / 'cache', path, 'konvid', 'resnet50')
    with path.open('a') as out:
        out.write('l1,livevqc,50,' + 'a' * 64 + ',1\n')
    with pytest.raises(ValueError):
        InventoryFeatureDataset(tmp_path / 'cache', path, 'livevqc', 'resnet50')
    text = path.read_text().replace('l1,livevqc,50', 'other,livevqc,nan', 1)
    path.write_text(text)
    with pytest.raises(ValueError):
        InventoryFeatureDataset(tmp_path / 'cache', path, 'livevqc', 'resnet50')


def test_prediction_context_and_dataset_do_not_load_live_labels(tmp_path, monkeypatch):
    from src.phase3 import load_phase3_context
    from src.data import InventoryFeatureDataset
    import src.data as data
    config, _ = source_fixture(tmp_path)
    live = tmp_path / 'livevqc.csv'
    live.write_text(live.read_text().replace(',50,', ',withheld,'))
    context = load_phase3_context(config, include_live_labels=False)
    assert all('mos' not in row for row in context['inventories']['livevqc'])
    monkeypatch.setattr(data, 'read_manifest', lambda root: [])
    monkeypatch.setattr(data, 'read_branch', lambda root, dataset, key, branch, records, row:
                        np.ones((2, 2048), dtype=np.float16))
    items = InventoryFeatureDataset(tmp_path / 'cache', live, 'livevqc', 'resnet50', include_labels=False)
    assert all('mos' not in items[i] for i in range(len(items)))
    with pytest.raises(ValueError):
        load_phase3_context(config)
