"""Verify every path->SHA256 entry listed in the Phase 1-4a gates is unchanged. Run: .venv/bin/python -m results.posthoc_dino.hash_check"""
import json
import re
from pathlib import Path

from src.cache import atomic_json
from src.features import sha256

GATES = ['results/phase1/phase_gate.json', 'results/phase2/phase_gate.json', 'results/phase3/phase_gate.json',
         'results/phase4a/phase_gate.json']
HEX = re.compile('[0-9a-f]{64}')
# Phase 1 names hashes by role (scripts/05_train.py prerequisites), not by path.
ROLES = {'config_feature_sha256': 'configs/features_v2.yaml', 'split_sha256': 'data/splits/konvid_folds.csv',
         'cache_sha256': 'cache/v2/manifest.csv', 'cache_meta_sha256': 'cache/v2/meta.json',
         'lock_sha256': 'env/requirements.lock', 'dino': 'weights/torch/checkpoints/dinov2_vitb14_pretrain.pth',
         'resnet50': 'weights/torch/checkpoints/resnet50-11ad3fa6.pth'}


def entries(value):
    """Yield (key, digest) for every mapping entry whose value is a SHA256 and whose key looks like a path."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and HEX.fullmatch(item) and ('/' in key or '.' in key or key in ROLES):
                yield ROLES.get(key, key), item
            else:
                yield from entries(item)
    elif isinstance(value, list):
        for item in value:
            yield from entries(item)


def main():
    record = {}
    for gate in GATES:
        root = Path(gate).parent
        checked, changed, unresolved = 0, [], []
        for key, digest in entries(json.loads(Path(gate).read_text())):
            path = next((p for p in (Path(key), root / key) if p.is_file()), None)
            if path is None:
                unresolved.append(key)
                continue
            checked += 1
            if sha256(path) != digest:
                changed.append(str(path))
        record[gate] = dict(checked=checked, changed=changed, unresolved=unresolved)
    record['all_unchanged'] = all(not r['changed'] for r in record.values() if isinstance(r, dict))
    atomic_json(Path('results/posthoc_dino/hash_check.json'), record)
    print(json.dumps({g: {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()} for g, r in record.items() if isinstance(r, dict)}))
    print('changed:', [p for r in record.values() if isinstance(r, dict) for p in r['changed']])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
