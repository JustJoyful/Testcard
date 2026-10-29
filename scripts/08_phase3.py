"""Check Phase 3 sources and identify branch-specific LIVE cache repairs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cache import atomic_write, validate_cache
from src.phase3 import live_cache_gaps, load_phase3_context


def check(config_path: Path):
    context = load_phase3_context(config_path)
    root = Path(context['config']['results_root'])
    root.mkdir(parents=True, exist_ok=True)
    gaps = live_cache_gaps(context)
    for branch, ids in gaps.items():
        path = root / f'repair_{branch}_ids.txt'
        if ids:
            atomic_write(path, ('\n'.join(ids) + '\n').encode())
            print(f'{branch}: repair {len(ids)} IDs via {path}')
        else:
            path.unlink(missing_ok=True)
    if any(gaps.values()):
        raise ValueError('LIVE cache incomplete; repair listed branches and rerun check')
    validate_cache(context['config']['cache_root'], context['config']['feature_config'])
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['check'], required=True)
    args = parser.parse_args()
    check(args.config)
    print('Phase 3 source and cache check passed')


if __name__ == '__main__':
    main()
