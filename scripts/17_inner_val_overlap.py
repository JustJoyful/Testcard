"""Count KoNViD videos used both for inner-validation selection and as held-out test videos (Rule 1 disclosure).

    .venv-serving/bin/python -m scripts.17_inner_val_overlap

Feature selection (Phase 2) and the post-hoc ridge alpha choices used the inner-validation role of every outer
fold; the head comparison scores each video once, in the fold where it is held out. Writes results/inner_val_overlap.json.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.cache import atomic_json
from src.features import sha256

SPLITS = Path('data/splits/konvid_folds.csv')
OUT = Path('results/inner_val_overlap.json')


def main():
    roles = defaultdict(lambda: defaultdict(set))
    for row in csv.DictReader(SPLITS.open()):
        roles[int(row['outer_fold'])][row['role']].add(row['video_id'])
    test_fold = {key: fold for fold, r in roles.items() for key in r['test']}
    folds, all_val = [], set()
    for fold in sorted(roles):
        val = roles[fold]['val']
        assert not val & roles[fold]['test'], 'a fold validates on its own held-out videos'
        all_val |= val
        folds.append(dict(fold=fold, train=len(roles[fold]['train']), val=len(val), test=len(roles[fold]['test']),
                          val_videos_held_out_in_another_fold=sum(test_fold.get(k) not in (None, fold) for k in val)))
    videos = len(test_fold)
    overlap = len(all_val & set(test_fold))
    record = dict(splits=str(SPLITS), splits_sha256=sha256(SPLITS), videos=videos, folds=folds,
                  inner_validation_rows=sum(f['val'] for f in folds), unique_inner_validation_videos=len(all_val),
                  unique_inner_validation_videos_also_held_out=overlap, fraction_of_videos=overlap / videos,
                  meaning='Every inner-validation video is a held-out test video in a different outer fold. Selections made '
                          'on inner-validation scores (feature set, ridge alpha, early stopping) therefore used videos whose '
                          'held-out predictions enter the head comparison.')
    atomic_json(OUT, record)
    print(json.dumps({k: v for k, v in record.items() if k != 'meaning'}))


if __name__ == '__main__':
    main()
