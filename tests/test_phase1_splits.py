from collections import Counter
import csv
import pytest


def rows():
    return [{'video_id':f'v{i:03}', 'mos':float(1+i%5), 'status':'ok'} for i in range(100)]


def test_split_membership_disjoint_and_each_video_test_once():
    from src.splits import make_splits, validate_splits
    actual = make_splits(rows())
    validate_splits(actual, rows())
    assert len(actual)==500
    assert Counter(r['video_id'] for r in actual if r['role']=='test') == {r['video_id']:1 for r in rows()}
    for fold in range(5):
        assert Counter(r['role'] for r in actual if r['outer_fold']==fold)=={'test':20,'val':12,'train':68}


def test_reordered_rows_give_identical_splits():
    from src.splits import make_splits
    assert make_splits(rows()) == make_splits(rows()[::-1])


def test_equal_mos_ties_are_deterministic():
    from src.splits import make_splits
    data = [{**r,'mos':3} for r in rows()]
    assert make_splits(data) == make_splits(data[::-1])


def test_existing_or_stale_split_cannot_be_regenerated(tmp_path):
    from src.splits import freeze_splits
    source, output = tmp_path/'inventory.csv',tmp_path/'folds.csv'
    def write(data):
        with source.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['video_id','mos','status'])
            writer.writeheader(); writer.writerows(data)
    write(rows())
    freeze_splits(source,output)
    before = output.read_bytes(),output.stat().st_mtime_ns
    freeze_splits(source,output)
    assert (output.read_bytes(),output.stat().st_mtime_ns)==before
    write([{**r,'mos':4} for r in rows()])
    with pytest.raises(ValueError,match='stale|changed'):
        freeze_splits(source,output)
