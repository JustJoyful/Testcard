import json
import numpy as np
import pytest


def fixture_meta():
    return {'feature_config_sha256': 'policy', 'expected_ids': {'konvid': ['v0'], 'livevqc': ['L0']},
            'expected_branches': ['dino_cls', 'dino_patchmean_L6', 'dino_patchstd_L6',
                                  'dino_patchmean_L9', 'dino_patchstd_L9', 'dino_patchmean_L12', 'dino_patchstd_L12', 'resnet50']}


def arrays():
    return {b: np.ones((2, 768), np.float16) * (i+1)
            for i, b in enumerate(fixture_meta()['expected_branches'][:-1])}


def row():
    return {'video_id': 'v0', 'source_sha256': 'source', 'source_size_bytes': '20', 'mos': '3', 'status': 'ok'}


def test_feature_order_and_missing_branch(tmp_path):
    from src.cache import write_video_features, load_cached_video
    with pytest.raises(ValueError):
        load_cached_video(tmp_path, 'konvid', 'v0', 'dino')
    write_video_features(tmp_path, 'konvid', 'v0', arrays(), row(), fixture_meta())
    x = load_cached_video(tmp_path, 'konvid', 'v0', 'dino')
    assert x.shape == (2, 1536)
    assert (x[:, :768] == 1).all() and (x[:, 768:] == 6).all()


def test_interrupted_multibranch_write_is_not_reusable(tmp_path, monkeypatch):
    import src.cache as cache
    original = cache.atomic_array
    calls = []
    def broken(path, value):
        calls.append(path)
        if len(calls) == 4:
            raise OSError('interruption')
        original(path, value)
    monkeypatch.setattr(cache, 'atomic_array', broken)
    with pytest.raises(OSError):
        cache.write_video_features(tmp_path, 'konvid', 'v0', arrays(), row(), fixture_meta())
    with pytest.raises(ValueError):
        cache.load_cached_video(tmp_path, 'konvid', 'v0', 'dino')
    monkeypatch.setattr(cache, 'atomic_array', original)
    cache.write_video_features(tmp_path, 'konvid', 'v0', arrays(), row(), fixture_meta())
    assert cache.load_cached_video(tmp_path, 'konvid', 'v0', 'dino').shape == (2, 1536)


def test_manifest_publication_is_atomic(tmp_path, monkeypatch):
    import src.cache as cache
    cache.write_video_features(tmp_path, 'konvid', 'v0', arrays(), row(), fixture_meta())
    before = (tmp_path/'manifest.csv').read_bytes()
    original = cache.os.replace
    def fail(src, dst):
        if str(dst).endswith('manifest.csv'):
            raise OSError('interruption')
        original(src, dst)
    monkeypatch.setattr(cache.os, 'replace', fail)
    with pytest.raises(OSError):
        cache.write_video_features(tmp_path, 'konvid', 'v0', arrays(), row(), fixture_meta())
    assert (tmp_path/'manifest.csv').read_bytes() == before


@pytest.mark.parametrize('mutation', ['checksum', 'dtype', 'shape', 'source'])
def test_checksum_shape_dtype_or_source_change_rejects_resume(tmp_path, mutation):
    from src.cache import write_video_features, load_cached_video, reusable
    write_video_features(tmp_path, 'konvid', 'v0', arrays(), row(), fixture_meta())
    path = tmp_path/'konvid/dino_cls/v0.npy'
    if mutation == 'source':
        assert not reusable(tmp_path, 'konvid', 'v0', list(arrays()), {**row(), 'source_sha256': 'changed'})
    else:
        np.save(path, np.ones((3 if mutation == 'shape' else 2, 768),
                             dtype=np.float32 if mutation == 'dtype' else np.float16)*9)
        with pytest.raises(ValueError, match='v0'):
            load_cached_video(tmp_path, 'konvid', 'v0', 'dino')


def test_subset_is_not_complete_dataset(tmp_path):
    from src.cache import write_video_features, coverage
    write_video_features(tmp_path, 'konvid', 'v0', arrays(), row(), fixture_meta())
    assert not coverage(tmp_path)['konvid']['complete']


def test_konvid_completion_does_not_certify_livevqc(tmp_path):
    from src.cache import write_video_features, coverage
    write_video_features(tmp_path, 'konvid', 'v0', {**arrays(), 'resnet50': np.ones((2,2048), np.float16)}, row(), fixture_meta())
    assert coverage(tmp_path)['konvid']['complete']
    assert not coverage(tmp_path)['livevqc']['complete']


def test_unsafe_video_id_is_rejected(tmp_path):
    from src.cache import write_video_features
    with pytest.raises(ValueError):
        write_video_features(tmp_path, 'konvid', '../x', arrays(), row(), fixture_meta())


def test_indexed_lookup_keeps_integrity_and_duplicate_checks(tmp_path):
    from src.cache import index_manifest,read_branch,read_manifest,write_video_features
    write_video_features(tmp_path,'konvid','v0',arrays(),row(),fixture_meta())
    records=read_manifest(tmp_path)
    index=index_manifest(records)
    np.testing.assert_array_equal(read_branch(tmp_path,'konvid','v0','dino_cls',index,row()),np.ones((2,768),np.float16))
    with pytest.raises(ValueError,match='source'):
        read_branch(tmp_path,'konvid','v0','dino_cls',index,{**row(),'source_sha256':'different'})
    duplicate=index_manifest(records+[records[0]])
    with pytest.raises(ValueError,match='manifest'):
        read_branch(tmp_path,'konvid','v0','dino_cls',duplicate,row())
    (tmp_path/'konvid/dino_cls/v0.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        read_branch(tmp_path,'konvid','v0','dino_cls',index,row())
