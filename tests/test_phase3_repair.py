import os

import numpy as np

from src.cache import DINO_BRANCHES, write_video_features


def test_dino_repair_preserves_valid_arrays(tmp_path):
    arrays = {name: np.ones((2, 768), dtype=np.float16) for name in DINO_BRANCHES}
    source = {'dataset': 'livevqc', 'video_id': 'l1', 'source_sha256': 'a' * 64,
              'source_size_bytes': '1', 'mos': '50'}
    write_video_features(tmp_path, 'livevqc', 'l1', arrays, source, {'cache_version': 'v2'})
    valid = tmp_path / 'livevqc' / DINO_BRANCHES[0] / 'l1.npy'
    broken = tmp_path / 'livevqc' / DINO_BRANCHES[1] / 'l1.npy'
    os.utime(valid, ns=(1000000000, 1000000000))
    broken.write_bytes(b'corrupt')
    write_video_features(tmp_path, 'livevqc', 'l1', arrays, source, {'cache_version': 'v2'})
    assert valid.stat().st_mtime_ns == 1000000000
    assert np.load(broken, allow_pickle=False).shape == (2, 768)
