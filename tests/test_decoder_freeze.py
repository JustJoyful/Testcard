import json

import pytest
import yaml


def test_versioned_freeze_is_selected_and_version_bound(tmp_path, monkeypatch):
    import src.cache as cache
    freeze_path = tmp_path / 'freeze-v2.json'
    config_path = tmp_path / 'features-v2.yaml'
    config_path.write_text(yaml.safe_dump({'cache_version': 'v2', 'freeze_record': str(freeze_path)}))
    frozen = {'status': 'frozen', 'cache_version': 'v2', 'feature_config_sha256': 'verified',
              'scope_config_sha256': 'verified', 'lock_sha256': 'verified',
              'readiness_snapshot_sha256': 'verified', 'source_sha256': {'src/preprocess.py': 'verified'},
              'weights_sha256': {'dino': 'verified', 'resnet50': 'verified'}}
    freeze_path.write_text(json.dumps(frozen))
    monkeypatch.setattr(cache, 'sha256', lambda path: 'verified')
    assert cache.validate_freeze(config_path)['cache_version'] == 'v2'
    freeze_path.write_text(json.dumps({**frozen, 'cache_version': 'v1'}))
    with pytest.raises(ValueError, match='version'):
        cache.validate_freeze(config_path)
