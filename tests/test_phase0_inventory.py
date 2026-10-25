import hashlib
import importlib
import struct
import zipfile

import pytest


def inventory():
    return importlib.import_module('scripts.00_inventory')


@pytest.mark.parametrize('name', ['../bad.mp4', '/bad.mp4', 'C:/bad.mp4', 'a\\bad.mp4'])
def test_unsafe_archive_member_rejected(tmp_path, name):
    path = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(name, b'video')
    with pytest.raises(ValueError, match='unsafe'):
        inventory().inspect_archive(path)


def test_crc_failure_reported_and_original_preserved(tmp_path):
    path = tmp_path / 'broken.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
        z.writestr('one.mp4', b'payload')
    data = bytearray(path.read_bytes())
    name_size, extra_size = struct.unpack_from('<HH', data, 26)
    data[30 + name_size + extra_size] ^= 1
    path.write_bytes(data)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = inventory().inspect_archive(path)
    assert result['status'] == 'failed'
    assert result['corrupt_member'] == 'one.mp4'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_duplicate_video_ids_and_missing_mos_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        inventory().validate_join(['one', 'one'], {'one': 3.0}, (1, 5))
    with pytest.raises(ValueError, match='missing'):
        inventory().validate_join(['one'], {}, (1, 5))
    with pytest.raises(ValueError, match='MOS'):
        inventory().validate_join(['one'], {'one': float('nan')}, (1, 5))


def test_failed_probe_preserves_video_id_and_source(tmp_path):
    path = tmp_path / 'clips.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('broken.mp4', b'not a video')
    before = path.read_bytes()
    with zipfile.ZipFile(path) as z:
        row = inventory().inventory_member(z, z.infolist()[0], 'konvid', 3.0, [1, 5], tmp_path)
    assert row['video_id'] == 'broken'
    assert row['status'] == 'failed'
    assert 'broken' in row['failure_reason']
    assert row['duration_s'] == ''
    assert row['source_sha256'] == hashlib.sha256(b'not a video').hexdigest()
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['clips.zip']


def test_materialize_only_selected_clips_and_reject_changed_raw(tmp_path, monkeypatch):
    import csv
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'clips.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a.mp4', b'one')
        z.writestr('b.mp4', b'two')
    (tmp_path / 'data/manifests').mkdir(parents=True)
    with (tmp_path / 'data/manifests/konvid.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'relative_path', 'source_sha256'])
        writer.writeheader()
        writer.writerow({'video_id': 'a', 'relative_path': 'a.mp4', 'source_sha256': hashlib.sha256(b'one').hexdigest()})
    config = {'samples': {'konvid': ['a']}, 'datasets': {'konvid': {'archive': str(path), 'raw_dir': 'data/raw/konvid'}}}
    paths = inventory().prepare_sample_paths(config, 'konvid')
    assert paths['a'].read_bytes() == b'one'
    assert not (tmp_path/'data/raw/konvid/b.mp4').exists()
    paths['a'].write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        inventory().prepare_sample_paths(config, 'konvid')
    assert paths['a'].read_bytes() == b'changed'


def test_deflate_corruption_is_identified_by_member(tmp_path):
    path = tmp_path/'deflated.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('bad.mp4', b'video'*100)
    data = bytearray(path.read_bytes())
    data[30+len('bad.mp4')] = 0xff
    path.write_bytes(data)
    result = inventory().inspect_archive(path)
    assert result['status'] == 'failed'
    assert result['corrupt_member'] == 'bad.mp4'


def test_interrupted_inventory_preserves_published_manifest(tmp_path, monkeypatch):
    import json
    import sys
    import yaml
    monkeypatch.chdir(tmp_path)
    path = tmp_path/'clips.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a.mp4', b'one'); z.writestr('b.mp4', b'two')
    directory = tmp_path/'data/manifests'; directory.mkdir(parents=True)
    (directory/'datasets.json').write_text(json.dumps({'konvid': {}}))
    manifest = directory/'konvid.csv'; manifest.write_text('previous complete inventory\n')
    config = {'datasets': {'konvid': {'archive': str(path), 'mos_scale': [1, 5]}},
              'feasibility': {'reserve_disk_bytes': 1}, 'samples': {'konvid': []}, 'seed': 0}
    settings = tmp_path/'config.yaml'; settings.write_text(yaml.safe_dump(config))
    module = inventory()
    monkeypatch.setattr(sys, 'argv', ['00_inventory.py', '--config', str(settings)])
    monkeypatch.setattr(module, 'read_labels', lambda *args: ({'a': 3., 'b': 3.}, 'hash'))
    calls = []
    def interrupted(*args):
        if calls:
            raise KeyboardInterrupt
        calls.append(1)
        return dict.fromkeys(module.FIELDS, '')
    monkeypatch.setattr(module, 'inventory_member', interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.main()
    assert manifest.read_text() == 'previous complete inventory\n'
