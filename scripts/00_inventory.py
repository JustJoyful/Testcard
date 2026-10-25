"""Validate archives, join labels, and inventory videos without bulk extraction."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import probe_video

FIELDS = ['video_id', 'dataset', 'relative_path', 'mos', 'mos_scale', 'source_size_bytes',
          'source_sha256', 'duration_s', 'fps', 'width', 'height', 'rotation_applied',
          'status', 'failure_reason']


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inspect_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for info in members:
            name = info.filename
            if (PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                    or '\\' in name or ':' in name or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'unsafe archive member: {name}')
        corrupt = None
        for info in members:
            try:
                with archive.open(info) as source:
                    while source.read(1024*1024):
                        pass
            except (zipfile.BadZipFile, zlib.error, RuntimeError, NotImplementedError):
                corrupt = info.filename
                break
        return {'sha256': file_hash(path), 'compressed_bytes': Path(path).stat().st_size,
                'uncompressed_bytes': sum(i.file_size for i in members),
                'member_count': len(members), 'corrupt_member': corrupt,
                'status': 'failed' if corrupt else 'ok'}


def validate_join(video_ids, labels, scale):
    duplicate = [key for key, count in Counter(video_ids).items() if count > 1]
    if duplicate:
        raise ValueError(f'duplicate video IDs: {duplicate}')
    missing, extra = set(video_ids) - labels.keys(), labels.keys() - set(video_ids)
    if missing or extra:
        raise ValueError(f'label join failed: missing={sorted(missing)} extra={sorted(extra)}')
    if any(not math.isfinite(float(v)) or not scale[0] <= float(v) <= scale[1] for v in labels.values()):
        raise ValueError('invalid MOS')


def read_labels(dataset, settings):
    archive_path = settings.get('labels') or settings['archive']
    with zipfile.ZipFile(archive_path) as archive:
        content = archive.read(settings['labels_member'])
    if dataset == 'konvid':
        rows = list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
        pairs = [(row['flickr_id'], float(row['mos'])) for row in rows]
    else:
        data = loadmat(io.BytesIO(content))
        pairs = [(Path(str(name.item())).stem, float(mos))
                 for name, mos in zip(data['video_list'].ravel(), data['mos'].ravel(), strict=True)]
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError(f'{dataset}: duplicate label IDs')
    return dict(pairs), hashlib.sha256(content).hexdigest()


def inventory_member(archive, info, dataset, mos, scale, temporary_root):
    video_id = PurePosixPath(info.filename).stem
    row = dict.fromkeys(FIELDS, '')
    row.update(video_id=video_id, dataset=dataset, relative_path=info.filename, mos=mos,
               mos_scale=f'{scale[0]}-{scale[1]}', source_size_bytes=info.file_size)
    try:
        with tempfile.TemporaryDirectory(prefix='inventory-', dir=temporary_root) as tmp:
            path = Path(tmp) / PurePosixPath(info.filename).name
            with archive.open(info) as source, path.open('wb') as target:
                shutil.copyfileobj(source, target)
            row['source_sha256'] = file_hash(path)
            row.update({k: v for k, v in probe_video(path, timestamps=False).items() if k != 'pts'})
        row['status'] = 'ok'
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        row.update(status='failed', failure_reason=f'{video_id}: {exc}')
    return row


def prepare_sample_paths(config, dataset):
    import os
    import re
    settings = config['datasets'][dataset]
    ids = config['samples']['konvid' if dataset == 'konvid' else 'livevqc_portrait']
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', key) for key in ids):
        raise ValueError('missing, duplicate or unsafe sample IDs')
    with Path(f'data/manifests/{dataset}.csv').open() as f:
        manifest = {row['video_id']: row for row in csv.DictReader(f)}
    target = Path(settings['raw_dir'])
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    with zipfile.ZipFile(settings['archive']) as archive:
        needed = sum(archive.getinfo(manifest[key]['relative_path']).file_size for key in ids)
        reserve = config.get('feasibility', {}).get('reserve_disk_bytes') or 0
        if shutil.disk_usage(target).free < needed + reserve:
            raise ValueError('insufficient disk for selected clips and reserve')
        for key in ids:
            row = manifest[key]
            path = target / (key + '.mp4')
            if path.exists():
                if file_hash(path) != row['source_sha256']:
                    raise ValueError(f'{key}: raw checksum mismatch; preserve and investigate')
            else:
                with tempfile.NamedTemporaryFile(dir=target, suffix='.partial', delete=False) as f:
                    partial = Path(f.name)
                    try:
                        with archive.open(row['relative_path']) as source:
                            shutil.copyfileobj(source, f)
                        f.flush()
                        os.fsync(f.fileno())
                        if file_hash(partial) != row['source_sha256']:
                            raise ValueError(f'{key}: extracted checksum mismatch')
                        os.replace(partial, path)
                    finally:
                        partial.unlink(missing_ok=True)
            paths[key] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = Path('data/manifests')
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path('data/raw')
    temporary.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((output / 'datasets.json').read_text())
    summary = {}
    failed = False
    for dataset, settings in config['datasets'].items():
        archive_path = Path(settings['archive'])
        audit = inspect_archive(archive_path)
        if audit['status'] != 'ok':
            raise ValueError(f'{dataset}: corrupt archive: {audit["corrupt_member"]}')
        labels, label_hash = read_labels(dataset, settings)
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted((i for i in archive.infolist() if i.filename.lower().endswith('.mp4')), key=lambda i: i.filename)
            validate_join([PurePosixPath(i.filename).stem for i in members], labels, settings['mos_scale'])
            reserve = config['feasibility']['reserve_disk_bytes']
            required = max(i.file_size for i in members)
            if shutil.disk_usage(temporary).free < required + (reserve or 0):
                raise ValueError(f'{dataset}: insufficient temporary disk capacity')
            rows = []
            published = output / f'{dataset}.csv'
            with tempfile.NamedTemporaryFile(mode='w', newline='', dir=output, suffix='.partial', delete=False) as f:
                partial = Path(f.name)
                try:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    for number, info in enumerate(members, 1):
                        row = inventory_member(archive, info, dataset, labels[PurePosixPath(info.filename).stem], settings['mos_scale'], temporary)
                        writer.writerow(row)
                        rows.append(row)
                        if number % 100 == 0:
                            f.flush()
                            print(f'{dataset}: {number}/{len(members)} probed', flush=True)
                    f.flush()
                    os.fsync(f.fileno())
                    os.replace(partial, published)
                finally:
                    partial.unlink(missing_ok=True)
        good = [r for r in rows if r['status'] == 'ok']
        failed |= len(good) != len(rows)
        summary[dataset] = {'manifest_sha256': file_hash(output / f'{dataset}.csv'), 'video_count': len(rows), 'successful_probes': len(good),
                            'failed_ids': [r['video_id'] for r in rows if r['status'] != 'ok'],
                            'archive': audit, 'largest_temporary_video_bytes': required,
                            'duration_total_s': sum(r['duration_s'] for r in good),
                            'distributions': {key: {'min': min(r[key] for r in good),
                                                  'median': float(np.median([r[key] for r in good])),
                                                  'max': max(r[key] for r in good)} for key in ['duration_s', 'fps', 'width', 'height']}}
        if dataset == 'konvid':
            candidates = sorted(r['video_id'] for r in good)
            if len(candidates) < 20:
                raise ValueError('fewer than 20 valid KoNViD candidates')
            sample = sorted(random.Random(config['seed']).sample(candidates, 20))
            if config['samples']['konvid'] and config['samples']['konvid'] != sample:
                raise ValueError('existing immutable KoNViD sample differs')
            config['samples']['konvid'] = sample
        else:
            portraits = sorted(r['video_id'] for r in good if
                               (r['width'] > r['height'] if r['rotation_applied'] % 180 else r['height'] > r['width']))
            if len(portraits) < 3:
                raise ValueError('fewer than three portrait clips')
            if not config['samples']['livevqc_portrait']:
                config['samples']['livevqc_portrait'] = portraits[:3]
        provenance[dataset].update(archive_identity=audit, label_member=settings['labels_member'], label_sha256=label_hash,
                                   accessed='2026-10-25', label_join='passed')
    summary['disk'] = {'free_bytes': shutil.disk_usage('.').free,
                       'reserve_bytes': config['feasibility']['reserve_disk_bytes'],
                       'reserve_confirmed': config['feasibility']['reserve_disk_bytes'] is not None,
                       'raw_policy': 'retain only 20 selected KoNViD clips and 3 portrait clips; original archives retained',
                       'cache_and_weights_projection': 'pending actual feature dimensions/weights'}
    Path('results/phase0/inventory.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output / 'datasets.json').write_text(json.dumps(provenance, indent=2)+'\n')
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    print('Inventory complete' if not failed else 'Inventory has failed probes', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
