import json
from pathlib import Path

import pytest
import yaml

from phase2_fixtures import phase2_tree


def test_context_derives_only_features_and_destination_and_preserves_sources(tmp_path,monkeypatch):
    from src.phase2 import load_phase2_context
    _,protocol,configs,_=phase2_tree(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in Path('results/phase1').rglob('*') if p.is_file()}
    c=load_phase2_context(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
    assert c['base_protocol']==protocol
    for head in configs:
        assert c['resnet_configs'][head]=={**configs[head],'feature_set':'resnet50','results_root':'results/phase2/resnet50'}
    assert all(p.read_bytes()==value for p,value in before.items())
    assert Path('results/phase2/effective_configs.json').exists()
    assert c['audit']['source_sha256'] and 'dirty_working_tree' in c['audit']


@pytest.mark.parametrize('damage',['gate','failed_gate','stale_gate','checkpoint','missing_seed','duplicate','mos','config','override','feature_version','snapshot'])
def test_invalid_sources_cannot_open_phase2(tmp_path,monkeypatch,damage):
    from src.phase2 import load_phase2_context
    phase2_tree(tmp_path,monkeypatch)
    if damage=='gate':
        Path('results/phase1/phase_gate.json').unlink()
    elif damage=='failed_gate':
        Path('results/phase1/phase_gate.json').write_text('{"status":"failed"}')
    elif damage=='stale_gate':
        p=Path('results/phase1/phase_gate.json'); d=json.loads(p.read_text());d['evidence']['cache_and_committed_splits']['split_sha256']='changed';p.write_text(json.dumps(d))
    elif damage=='checkpoint':
        Path('results/phase1/cv/mean_pool/fold0/seed0/best.pt').write_bytes(b'corrupt')
    elif damage=='missing_seed':
        Path('results/phase1/cv/mean_pool/fold0/seed0/run.json').unlink()
    elif damage in ('duplicate','mos'):
        from src.features import sha256
        p=Path('results/phase1/cv/mean_pool/fold0/seed0/predictions.json'); d=json.loads(p.read_text())
        if damage=='duplicate':d.append(d[0])
        else:d[0]['mos']=999
        p.write_text(json.dumps(d));m=p.with_suffix('.meta.json');meta=json.loads(m.read_text());meta['predictions_sha256']=sha256(p);m.write_text(json.dumps(meta))
    elif damage=='config':
        Path('results/phase1/cv/mean_pool/fold0/seed0/config.json').write_text('{}')
    elif damage in ('override','feature_version'):
        p=Path('configs/phase2.yaml');d=yaml.safe_load(p.read_text())
        if damage=='override':d['learning_rate']=.99
        else:d['features']='configs/features_v1.yaml'
        p.write_text(yaml.safe_dump(d))
    else:
        Path('results/phase2').mkdir(parents=True);Path('results/phase2/effective_configs.json').write_text('{"inputs":{}}')
    with pytest.raises((ValueError,OSError,KeyError)):
        load_phase2_context(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
