import copy
from pathlib import Path

import pytest

from phase2_fixtures import HEADS,make_cv,phase2_tree


def scores(dino=.5,resnet=.6):
    return {feature:[{'feature_set':feature,'head':head,'fold':fold,'seed':seed,'best_validation_srcc':value,
                      'config_sha256':'c'*64,'run_sha256':'a'*64}
                     for head in HEADS for fold in range(5) for seed in range(5)]
            for feature,value in [('dino',dino),('resnet50',resnet)]}


def test_equal_weighting_exact_tie_and_order_independence():
    from src.phase2 import select_feature
    result=select_feature(scores())
    assert result['selected_feature_set']=='resnet50'
    assert result['per_feature_validation_mean']=={'dino':.5,'resnet50':pytest.approx(.6)}
    reversed_rows={k:list(reversed(v)) for k,v in reversed(list(scores().items()))}
    assert select_feature(reversed_rows)==result
    assert select_feature(scores(.5,.5))['selected_feature_set']=='dino'


@pytest.mark.parametrize('damage',['missing','duplicate','nan','test_field','control','feature','seed','hash'])
def test_selector_rejects_incomplete_or_non_validation_inputs(damage):
    from src.phase2 import select_feature
    rows=scores();row=rows['dino'][0]
    if damage=='missing':rows['dino'].pop()
    elif damage=='duplicate':rows['dino'][-1]=copy.deepcopy(row)
    elif damage=='nan':row['best_validation_srcc']=float('nan')
    elif damage=='test_field':row['test_srcc']=.99
    elif damage=='control':row['label_mode']='shuffled'
    elif damage=='feature':row['feature_set']='resnet50'
    elif damage=='seed':row['seed']=5
    else:row['run_sha256']=''
    with pytest.raises(ValueError):select_feature(rows)


def test_selection_loader_never_reads_test_predictions_or_metrics(tmp_path,monkeypatch):
    from src.phase2 import load_phase2_context,load_validation_runs,select_feature,save_selection
    phase2_tree(tmp_path,monkeypatch)
    c=load_phase2_context(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
    make_cv(c['resnet_protocol'],c['resnet_configs'],c['base_provenance'])
    original=Path.read_text
    def read(path,*args,**kwargs):
        assert 'predictions' not in path.name and 'metrics' not in path.name
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    result=select_feature(load_validation_runs(c))
    assert result['selected_feature_set']=='dino'
    assert len(result['sources']['dino'])==100
    assert save_selection(c)['source_identity']==c['source_identity']
