import copy
import numpy as np
import pytest
import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x,lengths):
        return self.linear(x[:,0]).squeeze(-1)


def dataset(offset=0):
    return [{'video_id':f'v{i}','features':torch.tensor([[float(i)]]),'mos':float(i+offset)} for i in range(6)]


def config():
    return {'device':'cpu','cpu_threads':2,'batch_size':3,'max_epochs':3,'patience':2,'warmup_epochs':1,
            'learning_rate':.01,'weight_decay':.001,'gradient_clip_norm':1.,'rank_weight':.1,
            'deterministic_warn_only':True}


def test_normalization_uses_fit_rows_only(tmp_path):
    from src.training import fit_one
    result=fit_one(Tiny(),dataset(),dataset(100),config(),tmp_path,0,{})
    assert result['target_stats']['mean']==pytest.approx(2.5)
    assert result['target_stats']['std']==pytest.approx(np.std(np.arange(6)))


def test_validation_selects_best_checkpoint_not_last(tmp_path,monkeypatch):
    import src.training as training
    scores=iter([.9,.3,.1])
    monkeypatch.setattr(training,'correlations',lambda y,p:{'srcc':next(scores),'status':'ok'})
    model=Tiny()
    result=training.fit_one(model,dataset(),dataset(),config(),tmp_path,0,{})
    assert result['best_epoch']==1 and result['best_validation_srcc']==.9
    checkpoint=torch.load(result['checkpoint_path'],weights_only=True)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,checkpoint['state_dict'][key])


def test_test_label_perturbation_cannot_change_fit(tmp_path):
    from src.training import fit_one
    torch.manual_seed(5)
    first=Tiny(); second=copy.deepcopy(first)
    a=fit_one(first,dataset(),dataset(),config(),tmp_path/'a',0,{'heldout_audit':'original'})
    b=fit_one(second,dataset(),dataset(),config(),tmp_path/'b',0,{'heldout_audit':'changed'})
    assert a['best_epoch']==b['best_epoch'] and a['target_stats']==b['target_stats']
    for x,y in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_failed_or_incompatible_run_is_not_resumed(tmp_path):
    from src.training import fit_one
    result=fit_one(Tiny(),dataset(),dataset(),config(),tmp_path,0,{'split_sha256':'a'})
    with pytest.raises(ValueError,match='incompatible'):
        fit_one(Tiny(),dataset(),dataset(),config(),tmp_path,0,{'split_sha256':'b'})
    with open(result['checkpoint_path'],'ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        fit_one(Tiny(),dataset(),dataset(),config(),tmp_path,0,{'split_sha256':'a'})


def test_predictions_inverse_target_scale():
    from src.training import predict
    model=Tiny()
    with torch.no_grad():
        model.linear.weight.fill_(1); model.linear.bias.zero_()
    actual=predict(model,dataset(),{'mean':10.,'std':2.},'cpu')
    assert actual[2]=={'video_id':'v2','prediction':14.}
