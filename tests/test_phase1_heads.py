import pytest
import torch
import yaml
from pathlib import Path


def config():
    return {'tier':yaml.safe_load(Path('configs/head_tiers.yaml').read_text())['matched'],'input_dropout':.1,'head_dropout':.1}


@pytest.mark.parametrize('name',['mean_pool','bigru','transformer'])
def test_padding_invariance(name):
    from src.heads import build_head
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model=build_head(name,8,config()).eval()
    x=torch.randn(2,3,8)
    lengths=torch.tensor([1,3])
    expected=torch.cat([model(x[i:i+1,:n],torch.tensor([n])) for i,n in enumerate(lengths)])
    for pad in [999.,float('nan')]:
        padded=torch.full((2,7,8),pad)
        for i,n in enumerate(lengths):
            padded[i,:n]=x[i,:n]
        torch.testing.assert_close(model(padded,lengths),expected,atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('name',['mean_pool','bigru','transformer'])
def test_invalid_lengths_fail(name):
    from src.heads import build_head
    model=build_head(name,8,config())
    for lengths in [torch.tensor([0,3]),torch.tensor([1,4]),torch.tensor([1.,3.])]:
        with pytest.raises(ValueError):
            model(torch.randn(2,3,8),lengths)


@pytest.mark.parametrize('name',['mean_pool','bigru','transformer'])
def test_output_shape_and_finite_gradients(name):
    from src.heads import build_head
    model=build_head(name,8,config())
    value=model(torch.randn(2,5,8),torch.tensor([1,5]))
    assert value.shape==(2,)
    value.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_reverse_only_valid_prefix():
    from src.heads.common import reverse_valid
    value=reverse_valid(torch.tensor([[[1.],[2.],[3.],[999.],[999.]]]),torch.tensor([3]))
    torch.testing.assert_close(value[0,:3,0],torch.tensor([3.,2.,1.]))


def test_positions_beyond_training_length():
    from src.heads.transformer import sinusoidal_positions
    values=sinusoidal_positions(300,256,torch.device('cpu'),torch.float32)
    assert values.shape==(300,256) and torch.isfinite(values).all()
    assert not torch.equal(values[32],values[288])
