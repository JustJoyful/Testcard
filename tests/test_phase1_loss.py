import pytest
import torch


def test_rank_zero_equals_mean_l1():
    from src.losses import quality_loss
    torch.testing.assert_close(quality_loss(torch.tensor([0.,1.]),torch.tensor([0.,2.]),0.),torch.tensor(.5))


def test_rank_loss_ignores_ties():
    from src.losses import quality_loss
    x=torch.tensor([1.,2.],requires_grad=True)
    torch.testing.assert_close(quality_loss(x,torch.ones(2),1.),quality_loss(x,torch.ones(2),0.))


def test_correct_order_has_lower_rank_loss():
    from src.losses import quality_loss
    y=torch.tensor([-1.,1.])
    good=quality_loss(y,y,1.)-quality_loss(y,y,0.)
    bad=quality_loss(-y,y,1.)-quality_loss(-y,y,0.)
    assert good < bad


@pytest.mark.parametrize('weight,values',[(-1.,[0.,1.]),(.1,[float('nan'),1.])])
def test_invalid_loss_fails(weight,values):
    from src.losses import quality_loss
    with pytest.raises(ValueError):
        quality_loss(torch.tensor(values),torch.ones(2),weight)
