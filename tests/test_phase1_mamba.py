import math
import pytest
import torch
import yaml
from pathlib import Path


def test_scan_known_recurrence():
    from src.heads.mamba import selective_scan
    one=torch.ones(1,2,1,dtype=torch.float64)
    actual=selective_scan(torch.tensor([[[1.],[2.]]],dtype=torch.float64),one*math.log(2),-one[0,:1],one,one,torch.zeros(1,dtype=torch.float64),torch.tensor([2]))
    torch.testing.assert_close(actual,torch.tensor([[[math.log(2)],[2.5*math.log(2)]]],dtype=torch.float64),atol=1e-9,rtol=1e-7)


def test_scan_matches_independent_naive_values_and_gradients():
    from src.heads.mamba import selective_scan
    torch.manual_seed(7)
    sizes=[(2,3,2),(2,3,2),(2,2),(2,3,2),(2,3,2),(2,)]
    args=[torch.randn(s,dtype=torch.float64,requires_grad=True) for s in sizes]
    lengths=torch.tensor([1,3])
    actual=selective_scan(*args,lengths)
    u,delta,A,B,C,D=args
    rows=[]
    for b in range(2):
        times=[]
        states=[[torch.zeros((),dtype=torch.float64) for n in range(2)] for i in range(2)]
        for t in range(3):
            channels=[]
            for i in range(2):
                if t < lengths[b]:
                    for n in range(2):
                        states[i][n]=torch.exp(delta[b,t,i]*A[i,n])*states[i][n]+delta[b,t,i]*B[b,t,n]*u[b,t,i]
                    channels.append(sum(C[b,t,n]*states[i][n] for n in range(2))+D[i]*u[b,t,i])
                else:
                    channels.append(u[b,t,i]*0)
            times.append(torch.stack(channels))
        rows.append(torch.stack(times))
    expected=torch.stack(rows)
    torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-7)
    g1=torch.autograd.grad(actual.sum(),args,retain_graph=True)
    g2=torch.autograd.grad(expected.sum(),args,retain_graph=True)
    for a,b in zip(g1,g2):
        torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-7)
    tiny=[a[:1,:2].detach().requires_grad_() if a.ndim==3 else a.detach().requires_grad_() for a in args]
    assert torch.autograd.gradcheck(lambda *a:selective_scan(*a,torch.tensor([2])),tuple(tiny))


def tier():
    return yaml.safe_load(Path('configs/head_tiers.yaml').read_text())['matched']


def test_causal_sequence_prefix_ignores_future_suffix():
    from src.heads.mamba import SelectiveBlock
    block=SelectiveBlock(tier()).eval()
    x=torch.randn(2,7,256)
    y=x.clone(); y[:,4:]=999
    a=block.forward_sequence(x,torch.tensor([7,7]))
    b=block.forward_sequence(y,torch.tensor([7,7]))
    torch.testing.assert_close(a[:,:4],b[:,:4])


def test_mamba_padding_and_batch_invariance():
    from src.heads import build_head
    model=build_head('mamba',8,{'tier':tier(),'input_dropout':.1,'head_dropout':.1}).eval()
    x=torch.randn(2,3,8)
    expected=torch.cat([model(x[i:i+1,:n],torch.tensor([n])) for i,n in enumerate([1,3])])
    y=torch.full((2,7,8),float('nan')); y[0,:1]=x[0,:1]; y[1,:3]=x[1,:3]
    torch.testing.assert_close(model(y,torch.tensor([1,3])),expected,atol=1e-6,rtol=1e-5)
    model(y,torch.tensor([1,3])).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
