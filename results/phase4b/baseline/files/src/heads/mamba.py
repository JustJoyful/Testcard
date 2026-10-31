"""Mamba-1 variable-B/C recurrence in ordinary PyTorch; no extension kernels."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import sanitize, valid_mask, reverse_valid, masked_mean, readout


def selective_scan(u,delta,A,B,C,D,lengths):
    mask=valid_mask(lengths.to(u.device),u.shape[1])
    u=sanitize(u,lengths)
    delta=sanitize(delta,lengths)
    B=sanitize(B,lengths)
    C=sanitize(C,lengths)
    state=u.new_zeros(u.shape[0],u.shape[2],A.shape[1])
    outputs=[]
    for t in range(u.shape[1]):
        dt=delta[:,t,:,None]
        next_state=torch.exp(dt*A)*state+dt*B[:,t,None,:]*u[:,t,:,None]
        state=torch.where(mask[:,t,None,None],next_state,state)
        value=(state*C[:,t,None,:]).sum(-1)+D*u[:,t]
        outputs.append(torch.where(mask[:,t,None],value,torch.zeros_like(value)))
    return torch.stack(outputs,dim=1)


class SelectiveBlock(nn.Module):
    def __init__(self,tier):
        super().__init__()
        inner,state,rank=tier['mamba_inner'],tier['mamba_state'],tier['mamba_dt_rank']
        self.inner,self.state,self.rank=inner,state,rank
        self.norm=nn.LayerNorm(256)
        self.in_proj=nn.Linear(256,2*inner,bias=False)
        self.conv=nn.Conv1d(inner,inner,tier['mamba_conv'],groups=inner,padding=tier['mamba_conv']-1)
        self.x_proj=nn.Linear(inner,rank+2*state,bias=False)
        self.dt_proj=nn.Linear(rank,inner)
        self.A_log=nn.Parameter(torch.arange(1,state+1).float().log().repeat(inner,1))
        self.D=nn.Parameter(torch.ones(inner))
        self.out_proj=nn.Linear(inner,256,bias=False)
        # Mamba-1 reference: rank-scaled dt weights; inverse-softplus log-uniform dt bias.
        nn.init.uniform_(self.dt_proj.weight,-rank**-.5,rank**-.5)
        dt=torch.exp(torch.rand(inner)*(math.log(.1)-math.log(.001))+math.log(.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))

    def forward_sequence(self,x,lengths):
        x=sanitize(x,lengths)
        u,gate=self.in_proj(self.norm(x)).chunk(2,dim=-1)
        u=sanitize(u,lengths)
        u=F.silu(self.conv(u.transpose(1,2))[:,:,:x.shape[1]].transpose(1,2))
        dt,B,C=self.x_proj(u).split([self.rank,self.state,self.state],dim=-1)
        delta=F.softplus(self.dt_proj(dt))
        y=selective_scan(u,delta,-torch.exp(self.A_log),B,C,self.D,lengths)
        return sanitize(x+self.out_proj(y*F.silu(gate)),lengths)


class MambaHead(nn.Module):
    def __init__(self,tier,dropout):
        super().__init__()
        if tier['mamba_layers']!=1:
            raise ValueError('matched tier requires one Mamba layer')
        self.forward_block=SelectiveBlock(tier)
        self.backward_block=SelectiveBlock(tier)
        self.dropout=nn.Dropout(dropout)
        self.readout=readout(256,tier['readout_hidden'],dropout)

    def forward(self,x,lengths):
        forward=self.forward_block.forward_sequence(x,lengths)
        backward=reverse_valid(self.backward_block.forward_sequence(reverse_valid(x,lengths),lengths),lengths)
        return self.readout(self.dropout(masked_mean((forward+backward)/2,lengths)))
