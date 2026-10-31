import math
import torch
from torch import nn
from .common import masked_mean, valid_mask, readout


def sinusoidal_positions(T,H,device,dtype):
    positions=torch.arange(T,device=device,dtype=dtype)[:,None]
    frequencies=torch.exp(torch.arange(0,H,2,device=device,dtype=dtype)*(-math.log(10000.)/H))
    values=torch.zeros(T,H,device=device,dtype=dtype)
    values[:,0::2]=torch.sin(positions*frequencies)
    values[:,1::2]=torch.cos(positions*frequencies)
    return values


class TransformerHead(nn.Module):
    def __init__(self,tier,dropout):
        super().__init__()
        layer=nn.TransformerEncoderLayer(256,tier['transformer_nhead'],tier['transformer_ff'],dropout,
                                        activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,tier['transformer_layers'],enable_nested_tensor=False)
        self.readout=readout(256,tier['readout_hidden'],dropout)

    def forward(self,x,lengths):
        positions=sinusoidal_positions(x.shape[1],256,x.device,x.dtype)
        encoded=self.encoder(x+positions,src_key_padding_mask=~valid_mask(lengths.to(x.device),x.shape[1]))
        return self.readout(masked_mean(encoded,lengths))
