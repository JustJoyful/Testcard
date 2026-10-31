from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .common import masked_mean, readout


class BiGRUHead(nn.Module):
    def __init__(self,tier,dropout):
        super().__init__()
        self.encoder=nn.GRU(256,tier['gru_hidden'],batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(dropout)
        self.readout=readout(2*tier['gru_hidden'],tier['readout_hidden'],dropout)

    def forward(self,x,lengths):
        packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.encoder(packed)
        values,_=pad_packed_sequence(encoded,batch_first=True,total_length=x.shape[1])
        return self.readout(self.dropout(masked_mean(values,lengths)))
