import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack

class GradientReversalFunction(torch.autograd.Function):
    def __init__(self, lmbda):
        self.lmbda = lmbda

    def forward(self, x):
        return x.view_as(x)
    
    def backward(self, grad):
        return -1. * self.lmbda * grad

class GradientReversalLayer(nn.Module):
    def __init__(self, lmbda):
        self.fn = GradientReversalFunction(lmbda)

    def forward(self, x):
        return self.fn(x)

class DATLayer(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 n_layers,
                 n_accents):
        # Aggregating function
        self.agg = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_size, 
                           num_layers=n_layers)
        # Accent classifier
        self.clf = nn.Linear(n_layers*hidden_size, 
                             n_accents)

        self.logsftmx = nn.LogSoftmax(dim=-1)

    def forward(self, enc_out, enc_lengths):
        # enc_out -> [T, B, :]
        packed_enc_out = pack(enc_out, enc_lengths)
        _, (h, _) = self.agg(packed_enc_out)
        batch_size = h.size(1)
        h_concat = h.transpose(1, 0).view(batch_size, -1)
        return self.logsftmx(self.clf(h_concat))
