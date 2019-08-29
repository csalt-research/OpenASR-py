import torch
import torch.nn as nn

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
    def __init__(self, input_size, hidden_size):
        self.agg = nn.LSTM(input_size, hidden_size)
        self.clf = nn.Linear(hidden_size, 1)

    def forward(self, enc_out):
        # enc_out -> [T, B, :]
        _, out = self.agg(enc_out)
        logits = self.clf(out.squeeze(0))
        return torch.sigmoid(logits.squeeze(1))
