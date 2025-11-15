import torch
import torch.nn as nn

class RoundSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class RoundSTE(nn.Module):
    def __init__(self):
        super(RoundSTE, self).__init__()

    def forward(self, x):
        return RoundSTEFunction.apply(x)
