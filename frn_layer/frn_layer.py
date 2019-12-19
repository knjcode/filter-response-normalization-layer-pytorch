import torch
from torch import nn
from torch.nn import Parameter


class FilterResponseNormLayer(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FilterResponseNormLayer, self).__init__()
        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.eps = Parameter(torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.gamma)

    def forward(self, input):
        nu2 = torch.mean(input**2, dim=(2, 3), keepdim=True, out=None)
        input = input * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * input + self.beta, self.tau)

    def extra_repr(self):
        return '{}'.format(
            self.num_features
        )
