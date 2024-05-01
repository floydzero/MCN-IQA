import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
from config import cfg


def squash(inputs, axis=-1, lmd=1):
    """
    simplified squash-function in dynamic routing
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm / (lmd + norm ** 2)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    source routing of dynamic routing

    Attributes:
    in_num_caps: (int) number of input capsules
    in_dim_caps: (int) dim of input capsules
    out_num_caps: (int) number of output capsules
    out_dim_caps: (int) dim of output capsules
    routings: (int) number of routing iteration

    Input:
    primary capsule (shape like [batch, in_num_caps, in_dim_caps])

    Output:
    secondary capsule (shape like [batch, out_num_caps, out_dim_caps])
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps),
                                   requires_grad=True)

    def forward(self, x):
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        x_hat_detached = x_hat.detach()

        b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).cuda()

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            c = F.softmax(b, dim=1)

            if i == self.routings - 1:
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)
        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    fixed primary capsule layer

    Attributes:
    in_dim: (int) the dim will transform
    mode: (str) choice of 'feature_map' or 'channel'

    Input:
    feature (shape like [batch, channels, height, width])

    Output:
    primary capsules (shape like [batch, -1, in_dim])
    """
    def __init__(self, in_dim=8, mode=''):
        super(PrimaryCapsule, self).__init__()
        self.in_dim = in_dim
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_dim, 1)
        self.mode = mode
        if self.mode == 'feature_map':
            self.fc_channel = nn.Linear(cfg.CAPSNET.CONTENT.IN_CHANNELS, 1)
        else:
            self.fc_channel = nn.Linear(64, 1)

    def forward(self, x):
        if self.mode == 'feature_map':
            output = x.view(x.size(0), -1, self.in_dim)
            lmd_src = output
            lmd = self.sigmoid(self.fc(lmd_src))
            output = squash(output, lmd=lmd)
            return output
        elif self.mode == 'channel':
            output = x.view(x.size(0), -1, self.in_dim)
            output = output.transpose(1, 2)
            lmd_src = output
            lmd = self.sigmoid(self.fc_channel(lmd_src))
            output = squash(output, lmd=lmd)
            return output

