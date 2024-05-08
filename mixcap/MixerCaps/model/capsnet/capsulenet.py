import torch
from torch import nn
from .capsulelayers import *
from config import cfg
import torch.nn.init as init
import math
import os
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def squash(inputs, axis=-1):
    """
    source squash-function in dynamic routing
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs


class Squash(nn.Module):
    """
    learnable squash,SASF(to make length in [0, 1])
    Attributes:
    axis: (int) the squash dim
    lmd: (float) the learnable option
    learnable: (bool) if learnable

    Input:
    capsules (shape like [batch, caps_num, caps_dim])
    Output:
    capsules (shape like [batch, caps_num, caps_dim])
    """
    def __init__(self, axis=-1, lmd=1, learnable=True):
        super().__init__()
        self.learnable = learnable
        self.axis = axis
        if learnable == True:
            self.fc = nn.Linear(16, 1)
            #self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        if self.learnable:
            lmd = self.sigmoid(self.fc(x))
        else:
            lmd = 1

        norm = torch.norm(x, p=2, dim=self.axis, keepdim=True)
        scale = norm / (lmd + norm ** 2)
        return scale * x

class Affine(nn.Module):
    """
    FC to replace routing

    Attributes:
    num: (int) the number of group capsules
    in_dim: (int) the input dim
    out-dim: (int) the output dim

    Input:
    capsules: (shape like [batch, caps_num, caps_dim])

    Output:
    group capsules: (shape like [batch, num, caps_num, caps_dim])
    """
    def __init__(self, num, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.w = nn.Parameter(torch.randn(num, in_dim, out_dim))
        init.xavier_uniform_(self.w)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.in_dim)
        out = torch.einsum('b c d, o d n -> b o c n', [x, self.w])
        return out

# class ATR(nn.Module):
#     """
#     secondary layer of all capsnet

#     Attributes:
#     num: (int) group number
#     in_c: (int) channels of input feature
#     in_dim: (int) dim of intput feature
#     out_dim: (int) dim of output capsules
#     learnable: (bool) if learnable in squash

#     Input:
#     primary capusles (shape like [batch, in_caps_num, in_caps_dim])

#     Output:
#     secondary capsules (shape like [batch, num, out_caps_dim])
#     """
#     def __init__(self, num, in_c, in_dim, out_dim, leaxiernable):
#         super().__init__()

#         self.in_c = in_c
#         self.squash = Squash(axis=-1, learnable=learnable) # fixme: should be different 2 squash

#         self.v = Affine(num=num, in_dim=in_dim, out_dim=out_dim)
#         self.mask = nn.Parameter(F.softmax(torch.ones(num, num, self.in_c), dim=0), requires_grad=False)

#     def forward(self, x):
#         v = self.squash(self.v(x))
#         out = torch.einsum('c c n, b c n d -> b c d', [self.mask, v])

#         return self.squash(out)


class ATRMixer(nn.Module):
    """
    Mixer of QLR and QDR 

    Attributes:
    feature_in_c: (int) in_channels of feature branch
    spatial_in_c: (int) in_channels of spatial branch
    feature_in_dim: (int) in_dim of feature branch
    spatial_in_dim: (int) in_dim of spatial branch
    feature_out_dim: (int) out_dim of feature branch
    spatial_out_dim: (int) out_dim of spatial branch
    num: (int) group number
    learnable: (bool) if learnable in squash

    Input:
    feature of spatial and feature branch like ([[batch, spatial_channels, spatial_height, spatial_width], [batch, feature_channels, feature_height, feature_width]])

    Output:
    capsules (shape like [batch, group_num, spatial_out_dim + feature_out_dim])
    """
    def __init__(self, feature_in_c=2048, spatial_in_c=196, spatial_in_dim=1024, feature_in_dim=49, spatial_out_dim=16, feature_out_dim=16, num=16, num_c=64, learnable=True):
        super().__init__()

        self.atr_lr = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            ATR(num=num, in_c=spatial_in_c, in_dim=spatial_in_dim, num_c=num_c,out_dim=spatial_out_dim, learnable=learnable)
        )

        self.atr_dr = nn.Sequential(
            Rearrange('b c h w -> b c (h w)'),
            ATR(num=num, in_c=feature_in_c, in_dim=feature_in_dim, out_dim=feature_out_dim, learnable=learnable)
        )
        self.squash_final = Squash(learnable=False)
        self.fc1 = nn.Linear(16, 1)
        self.soft = nn.Softmax(dim=1)
    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        
        out1 = self.atr_lr(x1)
        out2 = self.atr_dr(x2)
        
        # The learnable W for QSM 
        out_m = self.fc1(out1)  
        out_m = nn.ReLU()(out_m)
        w = self.soft(out_m)+1 # w.shape(64,16,1)
        out1 = torch.mul(out1 , w)

        #Each of the two branches forms a global capsule。
        out1 = self.fc(out1.view(out1.size(0), -1))
        out2 = self.fc(out2.view(out2.size(0), -1))

        # mix the capsules from QLR and QDR
        out = torch.cat([out1, out2], dim=2)
        out = self.squash_final(out)
        out = out.norm(dim=-1)
        return out




class ContentCaps(nn.Module):
    """
    main capsnet

    Attributes:
    feature_in_c: (int) in_channels of feature branch
    spatial_in_c: (int) in_channels of spatial branch
    feature_in_dim: (int) in_dim of feature branch
    spatial_in_dim: (int) in_dim of spatial branch
    feature_out_dim: (int) out_dim of feature branch
    spatial_out_dim: (int) out_dim of spatial branch
    num: (int) group number
    learnable: (bool) if learnable in squash

    Input:
    feature of spatial and feature branch like ([[batch, spatial_channels, spatial_height, spatial_width], [batch, feature_channels, feature_height, feature_width]])

    Output:
    score (float): the prediction
    """
    def __init__(self, spatial_primary_in_c=1024, feature_primary_in_c=2048, feature_in_c=2048, spatial_in_c=196, spatial_in_dim=1024, feature_in_dim=49, spatial_out_dim=16, feature_out_dim=16, num=16, learnable=True):
        super().__init__()
        self.ce = ATRMixer(feature_in_c=feature_in_c, spatial_in_c=spatial_in_c, spatial_in_dim=spatial_in_dim, feature_in_dim=feature_in_dim, spatial_out_dim=spatial_out_dim, feature_out_dim=feature_out_dim, num=num, learnable=learnable)

        #self.fc = FC(num=num, dim=spatial_out_dim + feature_out_dim)

    def forward(self, x):
        out = self.ce(x)
        #x = self.fc(x)

        return out
