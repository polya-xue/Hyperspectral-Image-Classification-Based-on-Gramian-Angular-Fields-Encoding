# -*- coding: utf-8 -*-
# @Author: yang
# @Email: chengyangno1@gmail.com
# @File: heads.py
# @Date: 14/02/2022


import torch
import torch.nn as nn
from utils import weights_init_kaiming


class BNneckHead(nn.Module):
    def __init__(self, in_feat, num_classes):
        super(BNneckHead, self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)

    def forward(self, features):
        # [N, C, 1, 1](来自于 aggregation 模块) -> [N, C]
        return self.bnneck(features)[..., 0, 0]


class BNneckHead_Dropout(nn.Module):
    def __init__(self, in_feat, num_classes, dropout_rate=0.15):
        super(BNneckHead_Dropout, self).__init__()
        self.bnneck = nn.BatchNorm2d(in_feat)
        # self.bnneck.apply(weights_init_kaiming)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features):
        return self.dropout(self.bnneck(features)[..., 0, 0])


class IdentityHead(nn.Module):
    def __init__(self):
        super(IdentityHead, self).__init__()

    def forward(self, features):
        return features[..., 0, 0]


class ReductionHead(nn.Module):
    def __init__(self, in_feat, reduction_dim):
        super(ReductionHead, self).__init__()

        self.reduction_dim = reduction_dim

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_feat, reduction_dim, 1, 1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.bnneck = nn.BatchNorm2d(reduction_dim)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, features):
        global_feat = self.bottleneck(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]
        # if not self.training:
        #     return bn_feat,None
        return bn_feat


# Option-E, Pool + FC + BN + Dropout_FC( + BN)
class ReductionFCBNHead(nn.Module):
    def __init__(self, in_feat, reduction_dim=512, bn_shift=False):
        super(ReductionFCBNHead, self).__init__()

        self.reduction_dim = reduction_dim
        self.linear = nn.Linear(in_feat, reduction_dim, bias=False)
        self.bn_shift = bn_shift

        self.bn = nn.BatchNorm1d(reduction_dim)
        self.bn.apply(weights_init_kaiming)
        if not bn_shift:
            self.bn.bias.requires_grad_(False)  # no shift grad

    def forward(self, features):
        features = features[..., 0, 0]
        fc_feat = self.linear(features)
        bn_feat = self.bn(fc_feat)
        return bn_feat


# Option-E, Pool + (drop)FC +
class ReductionDropFCHead(nn.Module):
    def __init__(self, in_feat, reduction_dim=512, dropout_rate=0.2):
        super(ReductionDropFCHead, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.reduction_dim = reduction_dim
        self.linear = nn.Linear(in_feat, reduction_dim, bias=True)

    def forward(self, features):
        features = features[..., 0, 0]
        fc_feat = self.linear(self.dropout(features))
        return fc_feat


# Option-E, Pool + (drop)FC + BN
class ReductionDropFCBNHead(nn.Module):
    def __init__(self, in_feat, reduction_dim=512, dropout_rate=0.2, bn_shift=False):
        super(ReductionDropFCBNHead, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.reduction_dim = reduction_dim
        self.linear = nn.Linear(in_feat, reduction_dim, bias=True)

        self.bn_shift = bn_shift
        self.bn = nn.BatchNorm1d(reduction_dim)
        self.bn.apply(weights_init_kaiming)
        if not bn_shift:
            self.bn.bias.requires_grad_(False)  # no shift grad

    def forward(self, features):
        features = features[..., 0, 0]
        fc_feat = self.linear(self.dropout(features))
        bn_feat = self.bn(fc_feat)
        return bn_feat


class ReductionFCHead(nn.Module):
    def __init__(self, in_feat, reduction_dim=512):
        super(ReductionFCHead, self).__init__()
        self.reduction_dim = reduction_dim
        self.linear = nn.Linear(in_feat, reduction_dim, bias=True)

    def forward(self, features):
        features = features[..., 0, 0]
        fc_feat = self.linear(features)
        return fc_feat
