# -*- coding: utf-8 -*-
# @Author: yang
# @Email: chengyangno1@gmail.com
# @File: nets.py
# @Date: 14/02/2022


import torch
import torch.nn as nn
import timm
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

# from layers import *
from loss import *
from heads import *


class BuildModel(nn.Module):
    def __init__(self, config):
        super(BuildModel, self).__init__()

        # backbone, drop pool and fc
        backbone = timm.create_model(config['backbone'], pretrained=True, num_classes=0)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # TODO
        # layer for attention

        # pool
        self.pool = SelectAdaptivePool2d(pool_type=config['pool'])

        # head
        if config['head'].lower() == 'identity':                # identity mode, dims_head must equal dims_pool
            # flatten [N, C, 1, 1] - > [N, C]
            self.head = IdentityHead()
        elif config['head'].lower() == 'reduction_fc_bn':
            self.head = ReductionFCBNHead(in_feat=config['dims_pool'], reduction_dim=config['dims_head'])
        elif config['head'].lower() == 'reduction_drop_fc':
            self.head = ReductionDropFCHead(in_feat=config['dims_pool'], reduction_dim=config['dims_head'], dropout_rate=config['head_drop'])
        elif config['head'].lower() == 'reduction_drop_fc_bn':
            self.head = ReductionDropFCBNHead(in_feat=config['dims_pool'], reduction_dim=config['dims_head'], dropout_rate=config['head_drop'])
        elif config['head'].lower() == 'reduction_fc':
            self.head = ReductionFCHead(in_feat=config['dims_pool'], reduction_dim=config['dims_head'])
        # loss
        self.triplet = config['triplet']

        if config['loss'].lower() == 'ce':
            self.loss = CrossEntroy(config['dims_head'], config['num_classes'])
        elif config['loss'].lower() == 'arcv1':
            self.loss = ArcMarginProduct(config['dims_head'], config['num_classes'])
        elif config['loss'].lower() == 'arc':
            self.loss = ArcfaceLossDropout(config['dims_head'], config['num_classes'], reduction=config['loss_reduction'], dropout_rate=config['fc_drop'])
        elif config['loss'].lower() == 'arcsimple':
            self.loss = ArcfaceLossDropoutSimple(config['dims_head'], config['num_classes'], reduction=config['loss_reduction'], dropout_rate=config['fc_drop'])
        elif config['loss'].lower() == 'ldam':
            self.loss = LDAMLoss(config['dims_head'], config['num_classes'], reduction=config['loss_reduction'])
        if self.triplet:
            self.triLoss = TripletLoss(margin=0.3)

    def forward(self, inputs, targets=None, hook_feature=False, hook_feature_only=False, feature_pos='head'):
        assert feature_pos in ('backbone', 'pool', 'head', 'both')

        backbone_feature = self.backbone(inputs)
        if hook_feature_only and feature_pos == 'backbone':
            return backbone_feature

        pool_feature = self.pool(backbone_feature)
        if hook_feature_only and feature_pos == 'pool':
            return pool_feature

        head_feature = self.head(pool_feature)
        if hook_feature_only and feature_pos == 'head':
            return head_feature

        if hook_feature_only and feature_pos == 'both':
            return backbone_feature, pool_feature, head_feature

        pred = self.loss(head_feature, targets)

        if self.triplet and self.training:
            predTri = self.triLoss(head_feature, targets)
            pred += predTri

        if not hook_feature:
            return pred
        elif hook_feature and feature_pos == 'backbone':
            return pred, backbone_feature
        elif hook_feature and feature_pos == 'pool':
            return pred, pool_feature
        elif hook_feature and feature_pos == 'head':
            return pred, head_feature


def create_model(cfg, pretrain_path=''):
    model = BuildModel(cfg)

    if pretrain_path:
        model_state_dict = model.state_dict()
        state_dict = torch.load(pretrain_path, map_location='cpu')

        if 'ema' in state_dict.keys() and state_dict['ema'] is not None:
            state_dict = state_dict['ema'].state_dict()
        elif 'model' in state_dict.keys():
            state_dict = state_dict['model'].state_dict()

        for key in state_dict.keys():
            if key in model_state_dict.keys() and state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = state_dict[key]

        # assert
        assert len(state_dict.keys()) == len(model_state_dict.keys())
        model.load_state_dict(model_state_dict)

    return model
