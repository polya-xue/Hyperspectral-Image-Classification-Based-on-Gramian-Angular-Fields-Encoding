# -*- coding: utf-8 -*-
# @Author: yang
# @Email: chengyangno1@gmail.com
# @File: mixmethod.py
# @Date: 07/03/2022

import torch
import torch.nn as nn
import imp
import numpy as np
import mixops
import os
import torch.nn.functional as F
import random
import copy
from addict import Dict

from torch_utils import de_parallel


def get_spm(input, target, conf, model):
    model.eval()
    imgsize = (conf.cropsize, conf.cropsize)
    bs = input.size(0)
    with torch.no_grad():
        output, fms = model(input, hook_feature=True, feature_pos='backbone')
        head = de_parallel(model).head
        clsw = de_parallel(model).loss
        # arcface
        loss_name = clsw._get_name().lower()
        head_name = head._get_name().lower()
        if 'reduction' in head_name:
            weight = clsw.linear
            bias = None
            fms = F.silu(fms)
            poolfea = F.adaptive_avg_pool2d(fms, (1, 1))
            headfea = head(poolfea)
            clslogit = F.softmax(clsw.forward(headfea))
            logitlist = []
            weight = F.linear(head.linear.weight.T, weight)
            weight = weight.view(weight.size(1), weight.size(0), 1, 1)
        elif 'identity' in head_name:
            weight = clsw.linear.weight.data
            bias = clsw.linear.bias.data
            # 激活值筛选
            fms = F.silu(fms)       # Eff
            # fms = F.relu(fms)       # Res
            poolfea = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
            clslogit = F.softmax(clsw.forward(poolfea))
            logitlist = []
            #                       classes         gap
            weight = weight.view(weight.size(0), weight.size(1), 1, 1)

        for i in range(bs):
            # 获取对应类别的 logit
            logitlist.append(clslogit[i, target[i]])
        clslogit = torch.stack(logitlist)

        # classes 个 7 * 7 输出的 feature -> [bs, 1000, 7, 7]
        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i, target[i]]          # 7 * 7
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)          # bs x 7 x7
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(outmaps, imgsize, mode='bilinear', align_corners=False)

        # bs x imgsize x imgsize
        outmaps = outmaps.squeeze()

        # Norm
        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    model.train()
    # 每个 bs 的激活图, 经过激活值筛选后输出的 logit
    return outmaps, clslogit


def snapmix(input, target, conf, model=None):
    conf = Dict(conf)   # dict -> .
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        wfmaps, _ = get_spm(input, target, conf, model)
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        lam1 = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()          # 乱序 bs
        wfmaps_b = wfmaps[rand_index, :, :]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = mixops.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = mixops.rand_bbox(input.size(), lam1)

        area = (bby2 - bby1) * (bbx2 - bbx1)
        area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

        if area1 > 0 and area > 0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            # 裁剪区域在各自激活图上的权重占比
            lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (wfmaps.sum(2).sum(1) + 1e-8)
            lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (wfmaps_b.sum(2).sum(1) + 1e-8)
            tmp = lam_a.clone()
            # 相同标签需要保持更大的比重, 并且相同label在 lam_a, lam_b 上有相同的标签值
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam_a[same_label] /= 2
            lam_b[same_label] /= 2
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1 - lam

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def as_cutmix(input, target, conf, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]

        bbx1, bby1, bbx2, bby2 = mixops.rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = mixops.rand_bbox(input.size(), lam)

        if (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1) > 4 and (bby2 - bby1) * (bbx2 - bbx1) > 4:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2 - bbx1, bby2 - bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            # adjust lambda to exactly match pixel ratio
            lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a *= torch.ones(input.size(0))
    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def cutmix(input, target, conf, model=None):
    conf = Dict(conf)
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    target_b = target.clone()

    if r < conf.prob:
        bs = input.size(0)
        lam = np.random.beta(conf.beta, conf.beta)
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        input_b = input[rand_index].clone()
        bbx1, bby1, bbx2, bby2 = mixops.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input_b[:, :, bbx1:bbx2, bby1:bby2]

        # adjust lambda to exactly match pixel ratio
        lam_a = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        lam_a *= torch.ones(input.size(0))

    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def cutout(input, target, conf=None, model=None):
    r = np.random.rand(1)
    lam = torch.ones(input.size(0)).cuda()
    target_b = target.clone()
    lam_a = lam
    lam_b = 1 - lam

    if r < conf.prob:
        bs = input.size(0)
        lam = 0.75
        bbx1, bby1, bbx2, bby2 = mixops.rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = 0

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()


def mixup(input, target, conf, model=None):
    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0)).cuda()
    bs = input.size(0)
    target_a = target
    target_b = target

    if r < conf.prob:
        rand_index = torch.randperm(bs).cuda()
        target_b = target[rand_index]
        lam = np.random.beta(conf.beta, conf.beta)
        lam_a = lam_a * lam
        input = input * lam + input[rand_index] * (1 - lam)

    lam_b = 1 - lam_a

    return input, target, target_b, lam_a.cuda(), lam_b.cuda()
