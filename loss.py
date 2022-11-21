# -*- coding: utf-8 -*-
# @Author: yang
# @Email: chengyangno1@gmail.com
# @File: loss.py
# @Date: 14/02/2022


import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class CrossEntroy(nn.Module):
    def __init__(self, in_feat=512, num_classes=5, reduction='mean', weight=1.0):
        super(CrossEntroy, self).__init__()
        self.linear = nn.Linear(in_feat, num_classes, bias=True)
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.weight = weight

    def forward(self, predicts, targets=None):
        predicts = self.linear(predicts)

        if self.training:
            return self.criterion(predicts, targets.long()) * self.weight
        else:
            return predicts


# ArcfaceV1
# class ArcMarginProduct(nn.Module):
#     r"""Implement of large margin arc distance: :
#         Args:
#           in_features: size of each input sample
#           out_features: size of each output sample
#           s: norm of input feature
#           m: margin
#           cos(theta + m)
#       """
#
#     def __init__(self, in_feat=2048, num_classes=5, scale=30.0, margin=0.30, easy_margin=False, reduction='mean',
#                  weight=1.0):
#         super(ArcMarginProduct, self).__init__()
#         self.in_features = in_feat
#         self.out_features = num_classes
#         self._s = scale
#         self._m = margin
#         # self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
#         self.linear = nn.Parameter(torch.Tensor(num_classes, in_feat))
#         self.weight_loss = weight
#         self.criterion = nn.CrossEntropyLoss(reduction=reduction)
#
#         nn.init.xavier_uniform_(self.linear)
#
#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(self._m)
#         self.sin_m = math.sin(self._m)
#         self.th = math.cos(math.pi - self._m)
#         self.mm = math.sin(math.pi - self._m) * self._m
#
#     def forward(self, features, targets=None):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(features), F.normalize(self.linear))
#         if self.training:
#             sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#             phi = cosine * self.cos_m - sine * self.sin_m
#             phi = phi.type_as(cosine)
#             if self.easy_margin:
#                 phi = torch.where(cosine > 0, phi, cosine)
#             else:
#                 phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#             # --------------------------- convert targets to one-hot ---------------------------
#             # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#             one_hot = torch.zeros(cosine.size(), device='cuda')
#             one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
#             # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#             output = (one_hot * phi) + (
#                         (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#             output *= self._s
#             loss = self.criterion(output, targets) * self.weight_loss
#             return loss
#         else:
#             output = cosine * self._s
#             return output

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
            self,
            in_feat: int,
            num_classes: int,
            s: float = 30.0,
            m: float = 0.30,
            easy_margin: bool = False,
            ls_eps: float = 0.0,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_feat
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.criterion = nn.CrossEntropyLoss()

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor, targets: torch.Tensor = None, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if not self.training:
            pred_class_logits = cosine * self.s
            return pred_class_logits

        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = self.criterion(output, targets)
        return loss


class ArcfaceLossDropout(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, scale=30, margin=0.3, dropout_rate=0.1, reduction='mean',
                 weight=1.0):
        super(ArcfaceLossDropout, self).__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.weight_loss = weight

        # self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.linear = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))
        nn.init.kaiming_uniform_(self.linear, a=math.sqrt(1))

        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features, targets=None):
        # print(self._m)
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m
        # get cos(theta)
        cos_theta = F.linear(self.dropout(F.normalize(features)), F.normalize(self.linear))
        if self.training:
            cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

            target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            cos_theta_m = cos_theta_m.type_as(target_logit)
            mask = cos_theta > cos_theta_m
            final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            # import pdb; pdb.set_trace()
            cos_theta[mask] = (hard_example * (self.t + hard_example)).type_as(target_logit)
            cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
            pred_class_logits = cos_theta * self._s
            loss = self.criterion(pred_class_logits, targets) * self.weight_loss
            return loss
        else:
            return cos_theta * self._s


class ArcfaceLossDropoutSimple(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, scale=30, margin=0.3, dropout_rate=0.2, reduction='mean',
                 weight=1.0):
        super(ArcfaceLossDropoutSimple, self).__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.weight_loss = weight

        # self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.linear = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))
        nn.init.kaiming_uniform_(self.linear, a=math.sqrt(1))

        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features, targets=None):
        # print(self._m)
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m
        # get cos(theta)
        cos_theta = F.linear(self.dropout(F.normalize(features)), F.normalize(self.linear))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        if self.training:
            target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            cos_theta_m = cos_theta_m.type_as(target_logit)
            mask = cos_theta > cos_theta_m

            final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

            hard_example = cos_theta[mask]
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            # import pdb; pdb.set_trace()
            cos_theta[mask] = (hard_example * (self.t + hard_example)).type_as(target_logit)
            cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
            pred_class_logits = cos_theta * self._s
            loss = self.criterion(pred_class_logits, targets) * self.weight_loss
            return loss
        else:
            pred_class_logits = cos_theta * self._s
            return pred_class_logits


class LDAMLoss(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, scale=30, max_margin=0.5, num_class_list=[],
                 dropout_rate=0.2, reduction='mean', weight=1.0):
        super(LDAMLoss, self).__init__()
        self.in_features = in_feat
        self.out_features = num_classes
        self.s = scale
        self.max_m = max_margin
        self.betas = [0, 0.9999]
        num_class_list = np.load('./data/num_class_list.npy')
        self.num_class_list = num_class_list

        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (self.max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list

        idx = 1
        per_cls_weights = (1.0 - self.betas[idx]) / (1.0 - np.power(self.betas[idx], self.num_class_list))
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight_list = torch.cuda.FloatTensor(per_cls_weights)

        # self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.linear = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.weight_loss = weight

        nn.init.xavier_uniform_(self.linear)

    def forward(self, features, targets=None):
        features = F.linear(F.normalize(features), F.normalize(self.linear))
        if self.training:
            index = torch.zeros_like(features, dtype=torch.uint8)  # 和 x 维度一致全 0 的tensor
            index.scatter_(1, targets.data.view(-1, 1), 1)  # dim idx input
            index_float = index.type(torch.cuda.FloatTensor)  # 转 tensor
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))
            x_m = features - batch_m  # y 的 logit 减去 margin
            output = torch.where(index, x_m, features)  # 按照修改位置合并
            return F.cross_entropy(self.s * output, targets, weight=self.weight_list)
        else:
            pred_class_logits = features * self.s
            return pred_class_logits


class TripletLoss(nn.Module):
    def __init__(self, margin=0.6, weight=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数
        self.weight = weight

    def forward(self, inputs, labels):
        n = inputs.size(0)  # 获取batch_size
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维

        # Enable 16 bit precision
        inputs = inputs.to(torch.float32)

        dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(1, -2, inputs, inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 然后开方

        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 这里dist[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
        dist_ap, dist_an = [], []
        for i in range(n):
            if len(dist[i][mask[i]]) > 0 and len(dist[i][mask[i] == 0]) > 0:
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor

        loss = self.ranking_loss(dist_an, dist_ap, y)

        if self.training:
            return loss * self.weight
        else:
            return None
