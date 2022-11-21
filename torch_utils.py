import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def fix_bn(model):
    model = de_parallel(model)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


# for a few layer from backbone
def freeze_layers(model, opt, freeze_nums=[0, 0]):
    num_layers, num_stages = freeze_nums[0], freeze_nums[1]
    # for efficientNet, 前两层固定, 第三层固定前 num_layers 个 stage 尝试
    if 'efficientnet' in opt['backbone']:
        for k, v in model.backbone.named_children():
            if int(k) < num_layers:
                v.eval()
                for param in v.parameters():
                    param.requires_grad = False
            elif int(k) == num_layers:
                for i, j in v.named_children():
                    if int(i) <= num_stages:
                        j.eval()
                        for param in j.parameters():
                            param.requires_grad = False
                    else:
                        j.train()
                        for param in j.parameters():
                            param.requires_grad = True
            else:
                v.train()
                for param in v.parameters():
                    param.requires_grad = True
