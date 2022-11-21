# Pytorch Imports
import os
import time
from tqdm import tqdm
import numpy as np
import logging
import math
import random
from pathlib import Path
from copy import deepcopy
import pandas as pd
import warnings
from sklearn.preprocessing import normalize

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from torch.cuda import amp
from torch_utils import de_parallel, is_parallel, torch_distributed_zero_first
from torch.utils.tensorboard import SummaryWriter

import faiss

from utils import select_device, ModelEMA
from nets import create_model
from datasets import create_dataloader
from misc import init_seeds, increment_path, one_cycle
import transforms

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] [%(lineno)s]: %(message)s")

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

LOGGER.info(f'LOCAL_RANK: {LOCAL_RANK}, RANK: {RANK}, WORLD_SIZE: {WORLD_SIZE}')


def train(opt, device):
    # init params
    save_dir, epochs, batch_size, resume, workers, freeze, data_root, weights, log_period, retrieval_period = \
        Path(opt['save_dir']), opt['epochs'], opt['batch_size'], opt['resume'], opt['workers'], \
        opt['freeze'], opt['data_root'], opt['weights'], opt['log_period'], opt['retrieval_period']
    imgsz, fold, warmup_epochs, threshold = opt['imgsz'], opt['fold'], opt['warmup_epochs'], opt['threshold']
    backbone, pool, head, layer, dims_pool, dims_head, loss, num_classes, loss_reduction, ema_model, head_drop, fc_drop = \
        opt['backbone'], opt['pool'], opt['head'], opt['layer'], opt['dims_pool'], opt['dims_head'], opt['loss'], opt[
            'num_classes'], opt['loss_reduction'], opt['ema_model'], opt['head_drop'], opt['fc_drop']

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomErasing(p=0.2, sl=0.02, sh=0.1, rl=0.2),
            transforms.ColorJitter(p=0.2, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RescalePad(output_size=imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        ]),
        "val": transforms.Compose([
            transforms.RescalePad(output_size=imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        ])
    }

    # tensorboard
    with torch_distributed_zero_first(LOCAL_RANK):
        tb_writer = SummaryWriter(log_dir=f'{save_dir}/tensorboard')

    # for logging
    fh = logging.FileHandler(f'{save_dir}/result.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)
    LOGGER.addHandler(fh)

    if RANK in [-1, 0]:
        LOGGER.info(f'opt: {opt}')

    model_config = {
        'backbone': backbone,
        'pool': pool,
        'dims_pool': dims_pool,
        'head': head,
        'dims_head': dims_head,
        'head_drop': head_drop,
        'fc_drop': fc_drop,
        'layer': layer,
        'loss': loss,
        'num_classes': num_classes,
        'loss_reduction': loss_reduction,
        'triplet': opt['triplet']
    }


    train_pipeline = dict(
        fold=fold,
        dataloader=dict(
            batch_size=batch_size // WORLD_SIZE,
            num_workers=workers,
            drop_last=False,
            pin_memory=False,
            shuffle=True,
            # collate_fn="my_collate_fn",
        ),
        dataset=dict(
            root_dir=os.path.join(opt['data_root'], 'train20'),
        ),
        transforms=data_transforms['val'],
    )

    val_pipeline = dict(
        fold=fold,
        dataloader=dict(
            batch_size=batch_size // WORLD_SIZE * 2,
            num_workers=workers,
            drop_last=False,
            pin_memory=False,
            shuffle=False,
            # collate_fn="my_collate_fn",
        ),
        dataset=dict(
            root_dir=os.path.join(opt['data_root'], 'test_edge'),
        ),
        transforms=data_transforms['val'],
    )

    # LOGGER.info(f'train_pipeline: {train_pipeline}')
    # LOGGER.info(f'val_pipeline: {val_pipeline}')
    # if RANK in [-1, 0]:
    #     LOGGER.info(save_dir, opt)

    cuda = device.type != 'cpu'

    init_seeds(1 + RANK)
    pretrained = weights.endswith('.pt')
    if pretrained:
        model = create_model(model_config, pretrain_path=weights)
        LOGGER.info(f'Transferred all layers... ')
    else:
        raise ValueError("must be loaded pretrained model...")

    model.cuda()
    model.eval()

    # build dataloader
    train_loader, _ = create_dataloader(train_pipeline, rank=-1, test=True)
    # for oof
    val_loader, _ = create_dataloader(val_pipeline, rank=-1, val=True)

    with torch.no_grad():
        train_names = []
        train_features = []
        train_targets = []
        pbarTrain = enumerate(train_loader)
        nbt = len(train_loader)
        pbarTrain = tqdm(pbarTrain, total=nbt)  # progress bar
        LOGGER.info(f"preparing to extract train...")
        for i, batch in pbarTrain:
            imgs = batch['image']
            imgs = imgs.to(device, non_blocking=True).float()
            imgs_name = batch['image_name']
            targets = batch['target']

            fea = model(imgs, hook_feature=True, hook_feature_only=True, feature_pos='head')
            train_names.extend(imgs_name)
            train_features.extend(fea.cpu().numpy().tolist())
            train_targets.extend(targets)

    with torch.no_grad():
        val_names = []
        val_features = []
        val_targets = []
        val_logits = []
        nbv = len(val_loader)
        pbarVal = enumerate(val_loader)
        pbarVal = tqdm(pbarVal, total=nbv)  # progress bar
        LOGGER.info(f"preparing to extract val...")
        for i, batch in pbarVal:
            imgs = batch['image']
            imgs = imgs.to(device, non_blocking=True).float()
            imgs_name = batch['image_name']
            targets = batch['target']
            logit, fea = model(imgs, hook_feature=True, hook_feature_only=False, feature_pos='head')
            val_names.extend(imgs_name)
            val_features.extend(fea.cpu().numpy().tolist())
            val_targets.extend(targets)
            val_logits.extend(F.softmax(logit, dim=1).cpu().numpy().tolist())

    train_features = np.array(train_features).astype('float32')
    train_features = normalize(train_features, axis=1, norm='l2')
    train_names = np.array(train_names)
    train_targets = np.array(train_targets)
    np.save(f"{save_dir}/train_feature.npy", train_features)
    np.save(f"{save_dir}/train_name.npy", train_names)
    np.save(f"{save_dir}/train_targets.npy", train_targets)

    val_features = np.array(val_features).astype('float32')
    val_features = normalize(val_features, axis=1, norm='l2')
    val_names = np.array(val_names)
    val_targets = np.array(val_targets)
    val_logits = np.array(val_logits).astype('float32')
    np.save(f"{save_dir}/val_feature.npy", val_features)
    np.save(f"{save_dir}/val_name.npy", val_names)
    np.save(f"{save_dir}/val_targets.npy", val_targets)
    np.save(f"{save_dir}/val_logits.npy", val_logits)


def main(opt):
    # output
    if opt['resume']:
        ckpt = opt['resume']
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        LOGGER.info(f'Resuming training from {ckpt}')
        opt['weights'], opt['resume'] = ckpt, True

    # output
    opt['save_dir'] = str(increment_path(Path(opt['project']) / opt['name'], exist_ok=opt['resume']))

    # DDP mode
    device = select_device(opt['device'], batch_size=opt['batch_size'])
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt['batch_size'] % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    train(opt, device)
    if WORLD_SIZE > 1 and RANK == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup-epochs', type=int, default=3, help='nums of warmup')
    parser.add_argument('--fold', type=int, default=0, help='kth fold')
    parser.add_argument('--load-npy', action='store_true', help='load dataset from npy, for accelerating preprocess')
    parser.add_argument('--data-root-npy', action='store_true', help='not useful')
    parser.add_argument('--bbox-only', action='store_true', help='only using cropped images for training')
    parser.add_argument('--bbox-combine', action='store_true', help='using raw and cropped images for training')
    parser.add_argument('--bbox-val', action='store_true', help='using bbox for val')
    parser.add_argument('--triplet', action='store_true', help='using triplet loss')
    parser.add_argument('--retrieval-period', type=int, default=5, help='retrieval period')  # save time

    # build model
    # pipeline: backbone -> pool -> head -> loss
    parser.add_argument('--backbone', default='tf_efficientnet_b5', help='model backbone')
    parser.add_argument('--pool', default='avg', help='pool')
    parser.add_argument('--head', default='reduction_drop_fc', help='model identity in [identity, bnneck, bnneck_drop, '
                                                                    'reduction, reduction_fc_bn, reduction_fc, ]')
    parser.add_argument('--dims-pool', type=int, default=2048, help='the dims of pool layer or backbone dims')
    parser.add_argument('--layer', default=None, help='attention layer')
    parser.add_argument('--dims-head', type=int, default=512, help='the dims of head')
    parser.add_argument('--head-drop', type=float, default=0.0, help='setting dropout in embedded head')
    parser.add_argument('--fc-drop', type=float, default=0.0, help='setting dropout in fc head')
    parser.add_argument('--loss', default='ce', help='loss')
    parser.add_argument('--loss-reduction', default='mean', help='how to handle loss, in [mean, sum, none]')
    parser.add_argument('--num-classes', type=int, default=17, help='number of classes')
    parser.add_argument('--ema-model', action='store_true', help='use ema model')

    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learn rate')
    parser.add_argument('--lrf', type=float, default=0.1, help='final OneCycleLR learning rate (lr0 * lrf)')
    parser.add_argument('--warmup-momentum', type=float, default=0.8, help='warmup momentum')
    parser.add_argument('--momentum', type=float, default=0.937, help='# SGD momentum/Adam beta1')
    parser.add_argument('--threshold', type=float, default=0.4, help='the threshold for computing map5')
    parser.add_argument('--log-period', type=int, default=10, help='output log period')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--data-root', type=str, default='data', help='data root path')
    parser.add_argument('--epochs', type=int, default=20, help='epochs for training')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/infer', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--adamw', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')

    kwargs = vars(parser.parse_args())
    opt = {}
    for k, v in kwargs.items():
        opt[k] = v
    # set_seed(opt['seed'])
    main(opt)
