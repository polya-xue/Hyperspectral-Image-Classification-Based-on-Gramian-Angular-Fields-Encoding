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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from torch.cuda import amp
from torch_utils import de_parallel, is_parallel, torch_distributed_zero_first, fix_bn, freeze_layers
from torch.utils.tensorboard import SummaryWriter

import faiss

from utils import select_device, ModelEMA, get_cnf_matrix
from nets import create_model
from datasets import create_dataloader
from misc import init_seeds, increment_path, one_cycle, tf_cycle
from mixmethod import snapmix, cutmix, mixup
import transforms
import lr_find
import cv2

cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 16))  # NumExpr max threads

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False
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
    lr0, lrf, warmup_momentum, momentum = opt['lr0'], opt['lrf'], opt['warmup_momentum'], opt['momentum']
    backbone, pool, head, layer, dims_pool, dims_head, loss, num_classes, loss_reduction, ema_model, head_drop, fc_drop = \
        opt['backbone'], opt['pool'], opt['head'], opt['layer'], opt['dims_pool'], opt['dims_head'], opt['loss'], opt[
            'num_classes'], opt['loss_reduction'], opt['ema_model'], opt['head_drop'], opt['fc_drop']
    fixbn = opt['fix_bn']
    aug_decay, aug_prob = opt['aug_decay'], opt['prob']

    # optim
    multi_step = opt['multistep']
    milestones = opt['milestones']
    split_group = opt['split_group']

    data_transforms = {
        "train": transforms.Compose([
            # transforms.ShiftScaleRotate(p=0.2, shift_limit=0.1, scale_limit=(-0.5, 0.2), rotate_limit=15),
            # transforms.IAAPerspective(p=0.1, scale=(0.05, 0.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(p=0.2, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.ColorJitter(p=0.2, brightness=0.1, contrast=0.2, saturation=0.3, hue=0.01),
            # transforms.ColorJitter(p=0.2, hue=0.01),
            # transforms.ColorJitter(p=0.2, saturation=0.3),
            # transforms.ColorJitter(p=0.2, contrast=0.2),
            # transforms.ColorJitter(p=0.2, brightness=0.1),
            # transforms.Cutout(p=0.5, max_h_size=int(imgsz * 0.1), max_w_size=int(imgsz * 0.1), num_holes=5),
            # transforms.RandomErasing(p=0.2, sl=0.02, sh=0.1, rl=0.2),
            # transforms.RandomPatch(p=0.05, pool_capacity=1000, min_sample_size=100, patch_min_area=0.01,
            #                        patch_max_area=0.05, patch_min_ratio=0.1, p_rotate=0.5, p_flip_left_right=0.5),
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
        transforms=data_transforms['train'],
    )

    val_pipeline = dict(
        fold=fold,
        dataloader=dict(
            batch_size=batch_size // WORLD_SIZE * 2 if not opt['triplet'] else 32,
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

    allVal_pipeline = dict(
        fold=fold,
        dataloader=dict(
            batch_size=batch_size // WORLD_SIZE * 2 if not opt['triplet'] else 32,
            num_workers=workers,
            drop_last=False,
            pin_memory=False,
            shuffle=False,
            # collate_fn="my_collate_fn",
        ),
        dataset=dict(
            root_dir=os.path.join(opt['data_root'], 'All_images'),
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
        ckpt = torch.load(weights, map_location='cpu')
        model = create_model(model_config, pretrain_path=weights)
        LOGGER.info(f'Transferred all layers... ')
    else:
        model = create_model(model_config, pretrain_path='')

    model.cuda()

    # drop freeze
    for k, v in model.named_parameters():
        v.requires_grad = True

    # fix a fer layres in backbone
    if freeze[0] != 0:
        freeze_layers(model, opt, freeze_nums=freeze)
        if RANK in [-1, 0]:
            LOGGER.info(f'freezed first {freeze[0]} layers, and first {freeze[1]} blocks')

    # fix bn
    if fixbn:
        fix_bn(model)
        if RANK in [-1, 0]:
            LOGGER.info('freezed all bn layers in training phase!!!')

    # build dataloader
    train_loader, _ = create_dataloader(train_pipeline, rank=LOCAL_RANK, train=True, triplet=True if opt['triplet'] else False)

    nb = len(train_loader)
    nw = max(round(warmup_epochs * nb), 100)
    nbs = 64

    # TODO, calc downsample for model
    gs = 32
    accumulate_step = opt['accumulate_step'] if opt['accumulate_grad'] else 0
    # accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing

    # optimizer
    if not split_group:
        if opt['adam']:
            optimizer = Adam(model.parameters(), lr=lr0, betas=(momentum, 0.999), weight_decay=1e-6)
        elif opt['adamw']:
            optimizer = AdamW(model.parameters(), lr=lr0, weight_decay=1e-6)
        else:
            optimizer = SGD(model.parameters(), lr=lr0, momentum=momentum, weight_decay=5e-4, nesterov=True)
    else:
        param_groups = []
        param_groups.append({'params': model.backbone.parameters(), 'lr': lr0 * 0.1})
        param_groups.append({'params': model.pool.parameters(), 'lr': lr0})
        param_groups.append({'params': model.head.parameters(), 'lr': lr0})
        param_groups.append({'params': model.loss.parameters(), 'lr': lr0})
        if opt['adam']:
            optimizer = Adam(param_groups, lr=lr0, betas=(momentum, 0.999))
        elif opt['adamw']:
            optimizer = AdamW(param_groups, lr=lr0, betas=(momentum, 0.999))
        else:
            optimizer = SGD(param_groups, lr=lr0, momentum=momentum, nesterov=True, weight_decay=5e-4)

    # Scheduler
    if opt['tfcycle']:
        if RANK in [-1, 0]:
            LOGGER.info('using tf_cycle scheduler...')
        lf = tf_cycle(
            lr_max=5e-6 * batch_size if not opt['accumulate_grad'] else 5e-6 * batch_size * opt['accumulate_step'],
            steps=epochs)  # lr0 = 1 and cancel warmup
        # lf = tf_cycle(lr_max=5e-6 * batch_size, steps=epochs)         # lr0 = 1 and cancel warmup
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif opt['multistep']:
        milestones = opt['milestones']
        lf = lambda x: lr0
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        if RANK in [-1, 0]:
            LOGGER.info('using one_cycle scheduler...')
        # follow yolov5
        # lf = one_cycle(1., lrf, epochs)
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # native pytorch
        scheduler = lr_scheduler.OneCycleLR(optimizer, lr0, total_steps=None, epochs=epochs, steps_per_epoch=nb,
                                            pct_start=0.3)

    # EMA
    # ema = ModelEMA(deepcopy(model))
    if ema_model and RANK in [-1, 0]:
        tmp_model = deepcopy(model).cpu()
        ema = ModelEMA(deepcopy(tmp_model))
    if not ema_model:
        ema = None

    # Resume
    start_epoch, best_top1, best_top5, best_map5 = 0, 0.0, 0.0, 0.0

    if pretrained:
        # Optimizer
        if ckpt.get('optimizer'):
            optimizer.load_state_dict(ckpt['optimizer'])
            best_top1 = ckpt['best_top1']
            best_top5 = ckpt['best_top5']
            best_map5 = ckpt['best_map5']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1 if ckpt.get('optimizer') else 0
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt['sync_bn'] and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Process 0
    if RANK in [-1, 0]:
        val_loader, _ = create_dataloader(val_pipeline, rank=-1, val=True)
        allVal_loader, _ = create_dataloader(allVal_pipeline, rank=-1, val=True)

        # 从当前训练集中检索(不会将验证集也作为gallery), 所以 train=True.
        # gallery_loader, gallery_datasets = create_dataloader(gallery_pipeline, rank=-1, train=True)

    # DDP mode
    if cuda and RANK != -1:
        if freeze[0] == 0 and opt['augmethod']:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True,
                        broadcast_buffers=False)
        elif freeze[0] == 0 and not opt['augmethod']:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False,
                        broadcast_buffers=True)
        else:
            model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True,
                        broadcast_buffers=False)
        # model = DDP(model, device_ids=[LOCAL_RANK], broadcast_buffers=True)

    # Start training
    t0 = time.time()
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)

    if RANK in [-1, 0]:
        LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers} dataloader workers\n'
                    f'Loging results to {save_dir}\n'
                    f'Starting training for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):
        model.train()

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        pbar = enumerate(train_loader)
        optimizer.zero_grad()

        if opt['lr_search']:
            log_lrs, losses = lr_find.find_lr(train_loader, model, scaler, device, optimizer, tb_writer)
            LOGGER.info(f'log_lrs: {log_lrs}')
            LOGGER.info(f'losses: {losses}')
            exit(0)

        # Scheduler
        scheduler.step()

        # strong augmentation
        mixmethod = None
        if opt['augmethod']:
            mixmethod = opt['mixmethod']

        for i, batch in pbar:
            ni = i + nb * epoch
            imgs = batch['image']
            imgs = imgs.to(device, non_blocking=True).float()
            targets = batch['target']
            targets = targets.to(device)
            # branch
            if opt['augmethod']:
                imgs, targets_a, targets_b, lam_a, lam_b = eval(mixmethod)(imgs, targets, opt, model)

            # TODO use native one cycle
            # warmup--backbone
            if not opt['tfcycle']:
                if ni <= nw:
                    xi = [0, nw]
                    for j, x in enumerate(optimizer.param_groups):
                        # x['lr'] = np.interp(ni, xi, [0, x['initial_lr'] * lf(epoch)])
                        x['lr'] = np.interp(ni, xi, [0, x['initial_lr']])
                        # if 'momentum' in x:
                        #     x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])

            # Multi-scale
            if opt['multi_scale']:
                sz = random.randrange(imgsz * 0.75, imgsz * 1.25 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                if not opt['augmethod']:
                    loss = model(imgs, targets)
                else:
                    loss_a = model(imgs, targets_a)
                    loss_b = model(imgs, targets_b)
                    loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
                # loss 常规解算 mean, 仍然与bs有关, bs自带系数 WORLD_SIZE, 所以loss不用再次 scale
                # if RANK != -1:
                #     loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt['accumulate_grad']:
                    loss = loss / accumulate_step

            # Backward
            scaler.scale(loss).backward()

            # Optimizer
            if ni - last_opt_step >= accumulate_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # for Native Onecycle
                # scheduler.step()
                if RANK in [-1, 0]:
                    if ema:
                        ema.update(model)
                last_opt_step = ni

            if RANK in [-1, 0] and ni % log_period == 0:
                lr = [x['lr'] for x in optimizer.param_groups]
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss = loss if not accumulate_step else loss * accumulate_step
                LOGGER.info(f'[{epoch}/{epochs - 1}]' + f'[{i}/{nb}]' +
                            f'GPU: {mem}' + '\t' + f'imgsz: {imgsz}' + '\t' +
                            f'loss: {loss:.4}' + '\t' + f'LR: {lr[0]:.4}')
                tb_writer.add_scalar('total loss', loss.cpu().item(), ni)

        if aug_decay:
            opt['prob'] = aug_prob * (1 - epoch / epochs)  # linear decay

        # # Scheduler
        # scheduler.step()

        # torch.cuda.empty_cache()

        if RANK in [-1, 0]:
            torch.cuda.empty_cache()

            # calc top1, top5, mAP@5
            with torch.no_grad():
                train_top1, train_top5, _ = val(train_loader, model, epoch, device, opt, store=False)
                LOGGER.info(f'train top1: {train_top1}, train top5: {train_top5}')

                top1, top5, cnf_matrix = val(val_loader, model, epoch, device, opt, store=True)
                print('cnf: \n', cnf_matrix)
                LOGGER.info(f'top1: {top1}, top5: {top5}')

                allTop1, allTop5, _ = val(allVal_loader, model, epoch, device, opt, store=True)
                LOGGER.info(f'all_image top1: {top1}, top5: {top5}')

                # LOGGER.info(f'cnf_matrix:')
                # LOGGER.info(cnf_matrix)
                tb_writer.add_scalar('top1', top1, epoch)
                tb_writer.add_scalar('top5', top5, epoch)

                t1, t5, m5 = False, False, False
                # save model
                ckpt = {
                    'epoch': epoch,
                    'best_top1': best_top1,
                    'best_top5': best_top5,
                    'best_map5': best_map5,
                    'model': deepcopy(de_parallel(model)),
                    'ema': deepcopy(ema.ema) if ema else None,
                    'updates': ema.updates if ema else None,
                    'optimizer': optimizer.state_dict() if opt['save_optim'] else None,
                }
                # last epoch, drop optimizer weights for surviving model size
                if epoch == epochs - 1:
                    del ckpt['optimizer']
                    del ckpt['updates']
                    # TODO: when ema is valid
                    # if ema:
                    #     del ckpt['model']

                if top1 > best_top1:
                    t1 = True
                    torch.save(ckpt, f'{save_dir}/best_top1_{epoch}.pt')
                    best_top1 = top1

    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        LOGGER.info(f'\n the output of training has been output to {save_dir}, the logger in {save_dir}/result.txt')


def val(val_loader, model, epoch, device, opt, store=True):
    epochs = opt['epochs']
    save_dir = opt['save_dir']
    dims = opt['dims_head']
    threshold = opt['threshold']
    period = opt['retrieval_period']
    val_map = opt['val_map']

    model.eval()
    nbv = len(val_loader)
    LOGGER.debug(f'nbv: {nbv}')
    with torch.no_grad():
        query_names = []
        query_features = []
        pbarVal = enumerate(val_loader)
        pbarVal = tqdm(pbarVal, total=nbv)  # progress bar

        correct1, correct5, total = 0, 0, 0
        cnf_matrix = np.zeros((17, 17))
        for i, batch in pbarVal:
            imgs = batch['image']
            imgs = imgs.to(device, non_blocking=True).float()
            imgs_name = batch['image_name']
            targets = batch['target']
            targets = targets.to(device)
            if val_map:
                logits, fea = model(imgs, targets, hook_feature=True)
                query_names.extend(imgs_name)
                query_features.extend(fea.cpu().numpy().tolist())
            else:
                logits = model(imgs, targets)

            # top1
            pred = logits.argmax(dim=1)
            correct1 += torch.eq(pred, targets).sum().float().item()

            # top5
            maxk = max((1, 5))
            targets_resize = targets.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct5 += torch.eq(pred, targets_resize).sum().float().item()

            total += imgs.shape[0]

            p_t = get_cnf_matrix(logits, targets, 17)
            cnf_matrix += p_t

        # calc echo classes acc
        classesAcc = []
        for ci in range(len(cnf_matrix)):
            classesAcc.append(cnf_matrix[ci][ci] / (cnf_matrix[:, ci]).sum())
        print(classesAcc)

        top1, top5 = correct1 / total, correct5 / total
        torch.cuda.empty_cache()
        return top1, top5, cnf_matrix


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
    parser.add_argument('--retrieval-period', type=int, default=5, help='retrieval period')  # save time
    parser.add_argument('--val-period', type=int, default=5, help='val period')
    parser.add_argument('--val-map', action='store_true', help='calc map in val process')
    parser.add_argument('--accumulate-grad', action='store_true', help='accumulate grad in train process')
    parser.add_argument('--accumulate-step', type=int, default=4, help='accumulate batch number')
    parser.add_argument('--load-npy', action='store_true', help='load dataset from npy, for accelerating preprocess')
    parser.add_argument('--data-root-npy', action='store_true', help='not useful')
    parser.add_argument('--bbox-only', action='store_true', help='only using cropped images for training')
    parser.add_argument('--bbox-combine', action='store_true', help='using raw and cropped images for training')
    parser.add_argument('--bbox-val', action='store_true', help='using bbox for val')
    parser.add_argument('--triplet', action='store_true', help='using triplet loss')

    # build model
    # pipeline: backbone -> pool -> head -> loss
    parser.add_argument('--save-optim', action='store_true', help='save optimizer params for resuming')
    parser.add_argument('--backbone', default='tf_efficientnet_l2_ns', help='model backbone')
    parser.add_argument('--pool', default='avg', help='pool')
    parser.add_argument('--head', default='reduction_drop_fc', help='model identity in [identity, bnneck, bnneck_drop, '
                                                                    'reduction, reduction_fc_bn, reduction_fc, ]')
    parser.add_argument('--dims-pool', type=int, default=2560, help='the dims of pool layer or backbone dims')
    parser.add_argument('--layer', default=None, help='attention layer')
    parser.add_argument('--dims-head', type=int, default=512, help='the dims of head')
    parser.add_argument('--head-drop', type=float, default=0.0, help='setting dropout in embedded head')
    parser.add_argument('--fc-drop', type=float, default=0.0, help='setting dropout in fc head')
    parser.add_argument('--loss', default='arcv1', help='loss')
    parser.add_argument('--loss-reduction', default='mean', help='how to handle loss, in [mean, sum, none]')
    parser.add_argument('--num-classes', type=int, default=15587, help='number of classes')
    parser.add_argument('--ema-model', action='store_true', help='use ema model')

    # data augmentation
    parser.add_argument('--augmethod', action='store_true', help='use strong augmentation for training')
    parser.add_argument('--mixmethod', default='snapmix', help='use mixmethod augmentation for training, '
                                                               'in [snapmix, cutmix, mixup ...]')
    parser.add_argument('--prob', type=float, default=1.0, help='the prob of snapmix')
    parser.add_argument('--aug-decay', action='store_true', help='use decay in snapmix, like mosaic in yolox')
    parser.add_argument('--beta', type=float, default=5.0, help='beta distributed')
    parser.add_argument('--cropsize', default=512, type=int,
                        help='cropsize for snapmix, e.g. 768-384, 512-256, 640-320')

    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--tfcycle', action='store_true', help='reproduce tf version mAP')
    parser.add_argument('--multistep', action='store_true', help='using multisteps to update lr in training')
    parser.add_argument('--milestones', type=int, nargs='+', default=[14, 18], help='which steps to decay lr')
    parser.add_argument('--split-group', action='store_true', help='using differ group to update lr, e.g. backbone, fc')

    parser.add_argument('--lr-search', action='store_true', help='using finding best lr0')
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
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--fix-bn', action='store_true', help='whether fix all bn layers')
    parser.add_argument('--freeze', type=int, nargs='+', default=[0, 0],
                        help='Number of layers to freeze for backbone, [a(layers), b(blocks)], e.g. effb5 = [3, 3]')
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

# time cost
# dataloader cost: 0.007090330123901367
# preprecess cost: 0.011430978775024414
# forward cost: 0.2050943374633789
# backward cost: 0.6514279842376709
# optim cost: 0.03732776641845703
# post precess cost: 1.430511474609375e-06
