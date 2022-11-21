import math

import numpy as np
from torch.cuda import amp
from tqdm import tqdm


def find_lr(train_loader, model, scaler, device, optimizer, tb_writer, init_value=1e-6, final_value=1e-2, beta=0.98):
    epochs = 5
    num = len(train_loader) - 1
    num *= epochs
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0
    best_loss = 0
    losses = []
    log_lrs = []
    optimizer.zero_grad()
    pbar = enumerate(tqdm(train_loader))

    print(f'using finding best lr0, train_loader_nums: {len(train_loader)}')

    for j in range(epochs):
        for i, batch in pbar:

            batch_idx = i + 1
            imgs = batch['image']
            imgs = imgs.to(device, non_blocking=True).float()
            targets = batch['target']
            targets = targets.to(device)

            with amp.autocast(enabled=True):
                loss = model(imgs, targets)

            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_idx)

            # stop
            if batch_idx > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses

            # record
            if smoothed_loss < best_loss or batch_idx == 1:
                best_loss = smoothed_loss

            # store
            losses.append(smoothed_loss)
            log_lrs.append(lr)
            # log_lrs.append(math.log10(lr))

            # tb_writer.add_scalar('total loss', smoothed_loss.item(), math.log10(lr))
            tb_writer.add_scalar('total loss', smoothed_loss.item(), lr * 1e6)
            # tb_writer.add_scalar('total loss', loss.item(), lr * 1e6)

            # Backward
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # update
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

    return log_lrs, losses
