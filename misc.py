from pathlib import Path
import re
import glob

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import torch


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def tf_cycle(lr_min=1e-6, lr_max=5e-6 * 128, lr_start=1e-6, lr_warm=4, lr_decay=0.9, steps=24):
    return lambda x: (lr_max - lr_start) / lr_warm * x + lr_start if x < 4 else (lr_max - lr_min) * lr_decay ** (x - lr_warm) + lr_min


if __name__ == '__main__':
    import matplotlib.pyplot as ply
    batch_size = 24
    x = np.arange(0, 24)
    y = []

    # cosine
    # cosine = one_cycle(y1=1.0, y2=0.1, steps=24)
    # for i in x:
    #     y.append(cosine(i))

    # tf
    tf = tf_cycle(lr_min=1e-6, lr_max=5e-6 * batch_size, lr_start=1e-6, lr_warm=4, lr_decay=0.9, steps=24)
    for i in x:
        y.append(tf(i))
        print(tf(i))
    plt.plot(x, y)
    ply.show()

