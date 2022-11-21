import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _triple, _pair, _single


class Flatten(nn.Module):
    def forward(self, input):
        # [N, C, H, W] -> [N, C * H * W]
        return input.view(input.size(0), -1)


# p 可训练
class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """
    def __init__(self, p=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert p > 0
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)


# p fix住
class GeneralizedMeanPoolingP(nn.Module):
    def __init__(self, p=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__()
        assert p > 0
        self.p = float(p)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            # [N, C, H, W] -> [N, C, H * W]
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x_avg = self.avgpool(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, 1)
        x = x_max + x_avg
        return x


def soft_pool2d(x, kernel_size=None, stride=None, force_inplace=False):
    # Get input sizes
    _, c, h, w = x.size()
    if kernel_size is None:
        kernel_size = _pair(h)
    else:
        kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)

    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(
        F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))


class SoftPool2d(torch.nn.Module):
    def __init__(self, kernel_size=None, stride=None, force_inplace=False):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)


