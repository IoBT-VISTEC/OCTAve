import math
from typing import Optional

import torch
from torch.functional import Tensor


def get_same_padding_conv(input_size: int, kernel_size: int, stride: int):
    """Calculate padding size for square kernel with variable stride on square input.
    To get the same output feature size as input (n_in = n_out) with corr. k,s
    """
    padding_size = ((stride*(input_size-1)) - input_size + kernel_size) / 2
    return math.ceil(padding_size)


def get_same_padding_transpose(input_size: int, kernel_size: int, stride: int):
    padding_size = (stride - (input_size * (1 - stride)) + kernel_size) // 2
    return padding_size

def rand_uniform(x: Optional[Tensor] = None):
    rand = torch.FloatTensor(1).uniform_(0, 1).type_as(x)
    return rand