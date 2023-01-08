from typing import Literal, Optional, Sequence, Tuple

from einops import reduce
import numpy as np 
import torch
from torch import nn
from torch.functional import Tensor
from torch.nn import Conv2d, Softmax
from torch.nn import functional as F


class AdversarialAttentionGate(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """Adversarial Attention Gate Module

        Adversarial Attention Gate accept input from base residual module and perform
        dense convolution on feature maps with softmax to produce layers of segment and take hadamard product.

        params:

        in_channels     Number of input channels.
        out_channels    Number of output channels with each channel corresponding to each class segmentation.

        outputs:
        masked_x        Hadamard product of x and attention map.
        y_hat           Segmentor predicate.
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
        )
        self.softmax = Softmax(dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # -> Masked input, predicate 
        x_prime = self.conv1(x)
        y_hat = self.softmax(x_prime)
        # Collapse y_hat, treat layer 0 as a background layer.
        attention_mask = reduce(y_hat[:, 1 if y_hat.ndim > 1 else 0:, :, :], 'b c h w -> b 1 h w', 'sum')
        # Masking input with attention map
        masked_x = x * attention_mask
        return masked_x, y_hat


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    NOTE: Just a tensor clipping function
    """
    t = t.float()
    # t_min=t_min.float()
    # t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def create_mapping_kernel(kernel_size=7):
    """Mapping the kernel, creating k-square filter of the 1 x k x k kernel map.
    """
    # [kernel_size * kernel_size, kernel_size, kernel_size]
    # Generate kernel tensor
    kernel_arr = np.zeros((kernel_size * kernel_size, kernel_size, kernel_size), np.float32)
    # Initialize  of kernel map with 1
    for h in range(kernel_arr.shape[1]):
        for w in range(kernel_arr.shape[2]):
            # Sliding 1 along axis 1 with wrapping.
            kernel_arr[h * kernel_arr.shape[2] + w, h, w] = 1.0

    # [kernel_size * kernel_size, 1, kernel_size, kernel_size]
    kernel_tensor = torch.from_numpy(np.expand_dims(kernel_arr, axis=1))
    kernel_params = nn.parameter.Parameter(data=kernel_tensor.contiguous(), requires_grad=False)
    print(kernel_params.type())

    return kernel_params

def create_conv_kernel(in_channels, out_channels, kernel_size=3, avg=0.0, std=0.1):
    # [out_channels, in_channels, kernel_size, kernel_size]
    # Creating random valued conv kernel map, mostly a standard procedure with customable value distribution
    kernel_arr = np.random.normal(loc=avg, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
    kernel_arr = kernel_arr.astype(np.float32)
    kernel_tensor = torch.from_numpy(kernel_arr)
    kernel_params = nn.parameter.Parameter(data=kernel_tensor.contiguous(), requires_grad=True)
    print(kernel_params.type())
    return kernel_params

def create_conv_bias(channels):
    # [channels, ]
    bias_arr = np.zeros(channels, np.float32)
    assert bias_arr.shape[0] % 2 == 1

    bias_arr[bias_arr.shape[0] // 2] = 1.0
    bias_tensor = torch.from_numpy(bias_arr)
    bias_params = nn.parameter.Parameter(data=bias_tensor.contiguous(), requires_grad=True)

    return bias_params

class basePC(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        """
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        
        Documentation:
        this is a propagation basis class.
        """
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1_kernel = create_conv_kernel(in_channels=3, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv1 -> conv4 propagation kernel
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2*channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv4 -> conv 7 propagation kernel
        self.conv7_kernel = create_conv_kernel(in_channels=2*channels, out_channels=pn_size*pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size*pn_size)
        
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size*pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_src, input_thick, input_thin):
        input_all = torch.cat((input_src, input_thick, input_thin), dim=1)  # [b, 3, h, w] ##
        try:
            assert input_all.size()[1] == 3  # ##
        except AssertionError:
            raise ValueError(f'Expect concatenated confidence map to have channel depth of 3. Got {input_all.shape} instead.')
        # Full padding 
        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size//2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size//2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size//2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)

        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]

class adaptive_aggregationPC(nn.Module):
    def __init__(self, pn_size=5):
        """
        :param pn_size: the size of propagation neighbors.
        
        Aggregating confidence map with agg.coeff.
        """
        super().__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)

    def forward(self, input_thick, input_thin, agg_coeff):
        # agg.coeff is gained from progating 
        assert input_thick.size()[1] == 1 and input_thin.size()[1] == 1
        # Just maximize from both map.
        input_sal = torch.max(input_thick, input_thin)
        # Mapping...
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size//2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()

        # Calculate each pixels product.
        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)

        return prod_sal # [b, 1, h, w]


class baseC(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        """
        Base for single branch coarse stage
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        
        Documentation:
        this is a propagation basis class.
        """
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1_kernel = create_conv_kernel(in_channels=2, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv1 -> conv4 propagation kernel
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2*channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv4 -> conv 7 propagation kernel
        self.conv7_kernel = create_conv_kernel(in_channels=2*channels, out_channels=pn_size*pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size*pn_size)
        
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size*pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_src, input_thin):
        input_all = torch.cat((input_src, input_thin), dim=1)  # [b, 2, h, w] ##
        try:
            assert input_all.size()[1] == 2  # ##
        except AssertionError:
            raise ValueError(f'Expect concatenated confidence map to have channel depth of 3. Got {input_all.shape} instead.')
        # Full padding 
        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size//2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size//2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size//2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)

        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]


class adaptive_aggregationC(nn.Module):
    def __init__(self, pn_size=5):
        """
        :param pn_size: the size of propagation neighbors.
        
        Aggregating confidence map with agg.coeff.
        """
        super().__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)

    def forward(self, input_thin, agg_coeff):
        # agg.coeff is gained from progating 
        assert input_thin.size()[1] == 1
        # Just maximize from both map.
        input_sal = input_thin
        # Mapping...
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size//2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()

        # Calculate each pixels product.
        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)

        return prod_sal # [b, 1, h, w]

class baseMulti(nn.Module):
    def __init__(self, in_channels=3, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        """
        Base for single branch coarse stage
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        
        Documentation:
        this is a propagation basis class.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.conv1_kernel = create_conv_kernel(in_channels=in_channels, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv1 -> conv4 propagation kernel
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2*channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv4 -> conv 7 propagation kernel
        self.conv7_kernel = create_conv_kernel(in_channels=2*channels, out_channels=pn_size*pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size*pn_size)
        
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size*pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, basis: Sequence[Tensor]):
        input_all = torch.cat(basis[0:self.in_channels], dim=1)  # [b, 2, h, w] ##
        assert input_all.shape[1] == self.in_channels, f'Expect the input to had {self.in_channels} channels, got {input_all.shape[1]} instead.'
        # Full padding 
        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size//2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size//2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size//2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)

        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]


class adaptive_aggregationMulti(nn.Module):
    def __init__(self, pn_size=5):
        """
        :param pn_size: the size of propagation neighbors.
        
        Aggregating confidence map with agg.coeff.
        """
        super().__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)

    def forward(self, max_prob_input, agg_coeff):
        # agg.coeff is gained from progating 
        assert max_prob_input.shape[1] == 1
        input_sal = max_prob_input
        # Mapping...
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size//2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()

        # Calculate each pixels product.
        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)

        return prod_sal # [b, 1, h, w]


class GlobalAveragePooling2D(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor):
        return x.mean(dim=(2, 3))