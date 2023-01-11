from logging import warning
import warnings
from typing import List, Literal, Sequence

import torch
from architectures.utils import rand_uniform
from torch import nn
from torch._C import Size
from torch.functional import Tensor
from torch.nn.init import kaiming_normal_, xavier_uniform_
from torch.nn.utils import spectral_norm


class DiscriminatorBlock(nn.Module):

    def __init__(
        self, input_shape: Size, is_training: bool, depth: int = 3, num_filters: int =64,
        instance_noise: bool = True, label_noise: bool = True
        ):
        """Discriminator
        Original design by Gabriele Valvano.

        params:
        input_shape: torch.Size         Sample input size.
        is_training: bool               Model training state.
        num_filters: int                Number of base filter.
        instance_noise: bool            Enable instance noise.
        label_noise: bool               Enable label noise.

        Example:
        >>> x = torch.randn((1, 2, 304, 304))
        >>> attentions, predicate = segmentor(x)
        >>> discriminator = DiscriminatorBlock(x.shape, depth=3)
        >>> discriminator.forward(attentions)   # Or any list of tensor with shape of (1, 2, x.shape[3] // 2 ** i, x.shape[4] // 2 ** i)
        """
        super().__init__()
        self.num_filters = num_filters
        self.is_training = is_training
        self.depth = depth
        in_channels = input_shape[1]
        modules = []
        # Input noise
        if instance_noise:
            ins_0 = InstanceNoise(
                input_shape=input_shape,
                is_training=is_training, mean=.0, std=.2, clipping=True)
            modules.append(ins_0)
        conv_0 = nn.Conv2d(
            in_channels, num_filters, kernel_size=4, stride=2, padding=1)
        kaiming_normal_(conv_0.weight, nonlinearity='leaky_relu')
        modules.append(conv_0)
        modules.append(nn.LeakyReLU(negative_slope=0.2))
        self.stack_0 = nn.Sequential(*modules)

        # Discriminator blocks, top-down.
        squeeze_i = 'squeeze_{}'
        spectral_i = 'spectral_{}'
        squeeze_stack, spectral_stack = dict(), dict()
        for i in range(self.depth):
            squeeze, spectral = self._discriminator(
                in_channels=num_filters * (2 ** i), num_squeeze_filters=13, num_fake_channels=in_channels, num_sn_filters=num_filters * 2 * (2 ** i),
                sn_kernel_size=4,  num_sn_stride=2, sn_padding=1
            )
            squeeze_stack[squeeze_i.format(i)] = squeeze
            spectral_stack[spectral_i.format(i)] = spectral
        self.squeeze_dict = nn.ModuleDict(squeeze_stack)
        self.spectral_dict = nn.ModuleDict(spectral_stack)
        modules = []
        # Scalar Output
        h, w = [int(i)//(2 ** (self.depth + 1)) for i in input_shape[2:]]
        fc = nn.Conv2d(
            num_filters * (2 ** self.depth), out_channels=1, kernel_size=(h, w), stride=1
        )
        xavier_uniform_(fc.weight)
        modules.append(fc)
        flatten = nn.Flatten()
        modules.append(flatten)
        if label_noise:
            lbl = LabelNoise(0.1, 'sign')
            modules.append(lbl)
        self.out = nn.Sequential(*modules)

    def _discriminator(
            self,
            in_channels,
            num_squeeze_filters: int,
            num_fake_channels: int,
            num_sn_filters: int,
            sn_kernel_size: int,
            num_sn_stride: int,
            sn_padding: int):
        modules = []
        squeeze = nn.Conv2d(in_channels, num_squeeze_filters, kernel_size=1, stride=1)
        modules.append(squeeze)
        modules.append(nn.Sigmoid())
        squeezed = nn.Sequential(*modules)
        # Spectral Norm Conv
        modules = []
        conv = nn.Conv2d(
            num_squeeze_filters + num_fake_channels,
            num_sn_filters,
            kernel_size=sn_kernel_size,
            stride=num_sn_stride,
            padding=sn_padding
        )
        conv = spectral_norm(
            conv,
            n_power_iterations=1,
        )
        modules.append(conv)
        modules.append(nn.Tanh())
        spectral = nn.Sequential(*modules)

        return squeezed, spectral

    def forward(self, y: Sequence[Tensor]):
        """Forward method
        params: 
        y               List of multi-scale image tensor.
        """
        # Construct list of resized real
        s = self.stack_0(y[0])
        for i in range(self.depth):
            try:
                s = self.squeeze_dict[f'squeeze_{i}'](s)
                s = torch.cat((s, y[i+1]), dim=1)
                s = self.spectral_dict[f'spectral_{i}'](s)
            except Exception as e:
                raise Exception(f'Exception raised in depth = {i}') from e
        logits = self.out(s)

        return logits

    def predict(self, y: List[Tensor]):
        return self.forward(y)

class InstanceNoise(nn.Module):

    def __init__(
        self, input_shape: Size, mean: float, std: float, clipping: bool,
        is_training: bool):
        """Gaussian Noise Addition layer.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.clipping = clipping
        self.size = (input_shape[2], input_shape[3])
        self.is_training = is_training

    def forward(self, x: Tensor):
        noise = torch.normal(mean=self.mean, std=self.std, size=self.size).type_as(x)
        output = x + noise if self.is_training else x
        if self.clipping:
            output = torch.clip(output, 0, 1)
        return output

class LabelNoise(nn.Module):

    def __init__(self, prob: float = 0.1, mode: Literal['sign', 'label'] = 'sign'):
        """Label noise gradient reversal layer.
        """
        super().__init__()
        self.prob = prob
        self.mode = mode

    def flip_sign(self, x: Tensor):
        rand = rand_uniform(x)
        if rand < self.prob:
            return -1 * x
        else:
            return x

    def flip_label(self, x: Tensor):
        rand = rand_uniform(x)
        if rand < self.prob:
            return torch.abs(1. - x).type_as(x)
        else:
            return x

    def forward(self, x: Tensor):
       if self.mode == 'sign':
           return self.flip_sign(x)
       elif self.mode == 'label':
           return self.flip_label(x)
       else:
           raise NotImplementedError
