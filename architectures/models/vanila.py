"""Vanila Architectures
"""
from logging import warn
from typing import Any, Dict, Optional

from architectures.discriminator.blocks import DiscriminatorBlock
from architectures.discriminator.losses import (LSDiscriminatorialLoss,
                                                LSGeneratorLoss)
from architectures.segmentor.blocks import VanilaSegmentor
from architectures.segmentor.losses import WeightedPartialCE
from torch import nn
from torch._C import Size
from torch.functional import Tensor
from torchvision.transforms import functional as FT


class VanilaScribbleNet(nn.Module):

    def __init__(
        self,
        raw_input_shape: Size,
        mask_input_shape: Size,
        is_training: bool, num_classes: int = 1, num_filters: int = 32, instance_noise: bool = True, label_noise: bool = True):
        """Vanila ScribbleNet architecture designed by Valvano et al.
        params:
        raw_input_shape: torch.Size         Shape of sample raw input.
        mask_input_shape: torch.Size        Shape of sample mask target with channels responding to the number of classes.
        is_training: bool                   Model training state.
        num_classes: int                    Number of classes.
        instance_noise: bool                Enable instance noise in training.
        label_noise: bool                   Enable label noise in training
        """
        super().__init__()
        if mask_input_shape[1] != num_classes:
            warn('Number channels in mask input is not same as number of classes. Can cause an error when model discriminator is in use.')
        self.segmentor = VanilaSegmentor(input_shape=raw_input_shape, num_classes=num_classes, num_filters=num_filters, enable_batchnorm=True)
        self.discriminator = DiscriminatorBlock(
            input_shape=mask_input_shape, is_training=is_training, num_filters=num_filters, instance_noise=instance_noise, label_noise=label_noise
        )
        # Loss Definition
        self.supervised_loss = WeightedPartialCE(num_classes=num_classes, manual=True)
        self.discriminatorial_loss = LSDiscriminatorialLoss()
        self.generator_loss = LSGeneratorLoss() 
        self.is_train = is_training

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, Any]:
        raise NotImplementedError