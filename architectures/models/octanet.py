from logging import warn
from typing import Any, Dict, Optional

from architectures.discriminator.blocks import DiscriminatorBlock
from architectures.discriminator.losses import (LSDiscriminatorialLoss,
                                                LSGeneratorLoss)
from architectures.segmentor.compose import ResnestUNet
from architectures.segmentor.losses import WeightedPartialCE, DiceLoss
from torch import nn
from torch._C import Size
from torch.functional import Tensor
from torchvision.transforms import functional as FT


class OctaScribbleNet(nn.Module):

    def __init__(
        self,
        raw_input_shape: Size,
        mask_input_shape: Size,
        is_training: bool,
        pretrian: bool,
        weight_path: str = 'resnest50-528c19ca.pth',
        num_classes: int = 2, num_filters: int = 64,
        instance_noise: bool = True, label_noise: bool = True,
        segmentor_gating_level: int = 4,
        discriminator_depth: int = 4,
        encoder_gating: bool = False,
        weakly_supervise: bool = True,
        ):
        """Vanila ScribbleNet architecture designed by Valvano et al.
        params:
        raw_input_shape: torch.Size         Shape of sample raw input.
        mask_input_shape: torch.Size        Shape of sample mask target with channels responding to the number of classes.
        is_training: bool                   Model training state.
        num_classes: int                    Number of classes.
        instance_noise: bool                Enable instance noise in training.
        label_noise: bool                   Enable label noise in training.
        segmentor_gating_level              Attention gating level. Default is 4 (All layer). Negative value will disable all gating.
        discriminator_depth: int            Discrimator depth. Default is 4 (Equal to ResnestUnet depth).
        """
        super().__init__()
        if mask_input_shape[1] != num_classes:
            warn('Number channels in mask input is not same as number of classes. Can cause an error when model discriminator is in use.')
        self.segmentor = ResnestUNet(num_classes=num_classes, pretrain=pretrian, weight_path=weight_path, gating_level=segmentor_gating_level, encoder_gating=encoder_gating)
        # Attention depth from resnest u-net is 4.
        if discriminator_depth > 0:
            self.discriminator = DiscriminatorBlock(
                input_shape=mask_input_shape, is_training=is_training, depth=discriminator_depth, num_filters=num_filters, instance_noise=instance_noise, label_noise=label_noise
            )
        # Loss Definition
        if weakly_supervise:
            self.supervised_loss = WeightedPartialCE(num_classes=num_classes, manual=True)
        else:
            self.supervised_loss = DiceLoss()
        self.discriminatorial_loss = LSDiscriminatorialLoss()
        self.generator_loss = LSGeneratorLoss()
        self.is_train = is_training

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Dict[str, Any]:
        raise NotImplementedError
