from typing import Literal, Tuple

import torch
from architectures.extra.resnest import ResNestDecoder, Upsampling, resnest50
from architectures.segmentor.blocks import AdversarialAttentionGate, GlobalAveragePooling2D
from einops import rearrange
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F


class ResnestUNet(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrain: bool,
        weight_path: str = None,
        gating_level: int = 4,
        encoder_gating: bool = False):
        """Resnest with self-attention composed into u-net architecture based from ROSE: OCTA paper.
        """
        super().__init__()
        resnest = resnest50(pretrained=pretrain, model_path=weight_path)
        self.gating_level = gating_level
        self.encoder_gating = encoder_gating

        if self.encoder_gating:
            construct_gating = lambda _in, _out, kernel_size: nn.Sequential(
                nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=kernel_size),
                nn.Softmax(dim=1)
            )
            self.encoder_0_gate = construct_gating(64, 16, 1)
            self.encoder_1_gate = construct_gating(256, 16, 1)
            self.encoder_2_gate = construct_gating(512, 16, 1)
            self.encoder_3_gate = construct_gating(1024, 16, 1)
            self.encoder_4_gate = construct_gating(2048, 16, 1)

        # Depth 0
        self.encoder_0_1_2 = nn.Sequential(
            resnest.conv1,
            resnest.bn1,
            resnest.relu
        )
        self.encoder_0_2_2 = resnest.maxpool

        self.upsampling_0 = Upsampling(64, 64)
        self.decoder_0 = ResNestDecoder(64, 32)
        self.aag_0 = AdversarialAttentionGate(32, num_classes)

        # Depth 1
        self.encoder_1 = resnest.layer1
        
        self.upsampling_1 = Upsampling(256, 64)
        self.decoder_1 = ResNestDecoder(128, 64)
        self.aag_1 = AdversarialAttentionGate(64, num_classes)
        
        # Depth 2
        self.encoder_2 = resnest.layer2
        self.aag_2 = AdversarialAttentionGate(256, num_classes)

        self.upsampling_2 = Upsampling(512, 256)
        self.decoder_2 = ResNestDecoder(512, 256)

        # Depth 3
        self.encoder_3 = resnest.layer3

        self.upsampling_3 = Upsampling(1024, 512)
        self.decoder_3 = ResNestDecoder(1024, 512)
        self.aag_3 = AdversarialAttentionGate(512, num_classes)
        
        # Depth 4
        self.encoder_4 = resnest.layer4

        self.upsampling_4 = Upsampling(2048, 1024)
        self.decoder_4 = ResNestDecoder(2048, 1024)
        self.aag_4 = AdversarialAttentionGate(1024, num_classes)

        self.fc = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1)

        # Classification Head - For embedding forward
        self.linear_head_emb = nn.Sequential(
            GlobalAveragePooling2D(),
            nn.Linear(2048, num_classes),
        )

        # Classification Head - For decoder out
        self.linear_head_dec = nn.Sequential(
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(in_channels=num_classes, out_channels=64, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            GlobalAveragePooling2D(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Top-Down
        x_0_0 = self.encoder_0_1_2(x)
        x_0_1 = self.encoder_0_2_2(x_0_0)
        g_x = list()
        if self.encoder_gating:
            g_x_0_1 = self.encoder_0_gate(x_0_1)
            g_x.append(g_x_0_1)

        x_1 = self.encoder_1(x_0_1)
        if self.encoder_gating:
            g_x_1 = self.encoder_1_gate(x_1)
            g_x.append(g_x_1)
        x_2 = self.encoder_2(x_1)
        if self.encoder_gating:
            g_x_2 = self.encoder_2_gate(x_2)
            g_x.append(g_x_2)
        x_3 = self.encoder_3(x_2)
        if self.encoder_gating:
            g_x_3 = self.encoder_3_gate(x_3)
            g_x.append(g_x_3)

        down_padding = False
        right_padding = False

        if x_3.size()[2] % 2 == 1:
            x_3 = F.pad(x_3, (0, 0, 0, 1))
            down_padding = True
        if x_3.size()[3] % 2 == 1:
            x_3 = F.pad(x_3, (0, 1, 0, 0))
            right_padding = True

        x_4 = self.encoder_4(x_3)
        if self.encoder_gating:
            g_x_4 = self.encoder_4_gate(x_4)
            g_x.append(g_x_4)

        attentions = []

        # Bottom-Up
        d_4 = self.upsampling_4(x_4)
        d_4 = torch.cat((x_3, d_4), dim=1)
        if down_padding and (not right_padding):
            d_4 = d_4[:, :, :-1, :]
        if (not down_padding) and right_padding:
            d_4 = d_4[:, :, :, :-1]
        if down_padding and right_padding:
            d_4 = d_4[:, :, :-1, :-1]

        d_4 = self.decoder_4(d_4)
        if self.gating_level >= 4:
            d_4, y_4 = self.aag_4(d_4)
            attentions.append(y_4)

        d_3 = self.upsampling_3(d_4)
        d_3 = torch.cat((x_2, d_3), dim=1)
        d_3 = self.decoder_3(d_3)
        if self.gating_level >= 3:
            d_3, y_3 = self.aag_3(d_3)
            attentions.append(y_3)

        d_2 = self.upsampling_2(d_3)
        d_2 = torch.cat((x_1, d_2), dim=1)
        d_2 = self.decoder_2(d_2)
        if self.gating_level >= 2:
            d_2, y_2 = self.aag_2(d_2)
            attentions.append(y_2)

        d_1 = self.upsampling_1(d_2)
        d_1 = torch.cat((x_0_0, d_1), dim=1)
        d_1 = self.decoder_1(d_1)
        if self.gating_level >= 1:
            d_1, y_1 = self.aag_1(d_1)
            attentions.append(y_1)

        d_0 = self.upsampling_0(d_1)
        d_0 = self.decoder_0(d_0)
        if self.gating_level >= 0:
            d_0, y_0 = self.aag_0(d_0)
            attentions.append(y_0)

        agg_map = self.fc.forward(d_0)

        attentions.reverse()

        if self.encoder_gating:
            return tuple(g_x), tuple(attentions), agg_map, x_4
        return tuple(attentions), agg_map, x_4

    def predict(self, x: Tensor, method: Literal['softmax', 'one-hot', 'original', 'sigmoid'] = 'softmax'):
        attentions, agg_map, _ = self.forward(x)
        if method == 'softmax':
            predicate = nn.Softmax(dim=1)(agg_map)
        elif method == 'sigmoid':
            predicate = nn.Sigmoid()(agg_map)
        elif method == 'one-hot':
            predicate = rearrange(F.one_hot(torch.argmax(agg_map, dim=1)), 'b h w c -> b c h w')
        elif method == 'original':
            predicate = agg_map
        return attentions, predicate

    def classification_predict(self, x: Tensor, method: Literal['softmax', 'sigmoid'], mode: Literal['classic', 'classic-gating', 'ae-squash', 'ae-extract']):
        if mode == 'classic-gating' and self.encoder_gating:
            g_x, att, predicate, latent = self.forward(x)
        elif mode == 'classic-gating' and not self.encoder_gating:
            raise ValueError(f'{mode} is not valid if `encoder_gating` is not enabled.')
        else:
            att, predicate, latent = self.forward(x)

        predicate = nn.Softmax(dim=1)(predicate)

        if mode in ('classic', 'classic-gating'):
            emb = self.linear_head_emb(latent)
        elif mode == 'ae-squash':
            emb = GlobalAveragePooling2D()(predicate)
        elif mode == 'ae-extract':
            emb = self.linear_head_dec(predicate)
        else:
            raise NotImplementedError

        if method == 'softmax':
            class_pred = nn.Softmax(dim=1)(emb)
        elif method == 'sigmoid':
            class_pred = nn.Sigmoid()(emb)
        else:
            raise NotImplementedError

        if mode == 'classic-gating' and self.encoder_gating:
            return class_pred, g_x, att, predicate
        else:
            return class_pred, att, predicate


class ResnestUnetParallelHead(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrain: bool,
        weight_path: str = None):
        super().__init__()
        resnest = resnest50(pretrained=pretrain, model_path=weight_path)

        # Depth 0
        self.encoder_0_1_2 = nn.Sequential(
            resnest.conv1,
            resnest.bn1,
            resnest.relu
        )
        self.encoder_0_2_2 = resnest.maxpool

        self.upsampling_0 = Upsampling(64, 64)
        self.decoder_0 = ResNestDecoder(64, 32)

        # Depth 1
        self.encoder_1 = resnest.layer1
        
        self.upsampling_1 = Upsampling(256, 64)
        self.decoder_1 = ResNestDecoder(128, 64)

        # Depth 2
        self.encoder_2 = resnest.layer2


        self.upsampling_2 = Upsampling(512, 256)
        self.decoder_2 = ResNestDecoder(512, 256)

        # Depth 3
        self.encoder_3 = resnest.layer3

        self.upsampling_3 = Upsampling(1024, 512)
        self.decoder_3 = ResNestDecoder(1024, 512)

        # Depth 4
        self.encoder_4 = resnest.layer4

        self.upsampling_4 = Upsampling(2048, 1024)
        self.decoder_4 = ResNestDecoder(2048, 1024)

        # Depth 1 - Parallel Branch
        self.upsampling_1_c = Upsampling(256, 64)
        self.decoder_1_c = ResNestDecoder(128, 64)

        # Depth 0 - Parallel Branch
        self.upsampling_0_c = Upsampling(64, 64)
        self.decoder_0_c = ResNestDecoder(64, 32)

        self.fc = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1)
        self.fc_c = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1)


    def forward(self, x) -> Tensor:
        # Top-Down
        x_0_0 = self.encoder_0_1_2(x)
        x_0_1 = self.encoder_0_2_2(x_0_0)

        x_1 = self.encoder_1(x_0_1)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)

        down_padding = False
        right_padding = False

        if x_3.size()[2] % 2 == 1:
            x_3 = F.pad(x_3, (0, 0, 0, 1))
            down_padding = True
        if x_3.size()[3] % 2 == 1:
            x_3 = F.pad(x_3, (0, 1, 0, 0))
            right_padding = True

        x_4 = self.encoder_4(x_3)

        # Bottom-Up
        d_4 = self.upsampling_4(x_4)
        d_4 = torch.cat((x_3, d_4), dim=1)
        if down_padding and (not right_padding):
            d_4 = d_4[:, :, :-1, :]
        if (not down_padding) and right_padding:
            d_4 = d_4[:, :, :, :-1]
        if down_padding and right_padding:
            d_4 = d_4[:, :, :-1, :-1]

        d_4 = self.decoder_4(d_4)

        d_3 = self.upsampling_3(d_4)
        d_3 = torch.cat((x_2, d_3), dim=1)
        d_3 = self.decoder_3(d_3)

        d_2 = self.upsampling_2(d_3)
        d_2 = torch.cat((x_1, d_2), dim=1)
        d_2 = self.decoder_2(d_2)

        d_1 = self.upsampling_1(d_2)
        d_1 = torch.cat((x_0_0, d_1), dim=1)
        d_1 = self.decoder_1(d_1)

        d_0 = self.upsampling_0(d_1)
        d_0 = self.decoder_0(d_0)

        d_1_c = self.upsampling_1_c(x_1)
        d_1_c = torch.cat((x_0_0, d_1_c), dim=1)
        d_1_c = self.decoder_1_c(d_1_c)

        d_0_c = self.upsampling_0_c(d_1_c)
        d_0_c = self.decoder_0_c(d_0_c)


        agg_map = self.fc.forward(d_0)
        agg_map_c = self.fc_c.forward(d_0_c)

        return rearrange([agg_map, agg_map_c], 'k b c h w -> k b c h w')

    def predict(self, x: Tensor, method: Literal['softmax', 'sigmoid', 'one-hot', 'original'] = 'softmax'):
        agg_map = self.forward(x)
        if method == 'softmax':
            predicate = nn.Softmax(dim=2)(agg_map)
        elif method == 'sigmoid':
            predicate = nn.Sigmoid()(agg_map)
        elif method == 'one-hot':
            predicate = rearrange(F.one_hot(torch.argmax(agg_map, dim=2)), 'k b h w c -> k b c h w')
        elif method == 'original':
            predicate = agg_map
        return predicate


class ResnestUnetParallelHeadAttentionGate(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrain: bool,
        weight_path: str = None,
        gating_leveL: int = 3):
        super().__init__()
        self.gating_level = gating_leveL
        resnest = resnest50(pretrained=pretrain, model_path=weight_path)

        # Depth 0
        self.encoder_0_1_2 = nn.Sequential(
            resnest.conv1,
            resnest.bn1,
            resnest.relu
        )
        self.encoder_0_2_2 = resnest.maxpool

        self.upsampling_0 = Upsampling(64, 64)
        self.decoder_0 = ResNestDecoder(64, 32)
        self.aag_0 = AdversarialAttentionGate(32, num_classes)

        # Depth 1
        self.encoder_1 = resnest.layer1
        
        self.upsampling_1 = Upsampling(256, 64)
        self.decoder_1 = ResNestDecoder(128, 64)
        self.aag_1 = AdversarialAttentionGate(64, num_classes)

        # Depth 2
        self.encoder_2 = resnest.layer2


        self.upsampling_2 = Upsampling(512, 256)
        self.decoder_2 = ResNestDecoder(512, 256)
        self.aag_2 = AdversarialAttentionGate(256, num_classes)

        # Depth 3
        self.encoder_3 = resnest.layer3

        self.upsampling_3 = Upsampling(1024, 512)
        self.decoder_3 = ResNestDecoder(1024, 512)
        self.aag_3 = AdversarialAttentionGate(512, num_classes)

        # Depth 4
        self.encoder_4 = resnest.layer4

        self.upsampling_4 = Upsampling(2048, 1024)
        self.decoder_4 = ResNestDecoder(2048, 1024)
        self.aag_4 = AdversarialAttentionGate(1024, num_classes)

        # Depth 1 - Parallel Branch
        self.upsampling_1_c = Upsampling(256, 64)
        self.decoder_1_c = ResNestDecoder(128, 64)
        self.aag_1_c = AdversarialAttentionGate(64, num_classes)

        # Depth 0 - Parallel Branch
        self.upsampling_0_c = Upsampling(64, 64)
        self.decoder_0_c = ResNestDecoder(64, 32)
        self.aag_0_c = AdversarialAttentionGate(32, num_classes)

        self.fc = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1)
        self.fc_c = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1)


    def forward(self, x) -> Tensor:
        # Top-Down
        x_0_0 = self.encoder_0_1_2(x)
        x_0_1 = self.encoder_0_2_2(x_0_0)

        x_1 = self.encoder_1(x_0_1)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)

        down_padding = False
        right_padding = False

        if x_3.size()[2] % 2 == 1:
            x_3 = F.pad(x_3, (0, 0, 0, 1))
            down_padding = True
        if x_3.size()[3] % 2 == 1:
            x_3 = F.pad(x_3, (0, 1, 0, 0))
            right_padding = True

        x_4 = self.encoder_4(x_3)

        attentions = list()
        attentions_c = list()
        # Bottom-Up
        d_4 = self.upsampling_4(x_4)
        d_4 = torch.cat((x_3, d_4), dim=1)
        if down_padding and (not right_padding):
            d_4 = d_4[:, :, :-1, :]
        if (not down_padding) and right_padding:
            d_4 = d_4[:, :, :, :-1]
        if down_padding and right_padding:
            d_4 = d_4[:, :, :-1, :-1]

        d_4 = self.decoder_4(d_4)
        if self.gating_level > 3:
            d_4, y_4 = self.aag_4(d_4)
            attentions.append(y_4)

        d_3 = self.upsampling_3(d_4)
        d_3 = torch.cat((x_2, d_3), dim=1)
        d_3 = self.decoder_3(d_3)
        if self.gating_level >= 3:
            d_3, y_3 = self.aag_3(d_3)
            attentions.append(y_3)

        d_2 = self.upsampling_2(d_3)
        d_2 = torch.cat((x_1, d_2), dim=1)
        d_2 = self.decoder_2(d_2)
        if self.gating_level >= 2:
            d_2, y_2 = self.aag_2(d_2)
            attentions.append(y_2)

        d_1 = self.upsampling_1(d_2)
        d_1 = torch.cat((x_0_0, d_1), dim=1)
        d_1 = self.decoder_1(d_1)
        if self.gating_level >= 1:
            d_1, y_1 = self.aag_1(d_1)
            attentions.append(y_1)

        d_0 = self.upsampling_0(d_1)
        d_0 = self.decoder_0(d_0)
        if self.gating_level >= 0:
            d_0, y_0 = self.aag_0(d_0)
            attentions.append(y_0)

        d_1_c = self.upsampling_1_c(x_1)
        d_1_c = torch.cat((x_0_0, d_1_c), dim=1)
        d_1_c = self.decoder_1_c(d_1_c)
        if self.gating_level >= 1:
            d_1_c, y_1_c = self.aag_1_c(d_1_c)
            attentions_c.append(y_1_c)

        d_0_c = self.upsampling_0_c(d_1_c)
        d_0_c = self.decoder_0_c(d_0_c)
        if self.gating_level >= 0:
            d_0_c, y_0_c = self.aag_0_c(d_0_c)
            attentions_c.append(y_0_c)

        attentions.reverse()
        attentions_c.reverse()
        agg_map = self.fc.forward(d_0)
        agg_map_c = self.fc_c.forward(d_0_c)

        return (tuple(attentions), tuple(attentions_c)), rearrange([agg_map, agg_map_c], 'k b c h w -> k b c h w')

    def predict(self, x: Tensor, method: Literal['softmax', 'sigmoid', 'one-hot', 'original'] = 'softmax'):
        attentions, agg_map = self.forward(x)
        if method == 'softmax':
            predicate = nn.Softmax(dim=2)(agg_map)
        elif method == 'sigmoid':
            predicate = nn.Sigmoid()(agg_map)
        elif method == 'one-hot':
            predicate = rearrange(F.one_hot(torch.argmax(agg_map, dim=2)), 'k b h w c -> k b c h w')
        elif method == 'original':
            predicate = agg_map
        return attentions, predicate