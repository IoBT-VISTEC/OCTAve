from random import random
from typing import Any, Literal

from einops import rearrange
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.functional import Tensor
from torch_poly_lr_decay import PolynomialLRDecay
from architectures.segmentor.blocks import VanilaSegmentor
from architectures.segmentor.losses import WeightedPartialCE
from utils.logging import attention_predicate_plot, to_wandb_semantic_segmentation


class VanillaUnet(pl.LightningModule):

    def __init__(self,
     input_shape: torch.Size, 
     num_classes: int = 2,
     num_filters: int = 32,
     disable_weight_bg: bool = False,
     wpce_reduction: Literal['mean', 'sum'] = 'mean',
     wandb_logging: bool = True,
     evaluator = None,
     logging_frequency: int = 5):
        super().__init__()
        self.model = VanilaSegmentor(
            input_shape=input_shape,
            enable_attention_gates=False, # U-net, no ATTGate
            enable_batchnorm=True,
            num_classes=num_classes,
            num_filters=num_filters,
        )
        self.loss = WeightedPartialCE(num_classes=num_classes, manual=True)
        self.disable_weight_bg = disable_weight_bg
        self.wpce_reduction = wpce_reduction
        self.wandb_logging = wandb_logging
        self.evaluator = evaluator
        self.frequency = logging_frequency

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        return [optim]

    def forward(self, x: Tensor) -> Tensor:
        _, y_hat = self.model.predict(x, method='softmax')
        return y_hat

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        x = batch[self.pair_switch('x', 'projection')]
        y_segmentor = batch[self.pair_switch('y_weak', 'ground truth')]
        y = batch[self.pair_switch('y_unpair', 'ground truth')]
        gt_exclusion_mask = batch.get('ignore_y', None)

        # Segmentor forward pass
        predicate = self.forward(x)
        loss = self.loss.forward(y_hat=predicate, ys=y.float(), ignore_bg=self.disable_weight_bg, reduction=self.wpce_reduction)

        if self.wandb_logging:
            self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        y = batch['y_weak']
        y_hat = self.forward(x)
        loss = self.loss.forward(y_hat=y_hat[0], ys=y.float(), ignore_bg=self.disable_weight_bg, reduction=self.wpce_reduction)

        if not self.evaluator is None:
            self.evaluator.register(y_hat, y)
            accur = self.evaluator.accuracy()
            auc_scr = self.evaluator.auc()
            kappa = self.evaluator.cohen_kappa()
            iou_scr = self.evaluator.iou()
            dice_coef = self.evaluator.dice()
            sen, spe, fpr = self.evaluator.sensitivity_specificity_fpr()
            # Reshaping y for img plot
            re_y_hat = rearrange(torch.nn.functional.one_hot(torch.from_numpy(self.evaluator.thresholded_predicate[0]).long(), num_classes=2), 'h w code -> code h w')
            # Logging
            if self.wandb_logging:
                self.log('val/loss', loss)
                self.log('val/acc', accur)
                self.log('val/auroc', auc_scr)
                self.log('val/dice', dice_coef)
                self.log('val/iou', iou_scr)
                self.log('val/kappa', kappa)
                self.log('val/specificity', spe)
                self.log('val/sensitivity', sen)
                self.log('val/fpr', fpr)
        else:
            assert False, 'Evaluation without Evaluator is not supported.'
        return {'loss': loss, 'x': x, 'y': y, 'predicate': re_y_hat}

    def validation_epoch_end(self, outputs):
        if self.current_epoch % self.frequency == 0:
            if self.wandb_logging:
                masks = []
                samples = random.choices(outputs, k=5)
                for output in samples:
                    x, y, predicate = output['x'], output['y'], output['predicate']
                    mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel'}, ground_truth=y)
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})

