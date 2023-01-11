from architectures.segmentor.blocks import VanilaSegmentor
import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
from utils.evaluation import *
from utils.logging import attention_predicate_plot, to_wandb_semantic_segmentation


class SegmentorExperiment(pl.LightningModule):
    def __init__(self, segmentor: nn.Module, supervise_loss: nn.Module, wandb_logging: bool = False):
        """Segmentor Experimentation Frame
        params:
        segmentor: nn.Module                Segmentor used to test.
        supervise_loss: nn.Module           Loss function module used to test with segmentor.

        Example:
        >>> experiment = SegmentorExperiment(VanilaSegmentor(...), WeightedPartialCE(2, manual=True))
        >>> trainer.fit(
            experiment,
            train_loader,
            val_loader
        )
        """
        super().__init__()
        self.seg = segmentor
        self.seg_loss = supervise_loss
        self.wandb_logging = wandb_logging
        # Auto save all parameters.
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x = batch['projection']
        y = batch['ground truth']
        attentions, agg_map = self.seg.predict(x, method='softmax')
        loss = self.seg_loss.forward(agg_map, y.float())
        self.log('train/seg_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['projection']
        y = batch['ground truth']
        attentions, agg_map = self.seg.predict(x, method='softmax')
        loss = self.seg_loss.forward(agg_map, y.float())
        self.log('val/acc', accuracy(agg_map, y))
        self.log('val/dice', dice(agg_map, y))
        self.log('val/iou', iou(agg_map, y))
        self.log('val/auc', auc(agg_map, y))
        self.log('val/seg_loss', loss)
        return {'x': x, 'y': y, 'attentions': attentions, 'predicate': agg_map}

    def validation_epoch_end(self, outputs):
        if self.wandb_logging:
            masks = []
            samples = random.choices(outputs, k=5)
            for output in samples:
                x, y, attentions, predicate = output['x'], output['y'], output['attentions'], output['predicate']
                fig = attention_predicate_plot(x, attentions, predicate, y)
                self.logger.experiment.log({'rollout': fig})
                plt.close()
                mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel'}, ground_truth=y)
                masks.append(mask_img)
            self.logger.experiment.log({'predictions': masks})


class SegmentorExperimentOCTA(pl.LightningModule):
    def __init__(
        self, segmentor: VanilaSegmentor, supervise_loss: nn.Module, evaluator: Evaluator, wandb_logging: bool = False,
        simplistic_evaluation: bool = False, logging_frequency: int = 1):
        """Segmentor Experimentation Frame for OCTA-500 Dataloader
        params:
        segmentor: nn.Module                Segmentor used to test.
        supervise_loss: nn.Module           Loss function module used to test with segmentor.

        Example:
        >>> experiment = SegmentorExperimentOCTA500(VanilaSegmentor(...), WeightedPartialCE(2, manual=True))
        >>> trainer.fit(
            experiment,
            train_loader,
            val_loader
        )
        """
        super().__init__()
        self.seg = segmentor
        self.seg_loss = supervise_loss
        self.wandb_logging = wandb_logging
        self.simplistic_evaluation = simplistic_evaluation
        self.evaluator = evaluator
        self.frequency = logging_frequency
        # Auto save all parameters.
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y_weak']
        attentions, agg_map = self.seg.predict(x, method='softmax')
        loss = self.seg_loss.forward(agg_map, y.float())
        self.log('train/seg_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y_target = batch['y']
        attentions, predicate = self.seg.predict(x, method='softmax')
        loss = self.seg_loss.forward(predicate, y_target.float())
        self.log('val/seg_loss', loss)

        if self.simplistic_evaluation or self.evaluator is None:
            accur = accuracy(predicate, y_target)
            auc_scr = auc(predicate, y_target)
            kappa = cohen_kappa(predicate, y_target)
            iou_scr = iou(predicate, y_target)
            dice_coef = dice(predicate, y_target)
            sen, spe, fpr = sensitivity_specificity_fpr(predicate, y_target)
        else:
            self.evaluator.register(predicate, y_target)
            accur = self.evaluator.accuracy()
            auc_scr = self.evaluator.auc()
            kappa = self.evaluator.cohen_kappa()
            iou_scr = self.evaluator.iou()
            dice_coef = self.evaluator.dice()
            sen, spe, fpr = self.evaluator.sensitivity_specificity_fpr()

        if not self.evaluator is None:
            for c in range(self.evaluator.num_classes + (1 if self.evaluator.collapsible else 0)):
                self.log(f'val/acc_{c}', accur[c])
                self.log(f'val/auroc_{c}', auc_scr[c])
                self.log(f'val/dice_{c}', dice_coef[c])
                self.log(f'val/iou_{c}', iou_scr[c])
                self.log(f'val/kappa_{c}', kappa[c])
                self.log(f'val/specificity_{c}', spe[c])
                self.log(f'val/sensitivity_{c}', sen[c])
                self.log(f'val/fpr_{c}', fpr[c])

        return {'x': x, 'y': y_target, 'attentions': attentions, 'predicate': predicate}

    def validation_epoch_end(self, outputs) -> None:
        if self.current_epoch % self.frequency == 0:
            if self.wandb_logging:
                masks = []
                samples = random.choices(outputs, k=5)
                for output in samples:
                    x, y, attentions, predicate = output['x'], output['y'], output['attentions'], output['predicate']
                    if self.seg.enable_att:
                        fig = attention_predicate_plot(x, attentions, predicate, y, multi_class=False)
                        self.logger.experiment.log({'rollout': fig})
                        plt.close()
                    mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel'}, ground_truth=y)
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})