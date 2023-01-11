import logging
import random
from typing import Any, Literal, Optional, Union
from einops.einops import rearrange

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.autograd.grad_mode import enable_grad
from architectures.models.octanet import OctaScribbleNet as OCTANet
from architectures.segmentor.losses import DiceLoss, InterlayerDivergence, WeightedPartialCE
from interfaces.experiments.vanila import RoseOcta500
from interfaces.rose import SVC
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FT
from utils.evaluation import (Evaluator, accuracy, auc, cohen_kappa, dice, iou,
                              sensitivity_specificity_fpr)
from utils.logging import (attention_predicate_plot,
                           to_wandb_semantic_segmentation)


class OCTAScribbleNet(pl.LightningModule):

    def shared_dataloader(self, is_train: bool, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> Any:
        """Shared step dataloader
        Use centerline mode for training as scribble, pixel for performance evaluation
        """
        if self.pairing_option == 'unpaired':
            dataset = RoseOcta500(
                is_train=is_train,
                rose_datapath='data',
                rose_modality='thin_gt' if is_train and self.segmentor_learning_mode == 'weakly' else 'gt',
                octa500_datapath='data/OCTA-500',
                octa500_modality='3m',
                octa500_level='ILM_OPL',
                ground_truth_style='one-hot',
                rose_input_augmentation=self.enable_input_augmentation
            )
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        elif self.pairing_option == 'paired':
            # Use its own image as pair image
            dataset = SVC(
                datapath='data',
                is_train=is_train,
                ground_truth_mode='thin_gt' if is_train and self.segmentor_learning_mode == 'weakly' else 'gt',
                enable_augment=self.enable_input_augmentation,
                separate_vessel_capilary=False,
                ground_truth_style='one-hot'
            )
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def train_dataloader(self, **kwargs) -> Any:
        return self.shared_dataloader(True, **kwargs)

    def val_dataloader(self, **kwargs):
        return self.shared_dataloader(False, batch_size=1, shuffle=False, **kwargs)

    def __init__(
        self,
        is_training: bool, pretrain: bool, supervise_regulation: bool = True,
        base_filters: int = 64, wandb_logging: bool = False, weight_path: Optional[str] = None, enable_input_augmentation:bool = False, logging_frequency: int = 1,
        segmentor_learning_mode: Literal['fully', 'weakly', 'none'] = 'weakly',
        pairing_option: Literal['unpaired', 'paired'] = 'unpaired',
        supervise_loss: Any = None,
        simplistic_evaluation: bool = True,
        evaluator: Optional[Evaluator] = None):
        """Vanilla OCTA-net based ScribbleNet Adaptation.
        """
        super().__init__()
        # Disable Automatic Optimization
        self.automatic_optimization = False
        self.segmentor_learning_mode = segmentor_learning_mode
        self.pairing_option = pairing_option
        self.is_training = is_training
        # Basis network
        self.scribble_net = OCTANet(
            raw_input_shape=torch.zeros((1, 3, 304, 304)).shape,
            mask_input_shape=torch.zeros((1, 2, 304, 304)).shape,
            is_training=is_training,
            instance_noise=True,
            label_noise=True,
            num_filters=base_filters,
            num_classes=2,
            pretrian=pretrain,
            weight_path=weight_path,
        )
        self.lr1 = 0.1 if self.segmentor_learning_mode in ['weakly', 'fully'] else 0.2
        self.lr2 = 0.2
        self.lr3 = 0.2

        self.supervise_loss = supervise_loss
        self.wandb_logging = wandb_logging
        self.supervise_regulation = supervise_regulation
        self.frequency = logging_frequency
        self.enable_input_augmentation = enable_input_augmentation
        self.simplistic_evaluation = simplistic_evaluation
        self.evaluator = evaluator

        # Utilities
        self.pair_switch = lambda x, y: x if self.pairing_option == 'unpaired' else y

        self.save_hyperparameters()

    def configure_optimizers(self):
        opt_seg = torch.optim.Adam(self.scribble_net.segmentor.parameters(), lr=1e-5)
        opt_dis = torch.optim.Adam(self.scribble_net.discriminator.parameters(), lr=1e-5)
        lr_seg = torch.optim.lr_scheduler.CyclicLR(opt_seg, base_lr=1e-5, max_lr=1e-4, cycle_momentum=False)
        lr_dis = torch.optim.lr_scheduler.CyclicLR(opt_dis, base_lr=1e-5, max_lr=1e-4, cycle_momentum=False)
        return (
            {
                'optimizer': opt_seg,
                'lr_scheduler': {
                    'scheduler': lr_seg
                }
            },
            {
                'optimizer': opt_dis,
                'lr_scheduler': {
                    'scheduler': lr_dis
                }
            }
        )

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        opt_seg, opt_dis = self.optimizers()
        lr_seg, lr_dis = self.lr_schedulers()
        x = batch[self.pair_switch('x', 'projection')]
        y_segmentor = batch[self.pair_switch('y_weak', 'ground truth')]
        y = batch[self.pair_switch('y', 'ground truth')]

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Detach
        detached_attentions = [a.detach() for a in attentions]

        # Segmentor optimization
        fake_predicate = self.scribble_net.discriminator.forward(attentions)
        if self.segmentor_learning_mode in ['weakly', 'fully']:
            if self.supervise_loss is None:
                Lsup = self.scribble_net.supervised_loss.forward(y_hat=predicate, ys=y_segmentor)
            else:
                Lsup = self.supervise_loss(predicate, y_segmentor.float())
            Lgen = self.scribble_net.generator_loss.forward(fake_predicate)
            lr0 = Lgen.detach().norm() / Lsup.detach().norm() if self.supervise_regulation else 1.
            Lseg = (lr0 * Lsup) + (self.lr1 * Lgen)
            opt_seg.zero_grad()
            self.manual_backward(Lseg)
            opt_seg.step()
            lr_seg.step()
        
            # Logging
            self.log('train/gen_sup_loss', Lseg)
            self.log('train/sup_loss', Lsup)
            self.log('train/gen_loss', Lgen)
            if self.supervise_regulation:
                self.log('train/alpha_0', lr0)
        elif self.segmentor_learning_mode == 'none':
            Lgen = self.lr1 * self.scribble_net.generator_loss.forward(fake_predicate)
            opt_seg.zero_grad()
            self.manual_backward(Lgen)
            opt_seg.step()
            lr_seg.step()

            # Logging
            self.log('train/gen_loss', Lgen)

        # Discriminator optimization
        ylist = [y.type_as(x)]
        for i in range(1, len(attentions)):
            w, h = attentions[i].shape[2], attentions[i].shape[3]
            ylist.append(FT.resize(y, [w, h]).type_as(x))
        real_predicate = self.scribble_net.discriminator.forward(ylist)
        fake_predicate = self.scribble_net.discriminator.forward(detached_attentions)
        Ldis = self.lr2 * self.scribble_net.discriminatorial_loss.forward(real_predicate, fake_predicate)
        opt_dis.zero_grad()
        self.manual_backward(Ldis)
        opt_dis.step()
        lr_dis.step()

        # Logging
        self.log('train/dis_loss', Ldis)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y_target = batch[self.pair_switch('x', 'projection')], batch[self.pair_switch('y_target', 'ground truth')]

        # Segmentation
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        fake = self.scribble_net.discriminator.forward(attentions)
        # Performance evaluation
        # Loss calculation
        if self.supervise_loss is None:
            gen_loss = self.scribble_net.supervised_loss.forward(predicate, y_target)
        else:
            gen_loss = self.supervise_loss.forward(predicate, y_target.float())
        dis_loss = self.scribble_net.generator_loss.forward(fake)

        # Supervise loss
        self.log('val/sup_loss', gen_loss)
        # Generator loss
        self.log('val/gen_loss', dis_loss)

        if self.simplistic_evaluation:
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

        for c in range(self.evaluator.num_classes + 1 if self.evaluator.collapsible else 0):
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
        if self.current_epoch % self.frequency:
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

    def forward(self, x, *args, **kwargs) -> Any:
        return self.scribble_net.forward(x)


class OCTAScribbleNetVariantA(OCTAScribbleNet):

    def shared_dataloader(self, is_train: bool, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> Any:
        dataset = RoseOcta500(
            is_train=is_train,
            rose_datapath='data',
            rose_modality='thin_gt' if is_train else 'thick_gt',  # Evaluate vessel on pixel-level mask.
            octa500_datapath='data/OCTA-500',
            octa500_modality='3m',
            octa500_level='ILM_OPL',
            ground_truth_style='one-hot',
            separate_vessel_capillary=True,
            rose_input_augmentation=self.enable_input_augmentation
        )
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __init__(
        self, is_training: bool, pretrain: bool, supervise_regulation: bool,
        base_filters: int, wandb_logging: bool, weight_path: Optional[str] = None,
        enable_input_augmentation: bool = False, logging_frequency: int = 1,
        simplistic_evaluation: bool = False, evaluator: Optional[Evaluator] = None):
        """Multi-class, Singular Objective function.
        """
        super().__init__(
            is_training=is_training,
            pretrain=pretrain,
            supervise_regulation=supervise_regulation,
            base_filters=base_filters,
            wandb_logging=wandb_logging,
            weight_path=weight_path,
            logging_frequency=logging_frequency,
            enable_input_augmentation=enable_input_augmentation,
            simplistic_evaluation=simplistic_evaluation,
            evaluator=evaluator
        )
        self.automatic_optimization = False

        self.is_training = is_training
        # Basis network
        self.scribble_net = OCTANet(
            raw_input_shape=torch.zeros((1, 3, 304, 304)).shape,
            mask_input_shape=torch.zeros((1, 2, 304, 304)).shape, # Drop
            is_training=is_training,
            instance_noise=True,
            label_noise=True,
            num_filters=base_filters,
            num_classes=3,
            pretrian=pretrain,
            weight_path=weight_path,
        )
        self.lr1 = 0.1
        self.lr2 = 0.2
        self.lr3 = 0.2
        self.wandb_logging = wandb_logging

        assert self.segmentor_learning_mode == 'weakly', 'Variation does not support other than weakly mode.'
        assert self.pairing_option == 'unpaired', 'Variation only support paired option.'

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        opt_seg, opt_dis = self.optimizers()
        lr_seg, lr_dis = self.lr_schedulers()
        x = batch[self.pair_switch('x', 'projection')]
        y_segmentor = batch[self.pair_switch('y_weak', 'ground truth')]
        y = batch[self.pair_switch('y', 'ground truth')]

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Partial Attention - Drop Capillary Attention Map
        partial_attentions = [att[:, :2, ...] for att in attentions]

        # Detach - partial attentions
        detached_attentions = [a.detach() for a in partial_attentions]

        # Segmentor optimization
        fake_predicate = self.scribble_net.discriminator.forward(partial_attentions)
        if self.segmentor_learning_mode in ['weakly', 'fully']:
            Lsup = self.scribble_net.supervised_loss.forward(y_hat=predicate, ys=y_segmentor)
            Lgen = self.scribble_net.generator_loss.forward(fake_predicate)
            lr0 = Lgen.detach().norm() / Lsup.detach().norm() if self.supervise_regulation else 1.
            Lseg = (lr0 * Lsup) + (self.lr1 * Lgen)
            opt_seg.zero_grad()
            self.manual_backward(Lseg)
            opt_seg.step()
            lr_seg.step()

            # Logging
            self.log('train/gen_sup_loss', Lseg)
            self.log('train/sup_loss', Lsup)
            self.log('train/gen_loss', Lgen)
            if self.supervise_regulation:
                self.log('train/alpha_0', lr0)
        elif self.segmentor_learning_mode == 'none':
            Lgen = self.lr1 * self.scribble_net.generator_loss.forward(fake_predicate)
            opt_seg.zero_grad()
            self.manual_backward(Lgen)
            opt_seg.step()
            lr_seg.step()

            self.log('train/gen_loss', Lgen)

        # Discriminator optimization
        ylist = [y.type_as(x)]
        for i in range(1, len(attentions)):
            w, h = attentions[i].shape[2], attentions[i].shape[3]
            ylist.append(FT.resize(y, [w, h]).type_as(x))
        real_predicate = self.scribble_net.discriminator.forward(ylist)
        fake_predicate = self.scribble_net.discriminator.forward(detached_attentions)
        Ldis = self.lr2 * self.scribble_net.discriminatorial_loss.forward(real_predicate, fake_predicate)
        opt_dis.zero_grad()
        self.manual_backward(Ldis)
        opt_dis.step()
        lr_dis.step()

        # Logging
        self.log('train/dis_loss', Ldis)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y_target = batch[self.pair_switch('x', 'ground truth')], batch[self.pair_switch('y_target', 'ground truth')]

        # Segmentation
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        partial_attentions = [att[:, :2, ...] for att in attentions]
        fake = self.scribble_net.discriminator.forward(partial_attentions)
        # Performance evaluation
        # Loss calculation
        gen_loss = self.scribble_net.supervised_loss.forward(predicate, y_target)
        dis_loss = self.scribble_net.generator_loss.forward(fake)

        # Supervise loss
        self.log('val/sup_loss', gen_loss)
        # Generator loss
        self.log('val/gen_loss', dis_loss)

        if self.simplistic_evaluation:
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

        for c in range(self.evaluator.num_classes + 1 if self.evaluator.collapsible else 0):
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
                    fig = attention_predicate_plot(x, attentions, predicate, y, multi_class=True)
                    self.logger.experiment.log({'rollout': fig})
                    plt.close()
                    mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel', 2: 'capillary'}, ground_truth=y)
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})


class OCTAScribbleNetVariantAA(OCTAScribbleNetVariantA):

    def __init__(self, is_training: bool, pretrain: bool = False, supervise_regulation: bool = False, base_filters: int = 64, wandb_logging: bool = False, weight_path: Optional[str] = None, enable_input_augmentation: bool = False, logging_frequency: int = 1):
        """Multi-loss OCTA Scribble Net
        """
        super().__init__(is_training, pretrain, supervise_regulation, base_filters, wandb_logging, weight_path, enable_input_augmentation, logging_frequency)
        self.capillary_loss = WeightedPartialCE(2, manual=True)
        self.vessel_loss = WeightedPartialCE(2, manual=True)

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        opt_seg, opt_dis = self.optimizers()
        lr_seg, lr_dis = self.lr_schedulers()
        x = batch[self.pair_switch('x', 'projection')]
        y_segmentor = batch[self.pair_switch('y_weak', 'ground truth')]
        y = batch[self.pair_switch('y', 'ground truth')]

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Partial Attention - Drop Capillary Attention Map
        partial_attentions = [att[:, :2, ...] for att in attentions]

        # Detach - partial attentions
        detached_attentions = [a.detach() for a in partial_attentions]

        # Segmentor optimization
        fake_predicate = self.scribble_net.discriminator.forward(partial_attentions)
        Lsup_vessel = self.vessel_loss.forward(y_hat=predicate[:, :2, ...], ys=y_segmentor[:, :2, ...])
        Lsup_capill = self.capillary_loss.forward(predicate[:, [0, 2], :, :], y_segmentor[:, [0, 2], :, :].float(), full=True)  # Full-supervision
        Lsup = Lsup_vessel + Lsup_capill
        Lgen = self.scribble_net.generator_loss.forward(fake_predicate)
        lr0 = Lgen.detach().norm() / Lsup.detach().norm() if self.supervise_regulation else 1.
        Lseg = (lr0 * Lsup) + (self.lr1 * Lgen)
        opt_seg.zero_grad()
        self.manual_backward(Lseg)
        opt_seg.step()
        lr_seg.step()

        # Logging
        self.log('train/gen_sup_loss', Lseg)
        self.log('train/sup_loss', Lsup)
        self.log('train/gen_loss', Lgen)
        if self.supervise_regulation:
            self.log('train/alpha_0', lr0)

        # Discriminator optimization
        ylist = [y.type_as(x)]
        for i in range(1, len(attentions)):
            w, h = attentions[i].shape[2], attentions[i].shape[3]
            ylist.append(FT.resize(y, [w, h]).type_as(x))
        real_predicate = self.scribble_net.discriminator.forward(ylist)
        fake_predicate = self.scribble_net.discriminator.forward(detached_attentions)
        Ldis = self.lr2 * self.scribble_net.discriminatorial_loss.forward(real_predicate, fake_predicate)
        opt_dis.zero_grad()
        self.manual_backward(Ldis)
        opt_dis.step()
        lr_dis.step()

        # Logging
        self.log('train/dis_loss', Ldis)


class OCTAScribbleNetVariantB(OCTAScribbleNet):

    def shared_dataloader(self, is_train: bool, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> Any:
        dataset = RoseOcta500(
            is_train=is_train,
            rose_datapath='data',
            rose_modality='thin_gt' if is_train else 'thick_gt',  # Evaluate vessel on pixel-level mask.
            octa500_datapath='data/OCTA-500',
            octa500_modality='3m',
            octa500_level='ILM_OPL',
            ground_truth_style='one-hot',
            separate_vessel_capillary=True,
            rose_input_augmentation=self.enable_input_augmentation,
            vessel_capillary_gt='vessel',
            health_control_ratio=self.health_control_ratio
        )
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __init__(
        self, is_training: bool, pretrain: bool, supervise_regulation: bool,
        base_filters: int, wandb_logging: bool, weight_path: Optional[str] = None,
        enable_input_augmentation: bool = False, logging_frequency: int = 1,
        simplistic_evaluation: bool = False, evaluator: Optional[Evaluator] = None,
        health_control_ratio: Union[float, None] = None):
        super().__init__(
            is_training=is_training,
            pretrain=pretrain,
            supervise_regulation=supervise_regulation,
            base_filters=base_filters,
            wandb_logging=wandb_logging,
            weight_path=weight_path,
            logging_frequency=logging_frequency,
            enable_input_augmentation=enable_input_augmentation,
            simplistic_evaluation=simplistic_evaluation,
            evaluator=evaluator
        )
        self.automatic_optimization = False

        self.health_control_ratio = health_control_ratio

        self.is_training = is_training
        # Basis network
        self.scribble_net = OCTANet(
            raw_input_shape=torch.zeros((1, 3, 304, 304)).shape,
            mask_input_shape=torch.zeros((1, 2, 304, 304)).shape, # Drop
            is_training=is_training,
            instance_noise=True,
            label_noise=True,
            num_filters=base_filters,
            num_classes=2,
            pretrian=pretrain,
            weight_path=weight_path,
        )
        self.lr1 = 0.1
        self.lr2 = 0.2
        self.lr3 = 0.2
        self.wandb_logging = wandb_logging

        assert self.segmentor_learning_mode == 'weakly', 'Variation does not support other than weakly mode.'
        assert self.pairing_option == 'unpaired', 'Variation only support paired option.'

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        opt_seg, opt_dis = self.optimizers()
        lr_seg, lr_dis = self.lr_schedulers()
        x = batch[self.pair_switch('x', 'projection')]
        y_segmentor = batch[self.pair_switch('y_weak', 'ground truth')]
        y = batch[self.pair_switch('y', 'ground truth')]

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Detach - partial attentions
        detached_attentions = [a.detach() for a in attentions]

        # Segmentor optimization
        fake_predicate = self.scribble_net.discriminator.forward(attentions)
        if self.segmentor_learning_mode in ['weakly', 'fully']:
            Lsup = self.scribble_net.supervised_loss.forward(y_hat=predicate, ys=y_segmentor)
            Lgen = self.scribble_net.generator_loss.forward(fake_predicate)
            lr0 = Lgen.detach().norm() / Lsup.detach().norm() if self.supervise_regulation else 1.
            Lseg = (lr0 * Lsup) + (self.lr1 * Lgen)
            opt_seg.zero_grad()
            self.manual_backward(Lseg)
            opt_seg.step()
            lr_seg.step()

            # Logging
            self.log('train/gen_sup_loss', Lseg)
            self.log('train/sup_loss', Lsup)
            self.log('train/gen_loss', Lgen)
            if self.supervise_regulation:
                self.log('train/alpha_0', lr0)
        elif self.segmentor_learning_mode == 'none':
            Lgen = self.lr1 * self.scribble_net.generator_loss.forward(fake_predicate)
            opt_seg.zero_grad()
            self.manual_backward(Lgen)
            opt_seg.step()
            lr_seg.step()

            self.log('train/gen_loss', Lgen)

        # Discriminator optimization
        ylist = [y.type_as(x)]
        for i in range(1, len(attentions)):
            w, h = attentions[i].shape[2], attentions[i].shape[3]
            ylist.append(FT.resize(y, [w, h]).type_as(x))
        real_predicate = self.scribble_net.discriminator.forward(ylist)
        fake_predicate = self.scribble_net.discriminator.forward(detached_attentions)
        Ldis = self.lr2 * self.scribble_net.discriminatorial_loss.forward(real_predicate, fake_predicate)
        opt_dis.zero_grad()
        self.manual_backward(Ldis)
        opt_dis.step()
        lr_dis.step()

        # Logging
        self.log('train/dis_loss', Ldis)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y_target = batch[self.pair_switch('x', 'ground truth')], batch[self.pair_switch('y_target', 'ground truth')]

        # Segmentation
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        fake = self.scribble_net.discriminator.forward(attentions)
        # Performance evaluation
        # Loss calculation
        gen_loss = self.scribble_net.supervised_loss.forward(predicate, y_target)
        dis_loss = self.scribble_net.generator_loss.forward(fake)

        # Supervise loss
        self.log('val/sup_loss', gen_loss)
        # Generator loss
        self.log('val/gen_loss', dis_loss)

        if self.simplistic_evaluation:
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
                    fig = attention_predicate_plot(x, attentions, predicate, y, multi_class=False)
                    self.logger.experiment.log({'rollout': fig})
                    plt.close()
                    mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel'}, ground_truth=y)
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})



class OCTAScribbleNetVariantBA(OCTAScribbleNetVariantB):

    def shared_dataloader(self, is_train: bool, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> Any:
        dataset = RoseOcta500(
            is_train=is_train,
            rose_datapath='data',
            rose_modality='thin_gt' if is_train else 'thick_gt',  # Evaluate vessel on pixel-level mask.
            octa500_datapath='data/OCTA-500',
            octa500_modality='3m',
            octa500_level='ILM_OPL',
            ground_truth_style='one-hot',
            separate_vessel_capillary=True,
            rose_input_augmentation=self.enable_input_augmentation,
            vessel_capillary_gt='capillary'
        )
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class OCTAScribbleNetVariantBB(OCTAScribbleNetVariantBA):

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        opt_seg, opt_dis = self.optimizers()
        lr_seg, lr_dis = self.lr_schedulers()
        x = batch[self.pair_switch('x', 'projection')]
        y_segmentor = batch[self.pair_switch('y_weak', 'ground truth')]
        # y = batch[self.pair_switch('y', 'ground truth')]

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Detach - partial attentions
        detached_attentions = [a.detach() for a in attentions]

        # Segmentor optimization
        fake_predicate = self.scribble_net.discriminator.forward(attentions)
        if self.segmentor_learning_mode in ['weakly', 'fully']:
            Lsup = self.scribble_net.supervised_loss.forward(y_hat=predicate, ys=y_segmentor)
            Lgen = self.scribble_net.generator_loss.forward(fake_predicate)
            lr0 = Lgen.detach().norm() / Lsup.detach().norm() if self.supervise_regulation else 1.
            Lseg = (lr0 * Lsup) + (0.5 * Lgen)
            opt_seg.zero_grad()
            self.manual_backward(Lseg)
            opt_seg.step()
            lr_seg.step()

            # Logging
            self.log('train/gen_sup_loss', Lseg)
            self.log('train/sup_loss', Lsup)
            self.log('train/gen_loss', Lgen)
            if self.supervise_regulation:
                self.log('train/alpha_0', lr0)
        elif self.segmentor_learning_mode == 'none':
            Lgen = self.lr1 * self.scribble_net.generator_loss.forward(fake_predicate)
            opt_seg.zero_grad()
            self.manual_backward(Lgen)
            opt_seg.step()
            lr_seg.step()

            self.log('train/gen_loss', Lgen)

        # Discriminator optimization
        ylist = [y_segmentor.type_as(x)] # Pair
        for i in range(1, len(attentions)):
            w, h = attentions[i].shape[2], attentions[i].shape[3]
            ylist.append(FT.resize(y_segmentor, [w, h]).type_as(x))
        real_predicate = self.scribble_net.discriminator.forward(ylist)
        fake_predicate = self.scribble_net.discriminator.forward(detached_attentions)
        Ldis = self.lr2 * self.scribble_net.discriminatorial_loss.forward(real_predicate, fake_predicate)
        opt_dis.zero_grad()
        self.manual_backward(Ldis)
        opt_dis.step()
        lr_dis.step()

        # Logging
        self.log('train/dis_loss', Ldis)


class OCTAScribbleNetVariantC(OCTAScribbleNet):

    def shared_dataloader(self, is_train: bool, batch_size: int = 4, shuffle: bool = True, num_workers: int = 0) -> Any:
        dataset = RoseOcta500(
            is_train=is_train,
            rose_datapath='data',
            rose_modality='thin_gt' if is_train else 'thick_gt',  # Evaluate vessel on pixel-level mask.
            octa500_datapath='data/OCTA-500',
            octa500_modality='3m',
            octa500_level='ILM_OPL',
            ground_truth_style='one-hot',
            separate_vessel_capillary=True,
            rose_input_augmentation=self.enable_input_augmentation,
            vessel_capillary_gt='vessel',
            health_control_ratio=self.health_control_ratio
        )
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def __init__(
        self, is_training: bool, pretrain: bool, supervise_regulation: bool,
        base_filters: int, wandb_logging: bool, weight_path: Optional[str] = None,
        enable_input_augmentation: bool = False, logging_frequency: int = 1,
        simplistic_evaluation: bool = False, evaluator: Optional[Evaluator] = None,
        health_control_ratio: Union[float, None] = None,
        interlayer_divergence_weight: float = 0.2,
        enable_predicate_posterior: Literal['attentions', 'predicate', 'cross-entropy'] = 'attentions',
        kl_stop_gradient: bool = False):
        super().__init__(
            is_training=is_training,
            pretrain=pretrain,
            supervise_regulation=supervise_regulation,
            base_filters=base_filters,
            wandb_logging=wandb_logging,
            weight_path=weight_path,
            logging_frequency=logging_frequency,
            enable_input_augmentation=enable_input_augmentation,
            simplistic_evaluation=simplistic_evaluation,
            evaluator=evaluator
        )
        self.automatic_optimization = False

        self.health_control_ratio = health_control_ratio

        self.is_training = is_training
        # Basis network
        self.scribble_net = OCTANet(
            raw_input_shape=torch.zeros((1, 3, 304, 304)).shape,
            mask_input_shape=torch.zeros((1, 2, 304, 304)).shape, # Drop
            is_training=is_training,
            instance_noise=True,
            label_noise=True,
            num_filters=base_filters,
            num_classes=2,
            pretrian=pretrain,
            weight_path=weight_path,
            segmentor_gating_level=3, # Discarding ADS on the highest-level feature.
            discriminator_depth=3, # Discarding ADS on the highest-level feature.
        )
        self.kl_divergence = InterlayerDivergence('mean', stop_gradient=kl_stop_gradient)
        self.enable_predicate_posterior = enable_predicate_posterior
        
        self.lr1 = 0.1
        self.lr2 = 0.2
        self.lr3 = 0.2
        self.itl_div = interlayer_divergence_weight
        
        self.wandb_logging = wandb_logging

        assert self.segmentor_learning_mode == 'weakly', 'Variation does not support other than weakly mode.'
        assert self.pairing_option == 'unpaired', 'Variation only support paired option.'

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        opt_seg, opt_dis = self.optimizers()
        lr_seg, lr_dis = self.lr_schedulers()
        x = batch[self.pair_switch('x', 'projection')]
        y_segmentor = batch[self.pair_switch('y_weak', 'ground truth')]
        y = batch[self.pair_switch('y', 'ground truth')]

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Detach - partial attentions
        detached_attentions = [a.detach() for a in attentions]

        # Segmentor optimization
        fake_predicate = self.scribble_net.discriminator.forward(attentions)
        if self.segmentor_learning_mode in ['weakly', 'fully']:
            Lsup = self.scribble_net.supervised_loss.forward(y_hat=predicate, ys=y_segmentor)
            Lgen = self.scribble_net.generator_loss.forward(fake_predicate)
            if self.enable_predicate_posterior == 'attentions':
                kl_loss = self.kl_divergence(attentions)
            elif self.enable_predicate_posterior == 'predicate':
                kl_loss = self.kl_divergence([predicate, *attentions])
            elif self.enable_predicate_posterior == 'cross-entropy':
                kl_loss = self.kl_divergence([y_segmentor, *attentions])
            else:
                raise NotImplementedError(f'{self.enable_predicate_posterior} is not a valid mode.')

            lr0 = Lgen.detach().norm() / Lsup.detach().norm() if self.supervise_regulation else 1.

            Lseg = (lr0 * Lsup) + (self.lr1 * Lgen) + (self.itl_div * kl_loss)
            opt_seg.zero_grad()
            self.manual_backward(Lseg)
            opt_seg.step()
            lr_seg.step()

            # Logging
            self.log('train/gen_sup_loss', Lseg)
            self.log('train/sup_loss', Lsup)
            self.log('train/gen_loss', Lgen)
            self.log('train/kl_loss', kl_loss)
            if self.supervise_regulation:
                self.log('train/alpha_0', lr0)
        elif self.segmentor_learning_mode == 'none':
            Lgen = self.lr1 * self.scribble_net.generator_loss.forward(fake_predicate)
            opt_seg.zero_grad()
            self.manual_backward(Lgen)
            opt_seg.step()
            lr_seg.step()

            self.log('train/gen_loss', Lgen)

        # Discriminator optimization
        ylist = [y.type_as(x)]
        for i in range(1, len(attentions)):
            w, h = attentions[i].shape[2], attentions[i].shape[3]
            ylist.append(FT.resize(y, [w, h]).type_as(x))
        real_predicate = self.scribble_net.discriminator.forward(ylist)
        fake_predicate = self.scribble_net.discriminator.forward(detached_attentions)
        Ldis = self.lr2 * self.scribble_net.discriminatorial_loss.forward(real_predicate, fake_predicate)
        opt_dis.zero_grad()
        self.manual_backward(Ldis)
        opt_dis.step()
        lr_dis.step()

        # Logging
        self.log('train/dis_loss', Ldis)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y_target = batch[self.pair_switch('x', 'ground truth')], batch[self.pair_switch('y_target', 'ground truth')]

        # Segmentation
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        fake = self.scribble_net.discriminator.forward(attentions)
        # Performance evaluation
        # Loss calculation
        gen_loss = self.scribble_net.supervised_loss.forward(predicate, y_target)
        dis_loss = self.scribble_net.generator_loss.forward(fake)

        # Supervise loss
        self.log('val/sup_loss', gen_loss)
        # Generator loss
        self.log('val/gen_loss', dis_loss)

        if self.simplistic_evaluation:
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
                    fig = attention_predicate_plot(x, attentions, predicate, y, multi_class=False)
                    self.logger.experiment.log({'rollout': fig})
                    plt.close()
                    mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel'}, ground_truth=y)
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})