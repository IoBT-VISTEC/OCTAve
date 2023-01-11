import logging
from pathlib import Path
import random
from typing import List, Literal, Sequence, Union, Optional

from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule
from sklearn import metrics
import torch
from torch.functional import Tensor
from torch_poly_lr_decay import PolynomialLRDecay
import torchvision.transforms.functional as FT

from architectures.models.octanet import OctaScribbleNet as OCTANet
from architectures.models.vanila import VanilaScribbleNet
from architectures.models.cenet import CE_Net
from architectures.models.csnet import CSNet
from architectures.segmentor.blocks import adaptive_aggregationC, adaptive_aggregationMulti, adaptive_aggregationPC, baseC, baseMulti, basePC
from architectures.segmentor.compose import ResnestUNet, ResnestUnetParallelHead, ResnestUnetParallelHeadAttentionGate
from architectures.segmentor.losses import CELoss, ImageMseLoss, InterlayerDivergence, DiceLoss, WeightedPartialCE
from utils.logging import attention_predicate_plot, to_wandb_semantic_segmentation
from utils.evaluation import (
    accuracy, auc, cohen_kappa, dice, iou,
    sensitivity_specificity_fpr, Evaluator)
from utils.tools import upscale_tensors_like_2d


class OCTA500Experiment3M_A(LightningModule):
    """OCTAVeOCTA500 Experiment Module
    """
    def __init__(
        self, is_training: bool, pretrain: bool, supervise_regulation: bool,
        base_filters: int, wandb_logging: bool, weight_path: Optional[str] = None,
        segmentor_learning_mode: Literal['fully', 'weakly'] = 'weakly',
        pairing_option: Literal['unpaired', 'paired'] = 'unpaired',
        enable_input_augmentation: bool = False, logging_frequency: int = 1,
        simplistic_evaluation: bool = False, evaluator: Optional[Evaluator] = None,
        health_control_ratio: Union[float, None] = None,
        interlayer_divergence_weight: float = 0.2,
        enable_predicate_posterior: Literal['attentions', 'predicate', 'cross-entropy'] = 'attentions',
        kl_stop_gradient: bool = False,
        regulation_mode: Literal['origin', 'swap'] = 'origin',
        regulation_clip: float = 0.,
        dynamic_itl: bool = False,
        interlayer_divergense_type: Literal['KLD', 'JSD'] = 'KLD',
        interlayer_layer_contrib_weight: Optional[List[float]] = None,
        disable_discriminator: bool = False,
        segmentor_gating_level: int = 3,
        discriminator_depth: int = 3,
        disable_weight_on_bg: bool = False,
        wpce_reduction: Literal['mean', 'sum'] = 'mean',
        disable_unsupervise_itl: bool = False,
        weakly_supervise: bool = False,
        **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.segmentor_learning_mode = segmentor_learning_mode
        self.pairing_option = pairing_option
        self.is_training = is_training

        self.health_control_ratio = health_control_ratio

        self.is_training = is_training
        self.weakly_supervise = weakly_supervise
        # Basis network
        self.scribble_net = OCTANet(
            raw_input_shape=kwargs.get('raw_input_shape', torch.zeros((1, 3, 304, 304)).shape),
            mask_input_shape=kwargs.get('mask_input_shape', torch.zeros((1, 2, 304, 304)).shape), # Drop bg
            is_training=is_training,
            instance_noise=kwargs.get('instance_noise', True),
            label_noise=kwargs.get('label_noise', True),
            num_filters=base_filters,
            num_classes=2,
            pretrian=pretrain,
            weight_path=weight_path,
            segmentor_gating_level=segmentor_gating_level,
            discriminator_depth=discriminator_depth,
            weakly_supervise=weakly_supervise,
        )
        self.kl_divergence = InterlayerDivergence('mean', stop_gradient=kl_stop_gradient, divergence=interlayer_divergense_type)
        self.kl_layer_weights = interlayer_layer_contrib_weight
        self.enable_predicate_posterior = enable_predicate_posterior
        self.disable_unsupervise_itl = disable_unsupervise_itl

        self.wandb_logging = wandb_logging
        self.supervise_regulation = supervise_regulation
        self.frequency = logging_frequency
        self.enable_input_augmentation = enable_input_augmentation
        self.simplistic_evaluation = simplistic_evaluation
        self.evaluator = evaluator

        self.lr1 = 0.1
        self.lr2 = 0.2
        self.lr3 = 0.2
        self.itl_div = torch.tensor(interlayer_divergence_weight)
        self.regulation_mode = regulation_mode
        self.regulation_clip = regulation_clip
        self.dynamic_itl = dynamic_itl
        self.disable_weight_bg = disable_weight_on_bg
        self.wpce_reduction = wpce_reduction

        self.disable_discriminator = disable_discriminator
        self.segmentor_gating_level = segmentor_gating_level

        assert self.pairing_option == 'unpaired', 'Variation only support paired option.'

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
        x = batch[self.pair_switch('x', 'y')]
        if self.weakly_supervise:
            y_segmentor = batch[self.pair_switch('y_weak', 'y')]
        else:
            y_segmentor = batch['y']
        if self.weakly_supervise:
            y = batch[self.pair_switch('y_unpair', 'y')]
        else:
            y_segmentor = batch['y']
        gt_exclusion_mask = batch.get('ignore_y', torch.zeros(y_segmentor.shape[0]).int())

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Detach - partial attentions
        detached_attentions = [a.detach() for a in attentions]

        # Segmentor optimization
        if not self.disable_discriminator:
            fake_predicate = self.scribble_net.discriminator.forward(attentions)

        sup_predicate, sup_gt = predicate[torch.where(~gt_exclusion_mask)], y_segmentor[torch.where(~gt_exclusion_mask)]
        sup_attentions = tuple(map(lambda att: att[torch.where(~gt_exclusion_mask)], attentions))
        unsup_predicate = predicate[torch.where(gt_exclusion_mask)]
        unsup_attentions = tuple(map(lambda att: att[torch.where(gt_exclusion_mask)], attentions))

        opt_seg.zero_grad()

        # Supervise Branch
        supLseg = 0
        supLgen = 0
        kl_loss = torch.tensor(0.)
        if sup_predicate.size()[0] != 0:
            if self.segmentor_learning_mode in ['weakly', 'fully']:
                supLgen = torch.tensor(0.) if self.disable_discriminator else self.scribble_net.generator_loss.forward(fake_predicate[torch.where(~gt_exclusion_mask)])
                # Main supervised loss
                if self.weakly_supervise:
                    Lsup = self.scribble_net.supervised_loss.forward(y_hat=sup_predicate, ys=sup_gt, ignore_bg=self.disable_weight_bg, reduction=self.wpce_reduction)
                else:
                    Lsup = self.scribble_net.supervised_loss.forward(sup_predicate, sup_gt)
                self.log('train/sup_loss', Lsup)
                if self.itl_div != 0 or self.dynamic_itl:
                    if self.enable_predicate_posterior == 'attentions':
                        kl_loss = self.kl_divergence.forward(sup_attentions, weights=self.kl_layer_weights)
                    elif self.enable_predicate_posterior == 'predicate':
                        kl_loss = self.kl_divergence.forward([sup_predicate, *sup_attentions], weights=self.kl_layer_weights)
                    elif self.enable_predicate_posterior == 'cross-entropy':
                        kl_loss = self.kl_divergence.forward([sup_gt, *sup_attentions], weights=self.kl_layer_weights)
                    else:
                        raise NotImplementedError(f'{self.enable_predicate_posterior} is not a valid mode.')

                lr0 = torch.tensor(1.)
                if not self.disable_discriminator:
                    if self.supervise_regulation:
                        if self.regulation_mode == 'origin':
                            lr0: Tensor = (supLgen.detach().norm() / Lsup.detach().norm())
                        elif self.regulation_mode == 'swap':
                            lr0 = (Lsup.detach().norm() / supLgen.detach().norm())
                        else:
                            raise NotImplementedError(f'{self.regulation_mode} is invalid or not implemented.')
                        if self.regulation_clip > 0:
                            lr0 = lr0.clamp(max=self.regulation_clip)

                self.log('train/alpha_0', lr0)
                Lseg = (lr0 * Lsup) + (self.lr1 * supLgen)

                if self.dynamic_itl:
                    self.itl_div: Tensor = (kl_loss.detach().norm() / (Lsup.detach().norm() + supLgen.detach().norm()))
                    self.itl_div = self.itl_div.clamp(max=1) * 0.1

                supLseg =  Lseg + (self.itl_div * kl_loss)
            else:
                raise NotImplementedError('`none` mode no longer implemented')

        if self.weakly_supervise:
            # Unsupervise branch
            unsupLseg = 0
            unsupLgen = 0
            unsup_kl_loss = torch.tensor(0.)
            if unsup_predicate.size()[0] != 0:
                unsupLgen = torch.tensor(0.) if self.disable_discriminator else self.scribble_net.generator_loss.forward(fake_predicate[torch.where(gt_exclusion_mask)])
                # Self-supervision
                if self.itl_div != 0 or self.dynamic_itl:
                    if self.enable_predicate_posterior == 'attentions':
                        unsup_kl_loss = self.kl_divergence.forward(unsup_attentions, weights=self.kl_layer_weights)
                    elif self.enable_predicate_posterior == 'predicate':
                        unsup_kl_loss = self.kl_divergence.forward([unsup_predicate, *unsup_attentions], weights=self.kl_layer_weights)
                    elif self.enable_predicate_posterior == 'cross-entropy':
                        # No cross-entropy calculation for assume unsupervision
                        pass

                # Calculate weight.
                    if self.dynamic_itl:
                        inv_mean_confidence = 1 - unsup_predicate.flatten().mean().detach() # Good example ~ 0, Bad ~ 1
                        self.itl_div: Tensor = (unsup_kl_loss.detach().norm() / (unsupLgen.detach().norm() + inv_mean_confidence))
                        self.itl_div = self.itl_div.clamp(max=1) * 0.1
                unsupLseg = (self.lr3 * unsupLgen) + (0 if self.disable_unsupervise_itl else (self.itl_div * unsup_kl_loss))

            # Optimization Step
            aggLseg = supLseg + unsupLseg
        else:
            aggLseg = supLseg
        self.manual_backward(aggLseg)
        opt_seg.step()
        lr_seg.step()

        # Logging
        if self.weakly_supervise:
            self.log('train/gen_sup_loss', supLseg + unsupLseg)
            self.log('train/gen_loss', supLgen + unsupLgen)
            self.log('train/kl_loss', kl_loss + unsup_kl_loss)
            if self.dynamic_itl:
                self.log('train/itl_weight', self.itl_div)
        else:
            self.log('train/sup_loss', supLseg)

        if self.weakly_supervise:
            if not self.disable_discriminator:
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
        x, y_target = batch[self.pair_switch('x', 'ground truth')], batch[self.pair_switch('y', 'ground truth')]

        # Segmentation
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        # Performance evaluation
        # Loss calculation
        gen_loss = self.scribble_net.supervised_loss.forward(predicate, y_target)
        if self.weakly_supervise:
            if not self.disable_discriminator:
                fake = self.scribble_net.discriminator.forward(attentions)
                dis_loss = self.scribble_net.generator_loss.forward(fake)
                # Generator loss
                self.log('val/gen_loss', dis_loss)

        # Supervise loss
        self.log('val/sup_loss', gen_loss)

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
                self.log(f'val/acc_{c}', accur[c], on_step=True)
                self.log(f'val/auroc_{c}', auc_scr[c], on_step=True)
                self.log(f'val/dice_{c}', dice_coef[c], on_step=True)
                self.log(f'val/iou_{c}', iou_scr[c], on_step=True)
                self.log(f'val/kappa_{c}', kappa[c], on_step=True)
                self.log(f'val/specificity_{c}', spe[c], on_step=True)
                self.log(f'val/sensitivity_{c}', sen[c], on_step=True)
                self.log(f'val/fpr_{c}', fpr[c], on_step=True)

        return {'x': x, 'y': y_target, 'attentions': attentions, 'predicate': predicate}

    def validation_epoch_end(self, outputs) -> None:
        if self.current_epoch % self.frequency == 0:
            if self.wandb_logging:
                masks = []
                samples = random.choices(outputs, k=5)
                for output in samples:
                    x, y, attentions, predicate = output['x'], output['y'], output['attentions'], output['predicate']
                    if self.segmentor_gating_level >= 0:
                        fig = attention_predicate_plot(x, attentions, predicate, y, multi_class=False)
                        self.logger.experiment.log({'rollout': fig})
                        plt.close()
                    mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel'}, ground_truth=y)
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})

    def forward(self, x):
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        return attentions, predicate


class OCTA500ClassificationExperiment(LightningModule):
    """OCTAVeOCTA500 Experiment Module
    """
    def __init__(
        self,
        is_training: bool,
        pretrain: bool,
        supervise_regulation: bool,
        wandb_logging: bool, num_classes: int,
        class_weight: list,
        weight_path: Optional[str] = 'resnest50-528c19ca.pth',
        segmentor_gating_level: int = 3,
        kl_stop_gradient: bool = True,
        interlayer_divergence_type: Literal['KLD', 'JSD'] = 'KLD',
        interlayer_layer_contrib_weight: Sequence[float] = [1, 0.8, 0.4, 0.2],
        enable_predicate_posterior: Literal['attention', 'predicate'] = 'predicate',
        logging_frequency: int = 5,
        interlayer_divergence_weight: float = 0.1,
        regulation_mode: Literal['origin', 'swap'] = 'swap',
        regulation_clip: float = 1,
        dynamic_itl: bool = True,
        mode: str = 'classic',
        ild_start_epoch: int = 0,
        encoder_gating: bool = False,
        **kwargs):
        super().__init__()
        # self.automatic_optimization = False

        self.is_training = is_training
        # Basis network
        self.scribble_net = OCTANet(
            raw_input_shape=kwargs.get('raw_input_shape', torch.zeros((1, 3, 304, 304)).shape),
            mask_input_shape=kwargs.get('mask_input_shape', torch.zeros((1, 2, 304, 304)).shape), # Drop bg
            is_training=is_training,
            instance_noise=False,
            label_noise=False,
            num_classes=num_classes,
            pretrian=pretrain,
            weight_path=weight_path,
            segmentor_gating_level=segmentor_gating_level,
            discriminator_depth=-1,
            encoder_gating=encoder_gating,
        )
        self.kl_divergence = InterlayerDivergence('mean', stop_gradient=kl_stop_gradient, divergence=interlayer_divergence_type)
        self.kl_layer_weights = interlayer_layer_contrib_weight
        self.enable_predicate_posterior = enable_predicate_posterior
        self.class_weight = class_weight

        self.wandb_logging = wandb_logging
        self.supervise_regulation = supervise_regulation
        self.frequency = logging_frequency

        self.itl_div = torch.tensor(interlayer_divergence_weight)
        self.regulation_mode = regulation_mode
        self.regulation_clip = regulation_clip
        self.dynamic_itl = dynamic_itl
        self.mode = mode
        self.ild_start_epoch = ild_start_epoch
        self.encoder_gating = encoder_gating

        self.class_loss = CELoss(weight=torch.tensor(self.class_weight))

        # self.disable_discriminator = True
        self.segmentor_gating_level = segmentor_gating_level

        # Utilities
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt_seg = torch.optim.Adam(self.scribble_net.segmentor.parameters(), lr=1e-5)
        lr_seg = torch.optim.lr_scheduler.CyclicLR(opt_seg, base_lr=1e-5, max_lr=1e-4, cycle_momentum=False)
        return [opt_seg], [lr_seg]

    def training_step(self, batch, batch_idx, optimizer_idx = None, *args, **kwargs):
        x = batch['x']
        y = batch['y_class']

        # Segmentor forward pass
        if self.encoder_gating:
            y_hat, attentions, _, predicate_map = self.scribble_net.segmentor.classification_predict(x, method='softmax', mode=self.mode)
        else:
            y_hat, attentions, predicate_map = self.scribble_net.segmentor.classification_predict(x, method='softmax', mode=self.mode)

        # Calculate Classification Task Loss
        Lclass = self.class_loss.forward(y_hat, y)

        # Calculate KL
        kl_loss = torch.tensor(0.)
        if self.itl_div != 0 or self.dynamic_itl:
            if self.enable_predicate_posterior == 'attentions':
                kl_loss = self.kl_divergence.forward(attentions, weights=self.kl_layer_weights)
            elif self.enable_predicate_posterior == 'predicate':
                kl_loss = self.kl_divergence.forward([predicate_map, *attentions], weights=self.kl_layer_weights)
            else:
                raise NotImplementedError(f'{self.enable_predicate_posterior} is not a valid mode.')

        if self.dynamic_itl:
            self.itl_div: Tensor = (kl_loss.detach().norm() / (Lclass.detach().norm()))
            self.itl_div = self.itl_div.clamp(max=self.regulation_clip)

        if self.current_epoch >= self.ild_start_epoch:
            supL =  Lclass + (self.itl_div * kl_loss)
        else:
            supL =  Lclass
        # Optimization Step

        # Logging
        self.log('train/sup_loss', supL)
        self.log('train/cl_loss', Lclass)
        self.log('train/kl_loss', kl_loss)
        if self.dynamic_itl:
            self.log('train/itl_weight', self.itl_div)

        return {'loss': supL, 'cl_loss': Lclass, 'kl_loss': kl_loss}

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y_target = batch['x'], batch['y_class']

        # Segmentation
        if self.encoder_gating:
            y_hat, attentions, _, predicate = self.scribble_net.segmentor.classification_predict(x, method='softmax', mode=self.mode)
        else:
            y_hat, attentions, predicate = self.scribble_net.segmentor.classification_predict(x, method='softmax', mode=self.mode)

        # Performance evaluation
        # Loss calculation
        Lclass = self.class_loss.forward(y_hat, y_target)

        # Supervise loss
        self.log('val/cl_loss', Lclass)

        return {'x': x, 'y': y_target, 'y_hat': y_hat, 'attentions': attentions, 'predicate': predicate}

    def validation_epoch_end(self, outputs) -> None:
        # Calculate classification metric
        map_to_numpy = lambda x, key: np.concatenate(list(map(lambda k: k[key].cpu().numpy(), x)), axis=0)
        y, y_hat = map_to_numpy(outputs, 'y'), map_to_numpy(outputs, 'y_hat')
        y_numeral = np.argmax(y, axis=1)
        y_hat_numeral = np.argmax(y_hat, axis=1)
        acc = metrics.accuracy_score(y_true=y_numeral, y_pred=y_hat_numeral)
        f1 = metrics.f1_score(y_true=y_numeral, y_pred=y_hat_numeral, average='macro')
        precision = metrics.precision_score(y_true=y_numeral, y_pred=y_hat_numeral, average='macro')
        recall_score = metrics.recall_score(y_true=y_numeral, y_pred=y_hat_numeral, average='macro')
        # Non-normal sensitivity
        a_y_numeral = y_numeral[y_numeral!=0]
        a_y_numeral = a_y_numeral > 0
        a_y_hat_numeral = y_hat_numeral[y_numeral!=0]
        a_y_hat_numeral = a_y_hat_numeral > 0
        # tn, fp, fn, tp = metrics.confusion_matrix(y_true=a_y_numeral, y_pred=a_y_hat_numeral).ravel()

        self.log(f'val/acc', acc)
        self.log(f'val/f1', f1)
        self.log(f'val/precision', precision)
        self.log(f'val/recal', recall_score)
        # self.log(f'val/non_normal_sen', (tp / (tp + fn)))

        if self.current_epoch % self.frequency == 0:
            if self.wandb_logging:
                samples = random.choices(outputs, k=5)
                for output in samples:
                    x, y, attentions, predicate = output['x'], output['y'], output['attentions'], output['predicate']
                    if self.segmentor_gating_level >= 0 and not self.encoder_gating:
                        fig = attention_predicate_plot(x, attentions, predicate, predicate.detach(), multi_class=False)
                        self.logger.experiment[0].log({'rollout': fig})
                        plt.close()

    def forward(self, x):
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        return attentions, predicate


class ScribbleNet(LightningModule):
    """ Based on the implementation of Valvano et al. Learning from multiscale Adversarial Network.
    """
    segmentor_gating_level: int = 3
    def __init__(
        self, is_training: bool, supervise_regulation: bool,
        base_filters: int, wandb_logging: bool,
        segmentor_learning_mode: Literal['full', 'weakly'] = 'weakly',
        pairing_option: Literal['unpaired', 'paired'] = 'unpaired',
        enable_input_augmentation: bool = False, logging_frequency: int = 1,
        simplistic_evaluation: bool = False, evaluator: Optional[Evaluator] = None,
        health_control_ratio: Union[float, None] = None,
        interlayer_divergence_weight: float = 0.2,
        enable_predicate_posterior: Literal['attentions', 'predicate', 'cross-entropy'] = 'attentions',
        kl_stop_gradient: bool = False,
        regulation_mode: Literal['origin', 'swap'] = 'origin',
        regulation_clip: float = 0.,
        dynamic_itl: bool = False,
        interlayer_divergence_type: Literal['KLD', 'JSD'] = 'KLD',
        interlayer_layer_contrib_weight: Optional[List[float]] = None,
        disable_discriminator: bool = False,
        disable_weight_on_bg: bool = False,
        wpce_reduction: Literal['mean', 'sum'] = 'mean',
        disable_unsupervise_itl: bool = True,
        **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.segmentor_learning_mode = segmentor_learning_mode
        self.pairing_option = pairing_option
        self.is_training = is_training

        self.health_control_ratio = health_control_ratio

        self.is_training = is_training
        # Basis network
        self.scribble_net = VanilaScribbleNet(
            raw_input_shape=kwargs.get('raw_input_shape', torch.zeros((1, 3, 304, 304)).shape),
            mask_input_shape=kwargs.get('mask_input_shape', torch.zeros((1, 2, 304, 304)).shape), # Drop bg
            is_training=is_training,
            instance_noise=kwargs.get('instance_noise', True),
            label_noise=kwargs.get('label_noise', True),
            num_filters=base_filters,
            num_classes=2,
        )
        self.kl_divergence = InterlayerDivergence('mean', stop_gradient=kl_stop_gradient, divergence=interlayer_divergence_type)
        self.kl_layer_weights = interlayer_layer_contrib_weight
        self.enable_predicate_posterior = enable_predicate_posterior
        self.disable_unsupervise_itl = disable_unsupervise_itl

        self.wandb_logging = wandb_logging
        self.supervise_regulation = supervise_regulation
        self.frequency = logging_frequency
        self.enable_input_augmentation = enable_input_augmentation
        self.simplistic_evaluation = simplistic_evaluation
        self.evaluator = evaluator

        self.lr1 = 0.1
        self.lr2 = 0.2
        self.lr3 = 0.2
        self.itl_div = torch.tensor(interlayer_divergence_weight)
        self.regulation_mode = regulation_mode
        self.regulation_clip = regulation_clip
        self.dynamic_itl = dynamic_itl
        self.disable_weight_bg = disable_weight_on_bg
        self.wpce_reduction = wpce_reduction

        self.disable_discriminator = disable_discriminator

        # Below are irrelevant. Change the weight to 0 in order to turn off part of the optimization.
        # assert self.segmentor_learning_mode == 'weakly', 'Variation does not support other than weakly mode.'
        # assert self.pairing_option == 'unpaired', 'Variation only support paired option.'

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
        x = batch[self.pair_switch('x', 'x')]
        y_segmentor = batch[self.pair_switch('y_weak', 'y')]
        y = batch[self.pair_switch('y_unpair', 'y')]
        gt_exclusion_mask = batch.get('ignore_y', None)
        if gt_exclusion_mask is None:
            gt_exclusion_mask = torch.zeros(y.shape[0]).int()

        # Segmentor forward pass
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')

        # Detach - partial attentions
        detached_attentions = [a.detach() for a in attentions]

        # Segmentor optimization
        if not self.disable_discriminator:
            fake_predicate = self.scribble_net.discriminator.forward(attentions)

        sup_predicate, sup_gt = predicate[torch.where(~gt_exclusion_mask)], y_segmentor[torch.where(~gt_exclusion_mask)]
        sup_attentions = tuple(map(lambda att: att[torch.where(~gt_exclusion_mask)], attentions))
        unsup_predicate = predicate[torch.where(gt_exclusion_mask)]
        unsup_attentions = tuple(map(lambda att: att[torch.where(gt_exclusion_mask)], attentions))

        opt_seg.zero_grad()

        # Supervise Branch
        supLseg = torch.tensor(0.)
        supLgen = torch.tensor(0.)
        kl_loss = torch.tensor(0.)
        if sup_predicate.size()[0] != 0:
            if self.segmentor_learning_mode in ['weakly', 'fully']:
                supLgen = torch.tensor(0.) if self.disable_discriminator else self.scribble_net.generator_loss.forward(fake_predicate[torch.where(~gt_exclusion_mask)])
                Lsup = self.scribble_net.supervised_loss.forward(y_hat=sup_predicate, ys=sup_gt, ignore_bg=self.disable_weight_bg, reduction=self.wpce_reduction)
                self.log('train/sup_loss', Lsup)
                if self.itl_div != 0 or self.dynamic_itl:
                    if self.enable_predicate_posterior == 'attentions':
                        kl_loss = self.kl_divergence.forward(sup_attentions, weights=self.kl_layer_weights)
                    elif self.enable_predicate_posterior == 'predicate':
                        kl_loss = self.kl_divergence.forward([sup_predicate, *sup_attentions], weights=self.kl_layer_weights)
                    elif self.enable_predicate_posterior == 'cross-entropy':
                        kl_loss = self.kl_divergence.forward([sup_gt, *sup_attentions], weights=self.kl_layer_weights)
                    else:
                        raise NotImplementedError(f'{self.enable_predicate_posterior} is not a valid mode.')

                lr0 = torch.tensor(1.)
                if not self.disable_discriminator:
                    if self.supervise_regulation:
                        if self.regulation_mode == 'origin':
                            lr0: Tensor = (supLgen.detach().norm() / Lsup.detach().norm())
                        elif self.regulation_mode == 'swap':
                            lr0 = (Lsup.detach().norm() / supLgen.detach().norm())
                        else:
                            raise NotImplementedError(f'{self.regulation_mode} is invalid or not implemented.')
                        if self.regulation_clip > 0:
                            lr0 = lr0.clamp(max=self.regulation_clip)

                self.log('train/alpha_0', lr0)
                Lseg = (lr0 * Lsup) + (self.lr1 * supLgen)

                if self.dynamic_itl:
                    self.itl_div: Tensor = (kl_loss.detach().norm() / (Lsup.detach().norm() + supLgen.detach().norm()))
                    self.itl_div = self.itl_div.clamp(max=1) * 0.1

                supLseg =  Lseg + (self.itl_div * kl_loss)
            else:
                raise NotImplementedError('`none` mode no longer implemented')

        # Unsupervise branch
        unsupLseg = 0
        unsupLgen = 0
        unsup_kl_loss = torch.tensor(0.)
        if unsup_predicate.size()[0] != 0:
            unsupLgen = torch.tensor(0.) if self.disable_discriminator else self.scribble_net.generator_loss.forward(fake_predicate[torch.where(gt_exclusion_mask)])
            # Self-supervision
            if self.itl_div != 0 or self.dynamic_itl:
                if self.enable_predicate_posterior == 'attentions':
                    unsup_kl_loss = self.kl_divergence.forward(unsup_attentions, weights=self.kl_layer_weights)
                elif self.enable_predicate_posterior == 'predicate':
                    unsup_kl_loss = self.kl_divergence.forward([unsup_predicate, *unsup_attentions], weights=self.kl_layer_weights)
                elif self.enable_predicate_posterior == 'cross-entropy':
                    # No cross-entropy calculation for assume unsupervision
                    pass

            # Calculate weight.
                if self.dynamic_itl:
                    inv_mean_confidence = 1 - unsup_predicate.flatten().mean().detach() # Good example ~ 0, Bad ~ 1
                    self.itl_div: Tensor = (unsup_kl_loss.detach().norm() / (unsupLgen.detach().norm() + inv_mean_confidence))
                    self.itl_div = self.itl_div.clamp(max=1) * 0.1
            unsupLseg = (self.lr3 * unsupLgen) + (0 if self.disable_unsupervise_itl else (self.itl_div * unsup_kl_loss))

        # Optimization Step
        aggLseg = supLseg + unsupLseg
        self.manual_backward(aggLseg)
        opt_seg.step()
        lr_seg.step()

        # Logging
        self.log('train/gen_sup_loss', supLseg + unsupLseg)
        self.log('train/gen_loss', supLgen + unsupLgen)
        self.log('train/kl_loss', kl_loss + unsup_kl_loss)
        if self.dynamic_itl:
            self.log('train/itl_weight', self.itl_div)

        if not self.disable_discriminator:
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
        x, y_target = batch[self.pair_switch('x', 'x')], batch[self.pair_switch('y', 'y')]

        # Segmentation
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        # Performance evaluation
        # Loss calculation
        gen_loss = self.scribble_net.supervised_loss.forward(predicate, y_target)
        if not self.disable_discriminator:
            fake = self.scribble_net.discriminator.forward(attentions)
            dis_loss = self.scribble_net.generator_loss.forward(fake)
            # Generator loss
            self.log('val/gen_loss', dis_loss)

        # Supervise loss
        self.log('val/sup_loss', gen_loss)

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
                    if self.segmentor_gating_level >= 0:
                        fig = attention_predicate_plot(x, attentions, predicate, y, multi_class=False)
                        self.logger.experiment.log({'rollout': fig})
                        plt.close()
                    mask_img = to_wandb_semantic_segmentation(x, predicate, class_labels={0: 'bg', 1: 'vessel'}, ground_truth=y)
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})

    def forward(self, x):
        attentions, predicate = self.scribble_net.segmentor.predict(x, method='softmax')
        return attentions, predicate


class OCTAnetCoarse(LightningModule):
    weakly_supervise: bool = False

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        lr_scheduler = PolynomialLRDecay(optimizer=optim, max_decay_steps=self.num_epochs, end_learning_rate=1e-6, power=0.9)
        return [optim], [lr_scheduler]

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.model.predict(x, method='sigmoid')
        return y_hat

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        if self.weakly_supervise:
            y = batch['y_weak'][:, 1:]
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat[0], y.float())

        else:
            y = batch['y'][:, 1:] # Fully
            y_weak = batch['y_weak'][:, 1:] # For Centerline Branch
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat[0], y) + self.loss.forward(y_hat[1], y_weak)

        if self.wandb_logging:
            self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        if self.weakly_supervise:
            y = batch['y_weak'][:, 1:]
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat[0], y.float())
        else:
            y = batch['y'][:, 1:] # Fully
            y_weak = batch['y_weak'][:, 1:] # For Centerline Branch
            y_hat = self.forward(x)

            loss = self.loss.forward(y_hat[0], y) + self.loss.forward(y_hat[1], y_weak)

        if not self.evaluator is None:
            self.evaluator.register(y_hat[0], y) # Register pixel branch
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


class OCTAnetFine(LightningModule):
    weakly_supervise: bool = False

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = PolynomialLRDecay(optimizer=optim, max_decay_steps=self.num_epochs, end_learning_rate=1e-6, power=0.9)
        return [optim], [lr_scheduler]

    def forward(self, x: Tensor):
        with torch.no_grad():
            agg_map = self.coarse_model.forward(x)
        if self.weakly_supervise: # Assume single branch
            agg_coeff = self.backbone(x[:, :1, :, :], agg_map[0])
            prod_sal = self.adagg(agg_map[0], agg_coeff)
        else:
            agg_coeff = self.backbone(x[:, :1, :, :], agg_map[0], agg_map[1])
            prod_sal = self.adagg(agg_map[0], agg_map[1], agg_coeff)
        return prod_sal
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        if self.weakly_supervise:
            y = batch['y_weak'][:, 1:]
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat, y.float())
        else:
            y = batch['y'][:, 1:]
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat, y)
        
        if self.wandb_logging:
            self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        if self.weakly_supervise:
            y = batch['y_weak'][:, 1:]
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat, y.float())
        else:
            y = batch['y'][:, 1:]
            y_hat = self.forward(x)
            loss = self.loss.forward(y_hat, y)

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


class OCTAnetCoarseFullySupervise(OCTAnetCoarse):
    def __init__(self, is_training: bool,
                 pretrain: bool,
                 wandb_logging: bool,
                 weight_path: Optional[str] = None,
                 logging_frequency: int = 1,
                 evaluator: Optional[Evaluator] = None,
                 num_epochs: int = 200,
                 **kwargs):

        self.wandb_logging = wandb_logging
        self.frequency = logging_frequency
        self.evaluator = evaluator
        self.num_epochs =  num_epochs
        self.kwargs = kwargs
        super().__init__()

        self.is_training = is_training
        self.model = ResnestUnetParallelHead(num_classes=1,
                                 pretrain=pretrain,
                                 weight_path=weight_path)
        self.loss = ImageMseLoss()


        self.save_hyperparameters()


class OCTAnetFineFullySupervise(OCTAnetFine):
    def __init__(
        self, coarse_stage_path: str, channels: int = 64, pn_size: int = 3,
        kernel_size: int = 3, avg=0.0, std=0.1, lr=0.0005, weight_decay=0.001,
        num_epochs: int = 200, wandb_logging: bool = True, logging_frequency: int = 1 ,
        evaluator = None, **kwargs):
        self.wandb_logging = wandb_logging
        self.num_epochs = num_epochs
        self.frequency = logging_frequency
        self.evaluator = evaluator
        self.kwargs = kwargs
        super().__init__()

        self.backbone = basePC(channels, pn_size, kernel_size, avg, std)
        self.adagg = adaptive_aggregationPC(pn_size)
        coarse_path = Path(coarse_stage_path)
        if not coarse_path.is_file():
            # Try searching from alternative
            overrided_parent = kwargs.get('override_parent_dir', None)
            parent_level = kwargs.get('override_parent_level', None)
            dir_override = kwargs.get('model_dir_override', None)
            if overrided_parent:
                alternate_model_path = Path(overrided_parent).joinpath(dir_override, *coarse_path.parts[len(coarse_path.parts)-2:len(coarse_path.parts)])
                if not alternate_model_path.is_file():
                    assert kwargs.get('dryrun', False), f'Path to checkpoint is invalid {coarse_stage_path}, while use alternative path discovery {str(alternate_model_path)}'
                self.coarse_model = OCTAnetCoarseWeaklySupervise.load_from_checkpoint(alternate_model_path)
            else:
                assert kwargs.get('dryrun', False), f'Path to checkpoint is invalid, {coarse_stage_path}'
        else:
            self.coarse_model: OCTAnetCoarseFullySupervise = OCTAnetCoarseFullySupervise.load_from_checkpoint(coarse_stage_path)
        self.coarse_model.freeze()
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss = DiceLoss()

        self.save_hyperparameters()


class OCTAnetCoarseWeaklySupervise(OCTAnetCoarse):
    def __init__(self, is_training: bool,
                 pretrain: bool,
                 wandb_logging: bool,
                 weight_path: Optional[str] = None,
                 logging_frequency: int = 1,
                 evaluator: Optional[Evaluator] = None,
                 num_epochs: int = 200,
                 **kwargs):

        self.wandb_logging = wandb_logging
        self.frequency = logging_frequency
        self.evaluator = evaluator
        self.num_epochs =  num_epochs
        self.kwargs = kwargs
        super().__init__()
        self.weakly_supervise = True
        self.is_training = is_training
        self.model = ResnestUnetParallelHead(num_classes=1,
                                 pretrain=pretrain,
                                 weight_path=weight_path)
        self.loss = WeightedPartialCE(num_classes=1, manual=True)


        self.save_hyperparameters()


class OCTAnetFineWeaklySupervise(OCTAnetFine):
    def __init__(
        self, coarse_stage_path: str, channels: int = 64, pn_size: int = 3,
        kernel_size: int = 3, avg=0.0, std=0.1, lr=0.0005, weight_decay=0.001,
        num_epochs: int = 200, wandb_logging: bool = True, logging_frequency: int = 1 ,
        evaluator = None, **kwargs):
        self.wandb_logging = wandb_logging
        self.num_epochs = num_epochs
        self.frequency = logging_frequency
        self.evaluator = evaluator
        self.kwargs = kwargs
        super().__init__()

        self.backbone = baseC(channels, pn_size, kernel_size, avg, std)
        self.adagg = adaptive_aggregationC(pn_size)
        coarse_path = Path(coarse_stage_path)
        if not coarse_path.is_file():
            # Try searching from alternative
            overrided_parent = kwargs.get('override_parent_dir', None)
            parent_level = kwargs.get('override_parent_level', None)
            dir_override = kwargs.get('model_dir_override', None)
            if overrided_parent:
                alternate_model_path = Path(overrided_parent).joinpath(dir_override, *coarse_path.parts[len(coarse_path.parts)-2:len(coarse_path.parts)])
                if not alternate_model_path.is_file():
                    assert kwargs.get('dryrun', False), f'Path to checkpoint is invalid {coarse_stage_path}, while use alternative path discovery {str(alternate_model_path)}'
                self.coarse_model = OCTAnetCoarseWeaklySupervise.load_from_checkpoint(alternate_model_path)
            else:
                assert kwargs.get('dryrun', False), f'Path to checkpoint is invalid, {coarse_stage_path}'
        else:
            self.coarse_model: OCTAnetCoarseWeaklySupervise = OCTAnetCoarseWeaklySupervise.load_from_checkpoint(coarse_stage_path)
        self.coarse_model.freeze()
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss = WeightedPartialCE(num_classes=1, manual=True)

        self.weakly_supervise = True

        self.save_hyperparameters()


class OCTAnetCoarseAttentionGate(LightningModule):
    weakly_supervise: bool = False

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        lr_scheduler = PolynomialLRDecay(optimizer=optim, max_decay_steps=self.num_epochs, end_learning_rate=1e-6, power=0.9)
        return [optim], [lr_scheduler]

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.model.predict(x, method=self.prediction_act)
        return y_hat

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        kl_loss = torch.tensor(0.)
        y = batch['y'] # Fully
        y_weak = batch['y_weak'] # For Centerline Branch
        attentions, y_hat = self.forward(x)
        attentions_p, attentions_c = attentions # Split
        sup_loss = self.loss.forward(y_hat[0], y[:, :y_hat[0].shape[1]])
        sup_loss_c = self.loss.forward(y_hat[1], y_weak[:, :y_hat[0].shape[1]])

        kl_loss = 0
        kl_loss_c = 0
        if self.itl_div != 0 or self.dynamic_itl:
            if self.enable_predicate_posterior == 'attentions':
                kl_loss = self.kl_divergence.forward(attentions_p, weights=self.kl_layer_weights)
                kl_loss_c = self.kl_divergence.forward(attentions_c, weights=self.kl_layer_weights)
            elif self.enable_predicate_posterior == 'predicate':
                kl_loss = self.kl_divergence.forward([y_hat[0], *attentions_p], weights=self.kl_layer_weights)
                kl_loss_c = self.kl_divergence.forward([y_hat[1], *attentions_c], weights=self.kl_layer_weights)
            elif self.enable_predicate_posterior == 'cross-entropy':
                kl_loss = self.kl_divergence.forward([y, *attentions_p], weights=self.kl_layer_weights)
                kl_loss_c = self.kl_divergence.forward([y_weak, *attentions_c], weights=self.kl_layer_weights)
            else:
                raise NotImplementedError(f'{self.enable_predicate_posterior} is not a valid mode.')

        if self.dynamic_itl:
            self.itl_div: Tensor = (kl_loss.detach().norm() / (sup_loss.detach().norm()))
            self.itl_div = self.itl_div.clamp(max=1)
            self.itl_div_c: Tensor = (kl_loss_c.detach().norm() / (sup_loss_c.detach().norm()))
            self.itl_div_c = self.itl_div.clamp(max=1)

        loss =  (sup_loss) + (self.itl_div * kl_loss) + (sup_loss_c) + (self.itl_div_c * kl_loss_c)

        if self.wandb_logging:
            self.log('train/loss', loss.detach())
            self.log('train/sup_loss', sup_loss.detach() + sup_loss_c.detach())
        if kl_loss > 0:
            self.log('train/kl_loss', kl_loss.detach())
        if kl_loss_c > 0:
            self.log('train/kl_loss_c', kl_loss_c.detach())

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        y = batch['y'] # Fully
        y_weak = batch['y_weak'] # For Centerline Branch
        attentions, y_hat = self.forward(x)
        try:
            sup_loss = self.loss.forward(y_hat[0], y[:, :y_hat[0].shape[1]])
            sup_loss_c = self.loss.forward(y_hat[1], y_weak[:, :y_hat[0].shape[1]])
        except Exception as e:
            logging.error('Error while attempting to calculate loss')
            logging.error(f'{y_hat[0].shape}, {y.shape}')
            logging.error(f'{y_hat[1].shape}, {y_weak.shape}')
            logging.exception(e)
            raise e

        loss = sup_loss + sup_loss_c
        if y_hat.shape[2] == 1: # Grayscale is not directly supported by visualize fn
            mask = 1 - y_hat
            y_hat = torch.cat([mask, y_hat], dim=2)
        # Merge Pixel-Centerline
        y_hat_merge = torch.max(y_hat, dim=0).values
        y_merge = torch.any(rearrange([y, y_weak], 'k b c h w -> k b c h w'), dim=0).int()

        if not self.evaluator is None:
            try:
                self.evaluator.register(y_hat_merge, y_merge) # Register pixel branch
            except Exception as e:
                logging.error('Error while registering result.')
                logging.error(f'{y_hat.shape}, {y.shape}')
                logging.exception(e)
                raise e
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
        return {'loss': loss, 'x': x, 'y': y, 'y_weak': y_weak, 'predicate': re_y_hat, 'attentions': attentions, 'predicate_merged': y_hat_merge, 'y_merged': y_merge}

    def validation_epoch_end(self, outputs):
        if self.current_epoch % self.frequency == 0:
            if self.wandb_logging:
                masks = []
                samples = random.choices(outputs, k=5)
                for output in samples:
                    x, y, y_weak, predicate, attentions = output['x'], output['y'], output['y_weak'], output['predicate'], output['attentions']
                    p_att = attentions[0]
                    a_att = attentions[1]
                    if self.gating_level >= 0:
                        fig = attention_predicate_plot(x, p_att, predicate, y, multi_class=False)
                        self.logger.experiment.log({'rollout_pixel': fig})
                        plt.close()
                        fig = attention_predicate_plot(x, a_att, output['predicate_merged'], y_weak, multi_class=False)
                        self.logger.experiment.log({'rollout_coarse': fig})
                        plt.close()
                    mask_img = to_wandb_semantic_segmentation(x, output['predicate_merged'], class_labels={0: 'bg', 1: 'vessel'}, ground_truth=output['y_merged'])
                    masks.append(mask_img)
                self.logger.experiment.log({'predictions': masks})


class OCTAnetAttentionCoarseFullySupervise(OCTAnetCoarseAttentionGate):
    def __init__(self, is_training: bool,
                 pretrain: bool,
                 wandb_logging: bool,
                 weight_path: Optional[str] = None,
                 logging_frequency: int = 1,
                 evaluator: Optional[Evaluator] = None,
                 num_epochs: int = 200,
                 gating_level: int = 3,
                 kl_stop_gradient: bool = True,
                 interlayer_divergence_type: Literal['KLD', 'JSD'] = 'KLD',
                 interlayer_layer_contrib_weight: Optional[List[float]] = None,
                 enable_predicate_posterior: bool = True,
                 interlayer_divergence_weight: float = 0.2,
                 dynamic_itl: bool = False,
                 regulation_clip: float = 2,
                 num_classes: int = 2,
                 prediction_act: Literal['sigmoid', 'softmax'] = 'softmax',
                 **kwargs):

        self.wandb_logging = wandb_logging
        self.frequency = logging_frequency
        self.evaluator = evaluator
        self.num_epochs =  num_epochs
        self.kwargs = kwargs
        self.kl_layer_weights = interlayer_layer_contrib_weight
        self.enable_predicate_posterior = enable_predicate_posterior
        super().__init__()
        self.itl_div = interlayer_divergence_weight
        self.itl_div_c = interlayer_divergence_weight
        self.kl_divergence = InterlayerDivergence('mean', stop_gradient=kl_stop_gradient, divergence=interlayer_divergence_type)
        self.regulation_clip = regulation_clip
        self.dynamic_itl = dynamic_itl
        self.gating_level = gating_level
        self.num_classes = num_classes
        self.prediction_act = prediction_act

        self.is_training = is_training
        self.model = ResnestUnetParallelHeadAttentionGate(num_classes=num_classes,
                                 pretrain=pretrain,
                                 weight_path=weight_path,
                                 gating_leveL=gating_level)
        self.loss = ImageMseLoss()


        self.save_hyperparameters()


class OCTAnetAAGFine(LightningModule):
    weakly_supervise: bool = False

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = PolynomialLRDecay(optimizer=optim, max_decay_steps=self.num_epochs, end_learning_rate=1e-6, power=0.9)
        return [optim], [lr_scheduler]

    def forward(self, x: Tensor):
        with torch.no_grad():
            attentions, agg_map = self.coarse_model.forward(x)
        p_att, a_att = [], []
        if len(attentions[0]) > 0:
            p_att = [a[:, 1:] for a in upscale_tensors_like_2d(x, attentions[0])]
        if len(attentions[1]) > 0:
            a_att = [a[:, 1:] for a in upscale_tensors_like_2d(x, attentions[1])]
        p_start, c_start = 1, 1
        if agg_map[0].shape[1] == 1:
            p_start = 0
        if agg_map[1].shape[1] == 1:
            c_start = 0
        agg_coeff = self.backbone([x[:, :1, :, :], agg_map[0][:, p_start:], agg_map[1][:, c_start:], *p_att, *a_att][0:self.in_channels])
        max_prob = torch.max(agg_map[0][:, p_start:], agg_map[1][:, c_start:])
        prod_sal = self.adagg(max_prob, agg_coeff)
        return prod_sal
 
    def training_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        y = batch['y']
        y_hat = self.forward(x)
        loss = self.loss.forward(y_hat, y[:, 1:])
        
        if self.wandb_logging:
            self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        y = batch['y']
        y_hat = self.forward(x)
        loss = self.loss.forward(y_hat, y[:, 1:])

        if not self.evaluator is None:
            with torch.no_grad():
                mask = 1 - y_hat.detach()
                y_mask = torch.cat((mask, y_hat), dim=1)
            self.evaluator.register(y_mask, y)
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
    
    def test_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        y = batch['y']
        y_hat = self.forward(x)
        loss = self.loss.forward(y_hat, y[:, 1:])

        if not self.evaluator is None:
            with torch.no_grad():
                mask = 1 - y_hat.detach()
                y_mask = torch.cat((mask, y_hat), dim=1)
            self.evaluator.register(y_mask, y)
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
                self.log('test/loss', loss)
                self.log('test/acc', accur)
                self.log('test/auroc', auc_scr)
                self.log('test/dice', dice_coef)
                self.log('test/iou', iou_scr)
                self.log('test/kappa', kappa)
                self.log('test/specificity', spe)
                self.log('test/sensitivity', sen)
                self.log('test/fpr', fpr)
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


class OCTAnetFineAAGFullySupervise(OCTAnetAAGFine):
    def __init__(
        self, coarse_stage_path: str, in_channels: int, channels: int = 64, pn_size: int = 3,
        kernel_size: int = 3, avg=0.0, std=0.1, lr=0.0005, weight_decay=0.001,
        num_epochs: int = 200, wandb_logging: bool = True, logging_frequency: int = 1,
        evaluator = None, **kwargs):
        self.wandb_logging = wandb_logging
        self.num_epochs = num_epochs
        self.frequency = logging_frequency
        self.evaluator = evaluator
        self.kwargs = kwargs
        self.in_channels = in_channels
        super().__init__()

        self.backbone = baseMulti(in_channels, channels, pn_size, kernel_size, avg, std)
        self.adagg = adaptive_aggregationMulti(pn_size)
        coarse_path = Path(coarse_stage_path)
        if not coarse_path.is_file():
            # Try searching from alternative
            overrided_parent = kwargs.get('override_parent_dir', None)
            parent_level = kwargs.get('override_parent_level', None)
            dir_override = kwargs.get('model_dir_override', None)
            if overrided_parent:
                alternate_model_path = Path(overrided_parent).joinpath(dir_override, *coarse_path.parts[len(coarse_path.parts)-2:len(coarse_path.parts)])
                assert alternate_model_path.is_file(), f'Path to checkpoint is invalid {coarse_stage_path}, while use alternative path discovery {str(alternate_model_path)}'
                self.coarse_model = OCTAnetAttentionCoarseFullySupervise.load_from_checkpoint(alternate_model_path)
            else:
                raise Exception(f'Path to checkpoint is invalid, {coarse_stage_path}')
        else:
            self.coarse_model: OCTAnetAttentionCoarseFullySupervise = OCTAnetAttentionCoarseFullySupervise.load_from_checkpoint(coarse_stage_path)
        self.coarse_model.freeze()
        self.lr = lr
        self.weight_decay = weight_decay

        self.loss = DiceLoss()

        self.save_hyperparameters()


class CECSNetExperimentTemplate(LightningModule):

    def __init__(self,
                 wandb_logging: bool,
                 logging_frequency: int = 1,
                 evaluator: Optional[Evaluator] = None,
                 num_epochs: int = 200,
                 *args, **kwargs):
        super().__init__()
        self.wandb_logging = wandb_logging
        self.frequency = logging_frequency
        self.evaluator = evaluator
        self.num_epochs =  num_epochs

        self.loss = WeightedPartialCE(num_classes=1, manual=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        lr_scheduler = PolynomialLRDecay(optimizer=optim, max_decay_steps=self.num_epochs, end_learning_rate=1e-6, power=0.9)
        return [optim], [lr_scheduler]

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.model.predict(x, method='sigmoid')
        return y_hat

    def training_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        y = batch['y_weak'][:, 1:]
        y_hat = self.forward(x)
        loss = self.loss.forward(y_hat[:, 1:], y.float())

        if self.wandb_logging:
            self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x = batch['x']
        y = batch['y']
        y_weak = batch['y_weak'][:, 1:]
        y_hat = self.forward(x)
        loss = self.loss.forward(y_hat[:, 1:], y_weak.float())

        if not self.evaluator is None:
            self.evaluator.register(y_hat, y) # Register pixel branch
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


class CENetExperiment(CECSNetExperimentTemplate):

    def __init__(self, pretrain_model_path: str = 'resnet34-333f7ec4.pth', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = CE_Net(num_classes=2, num_channels=3, model_path=pretrain_model_path)

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        # Resize to match original
        shape = x.shape[2:]
        y = self.model(x)
        return FT.resize(img=y, size=shape, interpolation=FT.InterpolationMode.NEAREST)


class CSNetExperiment(CECSNetExperimentTemplate):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = CSNet(classes=2, channels=3)

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        # Resize to match original
        shape = x.shape[2:]
        y = self.model(x)
        return FT.resize(img=y, size=shape, interpolation=FT.InterpolationMode.NEAREST)