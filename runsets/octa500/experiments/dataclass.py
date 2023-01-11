from dataclasses import dataclass, field
from interfaces.experiments.octa500 import OCTA500Datamodule
from typing import Literal, Union

import torch
from utils.evaluation import Evaluator

@dataclass
class OCTAveExperimentModuleConfig:
    """OCTAve Experimentation Module Configuration
    Default is OCTAVE-ILD 3M Config
    """
    is_training: bool = True
    pretrain: bool = True
    supervise_regulation: bool = True
    base_filters: int = 64
    wandb_logging: bool = True
    weight_path: str = 'resnest50-528c19ca.pth'
    evaluator: Evaluator = Evaluator(tolerance=None, thresholding_mode='adaptive', tolerance_classes=[2], num_classes=1, collapsible=False)
    logging_frequency: int = 5
    interlayer_divergence_weight: float = 0.0
    enable_predicate_posterior: Literal ['attentions', 'predicate', 'cross-entropy'] = 'predicate'
    kl_stop_gradient: bool = True
    instance_noise: bool = True
    label_noise: bool = True
    regulation_mode: Literal['origin', 'swap'] = 'swap'
    regulation_clip: float = 2
    dynamic_itl: bool = False
    interlayer_divergense_type: Literal['KLD', 'JSD'] = 'KLD'
    interlayer_layer_contrib_weight: list = field(default_factory=lambda: [1, 0.8, 0.6, 0.4])
    disable_discriminator: bool = False
    disable_weight_on_bg: bool = False
    wpce_reduction: Literal['mean', 'sum'] ='mean'
    segmentor_gating_level: int = 3
    raw_input_shap: torch.Size = torch.ones(1, 3, 304, 304).shape
    mask_input_shape: torch.Size = torch.ones(1, 2, 304, 304).shape
    disable_unsupervise_itl: bool = False


@dataclass
class ScribbleNetExperimentModuleConfig:
    """OCTAve Experimentation Module Configuration
    Default is OCTAVE-ILD 3M Config
    """
    is_training: bool = True
    supervise_regulation: bool = True
    base_filters: int = 64
    wandb_logging: bool = True
    evaluator: Evaluator = Evaluator(tolerance=None, thresholding_mode='adaptive', tolerance_classes=[2], num_classes=1, collapsible=False)
    logging_frequency: int = 5
    interlayer_divergence_weight: float = 0.0
    enable_predicate_posterior: Literal ['attentions', 'predicate', 'cross-entropy'] = 'predicate'
    kl_stop_gradient: bool = True
    instance_noise: bool = True
    label_noise: bool = True
    regulation_mode: Literal['origin', 'swap'] = 'swap'
    regulation_clip: float = 2
    dynamic_itl: bool = True
    interlayer_divergense_type: Literal['KLD', 'JSD'] = 'KLD'
    interlayer_layer_contrib_weight: list = field(default_factory=lambda: [1, 0.8, 0.6, 0.4])
    disable_discriminator: bool = False
    disable_weight_on_bg: bool = False
    wpce_reduction: Literal['mean', 'sum'] ='mean'
    raw_input_shap: torch.Size = torch.ones(1, 3, 304, 304).shape
    mask_input_shape: torch.Size = torch.ones(1, 2, 304, 304).shape
    disable_unsupervise_itl: bool = False


@dataclass
class OCTAnetCoarseExperimentModuleConfig:
    """OCTAnet Coarse Stage Experiment
    """
    is_training: bool = True
    pretrain: bool = True
    wandb_logging: bool = True
    weight_path: str = 'resnest50-528c19ca.pth'
    evaluator: Evaluator = Evaluator(tolerance=None, thresholding_mode='adaptive', tolerance_classes=[2], num_classes=1, collapsible=False, bg_start_idx=0)
    logging_frequency: int = 5
    num_epochs: int = 300
    weakly_supervise: bool = False


@dataclass
class OCTAnetFineExperimentModuleConfig:
    """OCTAnet Fine Stage Experiment
    """
    coarse_stage_path: str = ''
    evaluator: Evaluator = Evaluator(tolerance=None, thresholding_mode='adaptive', tolerance_classes=[2], num_classes=1, collapsible=False, bg_start_idx=0)
    logging_frequency: int = 5
    num_epochs: int = 200
    channels: int = 64
    pn_size: int = 3
    kernel_size: int = 3
    avg: float = 0.0
    std: float = 0.1
    lr: float = 0.0005
    weight_decay: float = 0.0001
    wandb_logging: bool = True
    logging_frequency: int = 1
    weakly_supervise: bool = False


@dataclass
class OCTAnetExperimentModuleConfig:
    """OCTAnet Module Configuration
    """
    coarse_stage_config: OCTAnetCoarseExperimentModuleConfig
    fine_stage_config: OCTAnetFineExperimentModuleConfig


@dataclass
class ExperimentConfig:
    """Experimentation Configuration
    """
    experiment_config: Union[
        OCTAveExperimentModuleConfig, OCTAnetExperimentModuleConfig,
        ScribbleNetExperimentModuleConfig]
    experiment_name: str
    fold: int
    datamodule: OCTA500Datamodule
    random_seed: int = 50
    models_dir: str = 'models/'
    wandb_project: str = 'octa-net'
    wandb_tags: list = field(default_factory=lambda: ['octa500'])
    num_epochs: int = 300
    gpus: list = field(default_factory=lambda: [0])


@dataclass
class UNetExperimentModuleConfig:
    """OCTAve Experimentation Module Configuration
    Default is OCTAVE-ILD 3M Config
    """
    num_classes: int = 2
    base_filters: int = 32
    wandb_logging: bool = True
    evaluator: Evaluator = Evaluator(tolerance=None, thresholding_mode='adaptive', tolerance_classes=[2], num_classes=1, collapsible=False)
    logging_frequency: int = 5
    disable_weight_on_bg: bool = False
    wpce_reduction: Literal['mean', 'sum'] ='mean'
    input_shape: torch.Size = torch.ones(1, 3, 304, 304).shape
