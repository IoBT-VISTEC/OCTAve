from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from interfaces.experiments.octa500 import OCTA500Classification, OCTA500Datamodule


@dataclass
class ClassificationOCTAVEExperimentConfig:

    is_training: bool
    pretrain: bool
    supervise_regulation: bool
    wandb_logging: bool
    num_classes: int
    class_weight: list
    weight_path: Optional[str] = 'resnest50-528c19ca.pth'
    segmentor_gating_level: int = 3
    kl_stop_gradient: bool = True
    interlayer_divergence_type: Literal['KLD', 'JSD'] = 'KLD'
    interlayer_layer_contrib_weight: list = field(default_factory=lambda: [1, 0.8, 0.6, 0.4])
    enable_predicate_posterior: Literal['attentions', 'predicate'] = 'predicate'
    logging_frequency: int = 5
    interlayer_divergence_weight: float = 0.1
    regulation_mode: Literal['origin', 'swap'] = 'swap'
    regulation_clip: float = 0.1
    dynamic_itl: bool = True
    mode: Literal['classic', 'ae-squash', 'ae-extract'] = 'classic'
    ild_start_epoch: int = 0
    encoder_gating: bool = False


@dataclass
class ExperimentConfig:
    """Experimentation Configuration
    """
    experiment_config: ClassificationOCTAVEExperimentConfig
    experiment_name: str
    fold: int
    datamodule: OCTA500Classification
    random_seed: int = 50
    models_dir: str = 'models/'
    wandb_project: str = 'octave-classification'
    wandb_tags: list = field(default_factory=lambda: ['octa500'])
    num_epochs: int = 300
    gpus: list = field(default_factory=lambda: [0])
