from cProfile import label
from pathlib import Path
from typing import Dict, List, Literal, Union

from interfaces.experiments.octa500 import OCTA500Classification
from runsets.classification.experiments.dataclass import ExperimentConfig, ClassificationOCTAVEExperimentConfig
from runsets.classification.octave import train_octave
from runsets.utils import job_distribute


@job_distribute
def submit_octave_experiment(experimentation_config, **kwargs):

    return train_octave(
        experiment_config=experimentation_config.experiment_config,
        experiment_name=experimentation_config.experiment_name,
        fold=experimentation_config.fold,
        datamodule=experimentation_config.datamodule,
        random_seed=experimentation_config.random_seed,
        models_dir=experimentation_config.models_dir,
        wandb_project=experimentation_config.wandb_project,
        wandb_tags=experimentation_config.wandb_tags,
        num_epochs=experimentation_config.num_epochs,
        gpus=[0],
    )


class OCTAVEOCTA500ClassificationExperimentConfig:


    def __init__(
        self, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        base_logging_frequency: int = 10, mode: Literal['classic', 'classic-gating', 'ae-squash', 'ae-extract'] = 'classic',
        ild_enable: bool = False, ild_start_epoch: int = 0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.num_epochs = num_epochs

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        self.label_3m = Path('data/OCTA-500/OCTA_3M/Text labels.xlsx')
        self.label_6m = Path('data/OCTA-500/OCTA_6M/Text labels.xlsx')
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA500('OCTA_3M', x, self.label_3m), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA500('OCTA_6M', x, self.label_6m), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]
        self.weight_3m = [1., 1.]
        # self.weight_6m = [0.61904, 1.31007, 1.60952] # NC-AMD-DR
        # self.weight_6m = [0.69230, 1.8] # NC-DR
        # self.weight_6m = [0.73626374, 1.55813953] # NC-AMD
        self.weight_6m = [1.64835165, 0.71770335] # NC - DISEASE

        self.mode = mode
        self.encoding_gating = True if self.mode == 'classic-gating' else False
        self.enable_predicate_posterior = 'attentions' if self.mode == 'classic-gating' else 'predicate'

        self.experiments = [
            # ExperimentConfig(
            #     datamodule=octa3m['FULL'],
            #     experiment_name='octave-classification/3M/FULL',
            #     random_seed=50,
            #     num_epochs=num_epochs,
            #     fold=0,
            #     experiment_config=ClassificationOCTAVEExperimentConfig(
            #         is_training=True, pretrain=True, supervise_regulation=True, wandb_logging=True,
            #         num_classes=2, class_weight=self.weight_3m, logging_frequency=base_logging_frequency,
            #         dynamic_itl=False, interlayer_divergence_weight=0.1, interlayer_layer_contrib_weight=[1, 0.8, 0.6, 0.4]
            #     )
            # ),
            # ExperimentConfig(
            #     datamodule=octa3m['ILM_OPL'],
            #     experiment_name='octave-classification/3M/ILM_OPL',
            #     random_seed=50,
            #     num_epochs=num_epochs,
            #     fold=0,
            #     experiment_config=ClassificationOCTAVEExperimentConfig(
            #         is_training=True, pretrain=True, supervise_regulation=True, wandb_logging=True,
            #         num_classes=2, class_weight=self.weight_3m, logging_frequency=base_logging_frequency,
            #         dynamic_itl=False, interlayer_divergence_weight=0.1, interlayer_layer_contrib_weight=[1, 0.8, 0.6, 0.4]
            #     )
            # ),
            # ExperimentConfig(
            #     datamodule=octa3m['OPL_BM'],
            #     experiment_name='octave-classification/3M/OPL_BM',
            #     random_seed=50,
            #     num_epochs=num_epochs,
            #     fold=0,
            #     experiment_config=ClassificationOCTAVEExperimentConfig(
            #         is_training=True, pretrain=True, supervise_regulation=True, wandb_logging=True,
            #         num_classes=2, class_weight=self.weight_3m, logging_frequency=base_logging_frequency,
            #         dynamic_itl=False, interlayer_divergence_weight=0.1, interlayer_layer_contrib_weight=[1, 0.8, 0.6, 0.4]
            #     )
            # ),
            ExperimentConfig(
                datamodule=octa6m['FULL'],
                experiment_name=f'octave-classification/6M/FULL_{mode}_ild={ild_enable}_ildepoch={ild_start_epoch}',
                random_seed=50,
                num_epochs=num_epochs,
                fold=0,
                experiment_config=ClassificationOCTAVEExperimentConfig(
                    is_training=True, pretrain=True, supervise_regulation=True, wandb_logging=True,
                    num_classes=2, class_weight=self.weight_6m, logging_frequency=base_logging_frequency,
                    dynamic_itl=ild_enable, interlayer_divergence_weight=0.0, interlayer_layer_contrib_weight=[1, 0.8, 0.6, 0.4],
                    mode=mode, ild_start_epoch=ild_start_epoch, encoder_gating=self.encoding_gating, enable_predicate_posterior=self.enable_predicate_posterior
                )
            ),
            ExperimentConfig(
                datamodule=octa6m['ILM_OPL'],
                experiment_name=f'octave-classification/6M/ILM_OPL_{mode}_{ild_enable}_ildepoch={ild_start_epoch}',
                random_seed=50,
                num_epochs=num_epochs,
                fold=0,
                experiment_config=ClassificationOCTAVEExperimentConfig(
                    is_training=True, pretrain=True, supervise_regulation=True, wandb_logging=True,
                    num_classes=2, class_weight=self.weight_6m, logging_frequency=base_logging_frequency,
                    dynamic_itl=ild_enable, interlayer_divergence_weight=0.0, interlayer_layer_contrib_weight=[1, 0.8, 0.6, 0.4],
                    mode=mode, ild_start_epoch=ild_start_epoch, encoder_gating=self.encoding_gating, enable_predicate_posterior=self.enable_predicate_posterior
                )
            ),
            ExperimentConfig(
                datamodule=octa6m['OPL_BM'],
                experiment_name=f'octave-classification/6M/OPL_BM_{mode}_ild={ild_enable}_ildepoch={ild_start_epoch}',
                random_seed=50,
                num_epochs=num_epochs,
                fold=0,
                experiment_config=ClassificationOCTAVEExperimentConfig(
                    is_training=True, pretrain=True, supervise_regulation=True, wandb_logging=True,
                    num_classes=2, class_weight=self.weight_6m, logging_frequency=base_logging_frequency,
                    dynamic_itl=ild_enable, interlayer_divergence_weight=0.0, interlayer_layer_contrib_weight=[1, 0.8, 0.6, 0.4],
                    mode=mode, ild_start_epoch=ild_start_epoch, encoder_gating=self.encoding_gating, enable_predicate_posterior=self.enable_predicate_posterior
                )
            ),
        ][start_index:end_index]


    def OCTA500(self, modality: Literal['OCTA_3M', 'OCTA_6M'], depth: Literal['FULL', 'ILM_OPL', 'OPL_BM'], labelpath: str):
        datamodule = OCTA500Classification(
            datapath='./data/OCTA-500',
            modality=modality,
            labelpath=labelpath,
            depth=depth,
            dump_path='data/OCTA-500-CL-NC-AB',
        )
        return datamodule

    def __len__(self):
        return len(self.experiments)

    def __iter__(self):
        for exp in self.experiments:
            yield exp