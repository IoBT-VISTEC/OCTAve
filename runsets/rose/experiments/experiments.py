# Describe experiment
from logging import log
from typing import Any, Dict, List, Literal, Union

from loguru import logger
import torch

from interfaces.experiments.rose import RoseDatamodule, RoseOcta500
from runsets.dataclass import (
    ExperimentConfig, OCTAnetCoarseExperimentModuleConfig, OCTAnetFineAttentionGateExperimentModuleConfig,
    OCTAnetFineExperimentModuleConfig, OCTAveExperimentModuleConfig,
    OCTAnetExperimentModuleConfig,
    ScribbleNetExperimentModuleConfig,
    OCTAnetCoarseAttentionGateExperimentModuleConfig,
    OCTAnetAttentionGateExperimentModuleConfig,
    UNetExperimentModuleConfig)

from runsets.rose.scribblenet import train_scribblenet
from runsets.rose.octanet import train_octanet_coarse, train_octanet_fine, test_octanet_fine
from runsets.utils import job_distribute
from utils.evaluation import Evaluator


@job_distribute
def submit_scribblenet_experiment(experimentation_config, **kwargs):

    jdx, rdx = kwargs.get('jdx', None), kwargs.get('rdx', None)

    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        import torch
        logger.info('Dryrunning experiment...')
        logger.info(f'jdx: {jdx}, rdx: {rdx}')
        logger.info(f'Devices: {torch.cuda.device_count()}')

    if type(experimentation_config) is ExperimentConfig and not dry_run:
        return train_scribblenet(
            experiment_config=experimentation_config.experiment_config,
            experiment_name=experimentation_config.experiment_name,
            fold=experimentation_config.fold,
            datamodule=experimentation_config.datamodule,
            random_seed=experimentation_config.random_seed,
            models_dir=experimentation_config.models_dir,
            wandb_project=experimentation_config.wandb_project,
            wandb_tags=experimentation_config.wandb_tags,
            num_epochs=experimentation_config.num_epochs,
            gpus=experimentation_config.gpus,
        )
    elif any([not p is None for p in (jdx, rdx)]):
        # Bound case
        if jdx > len(experimentation_config) - 1:
            logger.warning('Overallocation detected ...')
            return None
        experimentation_config: ExperimentConfig = experimentation_config[jdx]
        if not dry_run:
            return train_scribblenet(
                experiment_config=experimentation_config.experiment_config,
                experiment_name=experimentation_config.experiment_name,
                fold=experimentation_config.fold,
                datamodule=experimentation_config.datamodule,
                random_seed=experimentation_config.random_seed,
                models_dir=experimentation_config.models_dir,
                wandb_project=experimentation_config.wandb_project,
                wandb_tags=experimentation_config.wandb_tags,
                num_epochs=experimentation_config.num_epochs,
                gpus=[rdx],
            )
        else:
            logger.info(f"Dryrun: {experimentation_config.experiment_name}, job_rank: {jdx}, resource_rank: {rdx}")
            return None
    else:
        logger.error("Unbounded error")
        raise NotImplementedError('Unbounded case reached.')


@job_distribute
def submit_octanet_experiments(experiment_configs: List[ExperimentConfig], **kwargs):
    jdx, rdx = kwargs.get('jdx', None), kwargs.get('rdx', None)
    dry_run = kwargs.get('dry_run', False)
    experiment_config: ExperimentConfig = experiment_configs[jdx]

    # Coarse Stage
    coarse_stage_cfg = experiment_config.experiment_config.coarse_stage_config
    coarse_stage_checkpoint = train_octanet_coarse(
        experiment_config=coarse_stage_cfg,
        experiment_name=experiment_config.experiment_name,
        fold=experiment_config.fold,
        datamodule=experiment_config.datamodule,
        random_seed=experiment_config.random_seed,
        models_dir=experiment_config.models_dir,
        wandb_project=experiment_config.wandb_project,
        wandb_tags=experiment_config.wandb_tags,
        num_epochs=experiment_config.num_epochs,
        gpus=experiment_config.gpus,
        monitor=coarse_stage_cfg.monitor,
        dryrun=dry_run,
        )
    experiment_config.experiment_config.fine_stage_config.coarse_stage_path = coarse_stage_checkpoint
    fine_stage_cfg = experiment_config.experiment_config.fine_stage_config
    fine_stage_checkpoint = train_octanet_fine(
        experiment_config=fine_stage_cfg,
        experiment_name=experiment_config.experiment_name,
        fold=experiment_config.fold,
        datamodule=experiment_config.datamodule,
        random_seed=experiment_config.random_seed,
        models_dir=experiment_config.models_dir,
        wandb_project=experiment_config.wandb_project,
        wandb_tags=experiment_config.wandb_tags,
        num_epochs=experiment_config.num_epochs,
        gpus=experiment_config.gpus,
        monitor=fine_stage_cfg.monitor,
        dryrun=dry_run
    )
    return coarse_stage_checkpoint, fine_stage_checkpoint


@job_distribute
def submit_octanet_test(experiment_configs: List[ExperimentConfig], **kwargs):
    jdx, rdx = kwargs.get('jdx', None), kwargs.get('rdx', None)
    dry_run = kwargs.get('dry_run', False)
    experiment_config: ExperimentConfig = experiment_configs[jdx]

    fine_stage_checkpoint = test_octanet_fine(
        experiment_name=experiment_config.experiment_name,
        datamodule=experiment_config.datamodule,
        random_seed=experiment_config.random_seed,
        models_dir=experiment_config.models_dir,
        wandb_project=experiment_config.wandb_project,
        wandb_tags=experiment_config.wandb_tags,
        gpus=experiment_config.gpus,
        dryrun=dry_run
    )

    return fine_stage_checkpoint


class ScribbleNetRoseFullyExperimentConfig:

    def __init__(
        self, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        rose_modal = ['ROSE-1/SVC', 'ROSE-1/DVC', 'ROSE-1/SVC-DVC', 'ROSE-2']
        rose = {k: v for k, v in zip(rose_modal, list(map(lambda x: self.ROSE(x), rose_modal)))}

        self.experiments = [

            ExperimentConfig(
                experiment_config=ScribbleNetExperimentModuleConfig(
                    interlayer_divergence_weight=0.0, dynamic_itl=True, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False)
                    , instance_noise=False
                    , label_noise=False
                    ),
                experiment_name=f'SCRIBBLE-ROSE1-SVC-FULLY',
                datamodule=rose['ROSE-1/SVC'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['scribble-fully-compare', 'ROSE-1', 'SVC'],
                num_epochs=self.num_epochs,
                models_dir=f'models/scribble_fully_supervised/',
            ),
            ExperimentConfig(
                experiment_config=ScribbleNetExperimentModuleConfig(
                    interlayer_divergence_weight=0.0, dynamic_itl=True, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False)
                    , instance_noise=False
                    , label_noise=False
                    ),
                experiment_name=f'SCRIBBLE-ROSE1-DVC-FULLY',
                datamodule=rose['ROSE-1/DVC'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['scribble-fully-compare', 'ROSE-1', 'DVC'],
                num_epochs=self.num_epochs,
                models_dir=f'models/scribble_fully_supervised/',
            ),
            ExperimentConfig(
                experiment_config=ScribbleNetExperimentModuleConfig(
                    interlayer_divergence_weight=0.0, dynamic_itl=True, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False)
                    , instance_noise=False
                    , label_noise=False
                    ),
                experiment_name=f'SCRIBBLE-ROSE1-SVC-DVC-FULLY',
                datamodule=rose['ROSE-1/SVC-DVC'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['scribble-fully-compare', 'ROSE-1', 'SVC-DVC'],
                num_epochs=self.num_epochs,
                models_dir=f'models/scribble_fully_supervised/',
            ),
            ExperimentConfig(
                experiment_config=ScribbleNetExperimentModuleConfig(
                    interlayer_divergence_weight=0.0, dynamic_itl=True, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False)
                    , instance_noise=False
                    , label_noise=False
                    ),
                experiment_name=f'SCRIBBLE-ROSE2-FULLY',
                datamodule=rose['ROSE-2'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['scribble-fully-compare', 'ROSE-2'],
                num_epochs=self.num_epochs,
                models_dir=f'models/scribble_fully_supervised/',
            ),

        ][start_index:end_index]

        # In case some experiment get rescheduled.
        # self.experiments = list(filter(lambda exp: exp.experiment_name in fix_list, self.experiments))

    def ROSE(self, modality: Literal['ROSE-1/SVC', 'ROSE-1/DVC', 'ROSE-1/SVC-DVC', 'ROSE-2']):
        datamodule = RoseDatamodule(
            source_dataset_path='data',
            rose_modal=modality,
            rose_label_modality='gt',
            rose_input_augmentation=False,
            vessel_capillary_gt='both',
            weakly_gt=False,
        )
        return datamodule


    @staticmethod
    def prepare_data(datamodules: Any):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        pass

    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class OCTANetAttentionGateRoseFullyExperimentConfig:


    def __init__(
        self, coarse_num_epochs: int = 300, fine_num_epochs: int = 100,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        gating_level = 3, ssds: bool = True, fine_in_channels: int = 9,
        num_classes: int = 2, prediction_act: Literal['sigmoid', 'softmax'] = 'softmax',
        monitor: str = 'val/dice'
        ):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function

        Note:
        Turn the `gating_level` <= 0, the `ssds` = False, and `fine_in_channels` = 3 will resultd in a least-modified OCTA-net.
        Turn the `num_classes` = 1, the `prediction_act` = 'sigmoid' will replicate OCTA-net architecture entirely.
        """
        self.coarse_num_epochs = coarse_num_epochs
        self.fine_num_epochs = fine_num_epochs
        self.get_parallel_job = get_parallel_job
        self.gating_level = gating_level
        self.ssds = ssds
        self.fine_in_channels = fine_in_channels
        self.num_classes = num_classes
        self.prediction_act = prediction_act
        self.monitor = monitor

        rose_modal = ['ROSE-1/SVC', 'ROSE-1/DVC', 'ROSE-1/SVC-DVC', 'ROSE-2']
        rose = {k: v for k, v in zip(rose_modal, list(map(lambda x: self.ROSE(x), rose_modal)))}

        self.experiments = [

            ExperimentConfig(
                experiment_config= OCTAnetAttentionGateExperimentModuleConfig(
                    coarse_stage_config=OCTAnetCoarseAttentionGateExperimentModuleConfig(
                    dynamic_itl=self.ssds, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False
                    , tolerance_kernel=(1, 1))
                    , gating_level=self.gating_level
                    , num_epochs=coarse_num_epochs
                    , num_classes=self.num_classes
                    , prediction_act=self.prediction_act
                    , monitor=self.monitor
                    ),
                    fine_stage_config=OCTAnetFineAttentionGateExperimentModuleConfig(
                        evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False
                        , tolerance_kernel=(1, 1)),
                        num_epochs=fine_num_epochs, logging_frequency=base_logging_frequency, in_channels=self.fine_in_channels
                    )
                ),
                experiment_name=f'OCTANet{"AAG" if self.gating_level > 0 else ""}-ROSE1-SVC-FULLY',
                datamodule=rose['ROSE-1/SVC'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['octanetaag-fully-compare', 'ROSE-1', 'SVC'],
                num_epochs=self.coarse_num_epochs,
                models_dir=f'models/octanet{"aag" if self.gating_level > 0 else ""}_fully_supervised/',
            ),
            ExperimentConfig(
                experiment_config= OCTAnetAttentionGateExperimentModuleConfig(
                    coarse_stage_config=OCTAnetCoarseAttentionGateExperimentModuleConfig(
                    dynamic_itl=self.ssds, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False)
                    , gating_level=self.gating_level
                    , num_classes=self.num_classes
                    , prediction_act=self.prediction_act
                    , monitor=self.monitor
                    ),
                fine_stage_config=OCTAnetFineAttentionGateExperimentModuleConfig(
                        evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False),
                        num_epochs=fine_num_epochs, logging_frequency=base_logging_frequency, in_channels=self.fine_in_channels
                        , monitor=self.monitor
                )
                ),
                experiment_name=f'OCTANet{"AAG" if self.gating_level > 0 else ""}-ROSE1-DVC-FULLY',
                datamodule=rose['ROSE-1/DVC'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['octanetaag-fully-compare', 'ROSE-1', 'DVC'],
                num_epochs=self.coarse_num_epochs,
                models_dir=f'models/octanet{"aag" if self.gating_level > 0 else ""}_fully_supervised/',
            ),
            ExperimentConfig(
                experiment_config= OCTAnetAttentionGateExperimentModuleConfig(
                    coarse_stage_config=OCTAnetCoarseAttentionGateExperimentModuleConfig(
                    dynamic_itl=self.ssds, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False,
                    tolerance_kernel=(1, 1))
                    , gating_level=self.gating_level
                    , num_classes=self.num_classes
                    , prediction_act=self.prediction_act
                    , monitor=self.monitor
                    ),
                fine_stage_config=OCTAnetFineAttentionGateExperimentModuleConfig(
                        evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False),
                        num_epochs=fine_num_epochs, logging_frequency=base_logging_frequency, in_channels=self.fine_in_channels
                        , monitor=self.monitor
                )
                ),
                experiment_name=f'OCTANet{"AAG" if self.gating_level > 0 else ""}-ROSE1-SVC-DVC-FULLY',
                datamodule=rose['ROSE-1/SVC-DVC'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['octanetaag-fully-compare', 'ROSE-1', 'SVC-DVC'],
                num_epochs=self.coarse_num_epochs,
                models_dir=f'models/octanet{"aag" if self.gating_level > 0 else ""}_fully_supervised/',
            ),
            ExperimentConfig(
                experiment_config= OCTAnetAttentionGateExperimentModuleConfig(
                    coarse_stage_config=OCTAnetCoarseAttentionGateExperimentModuleConfig(
                    dynamic_itl=self.ssds, logging_frequency=base_logging_frequency,
                    disable_discriminator=True, pairing_option='paired'
                    , evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False
                    , tolerance_kernel=(3, 3))
                    , gating_level=self.gating_level
                    , num_classes=self.num_classes
                    , prediction_act=self.prediction_act
                    , monitor=self.monitor
                    ),
                fine_stage_config=OCTAnetFineAttentionGateExperimentModuleConfig(
                        evaluator=Evaluator(tolerance='gt-tolerance', thresholding_mode='adaptive', tolerance_classes=[1], num_classes=1, collapsible=False),
                        num_epochs=fine_num_epochs, logging_frequency=base_logging_frequency, in_channels=self.fine_in_channels
                        , monitor=self.monitor
                )
                ),
                experiment_name=f'OCTANet{"AAG" if self.gating_level > 0 else ""}-ROSE2-FULLY',
                datamodule=rose['ROSE-2'],
                fold=0,
                wandb_project='octave-run',
                wandb_tags=['octanetaag-fully-compare', 'ROSE-2'],
                num_epochs=self.coarse_num_epochs,
                models_dir=f'models/octanet{"aag" if self.gating_level > 0 else ""}_fully_supervised/',
            ),

        ][start_index:end_index]

        # In case some experiment get rescheduled.
        # self.experiments = list(filter(lambda exp: exp.experiment_name in fix_list, self.experiments))

    def ROSE(self, modality: Literal['ROSE-1/SVC', 'ROSE-1/DVC', 'ROSE-1/SVC-DVC', 'ROSE-2']):
        datamodule = RoseDatamodule(
            source_dataset_path='data',
            rose_modal=modality,
            rose_label_modality='gt',
            rose_input_augmentation=False,
            vessel_capillary_gt='both',
            weakly_gt=True,
        )
        return datamodule


    @staticmethod
    def prepare_data(datamodules: Any):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        pass

    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list