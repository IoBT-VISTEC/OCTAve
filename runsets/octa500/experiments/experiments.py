# Describe experiment
from typing import Dict, List, Literal, Union

from loguru import logger
import torch

from interfaces.experiments.octa500 import OCTA500Datamodule
from runsets.dataclass import (
    ExperimentConfig, OCTAnetCoarseExperimentModuleConfig,
    OCTAnetFineExperimentModuleConfig, OCTAveExperimentModuleConfig,
    OCTAnetExperimentModuleConfig,
    ScribbleNetExperimentModuleConfig,
    UNetExperimentModuleConfig, CECSNetExperimentModuleConfig)
from runsets.octa500.octave import train_octave
from runsets.octa500.octanet import train_octanet_coarse, train_octanet_fine
from runsets.octa500.scribblenet import train_scribblenet
from runsets.octa500.unet import train_unet
from runsets.octa500.cenet import train_cenet
from runsets.octa500.csnet import train_csnet
from runsets.utils import job_distribute


@job_distribute
def submit_octave_experiment(experimentation_config, **kwargs):

    jdx, rdx = kwargs.get('jdx', None), kwargs.get('rdx', None)

    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        import torch
        logger.info('Dryrunning experiment...')
        logger.info(f'jdx: {jdx}, rdx: {rdx}')
        logger.info(f'Devices: {torch.cuda.device_count()}')

    if type(experimentation_config) is ExperimentConfig and not dry_run:
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
            gpus=experimentation_config.gpus,
        )
    elif any([not p is None for p in (jdx, rdx)]):
        # Bound case
        if jdx > len(experimentation_config) - 1:
            logger.warning('Overallocation detected ...')
            return None
        experimentation_config: ExperimentConfig = experimentation_config[jdx]
        if not dry_run:
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
                gpus=[rdx],
            )
        else:
            logger.info(f"Dryrun: {experimentation_config.experiment_name}, job_rank: {jdx}, resource_rank: {rdx}")
            return None
    else:
        logger.error("Unbounded error")
        raise NotImplementedError('Unbounded case reached.')


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
        dryrun=dry_run
    )
    return coarse_stage_checkpoint, fine_stage_checkpoint


@job_distribute
def submit_unet_experiment(experimentation_config, **kwargs):

    jdx, rdx = kwargs.get('jdx', None), kwargs.get('rdx', None)

    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        import torch
        logger.info('Dryrunning experiment...')
        logger.info(f'jdx: {jdx}, rdx: {rdx}')
        logger.info(f'Devices: {torch.cuda.device_count()}')

    if type(experimentation_config) is ExperimentConfig and not dry_run:
        return train_unet(
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
            return train_unet(
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
def submit_cenet_experiment(experimentation_config, **kwargs):

    jdx, rdx = kwargs.get('jdx', None), kwargs.get('rdx', None)

    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        import torch
        logger.info('Dryrunning experiment...')
        logger.info(f'jdx: {jdx}, rdx: {rdx}')
        logger.info(f'Devices: {torch.cuda.device_count()}')

    if type(experimentation_config) is ExperimentConfig and not dry_run:
        return train_cenet(
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
            return train_cenet(
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
def submit_csnet_experiment(experimentation_config, **kwargs):

    jdx, rdx = kwargs.get('jdx', None), kwargs.get('rdx', None)

    dry_run = kwargs.get('dry_run', False)
    if dry_run:
        import torch
        logger.info('Dryrunning experiment...')
        logger.info(f'jdx: {jdx}, rdx: {rdx}')
        logger.info(f'Devices: {torch.cuda.device_count()}')

    if type(experimentation_config) is ExperimentConfig and not dry_run:
        return train_csnet(
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
            return train_csnet(
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


class OCTAVEOCTA500ExperimentConfig:


    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        self.experiments = [

            # # Baseline OCTAVE 3M
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-3M-FULL-BASELINE-FOLD_{fold}',
            #         datamodule=octa3m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '3m-3m-full', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-3M-ILM_OPL-BASELINE-FOLD_{fold}',
            #         datamodule=octa3m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-3M-OPL_BM-BASELINE-FOLD_{fold}',
            #         datamodule=octa3m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],

            # # Baseline OCTAVE 6M
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
            #             logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-FULL-BASELINE-FOLD_{fold}',
            #         datamodule=octa6m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
            #             logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-ILM_OPL-BASELINE-FOLD_{fold}',
            #         datamodule=octa6m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
            #             logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-OPL_BM-BASELINE-FOLD_{fold}',
            #         datamodule=octa6m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],

            # OCTAVE-ILD 3M
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'OCTAVE-3M-FULL-ILD-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octave-ild', '3m-3m-full', 'origin', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'OCTAVE-3M-ILM_OPL-ILD-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octave-ild', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'OCTAVE-3M-OPL_BM-ILD-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octave-ild', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # OCTAVE-ILD 6M
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-FULL-ILD-FOLD_{fold}',
            #         datamodule=octa6m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-ILM_OPL-ILD-FOLD_{fold}',
            #         datamodule=octa6m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-OPL_BM-ILD-FOLD_{fold}',
            #         datamodule=octa6m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],

            # # OCTAVE-ILD 3M NO ITL ON UNSUPERVISED
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             logging_frequency=base_logging_frequency, disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-3M-FULL-ILD-FOLD_{fold}',
            #         datamodule=octa3m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '3m-3m-full', 'origin', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             logging_frequency=base_logging_frequency, disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-3M-ILM_OPL-ILD-FOLD_{fold}',
            #         datamodule=octa3m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             logging_frequency=base_logging_frequency, disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-3M-OPL_BM-ILD-FOLD_{fold}',
            #         datamodule=octa3m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],

            # # OCTAVE-ILD 6M NO ITL ON UNSUPERVISED
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency,
            #             disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-6M-FULL-ILD-FOLD_{fold}',
            #         datamodule=octa6m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-full', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency,
            #             disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-6M-ILM_OPL-ILD-FOLD_{fold}',
            #         datamodule=octa6m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency,
            #             disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-6M-OPL_BM-ILD-FOLD_{fold}',
            #         datamodule=octa6m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
        ][start_index:end_index]


    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule

    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule

    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)

    def __len__(self):
        return len(self.experiments)

    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class OCTAnetOCTA500ExperimentConfig:


    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        self.experiments = [

            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            )
                        ),
                    experiment_name=f'OCTAnet-FULLY-3M-FULL-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '3m-3m-full'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/fully-supervised/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            )
                        ),
                    experiment_name=f'OCTAnet-FULLY-3M-ILM_OPL-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '3m-3m-ilm_opl'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/fully-supervised/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            )
                        ),
                    experiment_name=f'OCTAnet-FULLY-3M-OPL_BM-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '3m-3m-opl_bm', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/fully-supervised/',
                ) for fold in range(self.n_folds)
            ],

            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            )
                        ),
                    experiment_name=f'OCTAnet-FULLY-6M-FULL-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '6m-6m-full', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/fully-supervised/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            )
                        ),
                    experiment_name=f'OCTAnet-FULLY-6M-ILM_OPL-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '6m-6m-ilm_opl', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/fully-supervised/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs
                            )
                        ),
                    experiment_name=f'OCTAnet-FULLY-6M-OPL_BM-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '6m-6m-opl_bm', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/fully-supervised/',
                ) for fold in range(self.n_folds)
            ],

        ][start_index:end_index]


    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=False,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=False,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)


    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class OCTAnetWeaklyOCTA500ExperimentConfig:


    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        self.experiments = [

            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            )
                        ),
                    experiment_name=f'OCTAnet-WEAKLY-3M-FULL-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '3m-3m-full'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/octanet-weakly-supervised-scribble-{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            )
                        ),
                    experiment_name=f'OCTAnet-WEAKLY-3M-ILM_OPL-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '3m-3m-ilm_opl'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/octanet-weakly-supervised-scribble-{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            )
                        ),
                    experiment_name=f'OCTAnet-WEAKLY-3M-OPL_BM-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '3m-3m-opl_bm', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/octanet-weakly-supervised-scribble-{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            )
                        ),
                    experiment_name=f'OCTAnet-WEAKLY-6M-FULL-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '6m-6m-full', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/octanet-weakly-supervised-scribble-{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            )
                        ),
                    experiment_name=f'OCTAnet-WEAKLY-6M-ILM_OPL-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '6m-6m-ilm_opl', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/octanet-weakly-supervised-scribble-{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAnetExperimentModuleConfig(
                            coarse_stage_config=OCTAnetCoarseExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            ),
                            fine_stage_config=OCTAnetFineExperimentModuleConfig(
                                logging_frequency=base_logging_frequency,
                                num_epochs=num_epochs,
                                weakly_supervise=True,
                            )
                        ),
                    experiment_name=f'OCTAnet-WEAKLY-6M-OPL_BM-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['octa-net', '6m-6m-opl_bm', ],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/octanet-weakly-supervised-scribble-{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

        ][start_index:end_index]

        # In case some experiment get rescheduled.
        # fix_list = [
        # ]
        # self.experiments = list(filter(lambda exp: exp.experiment_name in fix_list, self.experiments))


    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)


    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class ScribbleNetOCTA500ExperimentConfig:


    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        self.experiments = [

            # Baseline SCRIBBLE 3M
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-3M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-baseline', '3m-3m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-3M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-baseline', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-3M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-baseline', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # Baseline SCRIBBLE 6M
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-6M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-baseline', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-6M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-baseline', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-6M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-baseline', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # ILD
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-3M-FULL-ILD-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-ild', '3m-3m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-3M-ILM_OPL-ILD-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-ild', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-3M-OPL_BM-ILD-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-ild', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # ILD SCRIBBLE 6M
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-6M-FULL-ILD-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-ild', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-6M-ILM_OPL-ILD-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-ild', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=ScribbleNetExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'SCRIBBLE-6M-OPL_BM-ILD-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['scribble-ild', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

        ][start_index:end_index]

        # In case some experiment get rescheduled.
        # self.experiments = list(filter(lambda exp: exp.experiment_name in fix_list, self.experiments))

    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule

    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)


    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class UNetOCTA500ExperimentConfig:

    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        self.experiments = [

            # Baseline UNET 3M
            *[
                ExperimentConfig(
                    experiment_config=UNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'UNET-3M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['unet-baseline', '3m-3m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/unet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=UNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'UNET-3M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['unet-baseline', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/unet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=UNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'UNET-3M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['unet-baseline', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/unet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # Baseline UNET 6M
            *[
                ExperimentConfig(
                    experiment_config=UNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'UNET-6M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['unet-baseline', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/unet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=UNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'UNET-6M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['unet-baseline', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/unet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=UNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'UNET-6M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['unet-baseline', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/unet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

        ][start_index:end_index]

        # In case some experiment get rescheduled.
        # fix_list = []
        # self.experiments = list(filter(lambda exp: exp.experiment_name in fix_list, self.experiments))

    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule

    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)


    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class CENetOCTA500ExperimentConfig:

    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        self.experiments = [

            # Baseline CENet 3M
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CENet-3M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['cenet-baseline', '3m-3m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/cenet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CENet-3M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['cenet-baseline', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/cenet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CENet-3M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['cenet-baseline', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/cenet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # Baseline CENet 6M
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CENet-6M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['cenet-baseline', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/cenet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CENet-6M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['cenet-baseline', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/cenet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CENet-6M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['cenet-baseline', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/cenet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

        ][start_index:end_index]

        # In case some experiment get rescheduled.
        # fix_list = []
        # self.experiments = list(filter(lambda exp: exp.experiment_name in fix_list, self.experiments))

    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule

    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)


    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class CSNetOCTA500ExperimentConfig:

    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        self.experiments = [

            # Baseline CSNet 3M
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CSNet-3M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['csnet-baseline', '3m-3m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/csnet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CSNet-3M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['csnet-baseline', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/csnet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CSNet-3M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['csnet-baseline', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/csnet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # Baseline CSNet 6M
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CSNet-6M-FULL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['csnet-baseline', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/csnet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CSNet-6M-ILM_OPL-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['csnet-baseline', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/csnet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=CECSNetExperimentModuleConfig(
                        input_shape=torch.ones(1, 3, 400, 400).shape,
                        logging_frequency=base_logging_frequency),
                    experiment_name=f'CSNet-6M-OPL_BM-BASELINE-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-run',
                    wandb_tags=['csnet-baseline', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/csnet_weakly_scribble_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

        ][start_index:end_index]

        # In case some experiment get rescheduled.
        # fix_list = []
        # self.experiments = list(filter(lambda exp: exp.experiment_name in fix_list, self.experiments))

    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule


    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=True,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
        )
        return datamodule

    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)


    def __len__(self):
        return len(self.experiments)


    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list


class OCTAVEOCTA500FAZExperimentConfig:


    def __init__(
        self, n_folds: int = 3, num_epochs: int = 300,
        start_index: int = 0, end_index: Union[int, None] = None,
        get_parallel_job: int = 1, base_logging_frequency: int = 10,
        scribble_presence_ratio: float = 1.0, weakly_supervise: bool = True):
        """Describing experimentation configuration
        This class will yield necessary output for inputing the training function
        """
        self.n_folds = n_folds
        self.num_epochs = num_epochs
        self.get_parallel_job = get_parallel_job
        self.scribble_presence_ratio = scribble_presence_ratio

        level_3m = ['FULL', 'ILM_OPL', 'OPL_BM']
        level_6m = ['FULL', 'ILM_OPL', 'OPL_BM']
        octa3m = {k: v for k, v in zip(level_3m, list(map(lambda x: self.OCTA3M(x), level_3m)))}
        octa6m = {k: v for k, v in zip(level_6m, list(map(lambda x: self.OCTA6M(x), level_6m)))}
        self.dataset: List[Dict] = [octa3m, octa6m]

        experiment_prefix = 'scribble_faz' if weakly_supervise else 'fully_faz'

        self.experiments = [

            # # Baseline OCTAVE 3M
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-3M-FULL-BASELINE-FOLD_{fold}',
            #         datamodule=octa3m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '3m-3m-full', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-3M-ILM_OPL-BASELINE-FOLD_{fold}',
            #         datamodule=octa3m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-3M-OPL_BM-BASELINE-FOLD_{fold}',
            #         datamodule=octa3m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],

            # # Baseline OCTAVE 6M
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
            #             logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-FULL-BASELINE-FOLD_{fold}',
            #         datamodule=octa6m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
            #             logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-ILM_OPL-BASELINE-FOLD_{fold}',
            #         datamodule=octa6m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=False, raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape,
            #             logging_frequency=base_logging_frequency),
            #         experiment_name=f'OCTAVE-6M-OPL_BM-BASELINE-FOLD_{fold}',
            #         datamodule=octa6m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-baseline', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],

            # OCTAVE-ILD 3M
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        logging_frequency=base_logging_frequency, weakly_supervise=weakly_supervise),
                    experiment_name=f'OCTAVE-FAZ-3M-FULL-ILD-FOLD_{fold}',
                    datamodule=octa3m['FULL'],
                    fold=fold,
                    wandb_project='octave-faz-run',
                    wandb_tags=['octave-faz-ild', '3m-3m-full', 'origin', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/{experiment_prefix}_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        logging_frequency=base_logging_frequency, weakly_supervise=weakly_supervise),
                    experiment_name=f'OCTAVE-FAZ-3M-ILM_OPL-ILD-FOLD_{fold}',
                    datamodule=octa3m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-faz-run',
                    wandb_tags=['octave-faz-ild', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/{experiment_prefix}_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        logging_frequency=base_logging_frequency, weakly_supervise=weakly_supervise),
                    experiment_name=f'OCTAVE-FAZ-3M-OPL_BM-ILD-FOLD_{fold}',
                    datamodule=octa3m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-faz-run',
                    wandb_tags=['octave-faz-ild', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/{experiment_prefix}_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # OCTAVE-ILD 6M
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency, weakly_supervise=weakly_supervise),
                    experiment_name=f'OCTAVE-FAZ-6M-FULL-ILD-FOLD_{fold}',
                    datamodule=octa6m['FULL'],
                    fold=fold,
                    wandb_project='octave-faz-run',
                    wandb_tags=['octave-faz-ild', '6m-6m-full', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/{experiment_prefix}_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency, weakly_supervise=weakly_supervise),
                    experiment_name=f'OCTAVE-FAZ-6M-ILM_OPL-ILD-FOLD_{fold}',
                    datamodule=octa6m['ILM_OPL'],
                    fold=fold,
                    wandb_project='octave-faz-run',
                    wandb_tags=['octave-faz-ild', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/{experiment_prefix}_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],
            *[
                ExperimentConfig(
                    experiment_config=OCTAveExperimentModuleConfig(
                        interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
                        raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency, weakly_supervise=weakly_supervise),
                    experiment_name=f'OCTAVE-FAZ-6M-OPL_BM-ILD-FOLD_{fold}',
                    datamodule=octa6m['OPL_BM'],
                    fold=fold,
                    wandb_project='octave-faz-run',
                    wandb_tags=['octave-faz-ild', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}'],
                    num_epochs=self.num_epochs,
                    models_dir=f'models/{experiment_prefix}_{scribble_presence_ratio}/',
                ) for fold in range(self.n_folds)
            ],

            # # OCTAVE-ILD 3M NO ITL ON UNSUPERVISED
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             logging_frequency=base_logging_frequency, disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-3M-FULL-ILD-FOLD_{fold}',
            #         datamodule=octa3m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '3m-3m-full', 'origin', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             logging_frequency=base_logging_frequency, disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-3M-ILM_OPL-ILD-FOLD_{fold}',
            #         datamodule=octa3m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '3m-3m-ilm_opl', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             logging_frequency=base_logging_frequency, disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-3M-OPL_BM-ILD-FOLD_{fold}',
            #         datamodule=octa3m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '3m-3m-opl_bm', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],

            # # OCTAVE-ILD 6M NO ITL ON UNSUPERVISED
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency,
            #             disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-6M-FULL-ILD-FOLD_{fold}',
            #         datamodule=octa6m['FULL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-full', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency,
            #             disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-6M-ILM_OPL-ILD-FOLD_{fold}',
            #         datamodule=octa6m['ILM_OPL'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-ilm_opl', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
            # *[
            #     ExperimentConfig(
            #         experiment_config=OCTAveExperimentModuleConfig(
            #             interlayer_divergence_weight=0.0, dynamic_itl=True, enable_predicate_posterior='predicate', kl_stop_gradient=True, regulation_clip=2,
            #             raw_input_shap=torch.ones(1, 3, 400, 400).shape, mask_input_shape=torch.ones(1, 2, 400, 400).shape, logging_frequency=base_logging_frequency,
            #             disable_unsupervise_itl=True),
            #         experiment_name=f'OCTAVE-6M-OPL_BM-ILD-FOLD_{fold}',
            #         datamodule=octa6m['OPL_BM'],
            #         fold=fold,
            #         wandb_project='octave-run',
            #         wandb_tags=['octave-ild', '6m-6m-opl_bm', f'scribble-{scribble_presence_ratio}', 'disable-unsupervised-ild'],
            #         num_epochs=self.num_epochs,
            #         models_dir=f'models/scribble_{scribble_presence_ratio}/',
            #     ) for fold in range(self.n_folds)
            # ],
        ][start_index:end_index]


    def OCTA6M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-FAZ-6M-{modality}',
            train_modality='6M',
            train_annoation_type='Weak',
            unpaired_modality='6M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=False,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
            label_of_interest='faz',
        )
        return datamodule

    def OCTA3M(self, modality: Literal['FULL', 'ILM_OPL', 'OPL_BM']):
        datamodule = OCTA500Datamodule(
            source_dataset_path='data/OCTA-500',
            processed_dataset_path=f'data/OCTA-500-FAZ-3M-{modality}',
            train_modality='3M',
            train_annoation_type='Weak',
            unpaired_modality='3M',
            train_projection_level=modality,
            unpair_projection_level=modality,
            n_fold=self.n_folds,
            shuffle_cv=True,
            unpair_scribble_ratio=1,
            random_bg_crop=False,
            skeletonize_bg=False,
            crop_portions=1,
            unpair_augmentation=True,
            scribble_presence_ratio=self.scribble_presence_ratio,
            label_of_interest='faz',
        )
        return datamodule

    @staticmethod
    def prepare_data(datamodules: List[Dict[str, OCTA500Datamodule]]):
        """Data preparation subroutine.

        Submit this method to prepare the dataset for usage.
        """
        for datamodule in datamodules:
            for dataset in datamodule.values():
                dataset.prepare_data(override=True)

    def __len__(self):
        return len(self.experiments)

    def __iter__(self):
        n_item = self.get_parallel_job
        for experiment_list in (self.experiments[i:i+n_item] for i in range(0, self.__len__(), n_item)):
            yield experiment_list
