from interfaces.experiments.rose import RoseDatamodule
from runsets.dataclass import (
        OCTAnetCoarseAttentionGateExperimentModuleConfig,
        OCTAnetFineAttentionGateExperimentModuleConfig
)
from experiments.octa.octa500.lightning import OCTAnetAttentionCoarseFullySupervise, OCTAnetFineAAGFullySupervise


def train_octanet_coarse(
    experiment_config: OCTAnetCoarseAttentionGateExperimentModuleConfig, experiment_name: str, fold: int, datamodule: RoseDatamodule,
    random_seed=50, models_dir: str = 'models/', wandb_project: str = 'octa-net', wandb_tags=['octa500'],
    num_epochs: int = 300,
    gpus: list = [0], monitor: str = 'val/dice', validation_ratio: float = 0.2,
    dryrun=False):

    from dataclasses import asdict
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.trainer import Trainer

    from utils.logger import WandbSLURMLogger
    """Solid training function for slurm job submittion.

    params:
    experiment: LightningModule
    experiment_name: str
    fold: int
    datamodule: LightningDataModule
    random_seed: int
    models_dir: str
    wandb_project: str
    wandb_tags: list
    """
    import wandb
    wandb.login()
    seed_everything(random_seed)

    experiment_name = 'COARSE_' + experiment_name.format(fold)

    if monitor:
        ckpt_callback = ModelCheckpoint(
            monitor=monitor, # Monitoring on binary scoring.
            dirpath=models_dir + f'{experiment_name}',
            filename=experiment_name+r'_epoch={epoch:02d}-val_dice={val/dice:.4f}',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
        )
    else:
        # Save last, fixed epoch experiment
        ckpt_callback = ModelCheckpoint(
            dirpath=models_dir + f'{experiment_name}',
            filename=experiment_name+r'_epoch={epoch:02d}',
        )

    datamodule.setup(test_ratio=validation_ratio)

    logger = WandbSLURMLogger(name=experiment_name, project=wandb_project, tags=wandb_tags, reinit=True)

    experiment = OCTAnetAttentionCoarseFullySupervise(**asdict(experiment_config), dryrun=dryrun)

    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        max_epochs=num_epochs,
        callbacks=[ckpt_callback],
        fast_dev_run=dryrun
    )
    fit_kwargs = {
        'train_dataloader': datamodule.train_dataloader(batch_size=4)
    }
    if monitor:
        fit_kwargs['val_dataloaders'] = datamodule.val_dataloader(batch_size=1)
    trainer.fit(experiment, **fit_kwargs)

    return ckpt_callback.best_model_path

def train_octanet_fine(
    experiment_config: OCTAnetFineAttentionGateExperimentModuleConfig, experiment_name: str, fold: int, datamodule: RoseDatamodule,
    random_seed=50, models_dir: str = 'models/', wandb_project: str = 'octa-net', wandb_tags=['octa500'],
    num_epochs: int = 300,
    gpus: list = [0], monitor: str = 'val/dice', validation_ratio: float = 0.2,
    dryrun=False):

    from dataclasses import asdict
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.trainer import Trainer

    from utils.logger import WandbSLURMLogger
    """Solid training function for slurm job submittion.

    params:
    experiment: LightningModule
    experiment_name: str
    fold: int
    datamodule: LightningDataModule
    random_seed: int
    models_dir: str
    wandb_project: str
    wandb_tags: list
    """
    import wandb
    wandb.login()
    seed_everything(random_seed)

    experiment_name = 'FINE_' + experiment_name.format(fold)

    if monitor:
        ckpt_callback = ModelCheckpoint(
            monitor=monitor, # Monitoring on binary scoring.
            dirpath=models_dir + f'{experiment_name}',
            filename=experiment_name+r'_epoch={epoch:02d}-val_dice={val/dice:.4f}',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
        )
    else:
        # Save last, fixed epoch experiment
        ckpt_callback = ModelCheckpoint(
            dirpath=models_dir + f'{experiment_name}',
            filename=experiment_name+r'_epoch={epoch:02d}',
        )

    datamodule.setup(test_ratio=validation_ratio)

    logger = WandbSLURMLogger(name=experiment_name, project=wandb_project, tags=wandb_tags, reinit=True)

    experiment = OCTAnetFineAAGFullySupervise(**asdict(experiment_config), dryrun=dryrun)

    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        max_epochs=num_epochs,
        callbacks=[ckpt_callback],
        fast_dev_run=dryrun
    )
    fit_kwargs = {
        'train_dataloader': datamodule.train_dataloader(batch_size=4)
    }
    if monitor:
        fit_kwargs['val_dataloaders'] = datamodule.val_dataloader(batch_size=1)
    trainer.fit(experiment, **fit_kwargs)
    # Reload best model from checkpoint
    experiment = OCTAnetFineAAGFullySupervise.load_from_checkpoint(ckpt_callback.best_model_path, complete_load=True)
    trainer.test(experiment, test_dataloaders=datamodule.test_dataloader(batch_size=1))

    return ckpt_callback.best_model_path

def test_octanet_fine(
    experiment_name: str, datamodule: RoseDatamodule,
    random_seed=50, models_dir: str = 'models/', wandb_project: str = 'octa-net', wandb_tags=['octa500'],
    gpus: list = [0],
    dryrun=False):

    from pytorch_lightning import seed_everything
    from pytorch_lightning.trainer import Trainer

    from utils.logger import WandbSLURMLogger
    """Solid training function for slurm job submittion.

    params:
    experiment: LightningModule
    experiment_name: str
    fold: int
    datamodule: LightningDataModule
    random_seed: int
    models_dir: str
    wandb_project: str
    wandb_tags: list
    """
    from pathlib import Path
    import wandb
    wandb.login()
    seed_everything(random_seed)

    experiment_name = 'FINE_' + experiment_name
    models_dir_path = Path('./' + models_dir + f'{experiment_name}')
    for path in models_dir_path.iterdir():
        if path.is_file() and path.suffix == '.ckpt':
            models_path = str(path)
            break

    datamodule.setup()

    logger = WandbSLURMLogger(name=experiment_name, project=wandb_project, tags=wandb_tags, reinit=True)

    experiment = OCTAnetFineAAGFullySupervise.load_from_checkpoint(models_path, complete_load=True)

    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        fast_dev_run=dryrun
    )
    trainer.test(experiment, test_dataloaders=datamodule.test_dataloader(batch_size=1))

    return models_path