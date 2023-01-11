from interfaces.experiments.octa500 import OCTA500Classification
# from runsets.octa500.experiments.dataclass import OCTAveExperimentModuleConfig

from experiments.octa.octa500.lightning import OCTA500ClassificationExperiment


def train_octave(
    experiment_config: None, experiment_name: str, fold: int, datamodule: OCTA500Classification,
    random_seed=50, models_dir: str = 'models/', wandb_project: str = 'octa-net', wandb_tags=['octa500'],
    num_epochs: int = 300,
    gpus: list = [0]):

    from dataclasses import asdict
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.trainer import Trainer
    from pytorch_lightning.loggers import TestTubeLogger

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

    experiment_name = experiment_name.format(fold)

    ckpt_callback = ModelCheckpoint(
        dirpath=models_dir + f'{experiment_name}',
        filename=experiment_name+r'_epoch={epoch:02d}',
        auto_insert_metric_name=False,
        # mode='max',
        # save_top_k=1,
    )

    datamodule.setup(fold=fold)

    logger = WandbSLURMLogger(name=experiment_name, project=wandb_project, tags=wandb_tags, reinit=True)
    quick_logger = TestTubeLogger(save_dir='test_tube_logs', name=experiment_name)

    experiment = OCTA500ClassificationExperiment(**asdict(experiment_config))

    trainer = Trainer(
        gpus=gpus,
        logger=[logger, quick_logger],
        max_epochs=num_epochs,
        callbacks=[ckpt_callback],
        deterministic=True,
    )
    trainer.fit(experiment, train_dataloader=datamodule.train_dataloader(batch_size=16, balance=False), val_dataloaders=datamodule.val_dataloader(batch_size=1))
