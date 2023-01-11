from interfaces.experiments.octa500 import OCTA500Datamodule
from runsets.octa500.experiments.dataclass import ScribbleNetExperimentModuleConfig

from experiments.octa.octa500.lightning import ScribbleNet


def train_scribblenet(
    experiment_config: ScribbleNetExperimentModuleConfig, experiment_name: str, fold: int, datamodule: OCTA500Datamodule,
    random_seed=50, models_dir: str = 'models/', wandb_project: str = 'octa-net', wandb_tags=['octa500'],
    num_epochs: int = 300,
    gpus: list = [0]):

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

    experiment_name = experiment_name.format(fold)

    ckpt_callback = ModelCheckpoint(
        monitor='val/dice_0', # Monitoring on binary scoring.
        dirpath=models_dir + f'{experiment_name}',
        filename=experiment_name+r'_epoch={epoch:02d}-val_dice={val/dice_0:.4f}',
        auto_insert_metric_name=False,
        mode='max',
        save_top_k=1,
    )

    datamodule.setup(fold=fold)

    logger = WandbSLURMLogger(name=experiment_name, project=wandb_project, tags=wandb_tags, reinit=True)

    experiment = ScribbleNet(**asdict(experiment_config))

    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        max_epochs=num_epochs,
        callbacks=[ckpt_callback],
    )
    trainer.fit(experiment, train_dataloader=datamodule.train_dataloader(batch_size=4), val_dataloaders=datamodule.val_dataloader(batch_size=1))
