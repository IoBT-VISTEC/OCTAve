from argparse import Namespace
import os
from typing import Any, Dict, Optional, Union

from pytorch_lightning.loggers import WandbLogger
import wandb
from wandb.sdk.wandb_run import Run
from torch import nn


class WandbSLURMLogger(WandbLogger):
    """WandbLogger for SLURM tasks parallelism.
    """

    @property
    def experiment(self) -> Run:
        r"""

        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_wandb_function()

        """
        if self._experiment is None:
            if self._offline:
                os.environ['WANDB_MODE'] = 'dryrun'
            self._experiment = wandb.init(**self._wandb_init) if wandb.run is None else wandb.run

        # save checkpoints in wandb dir to upload on W&B servers
        if self._save_dir is None:
            self._save_dir = self._experiment.dir

        # define default x-axis (for latest wandb versions)
        if getattr(self._experiment, "define_metric", None):
            self._experiment.define_metric("trainer/global_step")
            self._experiment.define_metric("*", step_metric='trainer/global_step', step_sync=True)

        return self._experiment

    def watch(self, model: nn.Module, log: str = 'gradients', log_freq: int = 100):
        self.experiment.watch(model, log=log, log_freq=log_freq)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        params = self._sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        metrics = self._add_prefix(metrics)
        if step is not None:
            self.experiment.log({**metrics, 'trainer/global_step': step})
        else:
            self.experiment.log(metrics)

    def finalize(self, status: str) -> None:
        # upload all checkpoints from saving dir
        if self._log_model:
            wandb.save(os.path.join(self.save_dir, "*.ckpt"))
