import json
import logging
import os
from dataclasses import dataclass, field
from getpass import getuser
from typing import Optional, Union

import fsspec
import pytorch_lightning
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, GPUStatsMonitor, StochasticWeightAveraging, \
    ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from vision.data.data_modules import CIFAR10DataModule
from vision.lightning_modules.classifier import Classifier
from vision.utils.config_utils import ConfigClassMixin, REQUIRED


@dataclass
class TrainerConfig(ConfigClassMixin):
    """pytorch_lightning.Trainer parameters, for detailed help see the pytorch_lightning documentation"""

    limit_train_batches: Union[int, float] = field(
        default=1.0,
        metadata=dict(help="limit the number of training batches, can be useful for quick testing"),
    )
    limit_val_batches: Union[int, float] = field(
        default=1.0,
        metadata=dict(help="limit the number of validation batches, can be useful for quick testing"),
    )
    limit_test_batches: Union[int, float] = field(
        default=1.0,
        metadata=dict(help="limit the number of test batches, can be useful for quick testing"),
    )
    gpu: Optional[int] = field(
        default=1,  # use 1 gpu by default
        metadata=dict(help="gpu field for Trainer(gpu=).  1 means 1 gpu, None means cpu, 2 means 2 gpus"),
    )
    max_epochs: int = field(default=REQUIRED, metadata=dict(help="max epochs"))

    accumulate_grad_batches: Optional[Union[int, dict]] = field(
        default=None,
        metadata=dict(
            help="accumulate gradients over multiple batches."
        ),
    )


@dataclass
class TrainConfig(ConfigClassMixin):
    """Top level config for training a model"""

    # required params
    name: str = field(metadata=dict(help="experiment name"))

    base_output_dir: str = field(metadata=dict(help="output directory to store results in"))

    data_module: CIFAR10DataModule.Config = field(metadata=dict(help="DataModule Config"))
    trainer: TrainerConfig = field(metadata=dict(help="Parameters to pass to pytorch_lightning.Trainer"))
    lit_module: Union[Classifier.Config] = field(
        metadata=dict(help="pl.LightningModule Config")
    )

    model_checkpoint: dict = field(
        default=REQUIRED,
        metadata=dict(help="ModelCheckpoint params"),
    )

    # ex: dict(monitor="val/acc_macro", min_delta=0.00, patience=20, verbose=True, mode="max")
    early_stopping: Optional[dict] = field(
        default=REQUIRED,
        metadata=dict(help="Early stopping kwargs, if None, do not do early stopping."),
    )

    # ex: dict(swa_epoch_start=.8,swa_lrs=None,annealing_epochs=10,annealing_strategy='cos')
    swa: Optional[dict] = field(default=None, metadata=dict(help="params to StochasticWeightAveraging"))

    # Experiment params
    experiment_group: str = field(
        default=REQUIRED,
        metadata=dict(help="a name to group experiments by, for example in wandb"),
    )
    user: str = field(
        default=getuser(),
        metadata=dict(help="username of the person running this experiment, defaults to current unix user"),
    )
    logger: str = field(
        default="wandb",
        metadata=dict(help="lightning logger to use", choices=["wandb", "csv", "tensorboard"]),
    )

    seed: int = 42

    @property
    def output_dir(self):
        return os.path.join(self.base_output_dir, self.user, self.name)


def main(config: TrainConfig):
    seed_everything(config.seed)

    with fsspec.open(f"{config.output_dir}/train_config.yaml", "w") as fp:
        fp.write(config.to_yaml())

    # ### Setup Logger ####
    os.makedirs(config.output_dir, exist_ok=True)
    if config.logger == "wandb":
        raise NotImplementedError("WandbLogger not implemented")
    elif config.logger == "tensorboard":
        logger = TensorBoardLogger(save_dir=f"{config.output_dir}/logs", name=config.name,
                                   default_hp_metric=None)
    else:
        logger = CSVLogger(save_dir=f"{config.output_dir}/logs", name=config.name)

    data_module = config.get_data_module()
    data_module.prepare_data()
    lit_module = config.get_lit_module()

    # ## Setup Callbacks

    checkpoint_dirpath = f"{config.output_dir}/checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        **config.model_checkpoint,
    )

    callbacks = [
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    if config.early_stopping:
        callbacks.append(EarlyStopping(**config.early_stopping))
    if config.trainer.gpu is not None:
        callbacks.append(GPUStatsMonitor())
    if config.swa:
        callbacks.append(StochasticWeightAveraging(**config.swa))

    trainer = pytorch_lightning.Trainer(
        gpus=config.trainer.gpu,
        deterministic=True,
        weights_summary="top",
        max_time={"days": 4},
        terminate_on_nan=True,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        max_epochs=config.trainer.max_epochs,
        default_root_dir=config.output_dir,
        callbacks=callbacks,
        progress_bar_refresh_rate=1,
        gradient_clip_val=10.0,
        logger=logger,
        log_every_n_steps=30,
        accelerator="auto",
    )

    metrics = {}
    trainer.fit(lit_module, data_module)
    metrics.update(trainer.callback_metrics)
    trainer.test(ckpt_path="best", datamodule=data_module)
    metrics.update(trainer.callback_metrics)

    with fsspec.open(f"{config.output_dir}/results.json", "w") as fp:
        results = dict(
            best_model_path=str(checkpoint_callback.best_model_path),
            output_dir=config.output_dir,
            checkpoint_dirpath=checkpoint_dirpath,
            callback_metrics={k: float(v) for k, v in metrics.items()},
            dataset_info=data_module.dataset_info.to_dict(),
        )
        json.dump(results, fp=fp)

    logging.info(f"output_dir: {config.output_dir}")

    return results, trainer, data_module
