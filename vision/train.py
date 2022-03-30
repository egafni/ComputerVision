from dataclasses import dataclass, field
from getpass import getpass
from typing import Optional, Union

from pytorch_lightning import Trainer

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

    output_dir: str = field(metadata=dict(help="output directory to store results in"))

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
        default=getpass.getuser(),
        metadata=dict(help="username of the person running this experiment, defaults to current unix user"),
    )
    logger: str = field(
        default="wandb",
        metadata=dict(help="lightning logger to use", choices=["wandb", "csv", "tensorboard"]),
    )

    seed: int = 42


def main(config: TrainConfig):
    return

    trainer = Trainer(**config.trainer.to_dict())
