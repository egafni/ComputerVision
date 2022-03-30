from dataclasses import dataclass
from importlib import import_module

import pytorch_lightning
import torch.nn.functional as F
import torchmetrics
from torch import nn

from computervision.lightning_modules.backbones import Resnet
from computervision.utils.config_utils import ConfigMixin, REQUIRED
from computervision.utils.misc_utils import get_module_and_class_names


class Classifier(pytorch_lightning.LightningModule):
    @dataclass
    class Config(ConfigMixin):
        backbone: Resnet.Config = REQUIRED

        # ex: "torch.optim.AdamW"
        optimizer_class: str = REQUIRED
        # ex: dict(weight_decay=1e-4, lr=1e-3)
        optimizer_init_params: dict = REQUIRED

        # ex: "torch.optim.lr_scheduler.ReduceLROnPlateau"
        scheduler_class: str = REQUIRED
        # ex: dict(mode="min", factor=0.2, patience=10)
        scheduler_init_params: dict = REQUIRED
        # ex: dict(monitor="train/loss", interval="step")
        scheduler_lightning_cfg: dict = REQUIRED

        unique_config_id: str = 'ClassifierConfig'

    def __init__(self, config: Config):
        super(Classifier, self).__init__()

        self.config = config

        self.model = config.backbone
        self.save_hyperparameters()

        # Create backbone
        self.backbone = Resnet(config.backbone).create()

        # Create metrics
        metrics = dict()
        for stage in ['train', 'val', 'test']:
            metrics[f'{stage}/acc'] = torchmetrics.Accuracy()

        self.metrics = nn.ModuleDict(metrics)

    def forward(self, x):
        out = self.backbone(x)
        return F.log_softmax(out, dim=1)

    def _step(self, batch, batch_idx, stage):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        self.metrics[f'{stage}/acc'](logits, y)

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc", self.metrics[f'{stage}/acc'], prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        """
        Instantiates the optimizer and the scheduler from the classes and parameters specified in the config.
        """
        optim_module_name, optim_class_name = get_module_and_class_names(self.config.optimizer_class)
        optimizer_class = getattr(import_module(optim_module_name), optim_class_name)
        optimizer = optimizer_class(self.parameters(), **self.config.optimizer_init_params)

        if self.config.scheduler_class is None:
            return optimizer

        sched_module_name, sched_class_name = get_module_and_class_names(self.config.scheduler_class)
        scheduler_class = getattr(import_module(sched_module_name), sched_class_name)
        scheduler = scheduler_class(optimizer, **self.config.scheduler_init_params)
        scheduler_dict = {"scheduler": scheduler}
        if self.config.scheduler_lightning_cfg is not None:
            scheduler_dict.update(**self.config.scheduler_lightning_cfg)

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
