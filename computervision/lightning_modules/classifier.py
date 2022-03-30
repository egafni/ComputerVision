from dataclasses import dataclass
from importlib import import_module

import pytorch_lightning
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchmetrics.functional import accuracy

from computervision.utils.config_utils import ConfigClassMixin, REQUIRED
from computervision.utils.misc_utils import get_module_and_class_names


class Classifier(pytorch_lightning.LightningModule):
    @dataclass
    class Config(ConfigClassMixin):
        backbone: str

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

        self.model = config.backbone
        self.save_hyperparameters()

        # Create backbone
        backbone_func = getattr(torchvision.models, config.backbone)
        self.backbone = backbone_func(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        out = self.backbone(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

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
