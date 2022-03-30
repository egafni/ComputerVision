from dataclasses import dataclass
from typing import Tuple

import torchvision
from torch import nn


class Resnet:
    @dataclass
    class Config:
        input_size: Tuple[int, int, int]
        resnet_version: str  # ex 'resnet18'
        pretrained: bool
        num_classes: int

    def __init__(self, config: Config):
        self.config = config

    # 11 million parameters
    def create(self):
        model_creator = getattr(torchvision.models, self.config.resnet_version)

        backbone = model_creator(pretrained=self.config.pretrained, num_classes=self.config.num_classes)

        if self.config.input_size == (3, 32, 32):
            # cifar10 image sizes need a different first conv layer
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # backbone.maxpool = nn.Identity()

        return backbone
