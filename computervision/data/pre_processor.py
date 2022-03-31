from dataclasses import dataclass

from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose

from computervision.utils.config_utils import ConfigMixin


class Cifar10PreProc:
    @dataclass
    class Config(ConfigMixin):
        random_crop_size: int = 32
        unique_config_id: str = 'Cifar10PreProcess'

    def __init__(self, config):
        self.config = config

        self.transforms = dict(
            train=Compose([
                RandomCrop(config.random_crop_size, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=(0.49139968, 0.48215827, 0.44653124),
                          std=(0.24703233, 0.24348505, 0.26158768)),
            ]
            ),
            test=Compose([
                ToTensor(),
                Normalize(mean=(0.49139968, 0.48215827, 0.44653124),
                          std=(0.24703233, 0.24348505, 0.26158768)),
            ])
        )


class DTDPreProc:
    @dataclass
    class Config(ConfigMixin):
        random_crop_size: int = 40
        unique_config_id: str = 'DTDPreProcess'

    def __init__(self, config):
        self.config = config

        self.transforms = dict(
            train=Compose([
                RandomCrop(config.random_crop_size, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                # Normalize(mean=(0.49139968, 0.48215827, 0.44653124),
                #           std=(0.24703233, 0.24348505, 0.26158768)),
            ]
            ),
            test=Compose([
                ToTensor(),
                # Normalize(mean=(0.49139968, 0.48215827, 0.44653124),
                #           std=(0.24703233, 0.24348505, 0.26158768)),
            ])
        )
