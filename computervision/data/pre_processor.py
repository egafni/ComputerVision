from dataclasses import dataclass

from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from computervision.utils.config_utils import ConfigMixin


class PreProcessor:
    @dataclass
    class Config(ConfigMixin):
        transform_name: str

    def __init__(self, config: Config):
        self.config = config

        if config.transform_name == 'cifar10_default':
            self.transforms = dict(
                train=Compose([
                    RandomCrop(32, padding=4),
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
        else:
            raise ValueError(f'Unknown transform name: {config.transform_name}')


