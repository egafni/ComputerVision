import abc
import os
from dataclasses import dataclass
from typing import Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
# from torchvision.datasets import CIFAR10
from torchvision import datasets
from torchvision.transforms import transforms

from computervision.data.pre_processor import Cifar10PreProc, DTDPreProc
from computervision.utils.config_utils import ConfigMixin, REQUIRED


class DataModule(LightningDataModule):
    @dataclass
    class Config(ConfigMixin):
        # dataset_name: str  # ex CIFAR10 or DTD
        pre_processor: Union[Cifar10PreProc.Config, DTDPreProc.Config]  # FIXME automate building this Union
        num_workers: int = REQUIRED
        batch_size: int = REQUIRED
        data_dir: str = os.path.join(os.path.expanduser("~"), "ml_datasets", "cifar10")
        val_split: int = 5000
        seed: int = 42

    def __init__(
            self,
            config: Config,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config

        # FIXME automate instantiating the correct preprocessor
        if type(config.pre_processor) == Cifar10PreProc.Config:
            self.pre_processor = Cifar10PreProc(config.pre_processor)
        elif type(config.pre_processor) == DTDPreProc.Config:
            self.pre_processor = DTDPreProc(config.pre_processor)
        else:
            raise ValueError(f"Unknown preprocessor type: {type(self.pre_processor)}")

    @abc.abstractmethod
    def get_dataloader(self, mode: str):
        pass

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')


class Cifar10DataModule(DataModule):
    @dataclass
    class Config(DataModule.Config):
        unique_config_id: str = 'cifar10_datamodule'

    def __init__(self, config: Config):
        """
        Cifar10 Datamodule
        """
        super().__init__(config)

        self.config = config
        self.torchvision_dataset = datasets.CIFAR10
        self.num_samples = 60000 - config.val_split

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        self.torchvision_dataset(self.config.data_dir, train=True, download=True, transform=transforms.ToTensor())
        self.torchvision_dataset(self.config.data_dir, train=False, download=True, transform=transforms.ToTensor())

    def get_dataloader(self, mode: str):
        if mode == 'train':
            transform = self.pre_processor.transforms['train']
        else:
            transform = self.pre_processor.transforms['test']

        if mode in ['train', 'val']:
            dataset = self.torchvision_dataset(self.config.data_dir, train=True, download=False,
                                               transform=transform)
            train_length = len(dataset)
            dataset_train, dataset_val = random_split(
                dataset,
                [train_length - self.config.val_split, self.config.val_split],
                generator=torch.Generator().manual_seed(self.config.seed)
            )

            if mode == 'train':
                dataset = dataset_train
            elif mode == 'val':
                dataset = dataset_val

        elif mode == 'test':
            dataset = self.torchvision_dataset(self.config.data_dir, train=False, download=False,
                                               transform=transform, )
        else:
            raise AssertionError(f"Mode {mode} not recognized")

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True if mode == 'train' else False,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True
        )


class DTDDataModule(DataModule):
    @dataclass
    class Config(DataModule.Config):
        unique_config_id: str = 'DTD_datamodule'

    def __init__(self, config: Config):
        super().__init__(config)

        self.config = config
        self.torchvision_dataset = datasets.DTD

    def prepare_data(self):
        self.torchvision_dataset(self.config.data_dir, split='train', download=True, transform=transforms.ToTensor())
        self.torchvision_dataset(self.config.data_dir, split='val', download=True, transform=transforms.ToTensor())
        self.torchvision_dataset(self.config.data_dir, split='test', download=True, transform=transforms.ToTensor())

    def get_dataloader(self, mode: str):
        if mode == 'train':
            transform = self.pre_processor.transforms['train']
        else:
            transform = self.pre_processor.transforms['test']

        dataset = self.torchvision_dataset(self.config.data_dir, split=mode, download=False,
                                           transform=transform)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True if mode == 'train' else False,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True
        )
