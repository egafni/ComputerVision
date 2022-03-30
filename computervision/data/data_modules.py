import os
from dataclasses import dataclass

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from computervision.data.pre_processor import PreProcessor
from computervision.utils.config_utils import ConfigMixin


class CIFAR10DataModule(LightningDataModule):
    extra_args = {}

    @dataclass
    class Config(ConfigMixin):
        pre_processor: PreProcessor.Config
        data_dir: str = os.path.join(os.path.expanduser("~"), "ml_datasets", "cifar10")
        val_split: int = 5000
        num_workers: int = 16
        batch_size: int = 32
        seed: int = 42
        class_balance: bool = False  # apply class weights to the loss

    def __init__(
            self,
            config: Config,
            *args,
            **kwargs,
    ):
        """
        Cifar10 Datamodule
        """
        super().__init__(*args, **kwargs)
        # self.dims = (3, 32, 32)
        self.DATASET = CIFAR10
        self.config = config
        self.pre_processor = PreProcessor(self.config.pre_processor)

        if config.class_balance:
            raise NotImplementedError("Class balance not implemented, and CIFAR10 is class balanced anyway.")

        self.num_samples = 60000 - config.val_split

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        self.DATASET(self.config.data_dir, train=True, download=True, transform=transforms.ToTensor(),
                     **self.extra_args)
        self.DATASET(self.config.data_dir, train=False, download=True, transform=transforms.ToTensor(),
                     **self.extra_args)

    def get_dataloader(self, mode: str):
        if mode == 'train':
            transform = self.pre_processor.transforms['train']
        else:
            transform = self.pre_processor.transforms['test']

        if mode in ['train', 'val']:
            dataset = self.DATASET(self.config.data_dir, train=True, download=False,
                                   transform=transform,
                                   **self.extra_args)
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
            dataset = self.DATASET(self.config.data_dir, train=False, download=False,
                                   transform=transform, **self.extra_args)
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

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')