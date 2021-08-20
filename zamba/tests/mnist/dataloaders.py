from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from zamba_algorithms.settings import ROOT_DIRECTORY


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=64,
        img_transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor(),
        num_workers=max(cpu_count() - 1, 1),
        seed=22,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.num_workers = num_workers
        self.num_classes = 10
        self.seed = seed
        # support for tensorboard logging
        self.video_loader_config = dict()
        self.load_metadata_config = dict()
        self.dataset_name = "mnist"

        mnist_train = MNIST(
            ROOT_DIRECTORY,
            train=True,
            download=True,
            transform=self.img_transform,
            target_transform=self.target_transform,
        )
        mnist_test = MNIST(
            ROOT_DIRECTORY,
            train=False,
            download=True,
            transform=self.img_transform,
            target_transform=self.target_transform,
        )

        # train/val split
        mnist_train, mnist_val = random_split(
            mnist_train, [55000, 5000], generator=torch.Generator().manual_seed(self.seed)
        )

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
