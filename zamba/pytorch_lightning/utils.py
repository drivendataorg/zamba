import os
from multiprocessing import cpu_count
from typing import Callable, Optional, Union
import warnings

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.metrics import f1_score, top_k_accuracy_score
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import transforms

from zamba.data.metadata import LoadMetadataConfig
from zamba.data.video import VideoLoaderConfig
from zamba.pytorch.dataloaders import get_datasets
from zamba.pytorch.transforms import ConvertTHWCtoCTHW
from zamba.settings import ROOT_DIRECTORY

default_transform = transforms.Compose(
    [
        ConvertTHWCtoCTHW(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)


class ZambaDataModule(LightningDataModule):
    def __init__(
        self,
        data_directory: os.PathLike = ROOT_DIRECTORY,
        batch_size: int = 1,
        num_classes: Optional[int] = 24,
        num_workers: int = max(cpu_count() - 1, 1),
        transform: transforms.Compose = default_transform,
        video_loader_config: Optional[VideoLoaderConfig] = None,
        prefetch_factor: Optional[int] = 1,
        train_metadata: Optional[Union[os.PathLike, pd.DataFrame]] = None,
        predict_metadata: Optional[Union[os.PathLike, pd.DataFrame]] = None,
        load_metadata_config: Optional[Union[LoadMetadataConfig, dict]] = LoadMetadataConfig(
            zamba_label="original", subset="dev"
        ),
        *args,
        **kwargs,
    ):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers  # Number of parallel processes fetching data
        self.prefetch_factor = prefetch_factor
        self.video_loader_config = video_loader_config.dict()
        self.load_metadata_config = (
            load_metadata_config.dict()
            if isinstance(load_metadata_config, LoadMetadataConfig)
            else load_metadata_config
        )
        self.train_metadata = train_metadata
        self.predict_metadata = predict_metadata

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            self.predict_dataset,
        ) = get_datasets(
            train_metadata=train_metadata,
            predict_metadata=predict_metadata,
            load_metadata_config=load_metadata_config,
            video_dir=data_directory,
            transform=transform,
            video_loader_config=video_loader_config,
        )
        self.dataset_name = "zamba"

        super().__init__(*args, **kwargs)

    def train_dataloader(self):
        if self.train_dataset:
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                prefetch_factor=self.prefetch_factor,
            )

    def val_dataloader(self):
        if self.val_dataset:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                prefetch_factor=self.prefetch_factor,
            )

    def test_dataloader(self):
        if self.test_dataset:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                prefetch_factor=self.prefetch_factor,
            )

    def predict_dataloader(self):
        if self.predict_dataset:
            return torch.utils.data.DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                prefetch_factor=self.prefetch_factor,
            )


class ZambaVideoClassificationLightningModule(LightningModule):
    def __init__(
        self,
        num_classes: int,
        model: Optional = None,
        lr: float = 1e-3,
        scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, str]] = None,
        scheduler_params: Optional[dict] = None,
    ):
        super().__init__()
        if (scheduler is None) and (scheduler_params is not None):
            warnings.warn(
                "scheduler_params provided without scheduler. scheduler_params will have no effect."
            )
        self.model = model
        self.lr = lr
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params

        self.save_hyperparameters("scheduler", "scheduler_params")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss.detach())

        y_proba = torch.sigmoid(y_hat.cpu()).numpy()

        output = {
            "y_proba": y_proba,
            "y_pred": y_proba.round().astype(int),
            "y_true": y.cpu().numpy().astype(int),
        }
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs and calculate and log the
        following metrics:
          - macro f1_score
          - top 1, 3, 5, and 10 accuracies

        Args:
            outputs (List[dict]): list of output dictionaries from each validation step
                containing y_pred and y_true.
        """
        y_proba = np.vstack([output["y_proba"] for output in outputs])
        y_pred = np.vstack([output["y_pred"] for output in outputs])
        y_true = np.vstack([output["y_true"] for output in outputs])

        scores = {"val_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}

        for k in [1, 3, 5, 10]:
            scores[f"val_top_{k}_accuracy"] = top_k_accuracy_score(
                y_true.argmax(axis=1),  # top k accuracy only supports single label case
                y_proba,
                labels=np.arange(y_proba.shape[1]),
                k=k,
            )

        for metric, v in scores.items():
            self.log(metric, v)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self(x)
        pred = torch.sigmoid(y_hat).cpu().numpy()
        return pred

    def _get_optimizer(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        optim = self._get_optimizer()

        if self.scheduler is None:
            return optim
        else:
            return {
                "optimizer": optim,
                "lr_scheduler": self.scheduler(
                    optim, **({} if self.scheduler_params is None else self.scheduler_params)
                ),
            }
