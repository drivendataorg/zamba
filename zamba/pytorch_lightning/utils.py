import os
from multiprocessing import cpu_count
from multiprocessing.context import BaseContext
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import transforms

from zamba.data.video import VideoLoaderConfig
from zamba.metrics import compute_species_specific_metrics
from zamba.pytorch.dataloaders import get_datasets
from zamba.pytorch.transforms import ConvertTHWCtoCTHW


default_transform = transforms.Compose(
    [
        ConvertTHWCtoCTHW(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)

DEFAULT_TOP_K = (1, 3, 5, 10)


class ZambaDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = max(cpu_count() - 1, 1),
        transform: transforms.Compose = default_transform,
        video_loader_config: Optional[VideoLoaderConfig] = None,
        prefetch_factor: int = 2,
        train_metadata: Optional[pd.DataFrame] = None,
        predict_metadata: Optional[pd.DataFrame] = None,
        multiprocessing_context: Optional[str] = "forkserver",
        *args,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers  # Number of parallel processes fetching data
        self.prefetch_factor = prefetch_factor
        self.video_loader_config = (
            None if video_loader_config is None else video_loader_config.dict()
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
            transform=transform,
            video_loader_config=video_loader_config,
        )
        self.multiprocessing_context: BaseContext = (
            None
            if (multiprocessing_context is None) or (num_workers == 0)
            else multiprocessing_context
        )

        super().__init__(*args, **kwargs)

    def train_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.train_dataset:
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                multiprocessing_context=self.multiprocessing_context,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.num_workers > 0,
            )

    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.val_dataset:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                multiprocessing_context=self.multiprocessing_context,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.num_workers > 0,
            )

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.test_dataset:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                multiprocessing_context=self.multiprocessing_context,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.num_workers > 0,
            )

    def predict_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.predict_dataset:
            return torch.utils.data.DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                multiprocessing_context=self.multiprocessing_context,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=True,
            )


class ZambaVideoClassificationLightningModule(LightningModule):
    def __init__(
        self,
        species: List[str],
        lr: float = 1e-3,
        scheduler: Optional[str] = None,
        scheduler_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        if (scheduler is None) and (scheduler_params is not None):
            warnings.warn(
                "scheduler_params provided without scheduler. scheduler_params will have no effect."
            )

        self.lr = lr
        self.species = species
        self.num_classes = len(species)

        if scheduler is not None:
            self.scheduler = torch.optim.lr_scheduler.__dict__[scheduler]
        else:
            self.scheduler = scheduler

        self.scheduler_params = scheduler_params
        self.model_class = type(self).__name__

        self.save_hyperparameters("lr", "scheduler", "scheduler_params", "species")
        self.hparams["model_class"] = self.model_class

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        metrics = {"val_macro_f1": {}}

        if self.num_classes > 2:
            metrics.update(
                {f"val_top_{k}_accuracy": {} for k in DEFAULT_TOP_K if k < self.num_classes}
            )
        else:
            metrics.update({"val_accuracy": {}})

        # write hparams to hparams.yaml file, log metrics to tb hparams tab
        self.logger.log_hyperparams(self.hparams, metrics)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss.detach())
        self.training_step_outputs.append(loss)
        return loss

    def _val_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("val_loss", loss.detach())

        y_proba = torch.sigmoid(y_hat.cpu()).numpy()
        return {
            "y_true": y.cpu().numpy().astype(int),
            "y_pred": y_proba.round().astype(int),
            "y_proba": y_proba,
        }

    def validation_step(self, batch, batch_idx):
        output = self._val_step(batch, batch_idx)
        self.validation_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        output = self._val_step(batch, batch_idx)
        self.test_step_outputs.append(output)
        return output

    @staticmethod
    def aggregate_step_outputs(
        outputs: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true = np.vstack([output["y_true"] for output in outputs])
        y_pred = np.vstack([output["y_pred"] for output in outputs])
        y_proba = np.vstack([output["y_proba"] for output in outputs])

        return y_true, y_pred, y_proba

    def compute_and_log_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, subset: str
    ):
        self.log(
            f"{subset}_macro_f1",
            f1_score(y_true, y_pred, average="macro", zero_division=0),
        )

        # if only two classes, skip top_k accuracy since not enough classes
        if self.num_classes > 2:
            for k in DEFAULT_TOP_K:
                if k < self.num_classes:
                    self.log(
                        f"{subset}_top_{k}_accuracy",
                        top_k_accuracy_score(
                            y_true.argmax(
                                axis=1
                            ),  # top k accuracy only supports single label case
                            y_proba,
                            labels=np.arange(y_proba.shape[1]),
                            k=k,
                        ),
                    )
        else:
            self.log(f"{subset}_accuracy", accuracy_score(y_true, y_pred))

        for metric_name, label, metric in compute_species_specific_metrics(
            y_true, y_pred, self.species
        ):
            self.log(f"species/{subset}_{metric_name}/{label}", metric)

    def on_validation_epoch_end(self):
        """Aggregates validation_step outputs to compute and log the validation macro F1 and top K
        metrics.

        Args:
            outputs (List[dict]): list of output dictionaries from each validation step
                containing y_pred and y_true.
        """
        y_true, y_pred, y_proba = self.aggregate_step_outputs(self.validation_step_outputs)
        self.compute_and_log_metrics(y_true, y_pred, y_proba, subset="val")
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        y_true, y_pred, y_proba = self.aggregate_step_outputs(self.test_step_outputs)
        self.compute_and_log_metrics(y_true, y_pred, y_proba, subset="test")
        self.test_step_outputs.clear()  # free memory

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
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

    def to_disk(self, path: os.PathLike):
        """Save out model weights to a checkpoint file on disk.

        Note: this does not include callbacks, optimizer_states, or lr_schedulers.
        To include those, use `Trainer.save_checkpoint()` instead.
        """

        checkpoint = {
            "state_dict": self.state_dict(),
            "hyper_parameters": self.hparams,
            "global_step": self.global_step,
            "pytorch-lightning_version": pl.__version__,
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_disk(cls, path: os.PathLike, **kwargs):
        # note: we always load models onto CPU; moving to GPU is handled by `devices` in pl.Trainer
        return cls.load_from_checkpoint(path, map_location="cpu", **kwargs)
