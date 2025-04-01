import os
from typing import Any, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
import torch.utils
from zamba.images.evaluate import ClassificationEvaluator
from zamba.models.registry import register_model
from zamba.pytorch_lightning.base_module import ZambaClassificationLightningModule


@register_model
class ImageClassifierModule(ZambaClassificationLightningModule):
    def __init__(
        self,
        model_name: str,
        species: List[str],
        image_size: int,
        batch_size: int,
        num_training_batches: Optional[int] = None,
        lr: float = 1e-5,
        loss: Any = None,
        finetune_from: Optional[Union[os.PathLike, str]] = None,
        scheduler: Optional[LRScheduler] = None,
        scheduler_params: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            species=species,
            lr=lr,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            **kwargs,
        )

        self.image_size = image_size
        self.base_model_name = model_name
        self.num_training_batches = num_training_batches

        if finetune_from is None:
            self.model = timm.create_model(
                self.base_model_name, pretrained=True, num_classes=self.num_classes
            )
        else:
            self.model = self.from_disk(finetune_from).model
            self.model.head.fc = nn.Linear(self.model.head.fc.in_features, self.num_classes)

        self.lr = lr

        if loss is None:
            loss = nn.CrossEntropyLoss()
        self.loss_fn = loss

        self.evaluator = ClassificationEvaluator(species)

        self.save_hyperparameters(
            "lr",
            "image_size",
            "batch_size",
            "model_name",
            "species",
            "scheduler",
            "scheduler_params",
        )

    def configure_optimizers(self):
        # Use Adam optimizer
        # per https://arxiv.org/pdf/2405.13698, we set weight decay to
        # 1 / (lr * iter_per_epoch)
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=(
                1 / (self.lr * self.num_training_batches)
                if self.num_training_batches is not None
                else 0.01
            ),
        )

        # Reset CyclicLR params assuming learning rate was found with lr_find or other empirical method
        if "base_lr" in self.hparams.scheduler_params:
            self.hparams.scheduler_params["base_lr"] = self.lr / 10

        if "max_lr" in self.hparams.scheduler_params:
            self.hparams.scheduler_params["max_lr"] = self.lr * 10

        if self.scheduler is not None:
            scheduler = self.scheduler(
                optimizer,
                **self.hparams.scheduler_params,
            )

            return [optimizer], [scheduler]
        else:
            return [optimizer]

    @staticmethod
    def aggregate_step_outputs(outputs):
        y_true = np.vstack([output[0] for output in outputs])
        y_pred = np.vstack([output[1] for output in outputs])

        return y_true, y_pred

    def _log_metrics(self, y_true, y_pred, subset) -> None:
        metrics = self.evaluator.get_metrics(y_true, y_pred)
        for metric, value in metrics.items():
            self.log(f"{subset}_{metric}", value, logger=True, sync_dist=True, reduce_fx="mean")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).to(torch.float)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, logger=True, sync_dist=True, reduce_fx="mean")
        return loss

    def _val_step(self, batch, batch_idx, subset):
        x, y = batch
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).to(torch.float)

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{subset}_loss", loss.detach(), logger=True, sync_dist=True, reduce_fx="mean")

        return (
            y.cpu().numpy().astype(int),
            y_hat.cpu().numpy(),
        )

    def validation_step(self, batch, batch_idx):
        output = self._val_step(batch, batch_idx, "val")
        self.validation_step_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        output = self._val_step(batch, batch_idx, "test")
        self.test_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        y_true, y_pred = self.aggregate_step_outputs(self.validation_step_outputs)
        self._log_metrics(y_true, y_pred, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        y_true, y_pred = self.aggregate_step_outputs(self.test_step_outputs)

        self._log_metrics(y_true, y_pred, "test")

        confusion_matrix_image = self.evaluator.confusion_matrix_plot(y_true, y_pred)

        if confusion_matrix_image is not None:
            self.logger.experiment.log_image(
                self.logger.run_id, confusion_matrix_image, artifact_file="confusion_matrix.png"
            )

        self.test_step_outputs.clear()

    def on_train_end(self):
        # no checkpoint callback when tuning parameters (e.g., finding learning rate), so skip save in that case.
        if (
            getattr(self.trainer, "checkpoint_callback", None) is not None
            and self.trainer.checkpoint_callback.best_model_path
        ):
            self.logger.experiment.log_artifact(
                self.logger.run_id,
                self.trainer.checkpoint_callback.best_model_path,
                artifact_path="model",
            )

    def to_disk(self, path: os.PathLike):
        state_dict = self.state_dict()

        if "loss_fn.weight" in state_dict:
            # remove weights for loss if required
            del state_dict["loss_fn.weight"]

        checkpoint = {
            "state_dict": state_dict,
            "hyper_parameters": self.hparams,
            "global_step": self.global_step,
            "pytorch-lightning_version": pl.__version__,
        }
        torch.save(checkpoint, path)
