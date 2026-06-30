import os
import warnings
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.utils.data
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from zamba.pytorch.transforms import ConvertTHWCtoCTHW
from zamba.pytorch.utils import filter_scheduler_params

default_transform = transforms.Compose(
    [
        ConvertTHWCtoCTHW(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)


class ZambaClassificationLightningModule(LightningModule):
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

        self.scheduler_params = {} if scheduler_params is None else scheduler_params
        self.model_class = type(self).__name__

        self.save_hyperparameters("lr", "scheduler", "scheduler_params", "species")
        self.hparams["model_class"] = self.model_class

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

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
            scheduler_params = filter_scheduler_params(self.scheduler, self.scheduler_params)
            return {
                "optimizer": optim,
                "lr_scheduler": self.scheduler(optim, **scheduler_params),
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
        return cls.load_from_checkpoint(
            str(path), map_location="cpu", weights_only=False, **kwargs
        )

    def on_load_checkpoint(self, checkpoint):
        """Remap checkpoint state_dict keys so they match the keys this module currently
        expects. This reconciles two independent sources of key mismatch:

        - ``torch.compile`` inserts an ``_orig_mod.`` segment (e.g.
          ``model._orig_mod.stem.0.weight``). Depending on whether the *live* module is
          compiled, the checkpoint may need that segment added or removed. We match on the
          key with ``_orig_mod.`` removed, so it works in either direction (e.g. an
          in-process lr_find restore of a compiled module, or predict-time loading of a
          compiled checkpoint into an uncompiled module).
        - Some third-party models (notably speciesnet) save the backbone weights without
          the leading ``model.`` prefix.
        """
        sd = checkpoint.get("state_dict", {})
        if not sd:
            return

        expected = self.state_dict()

        def normalize(key: str) -> str:
            return key.replace("_orig_mod.", "")

        # Map each expected key by its normalized (compile-agnostic) form so we can
        # recover the exact key the live module wants regardless of compile state.
        normalized_to_expected = {normalize(k): k for k in expected.keys()}

        remapped = {}
        for k, v in sd.items():
            nk = normalize(k)
            if nk in normalized_to_expected:
                remapped[normalized_to_expected[nk]] = v
            elif f"model.{nk}" in normalized_to_expected:
                # checkpoint stored backbone weights without the 'model.' prefix
                remapped[normalized_to_expected[f"model.{nk}"]] = v
            else:
                remapped[k] = v

        checkpoint["state_dict"] = remapped
