import pytorch_lightning as pl
import torch
from torchvision.transforms import v2


class CutMixUpCallback(pl.Callback):
    """Applies torchvision v2.CutMix or v2.MixUp on the training batch inside a Lightning Callback."""

    def __init__(self, num_classes: int, p: float = 1.0):
        """
        Args:
            num_classes: number of classes to produce one-hot labels.
            p: probability of applying CutMix to a given batch.
        """
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        
        cutmix = v2.CutMix(num_classes=num_classes)
        mixup = v2.MixUp(num_classes=num_classes)
        self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        images, labels = batch

        # Decide if we apply CutMix this batch or not
        if torch.rand(1).item() < self.p:
            images, labels = self.cutmix_or_mixup(images, labels)

        # Return the possibly modified batch to the trainer
        return (images, labels)

