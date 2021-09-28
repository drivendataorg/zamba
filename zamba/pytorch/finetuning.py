from typing import Optional
import pytorch_lightning as pl


def multiplier_factory(rate: float):
    """Returns a function that returns a constant value for use in computing a constant learning
    rate multiplier.

    Args:
        rate (float): Constant multiplier.
    """

    def multiplier(*args, **kwargs):
        return rate

    return multiplier


class BackboneFinetuning(pl.callbacks.finetuning.BackboneFinetuning):
    r"""

    Derived from PTL's built-in ``BackboneFinetuning``, but during the backbone freeze phase,
    choose whether to freeze batch norm layers, even if ``train_bn`` is True (i.e., even if we train them
    during the backbone unfreeze phase).

    Finetune a backbone model based on a learning rate user-defined scheduling.
    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:

        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.

        lambda_func: Scheduling function for increasing backbone learning rate.

        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model

        backbone_initial_lr: Optional, Inital learning rate for the backbone.
            By default, we will use current_learning /  backbone_initial_ratio_lr

        should_align: Wheter to align with current learning rate when backbone learning
            reaches it.

        initial_denom_lr: When unfreezing the backbone, the intial learning rate will
            current_learning_rate /  initial_denom_lr.

        train_bn: Wheter to make Batch Normalization trainable.

        verbose: Display current learning rate for model and backbone

        round: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """

    def __init__(
        self, *args, multiplier: Optional[float] = 1, pre_train_bn: bool = False, **kwargs
    ):
        if multiplier is not None:
            kwargs["lambda_func"] = multiplier_factory(multiplier)
        super().__init__(*args, **kwargs)
        # choose whether to train batch norm layers prior to finetuning phase
        self.pre_train_bn = pre_train_bn

    def freeze_before_training(self, pl_module: "pl.LightningModule"):
        self.freeze(pl_module.backbone, train_bn=self.pre_train_bn)
