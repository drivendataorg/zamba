import os
from typing import Optional, Tuple, Union
from pytorchvideo.models.head import ResNetBasicHead
import torch

from zamba.models.registry import register_model
from zamba.pytorch.utils import build_multilayer_perceptron
from zamba.pytorch_lightning.utils import ZambaVideoClassificationLightningModule


@register_model
class SlowFast(ZambaVideoClassificationLightningModule):
    """Pretrained SlowFast model for fine-tuning with the following architecture:

    Input -> SlowFast Base (including trainable Backbone) -> Res Basic Head -> Output

    Attributes:
        backbone (torch.nn.Module): When scheduling the backbone to train with the
            `BackboneFinetune` callback, this indicates the trainable part of the base.
        base (torch.nn.Module): The entire model prior to the head.
        head (torch.nn.Module): The trainable head.
        _backbone_output_dim (int): Dimensionality of the backbone output (and head input).
    """

    _default_model_name = "slowfast"  # used to look up default configuration for checkpoints

    def __init__(
        self,
        backbone_mode: str = "train",
        post_backbone_dropout: Optional[float] = None,
        output_with_global_average: bool = True,
        head_dropout_rate: Optional[float] = None,
        head_hidden_layer_sizes: Optional[Tuple[int]] = None,
        finetune_from: Optional[Union[os.PathLike, str]] = None,
        **kwargs,
    ):
        """Initializes the SlowFast model.

        Args:
            backbone_mode (str): If "eval", treat the backbone as a feature extractor
                and set to evaluation mode in all forward passes.
            post_backbone_dropout (float, optional): Dropout that operates on the output of the
                backbone + pool (before the fully-connected layer in the head).
            output_with_global_average (bool): If True, apply an adaptive average pooling
                operation after the fully-connected layer in the head.
            head_dropout_rate (float, optional): Optional dropout rate applied after backbone and
                between projection layers in the head.
            head_hidden_layer_sizes (tuple of int): If not None, the size of hidden layers in the
                head multilayer perceptron.
            finetune_from (pathlike or str, optional): If not None, load an existing model from
                the path and resume training from an existing model.
        """
        super().__init__(**kwargs)

        if finetune_from is None:
            self.initialize_from_torchub()
        else:
            model = self.from_disk(finetune_from)
            self._backbone_output_dim = model.head.proj.in_features
            self.backbone = model.backbone
            self.base = model.base

        for param in self.base.parameters():
            param.requires_grad = False

        head = ResNetBasicHead(
            proj=build_multilayer_perceptron(
                self._backbone_output_dim,
                head_hidden_layer_sizes,
                self.num_classes,
                activation=torch.nn.ReLU,
                dropout=head_dropout_rate,
                output_activation=None,
            ),
            activation=None,
            pool=None,
            dropout=(
                None if post_backbone_dropout is None else torch.nn.Dropout(post_backbone_dropout)
            ),
            output_pool=torch.nn.AdaptiveAvgPool3d(1),
        )

        self.backbone_mode = backbone_mode
        self.head = head

        self.save_hyperparameters(
            "backbone_mode",
            "head_dropout_rate",
            "head_hidden_layer_sizes",
            "output_with_global_average",
            "post_backbone_dropout",
        )

    def initialize_from_torchub(self):
        """Loads SlowFast model from torchhub and prepares ZambaVideoClassificationLightningModule
        by removing the head and setting the backbone and base."""

        # workaround for pytorch bug
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        base = torch.hub.load(
            "facebookresearch/pytorchvideo:0.1.3", model="slowfast_r50", pretrained=True
        )
        self._backbone_output_dim = base.blocks[-1].proj.in_features

        base.blocks = base.blocks[:-1]  # Remove the pre-trained head

        # self.backbone attribute lets `BackboneFinetune` freeze and unfreeze that module
        self.backbone = base.blocks[-2:]
        self.base = base

    def forward(self, x, *args, **kwargs):
        if self.backbone_mode == "eval":
            self.base.eval()

        x = self.base(x)
        return self.head(x)
