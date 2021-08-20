from typing import Callable, Optional, Tuple, Union
from pytorchvideo.models.head import ResNetBasicHead
import torch

from zamba.pytorch.utils import build_multilayer_perceptron
from zamba.pytorch_lightning.utils import ZambaVideoClassificationLightningModule


class SlowFast(ZambaVideoClassificationLightningModule):
    """Pretrained SlowFast model for fine-tuning with the following architecture:

    Input -> SlowFast Base (including trainable Backbone) -> Res Basic Head -> Output

    Attributes:
        backbone (torch.nn.Module): When scheduling the backbone to train with the
            `BackboneFinetune` callback, this indicates the trainable part of the base.
        base (torch.nn.Module): The entire model prior to the head.
        head (torch.nn.Module): The trainable head.
    """

    def __init__(
        self,
        num_classes: int = 24,
        backbone_mode: str = "train",
        post_backbone_dropout: Optional[float] = None,
        output_with_global_average: bool = True,
        head_dropout_rate: Optional[float] = None,
        head_hidden_layer_sizes: Optional[Tuple[int]] = None,
        **kwargs,
    ):
        """Initializes the SlowFast model.

        Args:
            num_classes (int)
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
        """
        super().__init__(num_classes=num_classes, **kwargs)

        base = torch.hub.load(
            "facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True
        )
        backbone_out_features = base.blocks[-1].proj.in_features
        base.blocks = base.blocks[:-1]  # Remove the pre-trained head

        for param in base.parameters():
            param.requires_grad = False

        head = ResNetBasicHead(
            proj=build_multilayer_perceptron(
                backbone_out_features,
                head_hidden_layer_sizes,
                num_classes,
                activation=torch.nn.ReLU,
                dropout=head_dropout_rate,
                output_activation=None,
            ),
            activation=None,
            pool=None,
            dropout=post_backbone_dropout,
            output_pool=torch.nn.AdaptiveAvgPool3d(1),
        )

        # self.backbone attribute lets `BackboneFinetune` freeze and unfreeze that module
        self.backbone = base.blocks[-2:]
        self.base = base
        self.head = head
        self.backbone_mode = backbone_mode

        self.save_hyperparameters()

    def forward(self, x, *args, **kwargs):
        if self.backbone_mode == "eval":
            self.base.eval()

        x = self.base(x)
        return self.head(x)
