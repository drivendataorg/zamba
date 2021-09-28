from typing import Callable, Optional, Tuple, Union
from pytorchvideo.models.hub import i3d_r50
from pytorchvideo.models.head import create_res_basic_head

from zamba_algorithms.pytorch_lightning.utils import ZambaVideoClassificationLightningModule


class I3D(ZambaVideoClassificationLightningModule):
    """Pretrained i3d model.

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
        backbone_mode: str = "eval",
        post_backbone_pool: Optional[Callable] = None,
        post_backbone_pool_output_size: Union[int, Tuple[int]] = 1,
        post_backbone_dropout: float = 0.0,
        output_with_global_average: bool = True,
        **kwargs,
    ):
        """Initializes the SlowFast model.

        Args:
            num_classes (int)
            backbone_mode (str): If "eval", treat the backbone as a feature extractor
                and set to evaluation mode in all forward passes.
            post_backbone_pool (callable, optional): Pooling function that operates on the output
                of the backbone (before the fully-connected layer in the head).
            post_backbone_pool_output_size (int, tuple of ints): Dimensions of the post-backbone
                adaptive average pooling layer. If post_backbone_pool is False, this is not
                applicable.
            post_backbone_dropout (float): Dropout that operates on the output of the backbone +
                pool (before the fully-connected layer in the head).
            output_with_global_average (bool): If True, apply an adaptive average pooling
                operation after the fully-connected layer in the head.
        """
        super().__init__(num_classes=num_classes, **kwargs)

        base = i3d_r50(pretrained=True)
        backbone_out_features = base.blocks[-1].proj.in_features
        base.blocks = base.blocks[:-1]  # Remove the pre-trained head

        for param in base.parameters():
            param.requires_grad = False

        head = create_res_basic_head(
            in_features=backbone_out_features,
            out_features=num_classes,
            pool=post_backbone_pool,
            output_size=post_backbone_pool_output_size,
            dropout_rate=post_backbone_dropout,
            output_with_global_average=output_with_global_average,
        )

        # self.backbone attribute lets `BackboneFinetune` freeze and unfreeze that module
        self.backbone = base.blocks[-1:]
        self.base = base
        self.head = head
        self.backbone_mode = backbone_mode

        self.save_hyperparameters()

    def forward(self, x, *args, **kwargs):
        if self.backbone_mode == "eval":
            self.base.eval()

        x = self.base(x)
        return self.head(x)
