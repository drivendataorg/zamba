from pytorchvideo.models.hub import x3d_m
import torch

from zamba_algorithms.pytorch_lightning.utils import ZambaVideoClassificationLightningModule


class X3D(ZambaVideoClassificationLightningModule):
    """Pretrained i3d model.

    Input -> SlowFast Base (including trainable Backbone) -> Res Basic Head -> Output

    Attributes:
        backbone (torch.nn.Module): When scheduling the backbone to train with the
            `BackboneFinetune` callback, this indicates the trainable part of the base.
        base (torch.nn.Module): The entire model prior to the head.
        head (torch.nn.Module): The trainable head.
    """

    def __init__(self, num_classes: int = 24, backbone_mode: str = "eval", **kwargs):
        """Initializes the X3D model.

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

        model = x3d_m(pretrained=True)
        backbone_out_features = model.blocks[-1].proj.in_features

        for param in model.parameters():
            param.requires_grad = False

        # Replace linear layer for new output size
        model.blocks[-1].proj = torch.nn.Linear(
            in_features=backbone_out_features, out_features=num_classes
        )
        model.blocks[-1].activation = None

        self.model = model
        # self.backbone attribute lets `BackboneFinetune` freeze and unfreeze that module
        self.backbone = torch.nn.ModuleList(
            [model.blocks[-2].res_blocks[-1], model.blocks[-1].pool, model.blocks[-1].dropout]
        )
        self.head = torch.nn.ModuleList([model.blocks[-1].proj, model.blocks[-1].output_pool])
        self.backbone_mode = backbone_mode

        self.save_hyperparameters()

    def forward(self, x, *args, **kwargs):
        if self.backbone_mode == "eval":
            head_is_training = self.head.training
            for name, module in self.model.named_modules():
                module.eval()

            if head_is_training:
                self.head.train()

        return self.model(x)
