import os
from typing import Optional, Union

from fastai.layers import TimeDistributed
import timm
import torch
from torch import nn


from zamba.pytorch_lightning.utils import ZambaVideoClassificationLightningModule


class TimeDistributedEfficientNetMultiLayerHead(ZambaVideoClassificationLightningModule):
    def __init__(
        self,
        num_frames=16,
        num_classes=24,
        finetune_from: Optional[Union[os.PathLike, str]] = None,
        **kwargs,
    ):

        super().__init__(num_classes=num_classes, **kwargs)

        if finetune_from is None:
            efficientnet = timm.create_model("efficientnetv2_rw_m", pretrained=True)
            efficientnet.classifier = nn.Identity()
        else:
            efficientnet = self.load_from_checkpoint(finetune_from).base.module

        # freeze base layers
        for param in efficientnet.parameters():
            param.requires_grad = False

        num_backbone_final_features = efficientnet.num_features

        self.backbone = torch.nn.ModuleList(
            [
                efficientnet.get_submodule("blocks.5"),
                efficientnet.conv_head,
                efficientnet.bn2,
                efficientnet.act2,
                efficientnet.global_pool,
            ]
        )

        self.base = TimeDistributed(efficientnet, tdim=1)
        self.classifier = nn.Sequential(
            nn.Linear(num_backbone_final_features, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Flatten(),
            nn.Linear(64 * num_frames, num_classes),
        )

    def forward(self, x):
        self.base.eval()
        x = self.base(x)
        return self.classifier(x)
