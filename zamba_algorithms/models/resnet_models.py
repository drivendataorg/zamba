from torch import nn
import torchvision.models as models

from zamba_algorithms.pytorch.layers import TimeDistributed
from zamba_algorithms.pytorch_lightning.utils import ZambaVideoClassificationLightningModule


class SingleFrameResnet50(ZambaVideoClassificationLightningModule):
    def __init__(self, num_classes=24, **kwargs):

        super().__init__(num_classes=num_classes, **kwargs)

        # init a pretrained resnet
        resnet = models.resnet50(pretrained=True)

        # freeze base layers
        for param in resnet.parameters():
            param.requires_grad = False

        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.model = resnet

    def forward(self, x):
        # set to eval so we can use resnet as fixed feature extractor
        self.model.eval()
        return self.model(x)


class TimeDistributedResnet50(ZambaVideoClassificationLightningModule):
    def __init__(self, num_frames=16, num_classes=24, **kwargs):

        super().__init__(num_classes=num_classes, **kwargs)

        # init a pretrained resnet
        resnet = models.resnet50(pretrained=True)

        # freeze base layers
        for param in resnet.parameters():
            param.requires_grad = False

        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        self.time_distributed = TimeDistributed(resnet, tdim=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=num_frames * num_classes,
                out_features=num_classes,
            ),
        )

    def forward(self, x):
        # set to eval for dropout and batch normalization layers in resnet (fixed feature extractor)
        self.time_distributed.eval()
        x = self.time_distributed(x)
        return self.classifier(x)


class ResnetR2Plus1d18(ZambaVideoClassificationLightningModule):
    def __init__(self, num_classes=24, **kwargs):

        super().__init__(num_classes=num_classes, **kwargs)

        resnet = models.video.r2plus1d_18(pretrained=True)

        # freeze base layers
        for param in resnet.parameters():
            param.requires_grad = False

        resnet.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, num_classes),
        )

        self.model = resnet

    def forward(self, x):
        # set to eval so we can use resnet as fixed feature extractor
        self.model.eval()
        return self.model(x)
