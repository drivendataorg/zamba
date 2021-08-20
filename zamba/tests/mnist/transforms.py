from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from zamba.pytorch.transforms import (
    ConvertTCHWtoCTHW,
    PadDimensions,
    PackSlowFastPathways,
    Uint8ToFloat,
)


class RepeatAsChannels(torch.nn.Module):
    """Repeat grayscale image three times over channels dim.
    Expected input dimensions are (1, H, W). Ouput is (3, H, W).
    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img.repeat(3, 1, 1)


class RepeatAsFrames(torch.nn.Module):
    """Repeat image over new axis to create time dimension.
    Expected input dimensions are (C, H, W).
    Output dimensions are (C, num_frames, H, W).
    """

    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img.unsqueeze(1).repeat(1, self.num_frames, 1, 1)


class RotateFrames(torch.nn.Module):
    """Rotate every frame in place. Expected input dimensions are (C, T, H, W)
    where height and width are equal (square image). Output dimensions are unchanged.
    """

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        for i in np.arange(vid.shape[1]):
            vid[:, i, :, :] = torch.rot90(vid[:, i, :, :], k=i, dims=[1, 2])
        return vid


class MNISTOneHot(torch.nn.Module):
    """One hot encode labels."""

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        return F.one_hot(torch.tensor(label), 10).float()


def mnist_transforms(
    three_channels=True, repeat=None, time_first=False, resize: Optional[Tuple[int, int]] = None
):
    img_transforms = [transforms.ToTensor()]
    if resize:
        img_transforms += [transforms.Resize(resize)]
    if three_channels:
        img_transforms += [RepeatAsChannels()]
    if repeat:
        img_transforms += [RepeatAsFrames(num_frames=repeat)]
    if time_first:
        # swaps first and second dims, here it's going from CTHW to TCHW
        img_transforms += [ConvertTCHWtoCTHW()]
    return transforms.Compose(img_transforms)


def slowfast_mnist_transforms(repeat: int = 32):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            RepeatAsChannels(),
            RepeatAsFrames(num_frames=repeat),
            Uint8ToFloat(),
            NormalizeVideo(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            PadDimensions((None, 32, None, None)),
            PackSlowFastPathways(),
        ]
    )
