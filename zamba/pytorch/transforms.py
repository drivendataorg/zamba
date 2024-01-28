import itertools
from typing import Optional, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import Normalize


class ConvertTHWCtoCTHW(torch.nn.Module):
    """Convert tensor from (0:T, 1:H, 2:W, 3:C) to (3:C, 0:T, 1:H, 2:W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(3, 0, 1, 2)


class ConvertTHWCtoTCHW(torch.nn.Module):
    """Convert tensor from (T, H, W, C) to (T, C, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertTCHWtoCTHW(torch.nn.Module):
    """Convert tensor from (T, C, H, W) to (C, T, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class ConvertHWCtoCHW(torch.nn.Module):
    """Convert tensor from (0:H, 1:W, 2:C) to (2:C, 0:H, 1:W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(2, 0, 1)


class Uint8ToFloat(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / 255.0


class VideotoImg(torch.nn.Module):
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.squeeze(0)


class PadDimensions(torch.nn.Module):
    """Pads a tensor to ensure a fixed output dimension for a give axis.

    Attributes:
        dimension_sizes: A tuple of int or None the same length as the number of dimensions in the
            input tensor. If int, pad that dimension to at least that size. If None, do not pad.
    """

    def __init__(self, dimension_sizes: Tuple[Optional[int]]):
        super().__init__()
        self.dimension_sizes = dimension_sizes

    @staticmethod
    def compute_left_and_right_pad(original_size: int, padded_size: int) -> Tuple[int, int]:
        """Computes left and right pad size.

        Args:
            original_size (list, int): The original tensor size
            padded_size (list, int): The desired tensor size

        Returns:
           Tuple[int]: Pad size for right and left. For odd padding size, the right = left + 1
        """
        if original_size >= padded_size:
            return 0, 0
        pad = padded_size - original_size
        quotient, remainder = divmod(pad, 2)
        return quotient, quotient + remainder

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        padding = tuple(
            itertools.chain.from_iterable(
                (
                    (0, 0)
                    if padded_size is None
                    else self.compute_left_and_right_pad(original_size, padded_size)
                )
                for original_size, padded_size in zip(vid.shape, self.dimension_sizes)
            )
        )
        return torch.nn.functional.pad(vid, padding[::-1])


class PackSlowFastPathways(torch.nn.Module):
    """Creates the slow and fast pathway inputs for the slowfast model."""

    def __init__(self, alpha: int = 4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


imagenet_normalization_values = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def zamba_image_model_transforms(
    single_frame=False, normalization_values=imagenet_normalization_values, channels_first=False
):
    img_transforms = [
        ConvertTHWCtoTCHW(),
        Uint8ToFloat(),
        transforms.Normalize(**imagenet_normalization_values),
    ]

    if single_frame:
        img_transforms += [VideotoImg()]  # squeeze dim

    if channels_first:
        img_transforms += [ConvertTCHWtoCTHW()]

    return transforms.Compose(img_transforms)


def slowfast_transforms():
    return transforms.Compose(
        [
            ConvertTHWCtoTCHW(),
            Uint8ToFloat(),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ConvertTCHWtoCTHW(),
            PadDimensions((None, 32, None, None)),
            PackSlowFastPathways(),
        ]
    )
