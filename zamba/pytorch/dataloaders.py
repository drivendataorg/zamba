from typing import Optional, Tuple
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from pandas_path import path  # noqa: F401
import torch

import torchvision.datasets.video_utils
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.transforms

from zamba.data.video import npy_cache, load_video_frames, VideoLoaderConfig


def get_datasets(
    train_metadata: Optional[pd.DataFrame] = None,
    predict_metadata: Optional[pd.DataFrame] = None,
    transform: Optional[torchvision.transforms.transforms.Compose] = None,
    video_loader_config: Optional[VideoLoaderConfig] = None,
) -> Tuple[
    Optional["FfmpegZambaVideoDataset"],
    Optional["FfmpegZambaVideoDataset"],
    Optional["FfmpegZambaVideoDataset"],
    Optional["FfmpegZambaVideoDataset"],
]:
    """Gets training and/or prediction datasets.

    Args:
        train_metadata (pathlike, optional): Path to a CSV or DataFrame with columns:
          - filepath: path to a video, relative to `video_dir`
          - label:, label of the species that appears in the video
          - split (optional): If provided, "train", "val", or "holdout" indicating which dataset
            split the video will be included in. If not provided, and a "site" column exists,
            generate a site-specific split. Otherwise, generate a random split using
            `split_proportions`.
          - site (optional): If no "split" column, generate a site-specific split using the values
            in this column.
        predict_metadata (pathlike, optional): Path to a CSV or DataFrame with a "filepath" column.
        transform (torchvision.transforms.transforms.Compose, optional)
        video_loader_config (VideoLoaderConfig, optional)

    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset, predict_dataset) where each dataset
        can be None if not specified.
    """
    if predict_metadata is not None:
        # enable filtering the same way on all datasets
        predict_metadata["species_"] = 0

    def subset_metadata_or_none(
        metadata: Optional[pd.DataFrame] = None, subset: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if metadata is None:
            return None
        else:
            metadata_subset = metadata.loc[metadata.split == subset] if subset else metadata
            if len(metadata_subset) > 0:
                return FfmpegZambaVideoDataset(
                    annotations=metadata_subset.set_index("filepath").filter(regex="species"),
                    transform=transform,
                    video_loader_config=video_loader_config,
                )
            else:
                return None

    train_dataset = subset_metadata_or_none(train_metadata, "train")
    val_dataset = subset_metadata_or_none(train_metadata, "val")
    test_dataset = subset_metadata_or_none(train_metadata, "holdout")
    predict_dataset = subset_metadata_or_none(predict_metadata)

    return train_dataset, val_dataset, test_dataset, predict_dataset


class FfmpegZambaVideoDataset(VisionDataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        transform: Optional[torchvision.transforms.transforms.Compose] = None,
        video_loader_config: Optional[VideoLoaderConfig] = None,
    ):
        self.original_indices = annotations.index

        self.video_paths = annotations.index.tolist()
        self.species = [s.split("species_", 1)[1] for s in annotations.columns]
        self.targets = annotations

        self.transform = transform

        # get environment variable for cache if it exists
        if video_loader_config is None:
            video_loader_config = VideoLoaderConfig()

        self.video_loader_config = video_loader_config

        super().__init__(root=None, transform=transform)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index: int):
        try:
            cached_load_video_frames = npy_cache(
                cache_path=self.video_loader_config.cache_dir,
                cleanup=self.video_loader_config.cleanup_cache,
            )(load_video_frames)

            video = cached_load_video_frames(
                filepath=self.video_paths[index], config=self.video_loader_config
            )
        except Exception as e:
            if isinstance(e, IndexError):
                raise

            # show ffmpeg error
            logger.debug(e)
            logger.warning(
                f"Video {self.video_paths[index]} could not be loaded. Using an array of all zeros instead."
            )
            video = np.zeros(
                (
                    self.video_loader_config.total_frames,
                    (
                        self.video_loader_config.model_input_height
                        if self.video_loader_config.model_input_height is not None
                        else self.video_loader_config.frame_selection_height
                    ),
                    (
                        self.video_loader_config.model_input_width
                        if self.video_loader_config.model_input_width is not None
                        else self.video_loader_config.frame_selection_width
                    ),
                    3,
                ),
                dtype="int",
            )

        # ignore pytorch warning about non-writeable tensors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            video = torch.from_numpy(video)

        if self.transform is not None:
            video = self.transform(video)

        target = self.targets.iloc[index]
        target = torch.tensor(target).float()

        return video, target
