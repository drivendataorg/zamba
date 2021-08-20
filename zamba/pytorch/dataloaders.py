import os
import random
from typing import Callable, Dict, Optional, Tuple, Union
import warnings

from loguru import logger
import pandas as pd
from pandas_path import path  # noqa: F401
import torch

import torchvision.datasets.video_utils
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.transforms

from zamba_algorithms.data.metadata import (
    create_site_specific_splits,
    load_metadata,
    LoadMetadataConfig,
)
from zamba_algorithms.data.video import load_video_frames, VideoLoaderConfig


def get_datasets(
    train_metadata: Optional[Union[os.PathLike, pd.DataFrame]] = None,
    predict_metadata: Optional[os.PathLike] = None,
    load_metadata_config: Optional[Union[LoadMetadataConfig, dict]] = None,
    split_proportions: Optional[Dict[str, int]] = {"train": 3, "val": 1, "holdout": 1},
    video_dir: Optional[os.PathLike] = None,
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
        predict_metadata (pathlike, optional): Path to a CSV or DataFrame with a "filepath" column
            and an optional "label" column.
        load_metadata_config (dict or LoadMetadataConfig, optional): If `train_metadata` and
            `predict_metadata` are not provided, load the datasets by passing these parameters to
            the `src.data.metadata.load_metadata` function.
        split_proportions (dict, optional): If no "split" or "site" column, generate random splits
            with the following proportions.
        video_dir (pathlike, optional): The root filepath from which the relative paths in the
            metadata are loaded.
        transform (torchvision.transforms.transforms.Compose, optional)
        video_loader_config (VideoLoaderConfig, optional)

    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset, predict_dataset) where each dataset
        can be None if not specified.
    """
    if ((train_metadata is not None) or (predict_metadata is not None)) and load_metadata_config:
        if load_metadata_config:
            warnings.warn(
                "Ignoring `load_metadata_config` since `train_metadata` and/or `predict_metadata` "
                "are specified."
            )

    if (train_metadata is None) and (predict_metadata is None) and (load_metadata_config is None):
        raise ValueError(
            "Must provide either train_metadata, predict_metadata, or load_metadata_config"
        )

    if train_metadata is not None:
        if not isinstance(train_metadata, pd.DataFrame):
            logger.debug(f"Loading metadata from {train_metadata}")
            train_metadata = pd.read_csv(train_metadata, index_col="filepath")

        train_metadata = (
            pd.get_dummies(
                train_metadata.rename(columns={"label": "species"}), columns=["species"]
            )
            .groupby("filepath")
            .max()
        )

    if predict_metadata is not None:
        if not isinstance(predict_metadata, pd.DataFrame):
            predict_metadata = pd.read_csv(predict_metadata, index_col="filepath")

        if "label" not in predict_metadata:
            predict_metadata["label"] = None

        predict_metadata = (
            pd.get_dummies(
                predict_metadata.rename(columns={"label": "species"}), columns=["species"]
            )
            .groupby("filepath")
            .max()
        )
        predict_metadata["split"] = "predict"

    if (
        (train_metadata is None)
        and (predict_metadata is None)
        and (load_metadata_config is not None)
    ):
        train_metadata = load_metadata(
            **(
                load_metadata_config.dict()
                if isinstance(load_metadata_config, LoadMetadataConfig)
                else load_metadata_config
            )
        )
        train_metadata.set_index("local_path", inplace=True)
        train_metadata.index.rename("filepath", inplace=True)

    if (train_metadata is not None) and ("split" not in train_metadata):
        if "site" in train_metadata:
            train_metadata["split"] = create_site_specific_splits(
                train_metadata["site"], proportions=split_proportions
            )
        else:
            random.seed(4007)
            train_metadata["split"] = random.choices(
                list(split_proportions.keys()),
                weights=list(split_proportions.values()),
                k=len(train_metadata),
            )

    def subset_metadata_or_none(
        metadata: Optional[pd.DataFrame] = None, subset: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        if metadata is None:
            return None
        else:
            metadata_subset = metadata.loc[metadata.split == subset] if subset else metadata
            if len(metadata_subset) > 0:
                return FfmpegZambaVideoDataset(
                    video_dir=video_dir,
                    annotations=metadata_subset.filter(regex=r"^species_"),
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
        video_dir: os.PathLike,
        annotations: Union[pd.DataFrame, os.PathLike],
        transform: Optional[torchvision.transforms.transforms.Compose] = None,
        video_loader_config: Optional[VideoLoaderConfig] = None,
    ):
        if isinstance(annotations, os.PathLike):
            annotations = pd.read_csv(annotations, index_col="filepath")

        self.original_indices = annotations.index

        if video_dir is not None:
            annotations.index = str(video_dir) / annotations.index.path

        self.video_paths = annotations.index.tolist()
        self.targets = annotations

        self.transform = transform
        self.video_loader_config = video_loader_config

        super().__init__(root=video_dir, transform=transform)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index: int):
        video = load_video_frames(
            filepath=self.video_paths[index], config=self.video_loader_config
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
