import copy
from enum import Enum
from functools import lru_cache
import os
from pathlib import Path
from typing import Union

from cloudpathlib import S3Client, S3Path
import torch
import yaml

from zamba import MODELS_DIRECTORY

S3_BUCKET = "s3://drivendata-public-assets"


class RegionEnum(str, Enum):
    us = "us"
    eu = "eu"
    asia = "asia"


def download_weights(
    filename: str,
    destination_dir: Union[os.PathLike, str],
    weight_region: RegionEnum = RegionEnum("us"),
) -> Path:
    # get s3 bucket based on region
    if weight_region != "us":
        region_bucket = f"{S3_BUCKET}-{weight_region}"
    else:
        region_bucket = S3_BUCKET

    s3p = S3Path(
        f"{region_bucket}/zamba_official_models/{filename}",
        client=S3Client(local_cache_dir=destination_dir, no_sign_request=True),
    )

    s3p.download_to(destination_dir)
    return str(Path(destination_dir) / s3p.name)


def get_model_checkpoint_filename(model_name):
    if isinstance(model_name, Enum):
        model_name = model_name.value

    config_file = MODELS_DIRECTORY / model_name / "config.yaml"
    with config_file.open() as f:
        config_dict = yaml.safe_load(f)
    return Path(config_dict["public_checkpoint"])


def get_checkpoint_hparams(checkpoint):
    return copy.deepcopy(_cached_hparams(checkpoint))


@lru_cache()
def _cached_hparams(checkpoint):
    return torch.load(checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]