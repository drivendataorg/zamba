from enum import Enum
import os
from pathlib import Path
from typing import Union

from cloudpathlib import S3Client, S3Path


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
        f"{region_bucket}/{filename}",
        client=S3Client(local_cache_dir=destination_dir, no_sign_request=True),
    )
    return s3p.fspath
