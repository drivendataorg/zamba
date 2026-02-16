import os
from pathlib import Path

import appdirs

VIDEO_SUFFIXES = os.environ.get("VIDEO_SUFFIXES")
if VIDEO_SUFFIXES is not None:
    VIDEO_SUFFIXES = VIDEO_SUFFIXES.split(",")
else:
    VIDEO_SUFFIXES = [".avi", ".mp4", ".asf"]

# random seed to use for splitting data without site info into train / val / holdout sets
SPLIT_SEED = os.environ.get("SPLIT_SEED", 4007)


# experimental support for predicting on images
IMAGE_SUFFIXES = [
    ext.strip() for ext in os.environ.get("IMAGE_SUFFIXES", ".jpg,.jpeg,.png,.webp").split(",")
]


def get_model_cache_dir():
    model_cache_dir = Path(
        os.environ.get("MODEL_CACHE_DIR", Path(appdirs.user_cache_dir()) / "zamba")
    )
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    return model_cache_dir
