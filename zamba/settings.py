import os

from pathlib import Path


VIDEO_SUFFIXES = os.environ.get("VIDEO_SUFFIXES")
if VIDEO_SUFFIXES is not None:
    VIDEO_SUFFIXES = VIDEO_SUFFIXES.split(",")
else:
    VIDEO_SUFFIXES = [".avi", ".mp4", ".asf"]

# random seed to use for splitting data without site info into train / val / holdout sets
SPLIT_SEED = os.environ.get("SPLIT_SEED", 4007)
