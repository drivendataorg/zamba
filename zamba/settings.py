import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT_DIRECTORY = Path(__file__).parents[1].resolve()
MODELS_DIRECTORY = ROOT_DIRECTORY / "zamba" / "models" / "official_models"

LOAD_VIDEO_FRAMES_CACHE_DIR = os.environ.get("LOAD_VIDEO_FRAMES_CACHE_DIR")
if LOAD_VIDEO_FRAMES_CACHE_DIR == "":
    LOAD_VIDEO_FRAMES_CACHE_DIR = None
if LOAD_VIDEO_FRAMES_CACHE_DIR is not None:
    LOAD_VIDEO_FRAMES_CACHE_DIR = Path(LOAD_VIDEO_FRAMES_CACHE_DIR)

VIDEO_SUFFIXES = os.environ.get("VIDEO_SUFFIXES")
if VIDEO_SUFFIXES is not None:
    VIDEO_SUFFIXES = VIDEO_SUFFIXES.split(",")
else:
    VIDEO_SUFFIXES = [".avi", ".mp4", ".asf"]

# random seed to use for splitting data without site info into train / val / holdout sets
SPLIT_SEED = os.environ.get("SPLIT_SEED", 4007)
