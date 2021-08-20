import os

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

ROOT_DIRECTORY = Path(__file__).parents[1].resolve()

LOAD_VIDEO_FRAMES_CACHE_DIR = os.environ.get("LOAD_VIDEO_FRAMES_CACHE_DIR")
if LOAD_VIDEO_FRAMES_CACHE_DIR == "":
    LOAD_VIDEO_FRAMES_CACHE_DIR = None
if LOAD_VIDEO_FRAMES_CACHE_DIR is not None:
    LOAD_VIDEO_FRAMES_CACHE_DIR = Path(LOAD_VIDEO_FRAMES_CACHE_DIR)
