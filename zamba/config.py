import appdirs
import os
import logging
from pathlib import Path

log_levels = {None: logging.WARNING, True: logging.DEBUG}

cache_dir = os.getenv(
    "ZAMBA_CACHE_DIR",
    None
)
if cache_dir is None:
    cache_dir = appdirs.user_cache_dir()
    cache_dir = Path(cache_dir) / "zamba"
else:
    cache_dir = Path(cache_dir)

cache_dir.mkdir(exist_ok=True)

codeship = os.getenv("CI", False)
