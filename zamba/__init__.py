import os
from pathlib import Path
import sys

from loguru import logger

from zamba.models.efficientnet_models import TimeDistributedEfficientNet  # noqa: F401
from zamba.models.slowfast_models import SlowFast  # noqa: F401
from zamba.version import __version__

__version__

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.add(sys.stderr, level=log_level)

MODELS_DIRECTORY = Path(__file__).parents[1] / "zamba" / "models" / "official_models"
