import os
import sys
import logging
from pathlib import Path

# Optional loguru logger; fall back to std logging if unavailable
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Configure logger output (loguru or standard logging)
if hasattr(logger, "remove"):
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))
else:
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Export package version
from .version import __version__

# Public constants
MODELS_DIRECTORY = Path(__file__).parents[1] / "zamba" / "models" / "official_models"

