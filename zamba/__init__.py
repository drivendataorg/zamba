from dotenv import load_dotenv
import os
import sys

from loguru import logger

from zamba.version import __version__

__version__

load_dotenv()

logger.remove()
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.add(sys.stderr, level=log_level)
