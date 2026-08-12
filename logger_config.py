import os
import sys

from loguru import logger

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()

minimum_log_level = "DEBUG"

# Terminal output without tracebacks
logger.add(sys.stdout, level=minimum_log_level, backtrace=False, diagnose=False)

# Configuration of logging to a file
logger.add(
    os.path.join(LOG_DIR, "app.log"),
    rotation="10 MB",  # Rotate when file reaches 500 MB
    retention="30 days",  # Keep logs for 10 days
    compression="zip",  # Compress rotated logs
    level=minimum_log_level,
    backtrace=True,
    diagnose=True,
)

# Configuration of logging errors to a file
logger.add(
    os.path.join(LOG_DIR, "error.log"),
    rotation="5 MB",  # Rotate when file reaches 100 MB
    retention="30 days",  # Keep error logs longer
    compression="zip",
    level="ERROR",
    backtrace=True,
    diagnose=True,
)
