"""
Utility functions module.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

def setup_logging(log_file: str = "backtest.log", level: int = logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def ensure_directory(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")