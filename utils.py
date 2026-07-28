# utils.py
"""
Utility functions module for Bank Nifty ATM Option Backtest Engine.
Provides helper functions for logging, file operations, and data validation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

def setup_logging(log_file: str = "backtest.log", level: int = logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file: Name of the log file
        level: Logging level
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging configured")

def save_json(data: Dict[str, Any], filename: str):
    """
    Save data to JSON file.
    
    Args:
        data: Dictionary data to save
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filename: str) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file.
    
    Args:
        filename: Input filename
    
    Returns:
        Dictionary data or None if file not found
    """
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        return json.load(f)

def ensure_directory(path: str):
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger = logging.getLogger(__name__)
        logger.error(f"Missing columns: {missing_columns}")
        return False
    return True

def calculate_position_size(risk_amount: float, stop_loss: float, entry_price: float) -> int:
    """
    Calculate position size based on risk amount.
    
    Args:
        risk_amount: Maximum risk amount
        stop_loss: Stop loss price
        entry_price: Entry price
    
    Returns:
        Position size (quantity)
    """
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit == 0:
        return 0
    return int(risk_amount / risk_per_unit)

def format_currency(value: float, currency: str = "INR") -> str:
    """
    Format currency values.
    
    Args:
        value: Numeric value
        currency: Currency symbol
    
    Returns:
        Formatted currency string
    """
    if pd.isna(value):
        return f"{currency} 0.00"
    return f"{currency} {value:,.2f}"

def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")