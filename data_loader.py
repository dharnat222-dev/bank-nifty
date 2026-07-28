"""
Data loading module for Bank Nifty ATM Option Backtest Engine.
Handles downloading and loading of historical data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class for fetching and managing market data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_banknifty_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download Bank Nifty historical data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Download Bank Nifty data
            ticker = "^NSEBANK"
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                raise ValueError("No data downloaded")
            
            # Reset index to have Date as column
            df = df.reset_index()
            
            # Handle column names properly
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # If it's a tuple, take the first element
                    col_name = str(col[0]).lower().strip()
                else:
                    col_name = str(col).lower().strip()
                
                # Remove any extra suffixes
                if '^nsebank' in col_name:
                    col_name = col_name.replace('^nsebank', '').strip('_')
                
                # Map to standard names
                if 'date' in col_name:
                    col_name = 'date'
                elif 'open' in col_name:
                    col_name = 'open'
                elif 'high' in col_name:
                    col_name = 'high'
                elif 'low' in col_name:
                    col_name = 'low'
                elif 'close' in col_name:
                    col_name = 'close'
                elif 'volume' in col_name:
                    col_name = 'volume'
                
                new_columns.append(col_name)
            
            df.columns = new_columns
            
            # Ensure we have all required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in downloaded data")
                    # Try to find alternative
                    for alt_col in df.columns:
                        if col in alt_col.lower():
                            df.rename(columns={alt_col: col}, inplace=True)
                            break
            
            logger.info(f"Downloaded {len(df)} rows of Bank Nifty data")
            logger.info(f"Final columns: {df.columns.tolist()}")
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def generate_synthetic_option_data(self, spot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic option data for backtesting.
        
        Args:
            spot_data: DataFrame with spot price data
        
        Returns:
            DataFrame with synthetic option data
        """
        df = spot_data.copy()
        
        # Ensure we have all required columns
        required_cols = ['close', 'high', 'low', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data. Available: {df.columns.tolist()}")
        
        # Generate ATM strike prices (nearest 100)
        df['atm_strike'] = (df['close'] // 100) * 100
        
        # Generate option prices with realistic characteristics
        df['ce_price'] = np.maximum(df['close'] - df['atm_strike'], 0) + 20
        df['pe_price'] = np.maximum(df['atm_strike'] - df['close'], 0) + 20
        
        # Add volume
        df['ce_volume'] = df['volume'] * np.random.uniform(0.01, 0.05, len(df))
        df['pe_volume'] = df['volume'] * np.random.uniform(0.01, 0.05, len(df))
        
        # Option Greeks
        df['ce_delta'] = np.random.uniform(0.3, 0.7, len(df))
        df['pe_delta'] = np.random.uniform(-0.7, -0.3, len(df))
        
        return df
    
    def load_data(self, start_date: str, end_date: str, use_synthetic: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load or download data for backtesting.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_synthetic: If True, generate synthetic option data
        
        Returns:
            Tuple of (spot_data, option_data)
        """
        # Download data
        spot_data = self.download_banknifty_data(start_date, end_date)
        
        # Generate synthetic option data
        if use_synthetic:
            option_data = self.generate_synthetic_option_data(spot_data)
        else:
            option_data = self.generate_synthetic_option_data(spot_data)
        
        # Ensure date column exists
        if 'date' not in spot_data.columns:
            spot_data['date'] = spot_data.index
        
        logger.info(f"Data loaded successfully. Shape: {spot_data.shape}")
        
        return spot_data, option_data