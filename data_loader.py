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
            
            # Flatten MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Convert column names to lowercase
            df.columns = [str(col).lower().replace('^nsebank', '').strip('_') for col in df.columns]
            
            # Rename columns to standard format
            column_mapping = {
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            # Find correct column names
            for col in df.columns:
                for key, value in column_mapping.items():
                    if key in col.lower():
                        df.rename(columns={col: value}, inplace=True)
                        break
            
            # Ensure we have the required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in downloaded data")
                    # Try to find alternative column names
                    for alt_col in df.columns:
                        if col in alt_col.lower():
                            df.rename(columns={alt_col: col}, inplace=True)
                            break
            
            logger.info(f"Downloaded {len(df)} rows of Bank Nifty data")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def generate_synthetic_option_data(self, spot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic option data for backtesting.
        In production, this would be replaced with actual option data.
        
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
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Generate ATM strike prices (nearest 100)
        df['atm_strike'] = (df['close'] // 100) * 100
        
        # Generate option prices with realistic characteristics
        # CE prices (out-of-the-money to in-the-money based on strike)
        df['ce_price'] = np.maximum(df['close'] - df['atm_strike'], 0) + 20
        
        # PE prices
        df['pe_price'] = np.maximum(df['atm_strike'] - df['close'], 0) + 20
        
        # Add some volatility and volume
        df['ce_volume'] = df['volume'] * np.random.uniform(0.01, 0.05, len(df))
        df['pe_volume'] = df['volume'] * np.random.uniform(0.01, 0.05, len(df))
        
        # Option Greeks (simplified)
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
        # Create cache filename with safe characters
        cache_file = os.path.join(self.data_dir, f"banknifty_{start_date}_{end_date}.parquet")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            df = pd.read_parquet(cache_file)
            spot_data = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            option_data = df[['date', 'atm_strike', 'ce_price', 'pe_price', 'ce_volume', 'pe_volume']].copy()
            return spot_data, option_data
        
        # Download new data
        spot_data = self.download_banknifty_data(start_date, end_date)
        
        if use_synthetic:
            option_data = self.generate_synthetic_option_data(spot_data)
        else:
            # In production, you would load actual option data here
            option_data = self.generate_synthetic_option_data(spot_data)
        
        # Ensure date column exists for merging
        if 'date' not in spot_data.columns:
            spot_data['date'] = spot_data.index
        
        # Cache the combined data
        try:
            combined = pd.concat([spot_data, option_data[['atm_strike', 'ce_price', 'pe_price', 'ce_volume', 'pe_volume']]], axis=1)
            combined.to_parquet(cache_file)
            logger.info(f"Cached data to {cache_file}")
        except Exception as e:
            logger.warning(f"Could not cache data: {e}")
        
        return spot_data, option_data