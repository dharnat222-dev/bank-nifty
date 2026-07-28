"""
Data loading module.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader class."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_banknifty_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download Bank Nifty historical data."""
        try:
            ticker = "^NSEBANK"
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                raise ValueError("No data downloaded")
            
            # Reset index to have Date as column
            df = df.reset_index()
            
            # Print original columns for debugging
            logger.info(f"Original columns: {df.columns.tolist()}")
            
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
                elif 'adj close' in col_name or 'adj_close' in col_name:
                    col_name = 'adj_close'
                
                new_columns.append(col_name)
            
            df.columns = new_columns
            
            # Print columns after cleaning
            logger.info(f"Cleaned columns: {df.columns.tolist()}")
            
            # Ensure we have all required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # Check if any required column is missing
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                logger.info(f"Available columns: {df.columns.tolist()}")
                
                # Try to find alternative column names
                for missing_col in missing_cols:
                    for alt_col in df.columns:
                        if missing_col in str(alt_col).lower():
                            logger.info(f"Mapping '{alt_col}' to '{missing_col}'")
                            df.rename(columns={alt_col: missing_col}, inplace=True)
                            break
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                # If date column is missing, use index
                df['date'] = df.index
            
            # Ensure all required columns exist
            for col in required_cols:
                if col not in df.columns:
                    if col == 'volume':
                        df['volume'] = 0
                    else:
                        logger.error(f"Column '{col}' not found in data")
                        logger.error(f"Available columns: {df.columns.tolist()}")
                        raise ValueError(f"Column '{col}' not found in data")
            
            logger.info(f"Downloaded {len(df)} rows of Bank Nifty data")
            logger.info(f"Final columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def generate_synthetic_option_data(self, spot_data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic option data."""
        df = spot_data.copy()
        
        # Check if required columns exist
        if 'close' not in df.columns:
            raise ValueError("'close' column not found in spot data")
        
        if 'volume' not in df.columns:
            df['volume'] = 0
        
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
    
    def load_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load or download data."""
        # Download data
        spot_data = self.download_banknifty_data(start_date, end_date)
        
        # Generate synthetic option data
        option_data = self.generate_synthetic_option_data(spot_data)
        
        # Ensure date column exists
        if 'date' not in spot_data.columns:
            spot_data['date'] = spot_data.index
        
        logger.info(f"Data loaded successfully. Shape: {spot_data.shape}")
        logger.info(f"Spot data columns: {spot_data.columns.tolist()}")
        
        return spot_data, option_data