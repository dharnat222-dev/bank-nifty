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
            
            df = df.reset_index()
            
            # Clean column names
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    col_name = str(col[0]).lower().strip()
                else:
                    col_name = str(col).lower().strip()
                
                if '^nsebank' in col_name:
                    col_name = col_name.replace('^nsebank', '').strip('_')
                
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
            
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    for alt_col in df.columns:
                        if col in str(alt_col).lower():
                            df.rename(columns={alt_col: col}, inplace=True)
                            break
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Downloaded {len(df)} rows of Bank Nifty data")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def generate_synthetic_option_data(self, spot_data: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic option data."""
        df = spot_data.copy()
        
        df['atm_strike'] = (df['close'] // 100) * 100
        df['ce_price'] = np.maximum(df['close'] - df['atm_strike'], 0) + 20
        df['pe_price'] = np.maximum(df['atm_strike'] - df['close'], 0) + 20
        df['ce_volume'] = df['volume'] * np.random.uniform(0.01, 0.05, len(df))
        df['pe_volume'] = df['volume'] * np.random.uniform(0.01, 0.05, len(df))
        df['ce_delta'] = np.random.uniform(0.3, 0.7, len(df))
        df['pe_delta'] = np.random.uniform(-0.7, -0.3, len(df))
        
        return df
    
    def load_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load or download data."""
        spot_data = self.download_banknifty_data(start_date, end_date)
        option_data = self.generate_synthetic_option_data(spot_data)
        
        if 'date' not in spot_data.columns:
            spot_data['date'] = spot_data.index
        
        logger.info(f"Data loaded successfully. Shape: {spot_data.shape}")
        return spot_data, option_data