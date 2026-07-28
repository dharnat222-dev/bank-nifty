"""
Option selection module for Bank Nifty ATM Option Backtest Engine.
Handles ATM strike identification and actual historical option data loading.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptionSelector:
    """Option selector class for ATM strike identification and historical option data retrieval."""
    
    def __init__(self, spot_data: pd.DataFrame, option_data: pd.DataFrame):
        """
        Initialize OptionSelector.
        
        Args:
            spot_data: DataFrame with spot price data
            option_data: DataFrame with historical option data containing:
                - datetime: Timestamp
                - strike: Strike price
                - option_type: 'CE' or 'PE'
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Volume
        """
        self.spot_data = spot_data
        self.option_data = option_data
        self.current_position = None
        
        # Ensure option_data has required columns
        required_cols = ['datetime', 'strike', 'option_type', 'close']
        for col in required_cols:
            if col not in option_data.columns:
                raise ValueError(f"Option data missing required column: {col}")
        
        # Create lookup index for faster queries
        self._create_lookup_index()
    
    def _create_lookup_index(self):
        """
        Create a multi-index lookup for fast option price retrieval.
        """
        if self.option_data.empty:
            logger.warning("Option data is empty")
            return
        
        # Create a composite key for fast lookup
        self.option_data['lookup_key'] = (
            self.option_data['datetime'].astype(str) + '_' + 
            self.option_data['strike'].astype(str) + '_' + 
            self.option_data['option_type'].str.upper()
        )
        
        # Create dictionary for O(1) lookups
        self.lookup_dict = {}
        for _, row in self.option_data.iterrows():
            key = row['lookup_key']
            self.lookup_dict[key] = {
                'open': row.get('open', None),
                'high': row.get('high', None),
                'low': row.get('low', None),
                'close': row.get('close', None),
                'volume': row.get('volume', None)
            }
        
        logger.info(f"Created lookup index with {len(self.lookup_dict)} entries")
    
    def get_atm_strike(self, spot_price: float) -> float:
        """
        Get ATM strike price for given spot price.
        
        Args:
            spot_price: Current spot price
        
        Returns:
            ATM strike price (nearest 100)
        """
        atm_strike = np.round(spot_price / 100) * 100
        return float(atm_strike)
    
    def get_option_price(self, strike: float, option_type: str, row: pd.Series) -> Optional[float]:
        """
        Get actual historical option close price for given strike and option type.
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data (must contain 'date' or 'datetime')
        
        Returns:
            Option close price if found, None otherwise
        """
        # Get datetime from row
        if 'date' in row.index:
            dt = row['date']
        elif 'datetime' in row.index:
            dt = row['datetime']
        else:
            logger.error("Row must contain 'date' or 'datetime' column")
            return None
        
        # Convert to string for lookup
        dt_str = str(dt)
        strike_str = str(int(strike))
        option_type_str = option_type.upper()
        
        # Create lookup key
        lookup_key = f"{dt_str}_{strike_str}_{option_type_str}"
        
        # Look up in dictionary
        if hasattr(self, 'lookup_dict') and lookup_key in self.lookup_dict:
            option_data = self.lookup_dict[lookup_key]
            return option_data['close']
        
        # If not found, try without milliseconds
        if ' ' in dt_str:
            dt_str = dt_str.split('.')[0]  # Remove milliseconds
            lookup_key = f"{dt_str}_{strike_str}_{option_type_str}"
            
            if hasattr(self, 'lookup_dict') and lookup_key in self.lookup_dict:
                option_data = self.lookup_dict[lookup_key]
                return option_data['close']
        
        # Option data not found
        logger.debug(f"Option data not found for {lookup_key}")
        return None
    
    def get_option_data(self, strike: float, option_type: str, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Get complete option data (OHLCV) for given strike and option type.
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data (must contain 'date' or 'datetime')
        
        Returns:
            Dictionary with OHLCV data if found, None otherwise
        """
        # Get datetime from row
        if 'date' in row.index:
            dt = row['date']
        elif 'datetime' in row.index:
            dt = row['datetime']
        else:
            logger.error("Row must contain 'date' or 'datetime' column")
            return None
        
        # Convert to string for lookup
        dt_str = str(dt)
        strike_str = str(int(strike))
        option_type_str = option_type.upper()
        
        # Create lookup key
        lookup_key = f"{dt_str}_{strike_str}_{option_type_str}"
        
        # Look up in dictionary
        if hasattr(self, 'lookup_dict') and lookup_key in self.lookup_dict:
            return self.lookup_dict[lookup_key]
        
        # If not found, try without milliseconds
        if ' ' in dt_str:
            dt_str = dt_str.split('.')[0]  # Remove milliseconds
            lookup_key = f"{dt_str}_{strike_str}_{option_type_str}"
            
            if hasattr(self, 'lookup_dict') and lookup_key in self.lookup_dict:
                return self.lookup_dict[lookup_key]
        
        # Option data not found
        return None
    
    def select_option(self, spot_price: float, option_type: str, row: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Select option contract for entry using actual historical data.
        
        Args:
            spot_price: Current spot price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data
        
        Returns:
            Tuple of (strike, option_price, contract_info) or (None, None, None) if no data found
        """
        # Get ATM strike
        atm_strike = self.get_atm_strike(spot_price)
        
        # Get actual option price from historical data
        option_price = self.get_option_price(atm_strike, option_type, row)
        
        if option_price is None:
            logger.warning(f"No option data found for {atm_strike}{option_type} at {row.get('date', 'unknown')}")
            return None, None, None
        
        contract_info = f"{atm_strike:.0f}{option_type.upper()}"
        return atm_strike, option_price, contract_info
    
    def get_option_volume(self, strike: float, option_type: str, row: pd.Series) -> Optional[float]:
        """
        Get option volume for given strike and type from historical data.
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data
        
        Returns:
            Option volume if found, None otherwise
        """
        option_data = self.get_option_data(strike, option_type, row)
        
        if option_data is not None:
            return option_data.get('volume', None)
        
        return None
    
    def get_option_ohlc(self, strike: float, option_type: str, row: pd.Series) -> Optional[Dict[str, float]]:
        """
        Get option OHLC data for given strike and type from historical data.
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data
        
        Returns:
            Dictionary with OHLC data if found, None otherwise
        """
        option_data = self.get_option_data(strike, option_type, row)
        
        if option_data is not None:
            return {
                'open': option_data.get('open'),
                'high': option_data.get('high'),
                'low': option_data.get('low'),
                'close': option_data.get('close'),
                'volume': option_data.get('volume')
            }
        
        return None
    
    def find_nearest_strike(self, spot_price: float, option_type: str, row: pd.Series) -> Optional[float]:
        """
        Find the nearest available strike price for given spot price.
        
        Args:
            spot_price: Current spot price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data
        
        Returns:
            Nearest available strike price or None
        """
        # Get datetime from row
        if 'date' in row.index:
            dt = row['date']
        elif 'datetime' in row.index:
            dt = row['datetime']
        else:
            return None
        
        # Filter option data for this datetime and option type
        dt_str = str(dt)
        if ' ' in dt_str:
            dt_str = dt_str.split('.')[0]
        
        available_strikes = []
        
        for key in self.lookup_dict.keys():
            key_parts = key.split('_')
            if len(key_parts) >= 3:
                key_dt = key_parts[0] + '_' + key_parts[1]  # Date without milliseconds
                key_strike = float(key_parts[1] if len(key_parts) >= 2 else 0)
                key_type = key_parts[2] if len(key_parts) >= 3 else ''
                
                if key_dt.startswith(dt_str[:10]) and key_type.upper() == option_type.upper():
                    available_strikes.append(key_strike)
        
        if not available_strikes:
            return None
        
        # Find nearest strike
        available_strikes = sorted(available_strikes)
        nearest_strike = min(available_strikes, key=lambda x: abs(x - spot_price))
        
        return nearest_strike