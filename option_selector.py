# option_selector.py
"""
Option selection module for Bank Nifty ATM Option Backtest Engine.
Handles ATM strike selection and option contract management.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

class OptionSelector:
    """Option selector class for ATM strike identification and management."""
    
    def __init__(self, spot_data: pd.DataFrame, option_data: pd.DataFrame):
        """
        Initialize OptionSelector.
        
        Args:
            spot_data: DataFrame with spot price data
            option_data: DataFrame with option price data
        """
        self.spot_data = spot_data
        self.option_data = option_data
        self.current_position = None
    
    def get_atm_strike(self, spot_price: float) -> float:
        """
        Get ATM strike price for given spot price.
        
        Args:
            spot_price: Current spot price
        
        Returns:
            ATM strike price
        """
        # Round to nearest 100 (Bank Nifty strike interval)
        atm_strike = np.round(spot_price / 100) * 100
        return float(atm_strike)
    
    def get_option_price(self, strike: float, option_type: str, row: pd.Series) -> float:
        """
        Get option price for given strike and option type.
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            row: DataFrame row with option data
        
        Returns:
            Option price
        """
        if option_type == 'CE':
            # Check if strike matches ATM strike
            if row['atm_strike'] == strike:
                return row['ce_price']
            else:
                # For non-ATM strikes, calculate approximate price
                intrinsic = max(row['close'] - strike, 0)
                return intrinsic + 10  # Add some time value
        else:
            if row['atm_strike'] == strike:
                return row['pe_price']
            else:
                intrinsic = max(strike - row['close'], 0)
                return intrinsic + 10
    
    def select_option(self, spot_price: float, option_type: str, 
                     row: pd.Series) -> Tuple[float, float, str]:
        """
        Select option contract for entry.
        
        Args:
            spot_price: Current spot price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data
        
        Returns:
            Tuple of (strike, option_price, contract_info)
        """
        atm_strike = self.get_atm_strike(spot_price)
        option_price = self.get_option_price(atm_strike, option_type, row)
        
        contract_info = f"{atm_strike:.0f}{option_type}"
        
        return atm_strike, option_price, contract_info
    
    def get_option_volume(self, strike: float, option_type: str, row: pd.Series) -> float:
        """
        Get option volume for given strike and type.
        
        Args:
            strike: Strike price
            option_type: 'CE' or 'PE'
            row: DataFrame row with market data
        
        Returns:
            Option volume
        """
        if option_type == 'CE':
            return row['ce_volume'] if row['atm_strike'] == strike else row['ce_volume'] * 0.5
        else:
            return row['pe_volume'] if row['atm_strike'] == strike else row['pe_volume'] * 0.5