"""
Option selection module.
"""

import numpy as np
import pandas as pd
from typing import Tuple

class OptionSelector:
    """Option selector class."""
    
    def __init__(self, spot_data: pd.DataFrame, option_data: pd.DataFrame):
        self.spot_data = spot_data
        self.option_data = option_data
        self.current_position = None
    
    def get_atm_strike(self, spot_price: float) -> float:
        """Get ATM strike price."""
        return float(np.round(spot_price / 100) * 100)
    
    def get_option_price(self, strike: float, option_type: str, row: pd.Series) -> float:
        """Get option price."""
        if option_type == 'CE':
            if row['atm_strike'] == strike:
                return row['ce_price']
            else:
                return max(row['close'] - strike, 0) + 10
        else:
            if row['atm_strike'] == strike:
                return row['pe_price']
            else:
                return max(strike - row['close'], 0) + 10
    
    def select_option(self, spot_price: float, option_type: str, row: pd.Series) -> Tuple[float, float, str]:
        """Select option contract."""
        atm_strike = self.get_atm_strike(spot_price)
        option_price = self.get_option_price(atm_strike, option_type, row)
        contract_info = f"{atm_strike:.0f}{option_type}"
        return atm_strike, option_price, contract_info