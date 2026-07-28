"""
Strategy module.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

class Strategy:
    """Strategy class."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
    
    def check_buy_call_conditions(self, row: pd.Series) -> bool:
        """Check BUY CALL conditions."""
        return all([
            row['close'] > row['vwap'],
            row['ema_20'] > row['ema_50'],
            row['rsi'] > 55,
            row['volume_ratio'] > self.params['min_volume_ratio'],
            row['supertrend_direction'] == 1
        ])
    
    def check_buy_put_conditions(self, row: pd.Series) -> bool:
        """Check BUY PUT conditions."""
        return all([
            row['close'] < row['vwap'],
            row['ema_20'] < row['ema_50'],
            row['rsi'] < 45,
            row['volume_ratio'] > self.params['min_volume_ratio'],
            row['supertrend_direction'] == -1
        ])
    
    def check_entry(self, df: pd.DataFrame, index: int) -> Optional[Dict[str, Any]]:
        """Check for entry signals."""
        if index < 1:
            return None
        
        row = df.iloc[index]
        
        if pd.isna(row['vwap']) or pd.isna(row['rsi']):
            return None
        
        if self.check_buy_call_conditions(row):
            return {
                'direction': 'CALL',
                'entry_price': row['close'],
                'stop_loss': row['low'],
                'timestamp': index
            }
        
        if self.check_buy_put_conditions(row):
            return {
                'direction': 'PUT',
                'entry_price': row['close'],
                'stop_loss': row['high'],
                'timestamp': index
            }
        
        return None
    
    def check_exit(self, row: pd.Series, position: Dict[str, Any]) -> Tuple[bool, str]:
        """Check for exit signals."""
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        direction = position['direction']
        
        if direction == 'CALL':
            risk = entry_price - stop_loss
            target_1 = entry_price + (risk * self.params['risk_reward_1'])
            
            if row['low'] <= stop_loss:
                return True, 'Stop Loss Hit'
            if row['high'] >= target_1:
                return True, 'Target 1 Hit'
        else:
            risk = stop_loss - entry_price
            target_1 = entry_price - (risk * self.params['risk_reward_1'])
            
            if row['high'] >= stop_loss:
                return True, 'Stop Loss Hit'
            if row['low'] <= target_1:
                return True, 'Target 1 Hit'
        
        return False, ''