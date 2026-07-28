# strategy.py
"""
Strategy module for Bank Nifty ATM Option Backtest Engine.
Implements the entry and exit rules for the options strategy.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from indicators import calculate_all_indicators

class Strategy:
    """Strategy class implementing entry and exit rules."""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize Strategy.
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params
        self.entry_signals = []
        self.exit_signals = []
    
    def check_buy_call_conditions(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> bool:
        """
        Check conditions for BUY CALL entry.
        
        Args:
            row: Current row data
            prev_row: Previous row data (for volume comparison)
        
        Returns:
            True if conditions met, False otherwise
        """
        conditions = [
            row['close'] > row['vwap'],  # Close crosses above VWAP
            row['ema_20'] > row['ema_50'],  # EMA20 > EMA50
            row['rsi'] > 55,  # RSI > 55
            row['volume_ratio'] > self.params['min_volume_ratio'],  # Volume condition
            row['supertrend_direction'] == 1  # Supertrend bullish
        ]
        
        return all(conditions)
    
    def check_buy_put_conditions(self, row: pd.Series, prev_row: Optional[pd.Series] = None) -> bool:
        """
        Check conditions for BUY PUT entry.
        
        Args:
            row: Current row data
            prev_row: Previous row data (for volume comparison)
        
        Returns:
            True if conditions met, False otherwise
        """
        conditions = [
            row['close'] < row['vwap'],  # Close crosses below VWAP
            row['ema_20'] < row['ema_50'],  # EMA20 < EMA50
            row['rsi'] < 45,  # RSI < 45
            row['volume_ratio'] > self.params['min_volume_ratio'],  # Volume condition
            row['supertrend_direction'] == -1  # Supertrend bearish
        ]
        
        return all(conditions)
    
    def check_entry(self, df: pd.DataFrame, index: int) -> Optional[Dict[str, Any]]:
        """
        Check for entry signals at given index.
        
        Args:
            df: DataFrame with all data and indicators
            index: Current index
        
        Returns:
            Dictionary with entry signal or None
        """
        if index < 1:
            return None
        
        row = df.iloc[index]
        prev_row = df.iloc[index - 1]
        
        # Check if we have valid data
        if pd.isna(row['vwap']) or pd.isna(row['rsi']):
            return None
        
        # Check BUY CALL conditions
        if self.check_buy_call_conditions(row, prev_row):
            return {
                'direction': 'CALL',
                'entry_price': row['close'],
                'entry_time': row['date'] if 'date' in row else index,
                'stop_loss': row['low'],  # Entry candle low
                'timestamp': index
            }
        
        # Check BUY PUT conditions
        if self.check_buy_put_conditions(row, prev_row):
            return {
                'direction': 'PUT',
                'entry_price': row['close'],
                'entry_time': row['date'] if 'date' in row else index,
                'stop_loss': row['high'],  # Entry candle high
                'timestamp': index
            }
        
        return None
    
    def check_exit(self, row: pd.Series, position: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check for exit signals.
        
        Args:
            row: Current row data
            position: Current position dictionary
        
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        direction = position['direction']
        
        # Calculate targets based on risk-reward ratios
        if direction == 'CALL':
            risk = entry_price - stop_loss
            target_1 = entry_price + (risk * self.params['risk_reward_1'])
            target_2 = entry_price + (risk * self.params['risk_reward_2'])
            
            # Check stop loss
            if row['low'] <= stop_loss:
                return True, 'Stop Loss Hit'
            
            # Check targets
            if row['high'] >= target_1:
                return True, 'Target 1 Hit'
            if row['high'] >= target_2:
                return True, 'Target 2 Hit'
                
        else:  # PUT
            risk = stop_loss - entry_price
            target_1 = entry_price - (risk * self.params['risk_reward_1'])
            target_2 = entry_price - (risk * self.params['risk_reward_2'])
            
            # Check stop loss
            if row['high'] >= stop_loss:
                return True, 'Stop Loss Hit'
            
            # Check targets
            if row['low'] <= target_1:
                return True, 'Target 1 Hit'
            if row['low'] <= target_2:
                return True, 'Target 2 Hit'
        
        # Check for trailing stop if implemented
        if self.params.get('trailing_stop', False):
            if direction == 'CALL':
                if row['high'] >= entry_price + risk:  # 1R profit achieved
                    # Move SL to cost
                    if row['low'] <= entry_price:
                        return True, 'Trailing Stop Hit'
            else:  # PUT
                if row['low'] <= entry_price - risk:  # 1R profit achieved
                    if row['high'] >= entry_price:
                        return True, 'Trailing Stop Hit'
        
        return False, ''