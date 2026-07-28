# backtest.py
"""
Backtest module for Bank Nifty ATM Option Backtest Engine.
Orchestrates the entire backtesting process.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import logging

from data_loader import DataLoader
from option_selector import OptionSelector
from strategy import Strategy
from indicators import calculate_all_indicators
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Main backtest engine class."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize BacktestEngine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.data_loader = DataLoader()
        self.trades = []
        self.equity_curve = []
    
    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run the backtest for given date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Load data
        spot_data, option_data = self.data_loader.load_data(start_date, end_date)
        
        # Merge data
        df = pd.merge(spot_data, option_data, on='date', how='left')
        
        # Calculate indicators
        indicator_params = Config.get_indicator_params()
        df = calculate_all_indicators(df, indicator_params)
        
        # Initialize components
        option_selector = OptionSelector(spot_data, option_data)
        strategy_params = Config.get_entry_params()
        strategy = Strategy(strategy_params)
        
        # Run backtest
        trades, equity = self._run_backtest(df, option_selector, strategy)
        
        # Calculate metrics
        results = self._calculate_metrics(trades, equity)
        
        logger.info(f"Backtest completed. Total trades: {len(trades)}")
        return results
    
    def _run_backtest(self, df: pd.DataFrame, option_selector: OptionSelector, 
                     strategy: Strategy) -> Tuple[List[Dict], List[float]]:
        """
        Execute the backtest logic.
        
        Args:
            df: DataFrame with all data
            option_selector: OptionSelector instance
            strategy: Strategy instance
        
        Returns:
            Tuple of (trades_list, equity_curve)
        """
        trades = []
        equity = [0]  # Starting equity
        position = None
        current_equity = 0
        
        for i in range(50, len(df)):  # Start from 50 to ensure all indicators are calculated
            row = df.iloc[i]
            
            # Check for entry if not in position
            if position is None:
                signal = strategy.check_entry(df, i)
                
                if signal:
                    # Get option details
                    spot_price = row['close']
                    option_type = 'CE' if signal['direction'] == 'CALL' else 'PE'
                    strike, option_price, contract_info = option_selector.select_option(
                        spot_price, option_type, row
                    )
                    
                    # Create position
                    position = {
                        'entry_time': row['date'] if 'date' in row else i,
                        'direction': signal['direction'],
                        'strike': strike,
                        'atm_strike': strike,
                        'entry_price': option_price,
                        'stop_loss': signal['stop_loss'],
                        'target': None,  # Will be set later
                        'quantity': 1,  # Fixed quantity for now
                        'entry_index': i,
                        'contract_info': contract_info
                    }
                    
                    # Set target
                    if signal['direction'] == 'CALL':
                        risk = signal['entry_price'] - signal['stop_loss']
                        position['target'] = signal['entry_price'] + (risk * strategy.params['risk_reward_1'])
                    else:
                        risk = signal['stop_loss'] - signal['entry_price']
                        position['target'] = signal['entry_price'] - (risk * strategy.params['risk_reward_1'])
            
            # Check for exit if in position
            else:
                # Get current option price
                current_strike = position['strike']
                option_type = 'CE' if position['direction'] == 'CALL' else 'PE'
                current_price = option_selector.get_option_price(
                    current_strike, option_type, row
                )
                
                # Check exit conditions
                should_exit, exit_reason = strategy.check_exit(row, position)
                
                if should_exit:
                    # Calculate PnL
                    if position['direction'] == 'CALL':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # PUT
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    # Record trade
                    trade = {
                        'date': row['date'] if 'date' in row else i,
                        'time': '15:30:00',  # Default time
                        'direction': position['direction'],
                        'strike': position['strike'],
                        'atm_strike': position['atm_strike'],
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'sl': position['stop_loss'],
                        'target': position['target'],
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    # Update equity
                    current_equity += pnl
                    equity.append(current_equity)
                    
                    # Clear position
                    position = None
        
        return trades, equity
    
    def _calculate_metrics(self, trades: List[Dict], equity: List[float]) -> Dict[str, Any]:
        """
        Calculate performance metrics from trades.
        
        Args:
            trades: List of trade dictionaries
            equity: List of equity values
        
        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'trades': []
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic metrics
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        total_trades = len(df_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Average metrics
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in df_trades['pnl']:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Expectancy
        expectancy = df_trades['pnl'].mean() if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': df_trades['pnl'].sum(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'expectancy': expectancy,
            'trades': df_trades,
            'equity_curve': equity
        }

if __name__ == "__main__":
    # Run backtest
    engine = BacktestEngine()
    results = engine.run(Config.START_DATE, Config.END_DATE)
    
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Total PnL: {results['total_pnl']:.2f}")