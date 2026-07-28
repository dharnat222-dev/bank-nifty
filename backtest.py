"""
Backtest engine module.
"""

import pandas as pd
import numpy as np
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
        self.config = config or {}
        self.data_loader = DataLoader()
        self.trades = []
        self.equity_curve = []
    
    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run the backtest."""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Load data
        spot_data, option_data = self.data_loader.load_data(start_date, end_date)
        
        # Merge data
        df = pd.merge(spot_data, option_data, on='date', how='left')
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert to numeric
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=required_columns)
        
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
        """Execute the backtest logic."""
        trades = []
        equity = [0]
        position = None
        current_equity = 0
        
        if len(df) < 50:
            return trades, equity
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            if pd.isna(row['vwap']) or pd.isna(row['rsi']):
                continue
            
            if position is None:
                signal = strategy.check_entry(df, i)
                
                if signal:
                    spot_price = row['close']
                    option_type = 'CE' if signal['direction'] == 'CALL' else 'PE'
                    strike, option_price, contract_info = option_selector.select_option(
                        spot_price, option_type, row
                    )
                    
                    position = {
                        'entry_time': row['date'] if 'date' in row else i,
                        'direction': signal['direction'],
                        'strike': strike,
                        'atm_strike': strike,
                        'entry_price': option_price,
                        'stop_loss': signal['stop_loss'],
                        'quantity': 1,
                        'entry_index': i
                    }
            
            else:
                current_strike = position['strike']
                option_type = 'CE' if position['direction'] == 'CALL' else 'PE'
                current_price = option_selector.get_option_price(
                    current_strike, option_type, row
                )
                
                should_exit, exit_reason = strategy.check_exit(row, position)
                
                if should_exit:
                    if position['direction'] == 'CALL':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    trade = {
                        'date': row['date'] if 'date' in row else i,
                        'direction': position['direction'],
                        'strike': position['strike'],
                        'atm_strike': position['atm_strike'],
                        'entry': position['entry_price'],
                        'exit': current_price,
                        'sl': position['stop_loss'],
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    current_equity += pnl
                    equity.append(current_equity)
                    
                    position = None
        
        return trades, equity
    
    def _calculate_metrics(self, trades: List[Dict], equity: List[float]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'expectancy': 0,
                'trades': pd.DataFrame(),
                'equity_curve': equity
            }
        
        df_trades = pd.DataFrame(trades)
        
        winning = df_trades[df_trades['pnl'] > 0]
        losing = df_trades[df_trades['pnl'] < 0]
        
        total = len(df_trades)
        win_rate = len(winning) / total * 100 if total > 0 else 0
        
        gross_profit = winning['pnl'].sum() if not winning.empty else 0
        gross_loss = abs(losing['pnl'].sum()) if not losing.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_trades': total,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': df_trades['pnl'].sum(),
            'avg_win': winning['pnl'].mean() if not winning.empty else 0,
            'avg_loss': losing['pnl'].mean() if not losing.empty else 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'expectancy': df_trades['pnl'].mean() if total > 0 else 0,
            'trades': df_trades,
            'equity_curve': equity
        }

if __name__ == "__main__":
    try:
        engine = BacktestEngine()
        results = engine.run(Config.START_DATE, Config.END_DATE)
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Total PnL: {results['total_pnl']:.2f}")
        print(f"Avg Win: {results['avg_win']:.2f}")
        print(f"Avg Loss: {results['avg_loss']:.2f}")
        print(f"Expectancy: {results['expectancy']:.2f}")
        print("="*50)
        
        from report import ReportGenerator
        report_gen = ReportGenerator()
        report_gen.generate_comprehensive_report(results)
        
        print("\n✅ Backtest completed successfully!")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()