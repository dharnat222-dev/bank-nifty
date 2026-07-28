# report.py
"""
Report generation module for Bank Nifty ATM Option Backtest Engine.
Creates comprehensive reports in Excel, CSV, and visualization formats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Report generator class for creating backtest reports."""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize ReportGenerator.
        
        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_excel_report(self, results: Dict[str, Any], filename: str = "backtest_report.xlsx"):
        """
        Generate comprehensive Excel report.
        
        Args:
            results: Backtest results dictionary
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Trades', 'Win Rate (%)', 'Profit Factor', 'Total PnL'],
                'Value': [
                    results.get('total_trades', 0),
                    results.get('win_rate', 0),
                    results.get('profit_factor', 0),
                    results.get('total_pnl', 0)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Trades sheet
            if 'trades' in results and not results['trades'].empty:
                results['trades'].to_excel(writer, sheet_name='Tradebook', index=False)
            
            # Monthly P&L
            monthly_pnl = self._calculate_monthly_pnl(results)
            if monthly_pnl is not None:
                monthly_pnl.to_excel(writer, sheet_name='Monthly PnL')
            
            # Additional metrics
            metrics_data = {
                'Metric': ['Average Win', 'Average Loss', 'Max Consecutive Wins', 'Max Consecutive Losses', 'Expectancy'],
                'Value': [
                    results.get('avg_win', 0),
                    results.get('avg_loss', 0),
                    results.get('max_consecutive_wins', 0),
                    results.get('max_consecutive_losses', 0),
                    results.get('expectancy', 0)
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        logger.info(f"Excel report generated: {filepath}")
    
    def generate_csv_reports(self, results: Dict[str, Any]):
        """
        Generate CSV reports for all data.
        
        Args:
            results: Backtest results dictionary
        """
        # Tradebook CSV
        if 'trades' in results and not results['trades'].empty:
            trades_file = os.path.join(self.output_dir, 'tradebook.csv')
            results['trades'].to_csv(trades_file, index=False)
            logger.info(f"Tradebook CSV generated: {trades_file}")
        
        # Monthly P&L CSV
        monthly_pnl = self._calculate_monthly_pnl(results)
        if monthly_pnl is not None:
            monthly_file = os.path.join(self.output_dir, 'monthly_pnl.csv')
            monthly_pnl.to_csv(monthly_file)
            logger.info(f"Monthly P&L CSV generated: {monthly_file}")
    
    def generate_visualizations(self, results: Dict[str, Any]):
        """
        Generate visualization plots.
        
        Args:
            results: Backtest results dictionary
        """
        # Equity curve
        self._plot_equity_curve(results)
        
        # Drawdown
        self._plot_drawdown(results)
        
        # Distribution of PnL
        self._plot_pnl_distribution(results)
    
    def _calculate_monthly_pnl(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate monthly P&L from trades.
        
        Args:
            results: Backtest results dictionary
        
        Returns:
            DataFrame with monthly P&L
        """
        if 'trades' not in results or results['trades'].empty:
            return None
        
        trades = results['trades'].copy()
        
        # Ensure date column exists and is datetime
        if 'date' not in trades.columns:
            return None
        
        trades['date'] = pd.to_datetime(trades['date'])
        trades['month'] = trades['date'].dt.to_period('M')
        
        monthly_pnl = trades.groupby('month')['pnl'].sum().reset_index()
        monthly_pnl.columns = ['Month', 'PnL']
        
        return monthly_pnl
    
    def _plot_equity_curve(self, results: Dict[str, Any]):
        """
        Plot equity curve.
        
        Args:
            results: Backtest results dictionary
        """
        if 'equity_curve' not in results or not results['equity_curve']:
            return
        
        equity = results['equity_curve']
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'equity_curve.png'))
        plt.close()
        logger.info(f"Equity curve plot saved")
    
    def _plot_drawdown(self, results: Dict[str, Any]):
        """
        Plot drawdown curve.
        
        Args:
            results: Backtest results dictionary
        """
        if 'equity_curve' not in results or not results['equity_curve']:
            return
        
        equity = np.array(results['equity_curve'])
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        plt.plot(drawdown, color='darkred')
        plt.title('Drawdown')
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'drawdown.png'))
        plt.close()
        logger.info(f"Drawdown plot saved")
    
    def _plot_pnl_distribution(self, results: Dict[str, Any]):
        """
        Plot P&L distribution.
        
        Args:
            results: Backtest results dictionary
        """
        if 'trades' not in results or results['trades'].empty:
            return
        
        trades = results['trades']
        
        plt.figure(figsize=(10, 6))
        sns.histplot(trades['pnl'], bins=30, kde=True)
        plt.title('PnL Distribution')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at 0
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'pnl_distribution.png'))
        plt.close()
        logger.info(f"PnL distribution plot saved")
    
    def generate_comprehensive_report(self, results: Dict[str, Any]):
        """
        Generate all reports and visualizations.
        
        Args:
            results: Backtest results dictionary
        """
        logger.info("Generating comprehensive report...")
        
        # Generate Excel and CSV reports
        self.generate_excel_report(results)
        self.generate_csv_reports(results)
        
        # Generate visualizations
        self.generate_visualizations(results)
        
        logger.info(f"All reports generated in {self.output_dir}")