"""
Report generation module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Report generator class."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_excel_report(self, results: Dict[str, Any], filename: str = "backtest_report.xlsx"):
        """Generate Excel report."""
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            summary_data = {
                'Metric': ['Total Trades', 'Win Rate (%)', 'Profit Factor', 'Total PnL'],
                'Value': [
                    results.get('total_trades', 0),
                    results.get('win_rate', 0),
                    results.get('profit_factor', 0),
                    results.get('total_pnl', 0)
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            if 'trades' in results and not results['trades'].empty:
                results['trades'].to_excel(writer, sheet_name='Tradebook', index=False)
        
        logger.info(f"Excel report generated: {filepath}")
    
    def generate_csv_reports(self, results: Dict[str, Any]):
        """Generate CSV reports."""
        if 'trades' in results and not results['trades'].empty:
            trades_file = os.path.join(self.output_dir, 'tradebook.csv')
            results['trades'].to_csv(trades_file, index=False)
            logger.info(f"Tradebook CSV generated: {trades_file}")
    
    def generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualization plots."""
        if 'equity_curve' in results and results['equity_curve']:
            plt.figure(figsize=(12, 6))
            plt.plot(results['equity_curve'])
            plt.title('Equity Curve')
            plt.xlabel('Trade Number')
            plt.ylabel('Equity')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'equity_curve.png'))
            plt.close()
            logger.info("Equity curve plot saved")
    
    def generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate all reports."""
        logger.info("Generating comprehensive report...")
        self.generate_excel_report(results)
        self.generate_csv_reports(results)
        self.generate_visualizations(results)
        logger.info(f"All reports generated in {self.output_dir}")