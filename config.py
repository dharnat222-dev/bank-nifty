"""
Configuration module for the Bank Nifty ATM Option Backtest Engine.
"""

class Config:
    """Configuration class containing all backtest parameters."""
    
    # Data Configuration
    DATA_DIR = "data"
    OUTPUT_DIR = "reports"
    
    # Date Range
    START_DATE = "2022-01-01"
    END_DATE = "2026-12-31"
    
    # Option Configuration
    OPTION_TYPE = "CE"
    CONTRACT_SIZE = 25
    
    # Indicator Parameters
    VWAP_PERIOD = 20
    EMA_FAST = 20
    EMA_SLOW = 50
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    SUPERTREND_PERIOD = 10
    SUPERTREND_MULTIPLIER = 3.0
    
    # Entry Parameters
    MIN_VOLUME_RATIO = 1.0
    
    # Risk Parameters
    RISK_REWARD_RATIO_1 = 2.0
    RISK_REWARD_RATIO_2 = 3.0
    
    # Position Sizing
    MAX_POSITION_SIZE = 100
    
    @classmethod
    def get_date_range(cls):
        return (cls.START_DATE, cls.END_DATE)
    
    @classmethod
    def get_indicator_params(cls):
        return {
            'vwap_period': cls.VWAP_PERIOD,
            'ema_fast': cls.EMA_FAST,
            'ema_slow': cls.EMA_SLOW,
            'rsi_period': cls.RSI_PERIOD,
            'atr_period': cls.ATR_PERIOD,
            'supertrend_period': cls.SUPERTREND_PERIOD,
            'supertrend_multiplier': cls.SUPERTREND_MULTIPLIER
        }
    
    @classmethod
    def get_entry_params(cls):
        return {
            'min_volume_ratio': cls.MIN_VOLUME_RATIO,
            'risk_reward_1': cls.RISK_REWARD_RATIO_1,
            'risk_reward_2': cls.RISK_REWARD_RATIO_2
        }