# indicators.py
"""
Technical indicators module for Bank Nifty ATM Option Backtest Engine.
Implements all required indicators with proper calculations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
        period: Rolling period for VWAP calculation
    
    Returns:
        Series containing VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return vwap

def calculate_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        df: DataFrame with price data
        period: EMA period
        column: Column name for price data
    
    Returns:
        Series containing EMA values
    """
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        df: DataFrame with 'close' column
        period: RSI period
    
    Returns:
        Series containing RSI values
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period
    
    Returns:
        Series containing ATR values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Supertrend indicator.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period
        multiplier: ATR multiplier
    
    Returns:
        Tuple of (supertrend, direction) where direction is 1 for bullish, -1 for bearish
    """
    atr = calculate_atr(df, period)
    
    # Calculate basic upper and lower bands
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize supertrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(period, len(df)):
        if i == period:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > supertrend.iloc[i-1]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])
    
    return supertrend, direction

def calculate_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate all indicators for the strategy.
    
    Args:
        df: DataFrame with price and volume data
        params: Dictionary of indicator parameters
    
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    # Calculate VWAP
    df['vwap'] = calculate_vwap(df, params['vwap_period'])
    
    # Calculate EMAs
    df['ema_20'] = calculate_ema(df, params['ema_fast'])
    df['ema_50'] = calculate_ema(df, params['ema_slow'])
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df, params['rsi_period'])
    
    # Calculate ATR
    df['atr'] = calculate_atr(df, params['atr_period'])
    
    # Calculate Supertrend
    df['supertrend'], df['supertrend_direction'] = calculate_supertrend(
        df, 
        params['supertrend_period'], 
        params['supertrend_multiplier']
    )
    
    # Volume comparison
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    
    return df