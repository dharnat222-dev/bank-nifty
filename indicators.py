"""
Technical indicators module.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return vwap

def calculate_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator."""
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
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
    """Calculate all indicators for the strategy."""
    df = df.copy()
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['close', 'high', 'low', 'volume'])
    
    if len(df) < params.get('vwap_period', 20):
        logger.warning("Not enough data for indicator calculation")
        return df
    
    df['vwap'] = calculate_vwap(df, params['vwap_period'])
    df['ema_20'] = calculate_ema(df, params['ema_fast'])
    df['ema_50'] = calculate_ema(df, params['ema_slow'])
    df['rsi'] = calculate_rsi(df, params['rsi_period'])
    df['atr'] = calculate_atr(df, params['atr_period'])
    df['supertrend'], df['supertrend_direction'] = calculate_supertrend(
        df, params['supertrend_period'], params['supertrend_multiplier']
    )
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    
    return df