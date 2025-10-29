import pandas as pd
import numpy as np
from typing import Iterable
from parameter import EMA_LEN, RSI_LEN, MACD_FAST, MACD_SLOW, MACD_SIGNAL

# ───────────────────────── INDICATORS ───────────────────────

def calculate_rsi_series(close: pd.Series, period: int) -> pd.Series:
    """Return RSI series for specified period using Wilder's smoothing."""
    delta = close.astype(float).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicator_columns(
    df: pd.DataFrame,
    ema_lengths: Iterable[int] = (EMA_LEN,),
    rsi_periods: Iterable[int] = (RSI_LEN,),
    macd_params: Iterable[int] = (MACD_FAST, MACD_SLOW, MACD_SIGNAL),
) -> pd.DataFrame:
    """Return copy of df with EMA, RSI, and MACD columns added."""
    ema_lengths = tuple(dict.fromkeys(ema_lengths))  # remove duplicates, preserve order
    rsi_periods = tuple(dict.fromkeys(rsi_periods))
    fast, slow, signal = macd_params

    result = df.copy()
    close = result["close"]

    for span in ema_lengths:
        result[f"ema{span}"] = close.ewm(span=span, adjust=False).mean()

    for period in rsi_periods:
        result[f"rsi{period}"] = calculate_rsi_series(close, period)

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    result["macd"] = macd_line
    result["macd_signal"] = macd_line.ewm(span=signal, adjust=False).mean()

    return result


def calculate_atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    """Return Average True Range series for the provided period."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    alpha = 1 / period
    return true_range.ewm(alpha=alpha, adjust=False).mean()


def calculate_indicators(df: pd.DataFrame) -> pd.Series:
    """Calculate technical indicators and return the latest row."""
    enriched = add_indicator_columns(
        df,
        ema_lengths=(EMA_LEN,),
        rsi_periods=(RSI_LEN,),
        macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
    )
    enriched["rsi"] = enriched[f"rsi{RSI_LEN}"]
    return enriched.iloc[-1]