#!/usr/bin/env python3
"""
Functions for fetching market data and calculating technical indicators.
"""
import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import clients
from . import config


def calculate_rsi_series(close: pd.Series, period: int) -> pd.Series:
    """Return RSI series for specified period using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def add_indicator_columns(
    df: pd.DataFrame,
    ema_lengths: Iterable[int] = (config.EMA_LEN,),
    rsi_periods: Iterable[int] = (config.RSI_LEN,),
    macd_params: Iterable[int] = (config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL),
) -> pd.DataFrame:
    """Return copy of df with EMA, RSI, and MACD columns added."""
    ema_lengths = tuple(dict.fromkeys(ema_lengths))
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
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    alpha = 1 / period
    return true_range.ewm(alpha=alpha, adjust=False).mean()

def round_series(values: Iterable[Any], precision: int) -> List[float]:
    """Round numeric iterable to the given precision, skipping NaNs."""
    rounded: List[float] = []
    for value in values:
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        try:
            rounded.append(round(float(value), precision))
        except (TypeError, ValueError):
            continue
    return rounded

def collect_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Return rich market snapshot for a given symbol."""
    binance_client = clients.get_binance_client()
    if not binance_client:
        return None

    try:
        intraday_klines = binance_client.get_klines(symbol=symbol, interval=config.INTERVAL, limit=200)
        df_intraday = pd.DataFrame(
            intraday_klines,
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
            ],
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df_intraday[numeric_cols] = df_intraday[numeric_cols].astype(float)
        df_intraday["mid_price"] = (df_intraday["high"] + df_intraday["low"]) / 2
        df_intraday = add_indicator_columns(
            df_intraday,
            ema_lengths=(config.EMA_LEN,),
            rsi_periods=(7, config.RSI_LEN),
            macd_params=(config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL),
        )

        df_long = pd.DataFrame(
            binance_client.get_klines(symbol=symbol, interval="4h", limit=200),
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
            ],
        )
        df_long[numeric_cols] = df_long[numeric_cols].astype(float)
        df_long = add_indicator_columns(
            df_long,
            ema_lengths=(20, 50),
            rsi_periods=(14,),
            macd_params=(config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL),
        )
        df_long["atr3"] = calculate_atr_series(df_long, 3)
        df_long["atr14"] = calculate_atr_series(df_long, 14)

        try:
            oi_hist = binance_client.futures_open_interest_hist(symbol=symbol, period="5m", limit=30)
            open_interest_values = [float(entry["sumOpenInterest"]) for entry in oi_hist]
        except Exception:
            open_interest_values = []

        try:
            funding_hist = binance_client.futures_funding_rate(symbol=symbol, limit=30)
            funding_rates = [float(entry["fundingRate"]) for entry in funding_hist]
        except Exception:
            funding_rates = []

        intraday_tail = df_intraday.tail(10)
        long_tail = df_long.tail(10)

        return {
            "symbol": symbol,
            "coin": config.SYMBOL_TO_COIN[symbol],
            "price": float(df_intraday["close"].iloc[-1]),
            "ema20": float(df_intraday["ema20"].iloc[-1]),
            "rsi": float(df_intraday[f"rsi{config.RSI_LEN}"].iloc[-1]),
            "rsi7": float(df_intraday["rsi7"].iloc[-1]),
            "macd": float(df_intraday["macd"].iloc[-1]),
            "macd_signal": float(df_intraday["macd_signal"].iloc[-1]),
            "funding_rate": funding_rates[-1] if funding_rates else 0.0,
            "funding_rates": funding_rates,
            "open_interest": {
                "latest": open_interest_values[-1] if open_interest_values else None,
                "average": float(np.mean(open_interest_values)) if open_interest_values else None,
            },
            "intraday_series": {
                "mid_prices": round_series(intraday_tail["mid_price"], 3),
                "ema20": round_series(intraday_tail["ema20"], 3),
                "macd": round_series(intraday_tail["macd"], 3),
                "rsi7": round_series(intraday_tail["rsi7"], 3),
                "rsi14": round_series(intraday_tail[f"rsi{config.RSI_LEN}"], 3),
            },
            "long_term": {
                "ema20": float(df_long["ema20"].iloc[-1]),
                "ema50": float(df_long["ema50"].iloc[-1]),
                "atr3": float(df_long["atr3"].iloc[-1]),
                "atr14": float(df_long["atr14"].iloc[-1]),
                "current_volume": float(df_long["volume"].iloc[-1]),
                "average_volume": float(df_long["volume"].mean()),
                "macd": round_series(long_tail["macd"], 3),
                "rsi14": round_series(long_tail[f"rsi{config.RSI_LEN}"], 3),
            },
        }
    except Exception as exc:
        logging.error("Failed to build market snapshot for %s: %s", symbol, exc, exc_info=True)
        return None
