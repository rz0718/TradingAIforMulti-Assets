from typing import Optional, Dict, Any, Iterable, List
import pandas as pd
import numpy as np
import logging
from parameter import INTERVAL, EMA_LEN, RSI_LEN, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from indicators import calculate_indicators, calculate_atr_series, add_indicator_columns
from config_stock import SYMBOL_TO_COIN, API_KEY, API_SECRET
print(SYMBOL_TO_COIN)
import yfinance as yf
from requests.exceptions import RequestException, Timeout



def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch current market data for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval="1m", period="7d").tail(150)
        df = df.resample('3T').last()
        
        df['close'] = df['Close'].astype(float)
        df['high'] = df['High'].astype(float)
        df['low'] = df['Low'].astype(float)
        df['open'] = df['Open'].astype(float)
        
        last = calculate_indicators(df)
        
        # Get funding rate for perpetual futures
        funding_rate = 0
        
        return {
            'symbol': symbol,
            'price': last['close'],
            'ema20': last['ema20'],
            'rsi': last['rsi'],
            'macd': last['macd'],
            'macd_signal': last['macd_signal'],
            'funding_rate': funding_rate
        }
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None


def round_series(values: Iterable[Any], precision: int) -> List[float]:
    """Round numeric iterable to the given precision, skipping NaNs."""
    rounded: List[float] = []
    for value in values:
        try:
            if pd.isna(value):
                continue
        except TypeError:
            # Non-numeric/NA sentinel types fall back to ValueError later
            pass
        try:
            rounded.append(round(float(value), precision))
        except (TypeError, ValueError):
            continue
    return rounded


def collect_prompt_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Return rich market snapshot for prompt composition."""

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval="1m", period="7d").tail(600)
        df = df.resample('3T').last()
        df['close'] = df['Close'].astype(float)
        df['high'] = df['High'].astype(float)
        df['low'] = df['Low'].astype(float)
        df['open'] = df['Open'].astype(float)
        df["volume"] = df['Volume'].astype(float)

        df_intraday = df.copy()

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df_intraday[numeric_cols] = df_intraday[numeric_cols].astype(float)
        df_intraday["mid_price"] = (df_intraday["high"] + df_intraday["low"]) / 2
        df_intraday = add_indicator_columns(
            df_intraday,
            ema_lengths=(EMA_LEN,),
            rsi_periods=(7, RSI_LEN),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )

        df_long = ticker.history(interval='4h', period='3mo').tail(200)
        df_long['open'] = df_long['Open'].astype(float)
        df_long['high'] = df_long['High'].astype(float)
        df_long['low'] = df_long['Low'].astype(float)
        df_long['close'] = df_long['Close'].astype(float)
        df_long['volume'] = df_long['Volume'].astype(float)
        
        df_long[numeric_cols] = df_long[numeric_cols].astype(float)
        df_long = add_indicator_columns(
            df_long,
            ema_lengths=(20, 50),
            rsi_periods=(14,),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )
        df_long["atr3"] = calculate_atr_series(df_long, 3)
        df_long["atr14"] = calculate_atr_series(df_long, 14)


        open_interest_values = []
    
        funding_rates = []

        price = float(df_intraday["close"].iloc[-1])
        ema20 = float(df_intraday["ema20"].iloc[-1])
        rsi7 = float(df_intraday["rsi7"].iloc[-1])
        rsi14 = float(df_intraday["rsi14"].iloc[-1])
        macd_value = float(df_intraday["macd"].iloc[-1])
        funding_latest = funding_rates[-1] if funding_rates else 0.0

        intraday_tail = df_intraday.tail(10)
        long_tail = df_long.tail(10)

        open_interest_latest = open_interest_values[-1] if open_interest_values else None
        open_interest_average = (
            float(np.mean(open_interest_values)) if open_interest_values else None
        )
        funding_average = float(np.mean(funding_rates)) if funding_rates else None

        return {
            "symbol": symbol,
            "coin": SYMBOL_TO_COIN[symbol],
            "price": price,
            "ema20": ema20,
            "rsi": rsi14,
            "rsi7": rsi7,
            "macd": macd_value,
            "macd_signal": float(df_intraday["macd_signal"].iloc[-1]),
            "funding_rate": funding_latest,
            "funding_rates": funding_rates,
            "open_interest": {
                "latest": open_interest_latest,
                "average": open_interest_average,
            },
            "intraday_series": {
                "mid_prices": round_series(intraday_tail["mid_price"], 3),
                "ema20": round_series(intraday_tail["ema20"], 3),
                "macd": round_series(intraday_tail["macd"], 3),
                "rsi7": round_series(intraday_tail["rsi7"], 3),
                "rsi14": round_series(intraday_tail["rsi14"], 3),
            },
            "long_term": {
                "ema20": float(df_long["ema20"].iloc[-1]),
                "ema50": float(df_long["ema50"].iloc[-1]),
                "atr3": float(df_long["atr3"].iloc[-1]),
                "atr14": float(df_long["atr14"].iloc[-1]),
                "current_volume": float(df_long["volume"].iloc[-1]),
                "average_volume": float(df_long["volume"].mean()),
                "macd": round_series(long_tail["macd"], 3),
                "rsi14": round_series(long_tail["rsi14"], 3),
            },
        }
    except Exception as exc:
        logging.error("Failed to build market snapshot for %s: %s", symbol, exc, exc_info=True)
        return None

if __name__ == "__main__":
    print(fetch_market_data("MSFT"))
    print(collect_prompt_market_data("MSFT"))