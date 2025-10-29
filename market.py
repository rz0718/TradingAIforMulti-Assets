from typing import Optional, Dict, Any, Iterable, List
import pandas as pd
import numpy as np
import logging
from parameter import INTERVAL, EMA_LEN, RSI_LEN, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from indicators import calculate_indicators, calculate_atr_series, add_indicator_columns
from config import SYMBOL_TO_COIN
from config import API_KEY, API_SECRET
from binance.client import Client
from requests.exceptions import RequestException, Timeout

client: Optional[Client] = None

def get_binance_client() -> Optional[Client]:
    """Return a connected Binance client or None if initialization failed."""
    global client

    if client is not None:
        return client

    if not API_KEY or not API_SECRET:
        logging.error("BN_API_KEY and/or BN_SECRET missing; unable to initialize Binance client.")
        return None

    try:
        logging.info("Attempting to initialize Binance client...")
        client = Client(API_KEY, API_SECRET, testnet=False)
        logging.info("Binance client initialized successfully.")
    except Timeout as exc:
        logging.warning(
            "Timed out while connecting to Binance API: %s. Will retry automatically without exiting.",
            exc,
        )
        client = None
    except RequestException as exc:
        logging.error(
            "Network error while connecting to Binance API: %s. Will retry automatically.",
            exc,
        )
        client = None
    except Exception as exc:
        logging.error(
            "Unexpected error while initializing Binance client: %s",
            exc,
            exc_info=True,
        )
        client = None

    return client


def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch current market data for a symbol."""
    binance_client = get_binance_client()
    if not binance_client:
        logging.warning("Skipping market data fetch for %s: Binance client unavailable.", symbol)
        return None

    try:
        # Get recent klines
        klines = binance_client.get_klines(symbol=symbol, interval=INTERVAL, limit=50)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'
        ])
        
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['open'] = df['open'].astype(float)
        
        last = calculate_indicators(df)
        
        # Get funding rate for perpetual futures
        try:
            funding_info = binance_client.futures_funding_rate(symbol=symbol, limit=1)
            funding_rate = float(funding_info[0]['fundingRate']) if funding_info else 0
        except:
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
    binance_client = get_binance_client()
    if not binance_client:
        return None

    try:
        intraday_klines = binance_client.get_klines(symbol=symbol, interval=INTERVAL, limit=200)
        df_intraday = pd.DataFrame(
            intraday_klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )

        numeric_cols = ["open", "high", "low", "close", "volume"]
        df_intraday[numeric_cols] = df_intraday[numeric_cols].astype(float)
        df_intraday["mid_price"] = (df_intraday["high"] + df_intraday["low"]) / 2
        df_intraday = add_indicator_columns(
            df_intraday,
            ema_lengths=(EMA_LEN,),
            rsi_periods=(7, RSI_LEN),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )

        df_long = pd.DataFrame(
            binance_client.get_klines(symbol=symbol, interval="4h", limit=200),
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
        df_long[numeric_cols] = df_long[numeric_cols].astype(float)
        df_long = add_indicator_columns(
            df_long,
            ema_lengths=(20, 50),
            rsi_periods=(14,),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )
        df_long["atr3"] = calculate_atr_series(df_long, 3)
        df_long["atr14"] = calculate_atr_series(df_long, 14)

        try:
            oi_hist = binance_client.futures_open_interest_hist(symbol=symbol, period="5m", limit=30)
            open_interest_values = [float(entry["sumOpenInterest"]) for entry in oi_hist]
        except Exception as exc:
            logging.debug("Open interest history unavailable for %s: %s", symbol, exc)
            open_interest_values = []

        try:
            funding_hist = binance_client.futures_funding_rate(symbol=symbol, limit=30)
            funding_rates = [float(entry["fundingRate"]) for entry in funding_hist]
        except Exception as exc:
            logging.debug("Funding rate history unavailable for %s: %s", symbol, exc)
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