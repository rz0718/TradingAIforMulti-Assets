from typing import Optional, Dict, Any, Iterable, List
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import pytz 
import math
import yfinance as yf
from parameter import INTERVAL, EMA_LEN, RSI_LEN, MACD_FAST, MACD_SLOW, MACD_SIGNAL
from indicators import calculate_indicators, calculate_atr_series, add_indicator_columns
from config_stock import SYMBOL_TO_COIN, ALPACA_API_KEY, ALPACA_SECRET_KEY
print(SYMBOL_TO_COIN)
from requests.exceptions import RequestException, Timeout
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError

data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


def get_latest_realtime_candles(symbol: str, 
                                client: StockHistoricalDataClient, 
                                timeframe: TimeFrame,
                                limit: int) -> pd.DataFrame: # <-- CHANGED RETURN TYPE
    """
    Tries to get the LATEST 'n' candles from the 'iex' REAL-TIME feed.
    Returns a Pandas DataFrame.
    """
    results = {}
    # --- Attempt 1: Try Real-Time (IEX) ---
    try:
        req_realtime = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=datetime.now(pytz.utc) - timedelta(days=5),
            end=None, 
            feed='iex'
        )
        bars_realtime = client.get_stock_bars(req_realtime)
        if symbol in bars_realtime.data and len(bars_realtime.data[symbol]) > 0:
            # Convert list of bar objects to DataFrame
            return pd.DataFrame(
                [bar.__dict__ for bar in bars_realtime.data[symbol]]
            ).tail(limit)

    except (APIError, Exception) as e:
        print(f"[Info] Real-time 'iex' failed (market likely closed): {e}")

    # --- Attempt 2: Fallback to Delayed (SIP) ---
    try:
        req_historical = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            end=datetime.now(pytz.utc) - timedelta(minutes=16),
            limit=limit,
            feed='sip' 
        )
        bars_historical = client.get_stock_bars(req_historical)
        
        if symbol in bars_historical.data and len(bars_historical.data[symbol]) > 0:
            # Convert list of bar objects to DataFrame
            return pd.DataFrame(
                [bar.__dict__ for bar in bars_historical.data[symbol]]
            )
        else:
            print(f"[Error] No data found for {symbol} on 'sip' feed.")
            return pd.DataFrame() # Return empty DataFrame on failure

    except (APIError, Exception) as e:
        print(f"[Error] Error on 'sip' fallback: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure


def get_historical_n_candles(symbol: str, 
                             client: StockHistoricalDataClient, 
                             timeframe: TimeFrame,
                             limit: int) -> pd.DataFrame: # <-- CHANGED RETURN TYPE
    """
    Fetches the 'n' latest HISTORICAL candles from the 'sip' feed
    by explicitly providing a 'start' and 'end' date.
    Returns a Pandas DataFrame.
    """
    # 1. Define our 'end' time
    end_dt = datetime.now(pytz.utc) - timedelta(minutes=16)

    # 2. Calculate a safe 'start' time
    time_per_bar = timedelta() 
    if timeframe.unit == TimeFrameUnit.Minute:
        time_per_bar = timedelta(minutes=timeframe.amount)
    elif timeframe.unit == TimeFrameUnit.Hour:
        time_per_bar = timedelta(hours=timeframe.amount)
    elif timeframe.unit == TimeFrameUnit.Day:
        time_per_bar = timedelta(days=timeframe.amount)
    elif timeframe.unit == TimeFrameUnit.Week:
        time_per_bar = timedelta(weeks=timeframe.amount)
    elif timeframe.unit == TimeFrameUnit.Month:
        time_per_bar = timedelta(days=timeframe.amount * 30) 
    else:
        time_per_bar = timedelta(days=timeframe.amount)
    
    total_time_needed = time_per_bar * limit
    
    if timeframe.unit == TimeFrameUnit.Day:
        buffer_multiplier = 1.8 
    else:
        buffer_multiplier = 3.0
        
    start_dt = end_dt - (total_time_needed * buffer_multiplier)
    start_dt = start_dt - timedelta(days=5) 

    try:
        req_historical = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
            feed='sip' 
        )
        bars_historical = client.get_stock_bars(req_historical)
        
        if symbol in bars_historical.data and len(bars_historical.data[symbol]) > 0:
            
            bars_data = bars_historical.data[symbol]
            
            if len(bars_data) < limit:
                bars_to_return = bars_data
            else:
                # Slice the *last 'limit'* bars
                bars_to_return = bars_data[-limit:] 
                
            # Convert list of bar objects to DataFrame
            return pd.DataFrame(
                [bar.__dict__ for bar in bars_to_return]
            )
        else:
            print(f"[Error] No data found for {symbol} on 'sip' feed.")
            return pd.DataFrame() # Return empty DataFrame on failure
    except (APIError, Exception) as e:
        print(f"[Error] Error on 'sip' fetch: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure


def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch current market data for a symbol."""
    try:
        # ticker = yf.Ticker(symbol)
        # df = ticker.history(interval="1m", period="7d").tail(150)
        # df = df.resample('3T').last()
        
        # df['close'] = df['Close'].astype(float)
        # df['high'] = df['High'].astype(float)
        # df['low'] = df['Low'].astype(float)
        # df['open'] = df['Open'].astype(float)
        
        # last = calculate_indicators(df)
        
        # # Get funding rate for perpetual futures
        # funding_rate = 0
        
        # return {
        #     'symbol': symbol,
        #     'price': last['close'],
        #     'ema20': last['ema20'],
        #     'rsi': last['rsi'],
        #     'macd': last['macd'],
        #     'macd_signal': last['macd_signal'],
        #     'funding_rate': funding_rate
        # }
        timeframe_3_min = TimeFrame(3, TimeFrameUnit.Minute)
        df = get_latest_realtime_candles(
            symbol=symbol, 
            client=data_client, 
            timeframe=timeframe_3_min, 
            limit=100
        )
        df = df.set_index('timestamp')
        last = calculate_indicators(df)
        return {
            'symbol': symbol,
            'price': last['close'],
            'ema20': last['ema20'],
            'rsi': last['rsi'],
            'macd': last['macd'],
            'macd_signal': last['macd_signal'],
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
        timeframe_3_min = TimeFrame(3, TimeFrameUnit.Minute)
        df = get_latest_realtime_candles(
            symbol=symbol, 
            client=data_client, 
            timeframe=timeframe_3_min, 
            limit=100
        )
        df_intraday = df.copy()
        numeric_cols = ["open", "high", "low", "close", "volume", "vwap"]
        df_intraday[numeric_cols] = df_intraday[numeric_cols].astype(float)
        df_intraday["mid_price"] = (df_intraday["high"] + df_intraday["low"]) / 2
        df_intraday = add_indicator_columns(
            df_intraday,
            ema_lengths=(EMA_LEN,),
            rsi_periods=(7, RSI_LEN),
            macd_params=(MACD_FAST, MACD_SLOW, MACD_SIGNAL),
        )

        timeframe_4_hour = TimeFrame(4, TimeFrameUnit.Hour)
        df_long = get_historical_n_candles(
            symbol=symbol, 
            client=data_client, 
            timeframe=timeframe_4_hour, 
            limit=200
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

        price = float(df_intraday["close"].iloc[-1])
        ema20 = float(df_intraday["ema20"].iloc[-1])
        rsi7 = float(df_intraday["rsi7"].iloc[-1])
        rsi14 = float(df_intraday["rsi14"].iloc[-1])
        macd_value = float(df_intraday["macd"].iloc[-1])
        vwap = float(df_intraday["vwap"].iloc[-1])

        intraday_tail = df_intraday.tail(10)
        long_tail = df_long.tail(10)


        return {
            "symbol": symbol,
            "coin": SYMBOL_TO_COIN[symbol],
            "price": price,
            "ema20": ema20,
            "rsi": rsi14,
            "rsi7": rsi7,
            "macd": macd_value,
            "macd_signal": float(df_intraday["macd_signal"].iloc[-1]),
            "vwap": vwap,
            # "funding_rate": funding_latest,
            # "funding_rates": funding_rates,
            # "open_interest": {
            #     "latest": open_interest_latest,
            #     "average": open_interest_average,
            # },
            "intraday_series": {
                "mid_prices": round_series(intraday_tail["mid_price"], 3),
                "ema20": round_series(intraday_tail["ema20"], 3),
                "macd": round_series(intraday_tail["macd"], 3),
                "rsi7": round_series(intraday_tail["rsi7"], 3),
                "rsi14": round_series(intraday_tail["rsi14"], 3),
                "vwap": round_series(intraday_tail["vwap"], 3),
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
                "vwap": round_series(long_tail["vwap"], 3),
            },
        }
    except Exception as exc:
        logging.error("Failed to build market snapshot for %s: %s", symbol, exc, exc_info=True)
        return None

if __name__ == "__main__":
    data = collect_prompt_market_data("NVDA")
