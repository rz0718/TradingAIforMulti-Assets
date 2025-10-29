#!/usr/bin/env python3
"""
DeepSeek Multi-Asset Paper Trading Bot
Uses Binance API for market data and OpenRouter API for DeepSeek Chat V3.1 trading decisions
"""
from __future__ import annotations

import os
import re
import time
import json
import logging
import csv
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException, Timeout
from binance.client import Client
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init
from prompt import TRADING_RULES_PROMPT

colorama_init(autoreset=True)

BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"

if DOTENV_PATH.exists():
    dotenv_loaded = load_dotenv(dotenv_path=DOTENV_PATH, override=True)
else:
    dotenv_loaded = load_dotenv(override=True)

DEFAULT_DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────── CONFIG ─────────────────────────
API_KEY = os.getenv("BN_API_KEY", "")
API_SECRET = os.getenv("BN_SECRET", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Trading symbols to monitor
SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "DOGEUSDT", "BNBUSDT"]
SYMBOL_TO_COIN = {
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL", 
    "XRPUSDT": "XRP",
    "BTCUSDT": "BTC",
    "DOGEUSDT": "DOGE",
    "BNBUSDT": "BNB"
}


INTERVAL = "3m"  # 3-minute candles as per DeepSeek example
START_CAPITAL = 10000.0
CHECK_INTERVAL = 180  # Check every 3 minutes (when candle closes)
DEFAULT_RISK_FREE_RATE = 0.0  # Annualized baseline for Sortino ratio calculations

# Indicator settings
EMA_LEN = 20
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Binance fee structure (as decimals)
MAKER_FEE_RATE = 0.0         # 0.0000%
TAKER_FEE_RATE = 0.000275    # 0.0275%

# ───────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

def _resolve_risk_free_rate() -> float:
    """Determine the annualized risk-free rate used in Sortino calculations."""
    env_value = os.getenv("SORTINO_RISK_FREE_RATE")
    if env_value is None:
        env_value = os.getenv("RISK_FREE_RATE")
    if env_value is None:
        return DEFAULT_RISK_FREE_RATE
    try:
        return float(env_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid SORTINO_RISK_FREE_RATE/RISK_FREE_RATE value '%s'; using default %.4f",
            env_value,
            DEFAULT_RISK_FREE_RATE,
        )
        return DEFAULT_RISK_FREE_RATE

RISK_FREE_RATE = _resolve_risk_free_rate()

if not dotenv_loaded:
    logging.warning(f"No .env file found at {DOTENV_PATH}; falling back to system environment variables.")

if OPENROUTER_API_KEY:
    masked_key = (
        OPENROUTER_API_KEY
        if len(OPENROUTER_API_KEY) <= 12
        else f"{OPENROUTER_API_KEY[:6]}...{OPENROUTER_API_KEY[-4:]}"
    )
    logging.info(
        "OpenRouter API key detected: %s (length %d)",
        masked_key,
        len(OPENROUTER_API_KEY),
    )
else:
    logging.error("OPENROUTER_API_KEY not found; please check your .env file.")

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

# ──────────────────────── GLOBAL STATE ─────────────────────
balance: float = START_CAPITAL
positions: Dict[str, Dict[str, Any]] = {}  # coin -> position info
trade_history: List[Dict[str, Any]] = []
BOT_START_TIME = datetime.now(timezone.utc)
invocation_count: int = 0
iteration_counter: int = 0
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
current_iteration_messages: List[str] = []
equity_history: List[float] = []

# CSV files
STATE_CSV = DATA_DIR / "portfolio_state.csv"
STATE_JSON = DATA_DIR / "portfolio_state.json"
TRADES_CSV = DATA_DIR / "trade_history.csv"
DECISIONS_CSV = DATA_DIR / "ai_decisions.csv"
MESSAGES_CSV = DATA_DIR / "ai_messages.csv"
STATE_COLUMNS = [
    'timestamp',
    'total_balance',
    'total_equity',
    'total_return_pct',
    'num_positions',
    'position_details',
    'total_margin',
    'net_unrealized_pnl'
]

# ───────────────────────── CSV LOGGING ──────────────────────

def init_csv_files() -> None:
    """Initialize CSV files with headers."""
    if not STATE_CSV.exists():
        with open(STATE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(STATE_COLUMNS)
    
    if not TRADES_CSV.exists():
        with open(TRADES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'coin', 'action', 'side', 'quantity', 'price',
                'profit_target', 'stop_loss', 'leverage', 'confidence',
                'pnl', 'balance_after', 'reason'
            ])
    
    if not DECISIONS_CSV.exists():
        with open(DECISIONS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'coin', 'signal', 'reasoning', 'confidence'
            ])

    if not MESSAGES_CSV.exists():
        with open(MESSAGES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'direction', 'role', 'content', 'metadata'
            ])

def log_portfolio_state() -> None:
    """Log current portfolio state."""
    total_equity = calculate_total_equity()
    total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
    total_margin = calculate_total_margin()
    net_unrealized = total_equity - balance - total_margin
    
    position_details = "; ".join([
        f"{coin}:{pos['side']}:{pos['quantity']:.4f}@{pos['entry_price']:.4f}"
        for coin, pos in positions.items()
    ]) if positions else "No positions"
    
    with open(STATE_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            f"{balance:.2f}",
            f"{total_equity:.2f}",
            f"{total_return:.2f}",
            len(positions),
            position_details,
            f"{total_margin:.2f}",
            f"{net_unrealized:.2f}"
        ])

def log_trade(coin: str, action: str, details: Dict[str, Any]) -> None:
    """Log trade execution."""
    with open(TRADES_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            coin,
            action,
            details.get('side', ''),
            details.get('quantity', 0),
            details.get('price', 0),
            details.get('profit_target', 0),
            details.get('stop_loss', 0),
            details.get('leverage', 1),
            details.get('confidence', 0),
            details.get('pnl', 0),
            balance,
            details.get('reason', '')
        ])

def log_ai_decision(coin: str, signal: str, reasoning: str, confidence: float) -> None:
    """Log AI decision."""
    with open(DECISIONS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            coin,
            signal,
            reasoning,
            confidence
        ])


def log_ai_message(direction: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log raw messages exchanged with the AI provider."""
    with open(MESSAGES_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            direction,
            role,
            content,
            json.dumps(metadata) if metadata else ""
        ])

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes so Telegram receives plain text."""
    return ANSI_ESCAPE_RE.sub("", text)

def record_iteration_message(text: str) -> None:
    """Record console output for this iteration to share via Telegram."""
    if current_iteration_messages is not None:
        current_iteration_messages.append(strip_ansi_codes(text).rstrip())

def send_telegram_message(text: str) -> None:
    """Send a notification message to Telegram if credentials are configured."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
            },
            timeout=10,
        )
        if response.status_code != 200:
            logging.warning(
                "Telegram notification failed (%s): %s",
                response.status_code,
                response.text,
            )
    except Exception as exc:
        logging.error("Error sending Telegram message: %s", exc)
        
def notify_error(
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    log_error: bool = True,
) -> None:
    """Log an error and forward a brief description to Telegram."""
    if log_error:
        logging.error(message)
    log_ai_message(
        direction="error",
        role="system",
        content=message,
        metadata=metadata,
    )
    send_telegram_message(message)

# ───────────────────────── STATE MGMT ───────────────────────

def load_state() -> None:
    """Load persisted balance and positions if available."""
    global balance, positions, iteration_counter

    if not STATE_JSON.exists():
        logging.info("No existing state file found; starting fresh.")
        return

    try:
        with open(STATE_JSON, "r") as f:
            data = json.load(f)

        balance = float(data.get("balance", START_CAPITAL))
        try:
            iteration_counter = int(data.get("iteration", 0))
        except (TypeError, ValueError):
            iteration_counter = 0
        loaded_positions = data.get("positions", {})
        if isinstance(loaded_positions, dict):
            restored_positions: Dict[str, Dict[str, Any]] = {}
            for coin, pos in loaded_positions.items():
                if not isinstance(pos, dict):
                    continue
                fees_paid_raw = pos.get("fees_paid", pos.get("entry_fee", 0.0))
                if fees_paid_raw is None:
                    fees_paid_value = 0.0
                else:
                    try:
                        fees_paid_value = float(fees_paid_raw)
                    except (TypeError, ValueError):
                        fees_paid_value = 0.0

                fee_rate_raw = pos.get("fee_rate", TAKER_FEE_RATE)
                try:
                    fee_rate_value = float(fee_rate_raw)
                except (TypeError, ValueError):
                    fee_rate_value = TAKER_FEE_RATE

                restored_positions[coin] = {
                    "side": pos.get("side", "long"),
                    "quantity": float(pos.get("quantity", 0.0)),
                    "entry_price": float(pos.get("entry_price", 0.0)),
                    "profit_target": float(pos.get("profit_target", 0.0)),
                    "stop_loss": float(pos.get("stop_loss", 0.0)),
                    "leverage": float(pos.get("leverage", 1)),
                    "confidence": float(pos.get("confidence", 0.0)),
                    "invalidation_condition": pos.get("invalidation_condition", ""),
                    "margin": float(pos.get("margin", 0.0)),
                    "fees_paid": fees_paid_value,
                    "fee_rate": fee_rate_value,
                    "liquidity": pos.get("liquidity", "taker"),
                    "entry_justification": pos.get("entry_justification", ""),
                    "last_justification": pos.get("last_justification", pos.get("entry_justification", "")),
                }
            positions = restored_positions
        logging.info(
            "Loaded state from %s (balance: %.2f, positions: %d)",
            STATE_JSON,
            balance,
            len(positions),
        )
    except Exception as e:
        logging.error("Failed to load state from %s: %s", STATE_JSON, e, exc_info=True)
        balance = START_CAPITAL
        positions = {}

def save_state() -> None:
    """Persist current balance, open positions, and iteration counter."""
    try:
        with open(STATE_JSON, "w") as f:
            json.dump(
                {
                    "balance": balance,
                    "positions": positions,
                    "iteration": iteration_counter,
                    "updated_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logging.error("Failed to save state to %s: %s", STATE_JSON, e, exc_info=True)

def load_equity_history() -> None:
    """Populate the in-memory equity history for performance calculations."""
    equity_history.clear()
    if not STATE_CSV.exists():
        return
    try:
        df = pd.read_csv(STATE_CSV, usecols=["total_equity"])
    except ValueError:
        logging.warning(
            "%s missing 'total_equity' column; Sortino ratio unavailable until new data is logged.",
            STATE_CSV,
        )
        return
    except Exception as exc:
        logging.warning("Unable to load historical equity data: %s", exc)
        return

    values = pd.to_numeric(df["total_equity"], errors="coerce").dropna()
    if not values.empty:
        equity_history.extend(float(v) for v in values.tolist())

def register_equity_snapshot(total_equity: float) -> None:
    """Append the latest equity to the history if it is a finite value."""
    if total_equity is None:
        return
    if isinstance(total_equity, (int, float, np.floating)) and np.isfinite(total_equity):
        equity_history.append(float(total_equity))

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

# ───────────────────── AI DECISION MAKING ───────────────────

def format_prompt_for_deepseek() -> str:
    """Compose a rich prompt resembling the original DeepSeek in-context format."""
    global invocation_count
    invocation_count += 1

    now = datetime.now(timezone.utc)
    minutes_running = int((now - BOT_START_TIME).total_seconds() // 60)

    market_snapshots: Dict[str, Dict[str, Any]] = {}
    for symbol in SYMBOLS:
        snapshot = collect_prompt_market_data(symbol)
        if snapshot:
            market_snapshots[snapshot["coin"]] = snapshot

    total_margin = calculate_total_margin()
    total_equity = balance + total_margin
    for coin, pos in positions.items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        total_equity += calculate_unrealized_pnl(coin, current_price)

    total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100 if START_CAPITAL else 0.0
    net_unrealized_total = total_equity - balance - total_margin

    def fmt(value: Optional[float], digits: int = 3) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{digits}f}"

    def fmt_rate(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.6g}"

    prompt_lines: List[str] = []
    prompt_lines.append(
        f"It has been {minutes_running} minutes since you started trading. "
        f"The current time is {now.isoformat()} and you've been invoked {invocation_count} times. "
        "Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. "
        "Below that is your current account information, value, performance, positions, etc."
    )
    prompt_lines.append("ALL PRICE OR SIGNAL SERIES BELOW ARE ORDERED OLDEST → NEWEST.")
    prompt_lines.append(
        "Timeframe note: Intraday series use 3-minute intervals unless a different interval is explicitly mentioned."
    )
    prompt_lines.append("-" * 80)
    prompt_lines.append("CURRENT MARKET STATE FOR ALL COINS")

    for symbol in SYMBOLS:
        coin = SYMBOL_TO_COIN[symbol]
        data = market_snapshots.get(coin)
        if not data:
            continue

        intraday = data["intraday_series"]
        long_term = data["long_term"]
        open_interest = data["open_interest"]
        funding_rates = data.get("funding_rates", [])
        funding_avg_str = fmt_rate(float(np.mean(funding_rates))) if funding_rates else "N/A"

        prompt_lines.append(f"{coin} MARKET SNAPSHOT")
        prompt_lines.append(
            f"- Price: {fmt(data['price'], 3)}, EMA20: {fmt(data['ema20'], 3)}, MACD: {fmt(data['macd'], 3)}, RSI(7): {fmt(data['rsi7'], 3)}"
        )
        prompt_lines.append(
            f"- Open Interest (latest/avg): {fmt(open_interest.get('latest'), 2)} / {fmt(open_interest.get('average'), 2)}"
        )
        prompt_lines.append(
            f"- Funding Rate (latest/avg): {fmt_rate(data['funding_rate'])} / {funding_avg_str}"
        )
        prompt_lines.append("  Intraday series (3-minute, oldest → latest):")
        prompt_lines.append(f"    mid_prices: {json.dumps(intraday['mid_prices'])}")
        prompt_lines.append(f"    ema20: {json.dumps(intraday['ema20'])}")
        prompt_lines.append(f"    macd: {json.dumps(intraday['macd'])}")
        prompt_lines.append(f"    rsi7: {json.dumps(intraday['rsi7'])}")
        prompt_lines.append(f"    rsi14: {json.dumps(intraday['rsi14'])}")
        prompt_lines.append("  Longer-term context (4-hour timeframe):")
        prompt_lines.append(
            f"    EMA20 vs EMA50: {fmt(long_term['ema20'], 3)} / {fmt(long_term['ema50'], 3)}"
        )
        prompt_lines.append(
            f"    ATR3 vs ATR14: {fmt(long_term['atr3'], 3)} / {fmt(long_term['atr14'], 3)}"
        )
        prompt_lines.append(
            f"    Volume (current/average): {fmt(long_term['current_volume'], 3)} / {fmt(long_term['average_volume'], 3)}"
        )
        prompt_lines.append(f"    MACD series: {json.dumps(long_term['macd'])}")
        prompt_lines.append(f"    RSI14 series: {json.dumps(long_term['rsi14'])}")
        prompt_lines.append("-" * 80)

    prompt_lines.append("ACCOUNT INFORMATION AND PERFORMANCE")
    prompt_lines.append(f"- Total Return (%): {fmt(total_return, 2)}")
    prompt_lines.append(f"- Available Cash: {fmt(balance, 2)}")
    prompt_lines.append(f"- Margin Allocated: {fmt(total_margin, 2)}")
    prompt_lines.append(f"- Unrealized PnL: {fmt(net_unrealized_total, 2)}")
    prompt_lines.append(f"- Current Account Value: {fmt(total_equity, 2)}")
    prompt_lines.append("Open positions and performance details:")

    for coin, pos in positions.items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        quantity = pos["quantity"]
        gross_unrealized = calculate_unrealized_pnl(coin, current_price)
        leverage = pos.get("leverage", 1) or 1
        if pos["side"] == "long":
            liquidation_price = pos["entry_price"] * max(0.0, 1 - 1 / leverage)
        else:
            liquidation_price = pos["entry_price"] * (1 + 1 / leverage)
        notional_value = quantity * current_price
        position_payload = {
            "symbol": coin,
            "side": pos["side"],
            "quantity": quantity,
            "entry_price": pos["entry_price"],
            "current_price": current_price,
            "liquidation_price": liquidation_price,
            "unrealized_pnl": gross_unrealized,
            "leverage": pos.get("leverage", 1),
            "exit_plan": {
                "profit_target": pos.get("profit_target"),
                "stop_loss": pos.get("stop_loss"),
                "invalidation_condition": pos.get("invalidation_condition"),
            },
            "confidence": pos.get("confidence", 0.0),
            "risk_usd": pos.get("risk_usd"),
            "sl_oid": pos.get("sl_oid", -1),
            "tp_oid": pos.get("tp_oid", -1),
            "wait_for_fill": pos.get("wait_for_fill", False),
            "entry_oid": pos.get("entry_oid", -1),
            "notional_usd": notional_value,
        }
        prompt_lines.append(f"{coin} position data: {json.dumps(position_payload)}")

    sharpe_ratio = 0.0
    prompt_lines.append(f"Sharpe Ratio: {fmt(sharpe_ratio, 3)}")

    prompt_lines.append(
        """
INSTRUCTIONS:
For each coin, provide a trading decision in JSON format. You can either:
1. "hold" - Keep current position (if you have one)
2. "entry" - Open a new position (if you don't have one)
3. "close" - Close current position

Return ONLY a valid JSON object with this structure:
{
  "ETH": {
    "signal": "hold|entry|close",
    "side": "long|short",  // only for entry
    "quantity": 0.0,
    "profit_target": 0.0,
    "stop_loss": 0.0,
    "leverage": 10,
    "confidence": 0.75,
    "risk_usd": 500.0,
    "invalidation_condition": "If price closes below X on a 3-minute candle",
    "justification": "Reason for entry/close/hold"
  }
}

IMPORTANT:
- Only suggest entries if you see strong opportunities
- Use proper risk management
- Provide clear invalidation conditions
- Return ONLY valid JSON, no other text
""".strip()
    )

    return "\n".join(prompt_lines)

def call_deepseek_api(prompt: str) -> Optional[Dict[str, Any]]:
    """Call OpenRouter API with DeepSeek Chat V3.1."""
    try:
        log_ai_message(
            direction="sent",
            role="system",
            content=TRADING_RULES_PROMPT,
            metadata={
                "model": "deepseek/deepseek-chat-v3.1",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        )
        log_ai_message(
            direction="sent",
            role="user",
            content=prompt,
            metadata={
                "model": "deepseek/deepseek-chat-v3.1",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        )

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/crypto-trading-bot",
                "X-Title": "DeepSeek Trading Bot",
            },
            json={
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [
                    {
                        "role": "system",
                        "content": TRADING_RULES_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 4000
            },
            timeout=30
        )

        if response.status_code != 200:
            notify_error(
                f"OpenRouter API error: {response.status_code}",
                metadata={
                    "status_code": response.status_code,
                    "response_text": response.text,
                },
            )
            return None

        result = response.json()
        content = result['choices'][0]['message']['content']

        log_ai_message(
            direction="received",
            role="assistant",
            content=content,
            metadata={
                "status_code": response.status_code,
                "response_id": result.get("id"),
                "usage": result.get("usage")
            }
        )

        # Extract JSON from response (in case there's extra text)
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            try:
                decisions = json.loads(json_str)
                return decisions
            except json.JSONDecodeError as decode_err:
                snippet = json_str[:2000]
                notify_error(
                    f"DeepSeek JSON decode failed: {decode_err}",
                    metadata={
                        "response_id": result.get("id"),
                        "status_code": response.status_code,
                        "raw_json_excerpt": snippet,
                    },
                )
                return None
        else:
            notify_error(
                "No JSON found in DeepSeek response",
                metadata={
                    "response_id": result.get("id"),
                    "status_code": response.status_code,
                },
            )
            return None
            
    except Exception as e:
        logging.exception("Error calling DeepSeek API")
        notify_error(
            f"Error calling DeepSeek API: {e}",
            metadata={"context": "call_deepseek_api"},
            log_error=False,
        )
        return None

# ───────────────────── POSITION MANAGEMENT ──────────────────

def calculate_unrealized_pnl(coin: str, current_price: float) -> float:
    """Calculate unrealized PnL for a position."""
    if coin not in positions:
        return 0.0
    
    pos = positions[coin]
    if pos['side'] == 'long':
        pnl = (current_price - pos['entry_price']) * pos['quantity']
    else:  # short
        pnl = (pos['entry_price'] - current_price) * pos['quantity']
    
    return pnl

def calculate_net_unrealized_pnl(coin: str, current_price: float) -> float:
    """Calculate unrealized PnL after subtracting fees already paid."""
    gross_pnl = calculate_unrealized_pnl(coin, current_price)
    fees_paid = positions.get(coin, {}).get('fees_paid', 0.0)
    return gross_pnl - fees_paid

def calculate_pnl_for_price(pos: Dict[str, Any], target_price: float) -> float:
    """Return gross PnL for a hypothetical exit price."""
    try:
        quantity = float(pos.get('quantity', 0.0))
        entry_price = float(pos.get('entry_price', 0.0))
    except (TypeError, ValueError):
        return 0.0
    side = str(pos.get('side', 'long')).lower()
    if side == 'short':
        return (entry_price - target_price) * quantity
    return (target_price - entry_price) * quantity

def estimate_exit_fee(pos: Dict[str, Any], exit_price: float) -> float:
    """Estimate taker/maker fee required to exit the position at the given price."""
    try:
        quantity = float(pos.get('quantity', 0.0))
    except (TypeError, ValueError):
        quantity = 0.0
    fee_rate = pos.get('fee_rate', TAKER_FEE_RATE)
    try:
        fee_rate_value = float(fee_rate)
    except (TypeError, ValueError):
        fee_rate_value = TAKER_FEE_RATE
    estimated_fee = quantity * exit_price * fee_rate_value
    return max(estimated_fee, 0.0)

def format_leverage_display(leverage: Any) -> str:
    """Return leverage formatted as '<value>x' while handling strings gracefully."""
    if leverage is None:
        return "n/a"
    if isinstance(leverage, str):
        cleaned = leverage.strip()
        if not cleaned:
            return "n/a"
        if cleaned.lower().endswith('x'):
            return cleaned.lower()
        try:
            value = float(cleaned)
        except (TypeError, ValueError):
            return cleaned
    else:
        try:
            value = float(leverage)
        except (TypeError, ValueError):
            return str(leverage)
    if value.is_integer():
        return f"{int(value)}x"
    return f"{value:g}x"

def calculate_total_margin() -> float:
    """Return sum of margin allocated across all open positions."""
    return sum(float(pos.get('margin', 0.0)) for pos in positions.values())

def calculate_total_equity() -> float:
    """Calculate total equity (balance + unrealized PnL)."""
    total = balance + calculate_total_margin()
    
    for coin in positions:
        symbol = next((s for s, c in SYMBOL_TO_COIN.items() if c == coin), None)
        if not symbol:
            continue
        data = fetch_market_data(symbol)
        if data:
            total += calculate_unrealized_pnl(coin, data['price'])
    
    return total

def calculate_sortino_ratio(
    equity_values: Iterable[float],
    period_seconds: float,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Optional[float]:
    """
    Compute the annualized Sortino ratio from equity snapshots.

    Args:
        equity_values: Sequence of equity values in chronological order.
        period_seconds: Average period between snapshots (used to annualize).
        risk_free_rate: Annualized risk-free rate (decimal form).
    """
    values = [float(v) for v in equity_values if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]
    if len(values) < 2:
        return None

    returns = np.diff(values) / np.array(values[:-1], dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0:
        return None

    period_seconds = float(period_seconds) if period_seconds and period_seconds > 0 else CHECK_INTERVAL
    periods_per_year = (365 * 24 * 60 * 60) / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    per_period_rf = risk_free_rate / periods_per_year
    excess_return = returns.mean() - per_period_rf
    if not np.isfinite(excess_return):
        return None

    downside_diff = np.minimum(returns - per_period_rf, 0.0)
    downside_squared = downside_diff ** 2
    downside_deviation = np.sqrt(np.mean(downside_squared))
    if downside_deviation <= 0 or not np.isfinite(downside_deviation):
        return None

    sortino = (excess_return / downside_deviation) * np.sqrt(periods_per_year)
    if not np.isfinite(sortino):
        return None
    return float(sortino)

def execute_entry(coin: str, decision: Dict[str, Any], current_price: float) -> None:
    """Execute entry trade."""
    global balance
    
    if coin in positions:
        logging.warning(f"{coin}: Already have position, skipping entry")
        return
    
    side = decision.get('side', 'long')
    leverage = decision.get('leverage', 10)
    leverage_display = format_leverage_display(leverage)
    risk_usd = decision.get('risk_usd', balance * 0.01)
    
    # Calculate position size based on risk
    stop_distance = abs(current_price - decision['stop_loss'])
    if stop_distance == 0:
        logging.warning(f"{coin}: Invalid stop loss, skipping")
        return
    
    quantity = risk_usd / stop_distance
    position_value = quantity * current_price
    margin_required = position_value / leverage
    
    liquidity = decision.get('liquidity', 'taker').lower()
    fee_rate = decision.get('fee_rate')
    if fee_rate is not None:
        try:
            fee_rate = float(fee_rate)
        except (TypeError, ValueError):
            logging.warning(f"{coin}: Invalid fee_rate provided ({fee_rate}); defaulting to Binance schedule.")
            fee_rate = None
    if fee_rate is None:
        fee_rate = MAKER_FEE_RATE if liquidity == 'maker' else TAKER_FEE_RATE
    entry_fee = position_value * fee_rate
    
    total_cost = margin_required + entry_fee
    if total_cost > balance:
        logging.warning(
            f"{coin}: Insufficient balance ${balance:.2f} for margin ${margin_required:.2f} "
            f"and fees ${entry_fee:.2f}"
        )
        return
    
    # Open position
    raw_reason = str(decision.get('justification', '')).strip()
    positions[coin] = {
        'side': side,
        'quantity': quantity,
        'entry_price': current_price,
        'profit_target': decision['profit_target'],
        'stop_loss': decision['stop_loss'],
        'leverage': leverage,
        'confidence': decision.get('confidence', 0.5),
        'invalidation_condition': decision.get('invalidation_condition', ''),
        'margin': margin_required,
        'fees_paid': entry_fee,
        'fee_rate': fee_rate,
        'liquidity': liquidity,
        'risk_usd': risk_usd,
        'wait_for_fill': decision.get('wait_for_fill', False),
        'entry_oid': decision.get('entry_oid', -1),
        'tp_oid': decision.get('tp_oid', -1),
        'sl_oid': decision.get('sl_oid', -1),
        'entry_justification': raw_reason,
        'last_justification': raw_reason,
    }
    
    balance -= total_cost
    
    entry_price = current_price
    target_price = decision['profit_target']
    stop_price = decision['stop_loss']

    gross_at_target = calculate_pnl_for_price(positions[coin], target_price)
    gross_at_stop = calculate_pnl_for_price(positions[coin], stop_price)
    exit_fee_target = estimate_exit_fee(positions[coin], target_price)
    exit_fee_stop = estimate_exit_fee(positions[coin], stop_price)
    net_at_target = gross_at_target - (entry_fee + exit_fee_target)
    net_at_stop = gross_at_stop - (entry_fee + exit_fee_stop)

    expected_reward = max(gross_at_target, 0.0)
    expected_risk = max(-gross_at_stop, 0.0)
    if expected_risk > 0:
        rr_value = expected_reward / expected_risk if expected_reward > 0 else 0.0
        rr_display = f"{rr_value:.2f}:1"
    else:
        rr_display = "n/a"

    line = f"{Fore.GREEN}[ENTRY] {coin} {side.upper()} {leverage_display} @ ${entry_price:.4f}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Size: {quantity:.4f} {coin} | Margin: ${margin_required:.2f}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Risk: ${risk_usd:.2f} | Liquidity: {liquidity}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Target: ${target_price:.4f} | Stop: ${stop_price:.4f}"
    print(line)
    record_iteration_message(line)
    reason_text = raw_reason or "No justification provided."
    reason_text = " ".join(reason_text.split())

    gross_target_sign = '+' if gross_at_target >= 0 else '-'
    gross_stop_sign = '+' if gross_at_stop >= 0 else '-'
    net_target_sign = '+' if net_at_target >= 0 else '-'
    net_stop_sign = '+' if net_at_stop >= 0 else '-'

    line = (
        f"  ├─ PnL @ Target: {gross_target_sign}${abs(gross_at_target):.2f} "
        f"(Net: {net_target_sign}${abs(net_at_target):.2f})"
    )
    print(line)
    record_iteration_message(line)
    line = (
        f"  ├─ PnL @ Stop: {gross_stop_sign}${abs(gross_at_stop):.2f} "
        f"(Net: {net_stop_sign}${abs(net_at_stop):.2f})"
    )
    print(line)
    record_iteration_message(line)
    if entry_fee > 0:
        line = f"  ├─ Estimated Fee: ${entry_fee:.2f} ({liquidity} @ {fee_rate*100:.4f}%)"
        print(line)
        record_iteration_message(line)
    line = f"  ├─ Confidence: {decision.get('confidence', 0)*100:.0f}%"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Reward/Risk: {rr_display}"
    print(line)
    record_iteration_message(line)
    line = f"  └─ Reason: {reason_text}"
    print(line)
    record_iteration_message(line)
    
    log_trade(coin, 'ENTRY', {
        'side': side,
        'quantity': quantity,
        'price': current_price,
        'profit_target': decision['profit_target'],
        'stop_loss': decision['stop_loss'],
        'leverage': leverage,
        'confidence': decision.get('confidence', 0),
        'pnl': 0,
        'reason': f"{reason_text or 'AI entry signal'} | Fees: ${entry_fee:.2f}"
    })
    save_state()

def execute_close(coin: str, decision: Dict[str, Any], current_price: float) -> None:
    """Execute close trade."""
    global balance
    
    if coin not in positions:
        logging.warning(f"{coin}: No position to close")
        return
    
    pos = positions[coin]
    raw_reason = str(decision.get('justification', '')).strip()
    reason_text = raw_reason or pos.get('last_justification') or "AI close signal"
    reason_text = " ".join(reason_text.split())
    
    pnl = calculate_unrealized_pnl(coin, current_price)
    
    fee_rate = pos.get('fee_rate', TAKER_FEE_RATE)
    exit_fee = pos['quantity'] * current_price * fee_rate
    total_fees = pos.get('fees_paid', 0.0) + exit_fee
    net_pnl = pnl - total_fees
    
    # Return margin and add net PnL (after fees)
    balance += pos['margin'] + net_pnl
    
    color = Fore.GREEN if net_pnl >= 0 else Fore.RED
    line = f"{color}[CLOSE] {coin} {pos['side'].upper()} {pos['quantity']:.4f} @ ${current_price:.4f}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Entry: ${pos['entry_price']:.4f} | Gross PnL: ${pnl:.2f}"
    print(line)
    record_iteration_message(line)
    if total_fees > 0:
        line = f"  ├─ Fees Paid: ${total_fees:.2f} (includes exit fee ${exit_fee:.2f})"
        print(line)
        record_iteration_message(line)
    line = f"  ├─ Net PnL: ${net_pnl:.2f}"
    print(line)
    record_iteration_message(line)
    line = f"  ├─ Reason: {reason_text}"
    print(line)
    record_iteration_message(line)
    line = f"  └─ Balance: ${balance:.2f}"
    print(line)
    record_iteration_message(line)
    
    log_trade(coin, 'CLOSE', {
        'side': pos['side'],
        'quantity': pos['quantity'],
        'price': current_price,
        'profit_target': 0,
        'stop_loss': 0,
        'leverage': pos['leverage'],
        'confidence': 0,
        'pnl': net_pnl,
        'reason': (
            f"{reason_text} | "
            f"Gross: ${pnl:.2f} | Fees: ${total_fees:.2f}"
        )
    })
    
    del positions[coin]
    save_state()

def check_stop_loss_take_profit() -> None:
    """Check and execute stop loss / take profit for all positions."""
    for coin in list(positions.keys()):
        symbol = [s for s, c in SYMBOL_TO_COIN.items() if c == coin][0]
        data = fetch_market_data(symbol)
        if not data:
            continue
        
        current_price = data['price']
        pos = positions[coin]
        
        # Check stop loss
        if pos['side'] == 'long' and current_price <= pos['stop_loss']:
            execute_close(coin, {'justification': 'Stop loss hit'}, current_price)
        elif pos['side'] == 'short' and current_price >= pos['stop_loss']:
            execute_close(coin, {'justification': 'Stop loss hit'}, current_price)
        
        # Check take profit
        elif pos['side'] == 'long' and current_price >= pos['profit_target']:
            execute_close(coin, {'justification': 'Take profit hit'}, current_price)
        elif pos['side'] == 'short' and current_price <= pos['profit_target']:
            execute_close(coin, {'justification': 'Take profit hit'}, current_price)

# ─────────────────────────── MAIN ──────────────────────────

def main() -> None:
    """Main trading loop."""
    global current_iteration_messages, iteration_counter
    logging.info("Initializing DeepSeek Multi-Asset Paper Trading Bot...")
    init_csv_files()
    load_equity_history()
    load_state()
    
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY not found in .env file")
        return
    
    logging.info(f"Starting capital: ${START_CAPITAL:.2f}")
    logging.info(f"Monitoring: {', '.join(SYMBOL_TO_COIN.values())}")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        logging.info("Telegram notifications enabled (chat: %s).", TELEGRAM_CHAT_ID)
    else:
        logging.info("Telegram notifications disabled; missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")
    
    while True:
        try:
            iteration_counter += 1
            current_iteration_messages = []

            if not get_binance_client():
                retry_delay = min(CHECK_INTERVAL, 60)
                logging.warning(
                    "Binance client unavailable; retrying in %d seconds without exiting.",
                    retry_delay,
                )
                time.sleep(retry_delay)
                continue

            line = f"\n{Fore.CYAN}{'='*20}"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.CYAN}Iteration {iteration_counter} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.CYAN}{'='*20}\n"
            print(line)
            record_iteration_message(line)
            
            # Check stop loss / take profit first
            check_stop_loss_take_profit()
            
            # Get AI decisions
            logging.info("Requesting trading decisions from DeepSeek...")
            prompt = format_prompt_for_deepseek()
            decisions = call_deepseek_api(prompt)
            
            if not decisions:
                logging.warning("No decisions received from AI")
            else:
                # Process decisions for each coin
                for coin in SYMBOL_TO_COIN.values():
                    if coin not in decisions:
                        continue
                    
                    decision = decisions[coin]
                    signal = decision.get('signal', 'hold')
                    
                    # Log AI decision
                    log_ai_decision(
                        coin,
                        signal,
                        decision.get('justification', ''),
                        decision.get('confidence', 0)
                    )
                    
                    # Get current price
                    symbol = [s for s, c in SYMBOL_TO_COIN.items() if c == coin][0]
                    data = fetch_market_data(symbol)
                    if not data:
                        continue
                    
                    current_price = data['price']
                    
                    # Execute decision
                    if signal == 'entry':
                        execute_entry(coin, decision, current_price)
                    elif signal == 'close':
                        execute_close(coin, decision, current_price)
                    elif signal == 'hold':
                        if coin in positions:
                            pos = positions[coin]
                            raw_reason = str(decision.get('justification', '')).strip()
                            if raw_reason:
                                reason_text = " ".join(raw_reason.split())
                                pos['last_justification'] = reason_text
                            else:
                                existing_reason = str(pos.get('last_justification', '')).strip()
                                reason_text = existing_reason or "No justification provided."
                                if not existing_reason:
                                    pos['last_justification'] = reason_text
                            try:
                                quantity = float(pos.get('quantity', 0.0))
                            except (TypeError, ValueError):
                                quantity = 0.0
                            try:
                                fees_paid = float(pos.get('fees_paid', 0.0))
                            except (TypeError, ValueError):
                                fees_paid = 0.0
                            try:
                                entry_price = float(pos.get('entry_price', 0.0))
                            except (TypeError, ValueError):
                                entry_price = 0.0
                            try:
                                target_price = float(pos.get('profit_target', entry_price))
                            except (TypeError, ValueError):
                                target_price = entry_price
                            try:
                                stop_price = float(pos.get('stop_loss', entry_price))
                            except (TypeError, ValueError):
                                stop_price = entry_price
                            leverage_display = format_leverage_display(pos.get('leverage', 1.0))
                            try:
                                margin_value = float(pos.get('margin', 0.0))
                            except (TypeError, ValueError):
                                margin_value = 0.0
                            try:
                                risk_value = float(pos.get('risk_usd', 0.0))
                            except (TypeError, ValueError):
                                risk_value = 0.0

                            gross_unrealized = calculate_unrealized_pnl(coin, current_price)
                            estimated_exit_fee_now = estimate_exit_fee(pos, current_price)
                            total_fees_now = fees_paid + estimated_exit_fee_now
                            net_unrealized = gross_unrealized - total_fees_now

                            gross_at_target = calculate_pnl_for_price(pos, target_price)
                            exit_fee_target = estimate_exit_fee(pos, target_price)
                            net_at_target = gross_at_target - (fees_paid + exit_fee_target)

                            gross_at_stop = calculate_pnl_for_price(pos, stop_price)
                            exit_fee_stop = estimate_exit_fee(pos, stop_price)
                            net_at_stop = gross_at_stop - (fees_paid + exit_fee_stop)

                            expected_reward = max(gross_at_target, 0.0)
                            expected_risk = max(-gross_at_stop, 0.0)
                            if expected_risk > 0:
                                rr_value = expected_reward / expected_risk if expected_reward > 0 else 0.0
                                rr_display = f"{rr_value:.2f}:1"
                            else:
                                rr_display = "n/a"

                            pnl_color = Fore.GREEN if net_unrealized >= 0 else Fore.RED
                            net_sign = '+' if net_unrealized >= 0 else '-'
                            net_display = f"{net_sign}${abs(net_unrealized):.2f}"
                            gross_sign = '+' if gross_unrealized >= 0 else '-'
                            gross_display = f"{gross_sign}${abs(gross_unrealized):.2f}"

                            gross_target_sign = '+' if gross_at_target >= 0 else '-'
                            gross_target_display = f"{gross_target_sign}${abs(gross_at_target):.2f}"
                            gross_stop_sign = '+' if gross_at_stop >= 0 else '-'
                            gross_stop_display = f"{gross_stop_sign}${abs(gross_at_stop):.2f}"

                            net_target_sign = '+' if net_at_target >= 0 else '-'
                            net_target_display = f"{net_target_sign}${abs(net_at_target):.2f}"
                            net_stop_sign = '+' if net_at_stop >= 0 else '-'
                            net_stop_display = f"{net_stop_sign}${abs(net_at_stop):.2f}"

                            line = (
                                f"[HOLD] {coin} {pos['side'].upper()} {leverage_display} @ ${entry_price:.4f} | "
                                f"Current: ${current_price:.4f}"
                            )
                            print(line)
                            record_iteration_message(line)
                            line = f"  ├─ Size: {quantity:.4f} {coin} | Margin: ${margin_value:.2f}"
                            print(line)
                            record_iteration_message(line)
                            line = f"  ├─ TP: ${target_price:.4f} | SL: ${stop_price:.4f}"
                            print(line)
                            record_iteration_message(line)
                            line = (
                                f"  ├─ PnL: {pnl_color}{net_display}{Style.RESET_ALL} "
                                f"(Gross: {gross_display}, Fees: ${total_fees_now:.2f})"
                            )
                            print(line)
                            record_iteration_message(line)
                            line = (
                                f"  ├─ PnL @ Target: {gross_target_display} "
                                f"(Net: {net_target_display})"
                            )
                            print(line)
                            record_iteration_message(line)
                            line = (
                                f"  ├─ PnL @ Stop: {gross_stop_display} "
                                f"(Net: {net_stop_display})"
                            )
                            print(line)
                            record_iteration_message(line)
                            line = f"  ├─ Reward/Risk: {rr_display}"
                            print(line)
                            record_iteration_message(line)
                            line = f"  └─ Reason: {reason_text}"
                            print(line)
                            record_iteration_message(line)
            
            # Display portfolio summary
            total_equity = calculate_total_equity()
            total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
            equity_color = Fore.GREEN if total_return >= 0 else Fore.RED
            total_margin = calculate_total_margin()
            net_unrealized_total = total_equity - balance - total_margin
            net_color = Fore.GREEN if net_unrealized_total >= 0 else Fore.RED
            register_equity_snapshot(total_equity)
            sortino_ratio = calculate_sortino_ratio(
                equity_history,
                CHECK_INTERVAL,
                RISK_FREE_RATE,
            )
            
            line = f"\n{Fore.YELLOW}{'─'*20}"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.YELLOW}PORTFOLIO SUMMARY"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.YELLOW}{'─'*20}"
            print(line)
            record_iteration_message(line)
            line = f"Available Balance: ${balance:.2f}"
            print(line)
            record_iteration_message(line)
            if total_margin > 0:
                line = f"Margin Allocated: ${total_margin:.2f}"
                print(line)
                record_iteration_message(line)
            line = f"Total Equity: {equity_color}${total_equity:.2f} ({total_return:+.2f}%){Style.RESET_ALL}"
            print(line)
            record_iteration_message(line)
            line = f"Unrealized PnL: {net_color}${net_unrealized_total:.2f}{Style.RESET_ALL}"
            print(line)
            record_iteration_message(line)
            if sortino_ratio is not None:
                sortino_color = Fore.GREEN if sortino_ratio >= 0 else Fore.RED
                line = f"Sortino Ratio: {sortino_color}{sortino_ratio:+.2f}{Style.RESET_ALL}"
            else:
                line = "Sortino Ratio: N/A (need more data)"
            print(line)
            record_iteration_message(line)
            line = f"Open Positions: {len(positions)}"
            print(line)
            record_iteration_message(line)
            line = f"{Fore.YELLOW}{'─'*20}\n"
            print(line)
            record_iteration_message(line)

            if current_iteration_messages:
                send_telegram_message("\n".join(current_iteration_messages))
            
            # Log state
            log_portfolio_state()
            save_state()
            
            # Wait for next check
            logging.info(f"Waiting {CHECK_INTERVAL} seconds until next check...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\nShutting down bot...")
            save_state()
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
            save_state()
            time.sleep(60)

if __name__ == "__main__":
    main()
