#!/usr/bin/env python3
"""
Configuration loader for the trading bot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = BASE_DIR / ".env"

# --- ENV LOADING ---
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
else:
    load_dotenv(override=True)

DEFAULT_DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- BINANCE API ---
BN_API_KEY = os.getenv("BN_API_KEY", "")
BN_SECRET = os.getenv("BN_SECRET", "")

# --- LLM PROVIDER (OpenAI Compatible) ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)  # For proxies or other providers
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")

# --- TELEGRAM NOTIFICATIONS ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- TRADING PARAMETERS ---
SYMBOLS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "DOGEUSDT", "BNBUSDT"]
SYMBOL_TO_COIN = {
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "XRPUSDT": "XRP",
    "BTCUSDT": "BTC",
    "DOGEUSDT": "DOGE",
    "BNBUSDT": "BNB",
}
INTERVAL = "3m"  # 3-minute candles
START_CAPITAL = 10000.0
CHECK_INTERVAL = 3 * 60  # Check every 3 minutes (when candle closes)

# --- INDICATOR SETTINGS ---
EMA_LEN = 20
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- FEE & RISK ---
MAKER_FEE_RATE = 0.0  # 0.0000%
TAKER_FEE_RATE = 0.000275  # 0.0275%
DEFAULT_RISK_FREE_RATE = 0.0
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", DEFAULT_RISK_FREE_RATE))
