#!/usr/bin/env python3
"""
Configuration file for the Multi-LLM Trading Bot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


ASSET_MODE = os.getenv("ASSET_MODE", "crypto")

# --- DATA DIRECTORY ---
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

# --- BINANCE API CONFIGURATION ---
BN_API_KEY = os.getenv("BN_API_KEY", "")
BN_SECRET = os.getenv("BN_SECRET", "")

# --- OPENROUTER MULTI-LLM CONFIGURATION ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- ALPACA API CONFIGURATION ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Two LLM Models for Focused Testing (Claude Sonnet and Gemini Pro)
LLM_MODELS = {
    "deepseek_v3.1": {
        "model_id": "deepseek/deepseek-chat-v3.1",
        "name": "DeepSeek V3.1",
        "provider": "DeepSeek",
        "max_tokens": 4000,
        "temperature": 0.7,
    },
    "qwen3_max": {
        "model_id": "qwen/qwen3-max",
        "name": "Qwen3 Max",
        "provider": "Qwen",
        "max_tokens": 4000,
        "temperature": 0.7,
    },
}
# --- TELEGRAM NOTIFICATIONS ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

if ASSET_MODE.lower() == "crypto":
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
elif ASSET_MODE.lower() == "us_stock":
    SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "IBIT", "TQQQ", "SQQQ", "UVIX"]
    SYMBOL_TO_COIN = {
        "AAPL": "AAPL",
        "MSFT": "MSFT",
        "GOOG": "GOOG",
        "AMZN": "AMZN",
        "TSLA": "TSLA",
        "NVDA": "NVDA",
        "IBIT": "IBIT",
        "TQQQ": "TQQQ",
        "SQQQ": "SQQQ",
        "UVIX": "UVIX",
    }
INTERVAL = "3m"  # 3-minute candles
CHECK_INTERVAL = 3 * 60  # Check every 3 minutes (when candle closes)
START_CAPITAL = 10000.0
CAPITAL_PER_LLM = START_CAPITAL  # $10,000 per LLM (2 models total)

# --- INDICATOR SETTINGS ---
EMA_LEN = 20
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_LEN = 14

# --- RISK MANAGEMENT ---
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.05"))  # 5% annual risk-free rate
MAX_POSITION_SIZE = 0.1  # Max 10% of capital per position
STOP_LOSS_PCT = 0.02  # 2% stop loss
PROFIT_TARGET_PCT = 0.04  # 4% profit target

# --- TRADING RULES ---
MIN_CONFIDENCE = 0.6  # Minimum confidence for trade execution
MAX_LEVERAGE = 10  # Maximum leverage allowed
MAX_DAILY_TRADES = 20  # Maximum trades per day per LLM

# --- LOGGING ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
