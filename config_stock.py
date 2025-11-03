import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"

if DOTENV_PATH.exists():
    dotenv_loaded = load_dotenv(dotenv_path=DOTENV_PATH, override=True)
else:
    dotenv_loaded = load_dotenv(override=True)
# ───────────────────────── CONFIG ─────────────────────────
# Print all environment variables
API_KEY = os.getenv("BN_API_KEY", "")
API_SECRET = os.getenv("BN_SECRET", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Trading symbols to monitor
SYMBOLS = ["NVDA", "BYND", "TSLA", "PLTR", "MSTR", "SQQQ", "TQQQ", "SOXL", "SOXS"]
SYMBOL_TO_COIN = {
    "NVDA": "NVDA",
    "BYND": "BYND",
    "TSLA": "TSLA",
    "PLTR": "PLTR",
    "MSTR": "MSTR",
    "MSTZ": "MSTZ",
    "IBIT": "IBIT",
    "SQQQ": "SQQQ",
    "TQQQ": "TQQQ",
    "SOXL": "SOXL",
    "SOXS": "SOXS",
}