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