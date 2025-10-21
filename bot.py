#!/usr/bin/env python3
"""
DeepSeek Multi-Asset Paper Trading Bot
Uses Binance API for market data and OpenRouter API for DeepSeek Chat V3.1 trading decisions
"""
from __future__ import annotations

import os
import time
import json
import logging
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

BASE_DIR = Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"

if DOTENV_PATH.exists():
    dotenv_loaded = load_dotenv(dotenv_path=DOTENV_PATH, override=True)
else:
    dotenv_loaded = load_dotenv(override=True)

# ───────────────────────── CONFIG ─────────────────────────
API_KEY = os.getenv("BN_API_KEY", "")
API_SECRET = os.getenv("BN_SECRET", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

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

# Indicator settings
EMA_LEN = 20
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ───────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

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

client = Client(API_KEY, API_SECRET, testnet=False)

# ──────────────────────── GLOBAL STATE ─────────────────────
balance: float = START_CAPITAL
positions: Dict[str, Dict[str, Any]] = {}  # coin -> position info
trade_history: List[Dict[str, Any]] = []

# CSV files
STATE_CSV = "portfolio_state.csv"
STATE_JSON = "portfolio_state.json"
TRADES_CSV = "trade_history.csv"
DECISIONS_CSV = "ai_decisions.csv"
MESSAGES_CSV = "ai_messages.csv"

# ───────────────────────── CSV LOGGING ──────────────────────

def init_csv_files() -> None:
    """Initialize CSV files with headers."""
    if not os.path.exists(STATE_CSV):
        with open(STATE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'total_balance', 'total_equity', 'total_return_pct',
                'num_positions', 'position_details'
            ])
    
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'coin', 'action', 'side', 'quantity', 'price',
                'profit_target', 'stop_loss', 'leverage', 'confidence',
                'pnl', 'balance_after', 'reason'
            ])
    
    if not os.path.exists(DECISIONS_CSV):
        with open(DECISIONS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'coin', 'signal', 'reasoning', 'confidence'
            ])

    if not os.path.exists(MESSAGES_CSV):
        with open(MESSAGES_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'direction', 'role', 'content', 'metadata'
            ])

def log_portfolio_state() -> None:
    """Log current portfolio state."""
    total_equity = calculate_total_equity()
    total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
    
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
            position_details
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

# ───────────────────────── STATE MGMT ───────────────────────

def load_state() -> None:
    """Load persisted balance and positions if available."""
    global balance, positions

    if not os.path.exists(STATE_JSON):
        logging.info("No existing state file found; starting fresh.")
        return

    try:
        with open(STATE_JSON, "r") as f:
            data = json.load(f)

        balance = float(data.get("balance", START_CAPITAL))
        loaded_positions = data.get("positions", {})
        if isinstance(loaded_positions, dict):
            restored_positions: Dict[str, Dict[str, Any]] = {}
            for coin, pos in loaded_positions.items():
                if not isinstance(pos, dict):
                    continue
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
    """Persist current balance and open positions."""
    try:
        with open(STATE_JSON, "w") as f:
            json.dump(
                {
                    "balance": balance,
                    "positions": positions,
                    "updated_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logging.error("Failed to save state to %s: %s", STATE_JSON, e, exc_info=True)

# ───────────────────────── INDICATORS ───────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.Series:
    """Calculate technical indicators."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # EMA
    ema = close.ewm(span=EMA_LEN, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_LEN).mean()
    avg_loss = loss.rolling(RSI_LEN).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    
    # MACD
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    
    df["ema20"] = ema
    df["rsi"] = rsi
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    
    return df.iloc[-1]

def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch current market data for a symbol."""
    try:
        # Get recent klines
        klines = client.get_klines(symbol=symbol, interval=INTERVAL, limit=50)
        
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
            funding_info = client.futures_funding_rate(symbol=symbol, limit=1)
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

# ───────────────────── AI DECISION MAKING ───────────────────

def format_prompt_for_deepseek() -> str:
    """Format current portfolio state for DeepSeek."""
    market_data = {}
    for symbol in SYMBOLS:
        data = fetch_market_data(symbol)
        if data:
            coin = SYMBOL_TO_COIN[symbol]
            market_data[coin] = data
    
    # Calculate total equity
    total_equity = calculate_total_equity()
    total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
    
    prompt = f"""You are a cryptocurrency trading AI. Analyze the current market data and portfolio state, then provide trading decisions.

CURRENT PORTFOLIO STATE:
- Available Cash: ${balance:.2f}
- Total Equity: ${total_equity:.2f}
- Total Return: {total_return:.2f}%
- Number of Positions: {len(positions)}

CURRENT POSITIONS:
"""
    
    if positions:
        for coin, pos in positions.items():
            unrealized_pnl = calculate_unrealized_pnl(coin, market_data.get(coin, {}).get('price', 0))
            prompt += f"""
{coin}:
  - Side: {pos['side']}
  - Quantity: {pos['quantity']}
  - Entry Price: ${pos['entry_price']:.4f}
  - Current Price: ${market_data.get(coin, {}).get('price', 0):.4f}
  - Unrealized PnL: ${unrealized_pnl:.2f}
  - Profit Target: ${pos['profit_target']:.4f}
  - Stop Loss: ${pos['stop_loss']:.4f}
  - Leverage: {pos['leverage']}x
  - Invalidation: {pos['invalidation_condition']}
"""
    else:
        prompt += "  No open positions\n"
    
    prompt += "\nCURRENT MARKET DATA:\n"
    for coin, data in market_data.items():
        prompt += f"""
{coin}:
  - Price: ${data['price']:.4f}
  - RSI: {data['rsi']:.2f}
  - MACD: {data['macd']:.4f}
  - MACD Signal: {data['macd_signal']:.4f}
  - EMA20: {data['ema20']:.4f}
  - Funding Rate: {data['funding_rate']:.8f}
"""
    
    prompt += """
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
    "quantity": 0.0,  // calculated by system for entry, current for hold
    "profit_target": 0.0,
    "stop_loss": 0.0,
    "leverage": 10,
    "confidence": 0.75,
    "risk_usd": 500.0,
    "invalidation_condition": "If price closes below X on 3-minute candle",
    "justification": "Reason for entry/close"  // only for entry/close
  },
  // ... repeat for SOL, XRP, BTC, DOGE, BNB
}

IMPORTANT: 
- Only suggest entries if you see strong opportunities
- Use proper risk management
- Provide clear invalidation conditions
- Return ONLY valid JSON, no other text
"""
    
    return prompt

def call_deepseek_api(prompt: str) -> Optional[Dict[str, Any]]:
    """Call OpenRouter API with DeepSeek Chat V3.1."""
    try:
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
            logging.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            log_ai_message(
                direction="received",
                role="system",
                content=response.text,
                metadata={"status_code": response.status_code}
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
            decisions = json.loads(json_str)
            return decisions
        else:
            logging.error("No JSON found in response")
            log_ai_message(
                direction="error",
                role="system",
                content="No JSON found in response",
                metadata={"response_id": result.get("id")}
            )
            return None
            
    except Exception as e:
        logging.error(f"Error calling DeepSeek API: {e}", exc_info=True)
        log_ai_message(
            direction="error",
            role="system",
            content=str(e),
            metadata={"context": "call_deepseek_api"}
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

def calculate_total_equity() -> float:
    """Calculate total equity (balance + unrealized PnL)."""
    total = balance
    
    for coin in positions:
        symbol = [s for s, c in SYMBOL_TO_COIN.items() if c == coin][0]
        data = fetch_market_data(symbol)
        if data:
            total += calculate_unrealized_pnl(coin, data['price'])
    
    return total

def execute_entry(coin: str, decision: Dict[str, Any], current_price: float) -> None:
    """Execute entry trade."""
    global balance
    
    if coin in positions:
        logging.warning(f"{coin}: Already have position, skipping entry")
        return
    
    side = decision.get('side', 'long')
    leverage = decision.get('leverage', 10)
    risk_usd = decision.get('risk_usd', balance * 0.01)
    
    # Calculate position size based on risk
    stop_distance = abs(current_price - decision['stop_loss'])
    if stop_distance == 0:
        logging.warning(f"{coin}: Invalid stop loss, skipping")
        return
    
    quantity = risk_usd / stop_distance
    position_value = quantity * current_price
    margin_required = position_value / leverage
    
    if margin_required > balance:
        logging.warning(f"{coin}: Insufficient balance ${balance:.2f} for margin ${margin_required:.2f}")
        return
    
    # Open position
    positions[coin] = {
        'side': side,
        'quantity': quantity,
        'entry_price': current_price,
        'profit_target': decision['profit_target'],
        'stop_loss': decision['stop_loss'],
        'leverage': leverage,
        'confidence': decision.get('confidence', 0.5),
        'invalidation_condition': decision.get('invalidation_condition', ''),
        'margin': margin_required
    }
    
    balance -= margin_required
    
    print(f"{Fore.GREEN}[ENTRY] {coin} {side.upper()} {quantity:.4f} @ ${current_price:.4f}")
    print(f"  ├─ Leverage: {leverage}x | Margin: ${margin_required:.2f}")
    print(f"  ├─ Target: ${decision['profit_target']:.4f} | Stop: ${decision['stop_loss']:.4f}")
    print(f"  └─ Confidence: {decision.get('confidence', 0)*100:.0f}%")
    
    log_trade(coin, 'ENTRY', {
        'side': side,
        'quantity': quantity,
        'price': current_price,
        'profit_target': decision['profit_target'],
        'stop_loss': decision['stop_loss'],
        'leverage': leverage,
        'confidence': decision.get('confidence', 0),
        'pnl': 0,
        'reason': decision.get('justification', 'AI entry signal')
    })
    save_state()

def execute_close(coin: str, decision: Dict[str, Any], current_price: float) -> None:
    """Execute close trade."""
    global balance
    
    if coin not in positions:
        logging.warning(f"{coin}: No position to close")
        return
    
    pos = positions[coin]
    pnl = calculate_unrealized_pnl(coin, current_price)
    
    # Return margin and add PnL
    balance += pos['margin'] + pnl
    
    color = Fore.GREEN if pnl >= 0 else Fore.RED
    print(f"{color}[CLOSE] {coin} {pos['side'].upper()} {pos['quantity']:.4f} @ ${current_price:.4f}")
    print(f"  ├─ Entry: ${pos['entry_price']:.4f} | PnL: ${pnl:.2f}")
    print(f"  └─ Balance: ${balance:.2f}")
    
    log_trade(coin, 'CLOSE', {
        'side': pos['side'],
        'quantity': pos['quantity'],
        'price': current_price,
        'profit_target': 0,
        'stop_loss': 0,
        'leverage': pos['leverage'],
        'confidence': 0,
        'pnl': pnl,
        'reason': decision.get('justification', 'AI close signal')
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
    logging.info("Initializing DeepSeek Multi-Asset Paper Trading Bot...")
    init_csv_files()
    load_state()
    
    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY not found in .env file")
        return
    
    logging.info(f"Starting capital: ${START_CAPITAL:.2f}")
    logging.info(f"Monitoring: {', '.join(SYMBOL_TO_COIN.values())}")
    
    iteration = 0
    
    while True:
        try:
            iteration += 1
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"{Fore.CYAN}Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{Fore.CYAN}{'='*60}\n")
            
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
                            unrealized_pnl = calculate_unrealized_pnl(coin, current_price)
                            pnl_color = Fore.GREEN if unrealized_pnl >= 0 else Fore.RED
                            print(f"[HOLD] {coin} - Unrealized PnL: {pnl_color}${unrealized_pnl:.2f}{Style.RESET_ALL}")
            
            # Display portfolio summary
            total_equity = calculate_total_equity()
            total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
            equity_color = Fore.GREEN if total_return >= 0 else Fore.RED
            
            print(f"\n{Fore.YELLOW}{'─'*60}")
            print(f"{Fore.YELLOW}PORTFOLIO SUMMARY")
            print(f"{Fore.YELLOW}{'─'*60}")
            print(f"Available Balance: ${balance:.2f}")
            print(f"Total Equity: {equity_color}${total_equity:.2f} ({total_return:+.2f}%){Style.RESET_ALL}")
            print(f"Open Positions: {len(positions)}")
            print(f"{Fore.YELLOW}{'─'*60}\n")
            
            # Log state
            log_portfolio_state()
            
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
