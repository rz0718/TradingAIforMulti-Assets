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
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init
from prompt import TRADING_RULES_PROMPT
from parameter import INTERVAL, START_CAPITAL, DEFAULT_RISK_FREE_RATE, EMA_LEN, RSI_LEN, MACD_FAST, MACD_SLOW, MACD_SIGNAL, MAKER_FEE_RATE, TAKER_FEE_RATE, CHECK_INTERVAL
from config import API_KEY, API_SECRET, OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SYMBOLS, SYMBOL_TO_COIN
from telegram import send_telegram_message
# from market import fetch_market_data, collect_prompt_market_data, get_binance_client
from market import fetch_market_data, collect_prompt_market_data
from indicators import calculate_indicators, calculate_atr_series, add_indicator_columns


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


class LLMBot:
    def __init__(self, model: str = "deepseek-chat", bot_id: str = "default"):
        self.model = model
        self.bot_id = bot_id
        self.balance: float = START_CAPITAL
        self.positions: Dict[str, Dict[str, Any]] = {}  # coin -> position info
        self.trade_history: List[Dict[str, Any]] = []
        self.BOT_START_TIME = datetime.now(timezone.utc)
        self.invocation_count: int = 0
        self.iteration_counter: int = 0
        self.ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        self.current_iteration_messages: List[str] = []
        self.equity_history: List[float] = []
        
        # Use bot-specific data directory
        bot_data_dir = DATA_DIR / bot_id
        bot_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.STATE_CSV = bot_data_dir / "portfolio_state.csv"
        self.STATE_JSON = bot_data_dir / "portfolio_state.json"
        self.TRADES_CSV = bot_data_dir / "trade_history.csv"
        self.DECISIONS_CSV = bot_data_dir / "ai_decisions.csv"
        self.MESSAGES_CSV = bot_data_dir / "ai_messages.csv"
        self.STATE_COLUMNS = [
            'timestamp',
            'total_balance',
            'total_equity',
            'total_return_pct',
            'num_positions',
            'position_details',
            'total_margin',
            'net_unrealized_pnl'
        ]
        self.TRADES_COLUMNS = [
            'timestamp', 'coin', 'action', 'side', 'quantity', 'price',
            'profit_target', 'stop_loss', 'leverage', 'confidence',
            'pnl', 'balance_after', 'reason'
        ]
        self.DECISIONS_COLUMNS = [
            'timestamp', 'coin', 'signal', 'reasoning', 'confidence'
        ]
        self.MESSAGES_COLUMNS = [
            'timestamp', 'direction', 'role', 'content', 'metadata'
        ]

    def init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        if not self.STATE_CSV.exists():
            with open(self.STATE_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.STATE_COLUMNS)
        if not self.TRADES_CSV.exists():
            with open(self.TRADES_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.TRADES_COLUMNS)
        if not self.DECISIONS_CSV.exists():
            with open(self.DECISIONS_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.DECISIONS_COLUMNS)
        if not self.MESSAGES_CSV.exists():
            with open(self.MESSAGES_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.MESSAGES_COLUMNS)

    def log_portfolio_state(self) -> None:
        """Log current portfolio state."""
        total_equity = self.calculate_total_equity()
        total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
        total_margin = self.calculate_total_margin()
        net_unrealized = total_equity - self.balance - total_margin
        
        position_details = "; ".join([
            f"{coin}:{pos['side']}:{pos['quantity']:.4f}@{pos['entry_price']:.4f}"
            for coin, pos in self.positions.items()
        ]) if self.positions else "No positions"
        
        with open(self.STATE_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                f"{self.balance:.2f}",
                f"{total_equity:.2f}",
                f"{total_return:.2f}",
                len(self.positions),
                position_details,
                f"{total_margin:.2f}",
                f"{net_unrealized:.2f}"
            ])

    def log_trade(self, coin: str, action: str, details: Dict[str, Any]) -> None:
        """Log trade execution."""
        with open(self.TRADES_CSV, 'a', newline='') as f:
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
                self.balance,
                details.get('reason', '')
            ])

    def log_ai_decision(self, coin: str, signal: str, reasoning: str, confidence: float) -> None:
        """Log AI decision."""
        with open(self.DECISIONS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                coin,
                signal,
                reasoning,
                confidence
            ])
    
    def log_ai_message(self, direction: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log raw messages exchanged with the AI provider."""
        with open(self.MESSAGES_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                direction,
                role,
                content,
                json.dumps(metadata) if metadata else ""
            ])
    
    def strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI color codes so Telegram receives plain text."""
        return self.ANSI_ESCAPE_RE.sub("", text)
    
    def record_iteration_message(self, text: str) -> None:
        """Record console output for this iteration to share via Telegram."""
        if self.current_iteration_messages is not None:
            self.current_iteration_messages.append(self.strip_ansi_codes(text).rstrip())

    def notify_error(self, message: str, metadata: Optional[Dict[str, Any]] = None, *, log_error: bool = True) -> None:
        """Log an error and forward a brief description to Telegram."""
        if log_error:
            logging.error(message)
        self.log_ai_message(
            direction="error",
            role="system",
            content=message,
            metadata=metadata,

        )
        send_telegram_message(self, message)
    
    def load_state(self) -> None:
        """Load persisted balance and positions if available."""
        if not self.STATE_JSON.exists():
            logging.info("No existing state file found; starting fresh.")
            return

        try:
            with open(self.STATE_JSON, "r") as f:
                data = json.load(f)

            self.balance = float(data.get("balance", START_CAPITAL))
            try:
                self.iteration_counter = int(data.get("iteration", 0))
            except (TypeError, ValueError):
                self.iteration_counter = 0
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
                self.positions = restored_positions
            logging.info(
                "Loaded state from %s (balance: %.2f, positions: %d)",
                self.STATE_JSON,
                self.balance,
                len(self.positions),
            )
        except Exception as e:
            logging.error("Failed to load state from %s: %s", self.STATE_JSON, e, exc_info=True)
            self.balance = START_CAPITAL
            self.positions = {}

    def save_state(self) -> None:
        """Persist current balance, open positions, and iteration counter."""
        try:
            with open(self.STATE_JSON, "w") as f:
                json.dump(
                    {
                        "balance": self.balance,
                        "positions": self.positions,
                        "iteration": self.iteration_counter,
                        "updated_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logging.error("Failed to save state to %s: %s", self.STATE_JSON, e, exc_info=True)
    
    def load_equity_history(self) -> None:
        """Populate the in-memory equity history for performance calculations."""
        self.equity_history.clear()
        if not self.STATE_CSV.exists():
            return
        try:
            df = pd.read_csv(self.STATE_CSV, usecols=["total_equity"])
        except ValueError:
            logging.warning(
                "%s missing 'total_equity' column; Sortino ratio unavailable until new data is logged.",
                self.STATE_CSV,
            )
            return
        except Exception as exc:
            logging.warning("Unable to load historical equity data: %s", exc)
            return
        values = pd.to_numeric(df["total_equity"], errors="coerce").dropna()
        if not values.empty:
            self.equity_history.extend(float(v) for v in values.tolist())
    
    def register_equity_snapshot(self, total_equity: float) -> None:
        """Append the latest equity to the history if it is a finite value."""
        if total_equity is None:
            return
        if isinstance(total_equity, (int, float, np.floating)) and np.isfinite(total_equity):
            self.equity_history.append(float(total_equity))

    def format_prompt_for_deepseek(self) -> str:
        """Compose a rich prompt resembling the original DeepSeek in-context format."""
        self.invocation_count += 1

        now = datetime.now(timezone.utc)
        minutes_running = int((now - self.BOT_START_TIME).total_seconds() // 60)

        market_snapshots: Dict[str, Dict[str, Any]] = {}
        for symbol in SYMBOLS:
            snapshot = collect_prompt_market_data(symbol)
            if snapshot:
                market_snapshots[snapshot["coin"]] = snapshot

        total_margin = self.calculate_total_margin()
        total_equity = self.balance + total_margin
        for coin, pos in self.positions.items():
            current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
            total_equity += self.calculate_unrealized_pnl(coin, current_price)

        total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100 if START_CAPITAL else 0.0
        net_unrealized_total = total_equity - self.balance - total_margin

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
            f"The current time is {now.isoformat()} and you've been invoked {self.invocation_count} times. "
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
        prompt_lines.append(f"- Available Cash: {fmt(self.balance, 2)}")
        prompt_lines.append(f"- Margin Allocated: {fmt(total_margin, 2)}")
        prompt_lines.append(f"- Unrealized PnL: {fmt(net_unrealized_total, 2)}")
        prompt_lines.append(f"- Current Account Value: {fmt(total_equity, 2)}")
        prompt_lines.append("Open positions and performance details:")
        for coin, pos in self.positions.items():
            current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
            quantity = pos["quantity"]
            gross_unrealized = self.calculate_unrealized_pnl(coin, current_price)
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

    def call_llm_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call the LLM API with the given prompt."""
        try:
            self.log_ai_message(
                direction="sent",
                role="system",
                content=TRADING_RULES_PROMPT,
                metadata={
                    "model": self.model,
                    "temperature": 0.7,
                    "max_tokens": 4000
                }
            )
            self.log_ai_message(
                direction="sent",
                role="user",
                content=prompt,
                metadata={
                    "model": self.model,
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
                    "model": self.model,
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
                    "max_tokens": 6000,
                    "reasoning":
                    {
                        "effort": "medium",
                        "exclude": False,
                        "enabled": True,
                    }
                },
                timeout=30
            )

            print(response.json())
            if response.status_code != 200:
                self.notify_error(
                    f"OpenRouter API error: {response.status_code}",
                    metadata={
                        "status_code": response.status_code,
                        "response_text": response.text,
                    },
                )
                return None
            result = response.json()
            content = result['choices'][0]['message']['content']
            self.log_ai_message(
                direction="received",
                role="assistant",
                content=content,
                metadata={
                    "status_code": response.status_code,
                    "response_id": result.get("id"),
                    "usage": result.get("usage")
                }
            )
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                try:
                    decisions = json.loads(json_str)
                    return decisions
                except json.JSONDecodeError as decode_err:
                    snippet = json_str[:2000]
                    self.notify_error(
                        f"DeepSeek JSON decode failed: {decode_err}",
                        metadata={
                            "response_id": result.get("id"),
                            "status_code": response.status_code,
                            "raw_json_excerpt": snippet,
                        },
                    )
                    return None
            else:
                self.notify_error(
                    "No JSON found in DeepSeek response",
                    metadata={
                        "response_id": result.get("id"),
                        "status_code": response.status_code,
                    },
                )
                return None
        except Exception as e:
            logging.exception("Error calling LLM API")
            self.notify_error(
                f"Error calling LLM API: {e}",
                metadata={"context": "call_llm_api"},
                log_error=False,
            )
            return None
    
    def calculate_unrealized_pnl(self, coin: str, current_price: float) -> float:
        """Calculate unrealized PnL for a position."""
        if coin not in self.positions:
            return 0.0
        pos = self.positions[coin]
        if pos['side'] == 'long':
            pnl = (current_price - pos['entry_price']) * pos['quantity']
        else:  # short
            pnl = (pos['entry_price'] - current_price) * pos['quantity']
        return pnl

    def calculate_net_unrealized_pnl(self, coin: str, current_price: float) -> float:
        """Calculate unrealized PnL after subtracting fees already paid."""
        gross_pnl = self.calculate_unrealized_pnl(coin, current_price)
        fees_paid = self.positions.get(coin, {}).get('fees_paid', 0.0)
        return gross_pnl - fees_paid
    
    def calculate_pnl_for_price(self, pos: Dict[str, Any], target_price: float) -> float:
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
    
    def estimate_exit_fee(self, pos: Dict[str, Any], exit_price: float) -> float:
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
    
    def format_leverage_display(self, leverage: Any) -> str:
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
    
    def calculate_total_margin(self) -> float:
        """Return sum of margin allocated across all open positions."""
        return sum(float(pos.get('margin', 0.0)) for pos in self.positions.values())
    
    def calculate_total_equity(self) -> float:
        """Calculate total equity (balance + unrealized PnL)."""
        total = self.balance + self.calculate_total_margin()
        for coin in self.positions:
            symbol = next((s for s, c in SYMBOL_TO_COIN.items() if c == coin), None)
            if not symbol:
                continue
            data = fetch_market_data(symbol)
            if data:
                total += self.calculate_unrealized_pnl(coin, data['price'])
        return total
    
    def calculate_sortino_ratio(
        self,
        equity_values: Iterable[float],
        period_seconds: float,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> Optional[float]:
        """Compute the annualized Sortino ratio from equity snapshots."""
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
    
    def execute_entry(self, coin: str, decision: Dict[str, Any], current_price: float) -> None:
        """Execute entry trade."""
        if coin in self.positions:
            logging.warning(f"{coin}: Already have position, skipping entry")
            return
        side = decision.get('side', 'long')
        leverage = decision.get('leverage', 10)
        leverage_display = self.format_leverage_display(leverage)
        risk_usd = decision.get('risk_usd', self.balance * 0.01)
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
        if total_cost > self.balance:
            logging.warning(
                f"{coin}: Insufficient balance ${self.balance:.2f} for margin ${margin_required:.2f} "
                f"and fees ${entry_fee:.2f}"
            )
            return
        raw_reason = str(decision.get('justification', '')).strip()
        self.positions[coin] = {
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
        self.balance -= total_cost
        entry_price = current_price
        target_price = decision['profit_target']
        stop_price = decision['stop_loss']
        gross_at_target = self.calculate_pnl_for_price(self.positions[coin], target_price)
        gross_at_stop = self.calculate_pnl_for_price(self.positions[coin], stop_price)
        exit_fee_target = self.estimate_exit_fee(self.positions[coin], target_price)
        exit_fee_stop = self.estimate_exit_fee(self.positions[coin], stop_price)
        net_at_target = gross_at_target - (entry_fee + exit_fee_target)
        net_at_stop = gross_at_stop - (entry_fee + exit_fee_stop)
        expected_reward = max(gross_at_target, 0.0)
        expected_risk = max(-gross_at_stop, 0.0)
        if expected_risk > 0:
            rr_value = expected_reward / expected_risk if expected_reward > 0 else 0.0
            rr_display = f"{rr_value:.2f}:1"
        else:
            rr_display = "n/a"
        line = f"{Fore.GREEN}{self.model},{self.bot_id}: [ENTRY] {coin} {side.upper()} {leverage_display} @ ${entry_price:.4f}"
        print(line)
        self.record_iteration_message(line)
        line = f"  ├─ Size: {quantity:.4f} {coin} | Margin: ${margin_required:.2f}"
        print(line)
        self.record_iteration_message(line)
        line = f"  ├─ Risk: ${risk_usd:.2f} | Liquidity: {liquidity}"
        print(line)
        self.record_iteration_message(line)
        line = f"  ├─ Target: ${target_price:.4f} | Stop: ${stop_price:.4f}"
        print(line)
        self.record_iteration_message(line)
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
        self.record_iteration_message(line)
        line = (
            f"  ├─ PnL @ Stop: {gross_stop_sign}${abs(gross_at_stop):.2f} "
            f"(Net: {net_stop_sign}${abs(net_at_stop):.2f})"
        )
        print(line)
        self.record_iteration_message(line)
        if entry_fee > 0:
            line = f"  ├─ Estimated Fee: ${entry_fee:.2f} ({liquidity} @ {fee_rate*100:.4f}%)"
            print(line)
            self.record_iteration_message(line)
        line = f"  ├─ Confidence: {decision.get('confidence', 0)*100:.0f}%"
        print(line)
        self.record_iteration_message(line)
        line =f"  ├─ Reward/Risk: {rr_display}"
        print(line)
        self.record_iteration_message(line)
        line = f"  └─ Reason: {reason_text}"
        print(line)
        self.record_iteration_message(line)
        
        self.log_trade(coin, 'ENTRY', {
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
        self.save_state()

    def execute_close(self, coin: str, decision: Dict[str, Any], current_price: float) -> None:
        """Execute close trade."""
        if coin not in self.positions:
            logging.warning(f"{coin}: No position to close")
            return
        pos = self.positions[coin]
        raw_reason = str(decision.get('justification', '')).strip()
        reason_text = raw_reason or pos.get('last_justification') or "AI close signal"
        reason_text = " ".join(reason_text.split())
        pnl = self.calculate_unrealized_pnl(coin, current_price)
        fee_rate = pos.get('fee_rate', TAKER_FEE_RATE)
        exit_fee = pos['quantity'] * current_price * fee_rate
        total_fees = pos.get('fees_paid', 0.0) + exit_fee
        net_pnl = pnl - total_fees
        self.balance += pos['margin'] + net_pnl
        color = Fore.GREEN if net_pnl >= 0 else Fore.RED
        line = f"{color}{self.model},{self.bot_id}: [CLOSE] {coin} {pos['side'].upper()} {pos['quantity']:.4f} @ ${current_price:.4f}"
        print(line)
        self.record_iteration_message(line)
        line = f"  ├─ Entry: ${pos['entry_price']:.4f} | Gross PnL: ${pnl:.2f}"
        print(line)
        self.record_iteration_message(line)
        if total_fees > 0:
            line = f"  ├─ Fees Paid: ${total_fees:.2f} (includes exit fee ${exit_fee:.2f})"
            print(line)
            self.record_iteration_message(line)
        line = f"  ├─ Net PnL: ${net_pnl:.2f}"
        print(line)
        self.record_iteration_message(line)
        line = f"  ├─ Reason: {reason_text}"
        print(line)
        self.record_iteration_message(line)
        line = f"  └─ Balance: ${self.balance:.2f}"
        print(line)
        self.record_iteration_message(line)
        
        self.log_trade(coin, 'CLOSE', {
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
        del self.positions[coin]
        self.save_state()

    def check_stop_loss_take_profit(self) -> None:
        """Check and execute stop loss / take profit for all positions."""
        for coin in list(self.positions.keys()):
            symbol = [s for s, c in SYMBOL_TO_COIN.items() if c == coin][0]
            data = fetch_market_data(symbol)
            if not data:
                continue
            current_price = data['price']
            pos = self.positions[coin]
            if pos['side'] == 'long' and current_price <= pos['stop_loss']:
                self.execute_close(coin, {'justification': 'Stop loss hit'}, current_price)
            elif pos['side'] == 'short' and current_price >= pos['stop_loss']:
                self.execute_close(coin, {'justification': 'Stop loss hit'}, current_price)
            elif pos['side'] == 'long' and current_price >= pos['profit_target']:
                self.execute_close(coin, {'justification': 'Take profit hit'}, current_price)
            elif pos['side'] == 'short' and current_price <= pos['profit_target']:
                self.execute_close(coin, {'justification': 'Take profit hit'}, current_price) 

    def run(self) -> None:
        """Main trading loop."""
        logging.info("Initializing DeepSeek Multi-Asset Paper Trading Bot...")
        self.init_csv_files()
        self.load_equity_history()
        self.load_state()
        
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
                self.iteration_counter += 1
                self.current_iteration_messages = []
                
                # if not get_binance_client():
                #     retry_delay = min(CHECK_INTERVAL, 60)
                #     logging.warning(
                #         "Binance client unavailable; retrying in %d seconds without exiting.",
                #         retry_delay,
                #     )
                #     time.sleep(retry_delay)
                #     continue
                
                line = f"\n{Fore.CYAN}{'='*20}"
                print(line)
                self.record_iteration_message(line)
                line = f"{Fore.CYAN}Iteration {self.iteration_counter} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                print(line)
                self.record_iteration_message(line)
                line = f"{Fore.CYAN}{'='*20}\n"
                print(line)
                self.record_iteration_message(line)
                
                # Check stop loss / take profit first
                self.check_stop_loss_take_profit()
                
                # Get AI decisions
                logging.info("Requesting trading decisions from DeepSeek...")
                prompt = self.format_prompt_for_deepseek()
                decisions = self.call_llm_api(prompt)
                
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
                        self.log_ai_decision(
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
                            self.execute_entry(coin, decision, current_price)
                        elif signal == 'close':
                            self.execute_close(coin, decision, current_price)
                        elif signal == 'hold':
                            if coin in self.positions:
                                pos = self.positions[coin]
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
                                leverage_display = self.format_leverage_display(pos.get('leverage', 1.0))
                                try:
                                    margin_value = float(pos.get('margin', 0.0))
                                except (TypeError, ValueError):
                                    margin_value = 0.0
                                try:
                                    risk_value = float(pos.get('risk_usd', 0.0))
                                except (TypeError, ValueError):
                                    risk_value = 0.0
                                gross_unrealized = self.calculate_unrealized_pnl(coin, current_price)
                                estimated_exit_fee_now = self.estimate_exit_fee(pos, current_price)
                                total_fees_now = fees_paid + estimated_exit_fee_now
                                net_unrealized = gross_unrealized - total_fees_now
                                gross_at_target = self.calculate_pnl_for_price(pos, target_price)
                                exit_fee_target = self.estimate_exit_fee(pos, target_price)
                                net_at_target = gross_at_target - (fees_paid + exit_fee_target)
                                gross_at_stop = self.calculate_pnl_for_price(pos, stop_price)
                                exit_fee_stop = self.estimate_exit_fee(pos, stop_price)
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
                                self.record_iteration_message(line)
                                line = f"  ├─ Size: {quantity:.4f} {coin} | Margin: ${margin_value:.2f}"
                                print(line)
                                self.record_iteration_message(line)
                                line = f"  ├─ TP: ${target_price:.4f} | SL: ${stop_price:.4f}"
                                print(line)
                                self.record_iteration_message(line)
                                line = (
                                    f"  ├─ PnL: {pnl_color}{net_display}{Style.RESET_ALL} "
                                    f"(Gross: {gross_display}, Fees: ${total_fees_now:.2f})"
                                )
                                print(line)
                                self.record_iteration_message(line)
                                line = (
                                    f"  ├─ PnL @ Target: {gross_target_display} "
                                    f"(Net: {net_target_display})"
                                )
                                print(line)
                                self.record_iteration_message(line)
                                line = (
                                    f"  ├─ PnL @ Stop: {gross_stop_display} "
                                    f"(Net: {net_stop_display})"
                                )
                                print(line)
                                self.record_iteration_message(line)
                                line = f"  ├─ Reward/Risk: {rr_display}"
                                print(line)
                                self.record_iteration_message(line)
                                line = f"  └─ Reason: {reason_text}"
                                print(line)
                                self.record_iteration_message(line)
                                
                # Display portfolio summary
                total_equity = self.calculate_total_equity()
                total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
                equity_color = Fore.GREEN if total_return >= 0 else Fore.RED
                total_margin = self.calculate_total_margin()
                net_unrealized_total = total_equity - self.balance - total_margin
                net_color = Fore.GREEN if net_unrealized_total >= 0 else Fore.RED
                self.register_equity_snapshot(total_equity)
                sortino_ratio = self.calculate_sortino_ratio(
                    self.equity_history,
                    CHECK_INTERVAL,
                    DEFAULT_RISK_FREE_RATE,
                )
                
                line = f"\n{Fore.YELLOW}{'─'*20}"
                print(line)
                self.record_iteration_message(line)
                line = f"{Fore.YELLOW}PORTFOLIO SUMMARY"
                print(line)
                self.record_iteration_message(line)
                line = f"{Fore.YELLOW}{'─'*20}"
                print(line)
                self.record_iteration_message(line)
                line = f"Available Balance: ${self.balance:.2f}"
                print(line)
                self.record_iteration_message(line)
                if total_margin > 0:
                    line = f"Margin Allocated: ${total_margin:.2f}"
                    print(line)
                    self.record_iteration_message(line)
                line = f"Total Equity: {equity_color}${total_equity:.2f} ({total_return:+.2f}%){Style.RESET_ALL}"
                print(line)
                self.record_iteration_message(line)
                line = f"Unrealized PnL: {net_color}${net_unrealized_total:.2f}{Style.RESET_ALL}"
                print(line)
                self.record_iteration_message(line)
                if sortino_ratio is not None:
                    sortino_color = Fore.GREEN if sortino_ratio >= 0 else Fore.RED
                    line = f"Sortino Ratio: {sortino_color}{sortino_ratio:+.2f}{Style.RESET_ALL}"
                else:
                    line = "Sortino Ratio: N/A (need more data)"
                print(line)
                self.record_iteration_message(line)
                line = f"Open Positions: {len(self.positions)}"
                print(line)
                self.record_iteration_message(line)
                line = f"{Fore.YELLOW}{'─'*20}\n"
                print(line)
                self.record_iteration_message(line)
                
                if self.current_iteration_messages:
                    send_telegram_message("\n".join(self.current_iteration_messages))
                    
                self.log_portfolio_state()
                self.save_state()
                
                logging.info(f"Waiting {CHECK_INTERVAL} seconds until next check...")
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                logging.info("Shutting down bot...")
                self.save_state()
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}", exc_info=True)
                self.save_state()
                time.sleep(60)

if __name__ == "__main__":
    # Single bot mode - for running one bot instance
    import sys
    
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek/deepseek-chat-v3.1"
    bot_id = sys.argv[2] if len(sys.argv) > 2 else "deepseek-1"
    print(model, bot_id)
    
    print(f"Starting single bot: model={model}, bot_id={bot_id}")
    bot = LLMBot(model=model, bot_id=bot_id)
    bot.run()