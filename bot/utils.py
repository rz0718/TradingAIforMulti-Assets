#!/usr/bin/env python3
"""
Utility functions for the trading bot.
"""
import logging
import csv
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from . import config


# --- LOGGING ---
def setup_logging():
    """Initializes basic logging configuration."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
    )
    logging.info("Logging configured.")


# --- CSV & STATE MANAGEMENT ---
STATE_CSV = config.DATA_DIR / "portfolio_state.csv"
STATE_JSON = config.DATA_DIR / "portfolio_state.json"
TRADES_CSV = config.DATA_DIR / "trade_history.csv"
DECISIONS_CSV = config.DATA_DIR / "ai_decisions.csv"
MESSAGES_CSV = config.DATA_DIR / "ai_messages.csv"

STATE_COLUMNS = [
    "timestamp",
    "total_balance",
    "total_equity",
    "total_return_pct",
    "num_positions",
    "position_details",
    "total_margin",
    "net_unrealized_pnl",
]


def init_csv_files() -> None:
    """Initialize CSV files with headers if they don't exist."""
    files_to_init = {
        STATE_CSV: STATE_COLUMNS,
        TRADES_CSV: [
            "timestamp",
            "coin",
            "action",
            "side",
            "quantity",
            "price",
            "profit_target",
            "stop_loss",
            "leverage",
            "confidence",
            "pnl",
            "balance_after",
            "reason",
        ],
        DECISIONS_CSV: [
            "timestamp",
            "model",
            "coin",
            "signal",
            "reasoning",
            "confidence",
        ],
        MESSAGES_CSV: ["timestamp", "direction", "role", "content", "metadata"],
    }
    for path, header in files_to_init.items():
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)


def log_portfolio_state(state: Dict[str, Any]) -> None:
    """Log current portfolio state to CSV."""
    with open(STATE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([state.get(col, "") for col in STATE_COLUMNS])


def log_trade(trade_data: Dict[str, Any]) -> None:
    """Log trade execution to CSV."""
    header = [
        "timestamp",
        "coin",
        "action",
        "side",
        "quantity",
        "price",
        "profit_target",
        "stop_loss",
        "leverage",
        "confidence",
        "pnl",
        "balance_after",
        "reason",
    ]
    with open(TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trade_data.get(col, "") for col in header])


def log_ai_decision(decision_data: Dict[str, Any]) -> None:
    """Log AI decision to CSV."""
    header = ["timestamp", "model", "coin", "signal", "reasoning", "confidence"]
    with open(DECISIONS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([decision_data.get(col, "") for col in header])


def log_ai_message(message_data: Dict[str, Any]) -> None:
    """Log raw messages exchanged with the AI provider to CSV."""
    header = ["timestamp", "direction", "role", "content", "metadata"]
    with open(MESSAGES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([message_data.get(col, "") for col in header])


# --- TELEGRAM ---
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes so Telegram receives plain text."""
    return ANSI_ESCAPE_RE.sub("", text)


def send_telegram_message(text: str) -> None:
    """Send a notification message to Telegram if credentials are configured."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": config.TELEGRAM_CHAT_ID,
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
