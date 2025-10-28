#!/usr/bin/env python3
"""
Prompt generation for the LLM.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from . import config


def create_llm_specific_prompt(
    model_key: str, llm_state: Dict, market_snapshots: Dict[str, Any]
) -> str:
    """Create a prompt specific to the LLM with its own state."""
    model_config = config.LLM_MODELS[model_key]

    # Calculate current equity
    current_equity = calculate_llm_equity(model_key, llm_state, market_snapshots)
    total_return = (
        (current_equity - config.CAPITAL_PER_LLM) / config.CAPITAL_PER_LLM
    ) * 100

    prompt_lines = [
        f"You are {model_config['name']}, an independent crypto trader.",
        f"Your current balance: ${llm_state['balance']:.2f}",
        f"Your current equity: ${current_equity:.2f}",
        f"Your total return: {total_return:.2f}%",
        f"Your positions: {len(llm_state['positions'])}",
        f"Your trading performance: {llm_state['winning_trades']}/{llm_state['total_trades']} winning trades",
        f"Your total P&L: ${llm_state['total_pnl']:.2f}",
        "",
        "SHARED MARKET DATA (same for all traders):",
        "-" * 80,
    ]

    # Add shared market data
    for symbol in config.SYMBOLS:
        coin = config.SYMBOL_TO_COIN[symbol]
        data = market_snapshots.get(coin)
        if not data:
            continue

        prompt_lines.extend(
            [
                f"{coin} MARKET SNAPSHOT",
                f"- Price: {data['price']:.3f}, EMA20: {data['ema20']:.3f}, MACD: {data['macd']:.3f}, RSI(7): {data['rsi7']:.3f}",
                f"- Open Interest: {data['open_interest'].get('latest', 'N/A')}",
                f"- Funding Rate: {data['funding_rate']:.6g}",
                "  Intraday series (3-minute, oldest → latest):",
                f"    mid_prices: {json.dumps(data['intraday_series']['mid_prices'])}",
                f"    ema20: {json.dumps(data['intraday_series']['ema20'])}",
                f"    macd: {json.dumps(data['intraday_series']['macd'])}",
                f"    rsi7: {json.dumps(data['intraday_series']['rsi7'])}",
                f"    rsi14: {json.dumps(data['intraday_series']['rsi14'])}",
                "-" * 80,
            ]
        )

    # Add LLM-specific position data
    if llm_state["positions"]:
        prompt_lines.append("YOUR CURRENT POSITIONS:")
        for coin, pos in llm_state["positions"].items():
            current_price = market_snapshots.get(coin, {}).get(
                "price", pos["entry_price"]
            )
            pnl = calculate_position_pnl(pos, current_price)
            prompt_lines.append(
                f"{coin}: {pos['side']} {pos['quantity']} @ ${pos['entry_price']:.2f} (Current: ${current_price:.2f}, P&L: ${pnl:.2f})"
            )

    # Add decision instructions
    prompt_lines.extend(
        [
            "",
            "INSTRUCTIONS:",
            "For each coin, provide a trading decision in JSON format. You can either:",
            "1. 'hold' - Keep current position (if you have one)",
            "2. 'entry' - Open a new position (if you don't have one)",
            "3. 'close' - Close current position",
            "",
            "CRITICAL: Return ONLY a valid JSON object with this EXACT structure:",
            "DO NOT wrap in markdown code blocks or add any other text.",
            "DO NOT include these instructions in your response.",
            "DO NOT truncate your response - include ALL 6 coins.",
            "DO NOT repeat the formatting rules or instructions.",
            "Do not wrap in ```json, ```csv, or any code blocks.",
            "Do not escape quotes - use standard JSON format.",
            "",
            "{",
            '  "ETH": {',
            '    "signal": "hold",',
            '    "side": null,',
            '    "quantity": 0.0,',
            '    "confidence": 0.75,',
            '    "justification": "Your reasoning here"',
            "  },",
            '  "SOL": {',
            '    "signal": "entry",',
            '    "side": "long",',
            '    "quantity": 0.1,',
            '    "confidence": 0.8,',
            '    "justification": "Your reasoning here"',
            "  },",
            '  "XRP": {',
            '    "signal": "hold",',
            '    "side": null,',
            '    "quantity": 0.0,',
            '    "confidence": 0.5,',
            '    "justification": "Your reasoning here"',
            "  },",
            '  "BTC": {',
            '    "signal": "hold",',
            '    "side": null,',
            '    "quantity": 0.0,',
            '    "confidence": 0.5,',
            '    "justification": "Your reasoning here"',
            "  },",
            '  "DOGE": {',
            '    "signal": "hold",',
            '    "side": null,',
            '    "quantity": 0.0,',
            '    "confidence": 0.5,',
            '    "justification": "Your reasoning here"',
            "  },",
            '  "BNB": {',
            '    "signal": "hold",',
            '    "side": null,',
            '    "quantity": 0.0,',
            '    "confidence": 0.5,',
            '    "justification": "Your reasoning here"',
            "  }",
            "}",
            "",
            "JSON FORMATTING RULES:",
            "- Use double quotes for all strings",
            "- Use null for empty values, not undefined",
            "- No trailing commas",
            "- No comments in JSON",
            "- Include ALL coins: ETH, SOL, XRP, BTC, DOGE, BNB",
            "- Return ONLY the JSON object, no other text",
            "",
            "IMPORTANT:",
            "- Only suggest entries if you see strong opportunities",
            "- Use proper risk management",
            "- Provide clear justifications",
            "- Ensure valid JSON format",
        ]
    )

    return "\n".join(prompt_lines)


def calculate_llm_equity(
    model_key: str, llm_state: Dict, market_snapshots: Dict[str, Any]
) -> float:
    """Calculate current equity for an LLM."""
    equity = llm_state["balance"]

    for coin, pos in llm_state["positions"].items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        pnl = calculate_position_pnl(pos, current_price)
        equity += pnl

    return equity


def calculate_position_pnl(position: Dict, current_price: float) -> float:
    """Calculate P&L for a position."""
    if position["side"] == "long":
        return (current_price - position["entry_price"]) * position["quantity"]
    else:
        return (position["entry_price"] - current_price) * position["quantity"]


TRADING_RULES_PROMPT = """
You are a top level crypto trader focused on multiplying the account while safeguarding capital. Always apply these core rules:

Most Important Rules for Crypto Traders

Capital preservation is the foundation of successful crypto trading—your primary goal is to protect what you have so you can continue trading and growing.

Never Risk More Than 1-2% Per Trade
- Treat the 1% rule as non-negotiable; never risk more than 1-2% of total capital on a single trade.
- Survive losing streaks with enough capital to recover.

Use Stop-Loss Orders on Every Trade
- Define exit points before entering any position.
- Stop-loss orders are mandatory safeguards against emotional decisions.

Follow the Trend—Don't Fight the Market
- Buy rising coins and sell falling ones; the market is always right.
- Wait for confirmation before committing capital.

Stay Inactive Most of the Time
- Trade only when high-probability setups emerge.
- Avoid overtrading; patience and discipline preserve capital.

Cut Losses Quickly and Let Profits Run
- Close losing trades decisively; exit weak performers without hesitation.
- Let winning trades develop and grow when they show early profit.

Maintain a Written Trading Plan
- Know entry, exit, and profit targets before executing.
- Consistently follow the plan to keep emotions in check.

Control Leverage and Position Sizing
- Use leverage responsibly; ensure even a worst-case loss stays within the 1-2% risk cap.
- Proper sizing is central to risk management.

Focus on Small Consistent Wins
- Prioritize steady gains over chasing moonshots.
- Incremental growth compounds reliably and is easier to manage.

Think in Probabilities, Not Predictions
- Treat trading like a probability game with positive expectancy over many trades.
- Shift mindset from needing to be right to managing outcomes.

Stay Informed but Trade Less
- Track market-moving news but trade only when indicators align and risk-reward is favorable.
""".strip()


def create_trading_prompt(
    state: Dict[str, Any], market_snapshots: Dict[str, Dict[str, Any]]
) -> str:
    """Compose a rich prompt for the LLM based on current state and market data."""
    now = datetime.now(timezone.utc)
    minutes_running = int((now - state["start_time"]).total_seconds() // 60)

    def fmt(value: Optional[float], digits: int = 3) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{digits}f}"

    def fmt_rate(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.6g}"

    prompt_lines: List[str] = [
        f"It has been {minutes_running} minutes since you started trading. ",
        f"The current time is {now.isoformat()} and you've been invoked {state['invocation_count']} times. ",
        "Below is a variety of state data, price data, and predictive signals so you can discover alpha.",
        "ALL PRICE OR SIGNAL SERIES BELOW ARE ORDERED OLDEST → NEWEST.",
        "Timeframe note: Intraday series use 3-minute intervals unless a different interval is explicitly mentioned.",
        "-" * 80,
        "CURRENT MARKET STATE FOR ALL COINS",
    ]

    for symbol in config.SYMBOLS:
        coin = config.SYMBOL_TO_COIN[symbol]
        data = market_snapshots.get(coin)
        if not data:
            continue

        intraday = data["intraday_series"]
        long_term = data["long_term"]
        open_interest = data["open_interest"]
        funding_rates = data.get("funding_rates", [])
        funding_avg_str = (
            fmt_rate(float(np.mean(funding_rates))) if funding_rates else "N/A"
        )

        prompt_lines.extend(
            [
                f"{coin} MARKET SNAPSHOT",
                f"- Price: {fmt(data['price'], 3)}, EMA20: {fmt(data['ema20'], 3)}, MACD: {fmt(data['macd'], 3)}, RSI(7): {fmt(data['rsi7'], 3)}",
                f"- Open Interest (latest/avg): {fmt(open_interest.get('latest'), 2)} / {fmt(open_interest.get('average'), 2)}",
                f"- Funding Rate (latest/avg): {fmt_rate(data['funding_rate'])} / {funding_avg_str}",
                "  Intraday series (3-minute, oldest → latest):",
                f"    mid_prices: {json.dumps(intraday['mid_prices'])}",
                f"    ema20: {json.dumps(intraday['ema20'])}",
                f"    macd: {json.dumps(intraday['macd'])}",
                f"    rsi7: {json.dumps(intraday['rsi7'])}",
                f"    rsi14: {json.dumps(intraday['rsi14'])}",
                "  Longer-term context (4-hour timeframe):",
                f"    EMA20 vs EMA50: {fmt(long_term['ema20'], 3)} / {fmt(long_term['ema50'], 3)}",
                f"    ATR3 vs ATR14: {fmt(long_term['atr3'], 3)} / {fmt(long_term['atr14'], 3)}",
                f"    Volume (current/average): {fmt(long_term['current_volume'], 3)} / {fmt(long_term['average_volume'], 3)}",
                f"    MACD series: {json.dumps(long_term['macd'])}",
                f"    RSI14 series: {json.dumps(long_term['rsi14'])}",
                "-" * 80,
            ]
        )

    prompt_lines.extend(
        [
            "ACCOUNT INFORMATION AND PERFORMANCE",
            f"- Total Return (%): {fmt(state['total_return_pct'], 2)}",
            f"- Available Cash: {fmt(state['balance'], 2)}",
            f"- Margin Allocated: {fmt(state['total_margin'], 2)}",
            f"- Unrealized PnL: {fmt(state['net_unrealized_pnl'], 2)}",
            f"- Current Account Value: {fmt(state['total_equity'], 2)}",
            "Open positions and performance details:",
        ]
    )

    for coin, pos in state["positions"].items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        pnl = (
            (current_price - pos["entry_price"]) * pos["quantity"]
            if pos["side"] == "long"
            else (pos["entry_price"] - current_price) * pos["quantity"]
        )
        leverage = pos.get("leverage", 1) or 1
        liquidation_price = (
            pos["entry_price"] * max(0.0, 1 - 1 / leverage)
            if pos["side"] == "long"
            else pos["entry_price"] * (1 + 1 / leverage)
        )

        position_payload = {
            "symbol": coin,
            "side": pos["side"],
            "quantity": pos["quantity"],
            "entry_price": pos["entry_price"],
            "current_price": current_price,
            "liquidation_price": liquidation_price,
            "unrealized_pnl": pnl,
            "leverage": pos.get("leverage", 1),
            "notional_usd": pos["quantity"] * current_price,
        }
        prompt_lines.append(f"{coin} position data: {json.dumps(position_payload)}")

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
    "justification": "Reason for entry/close"  // only for entry/close
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
