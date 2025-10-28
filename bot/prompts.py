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

    # Calculate current equity and performance metrics
    current_equity = calculate_llm_equity(model_key, llm_state, market_snapshots)
    total_return_pct = (
        (current_equity - config.CAPITAL_PER_LLM) / config.CAPITAL_PER_LLM
    ) * 100

    # Calculate total margin and unrealized PnL
    total_margin = sum(
        float(pos.get("margin", 0.0)) for pos in llm_state["positions"].values()
    )
    net_unrealized_pnl = 0.0
    for coin, pos in llm_state["positions"].items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        pnl = calculate_position_pnl(pos, current_price)
        net_unrealized_pnl += pnl

    # Create state summary similar to trading_workflow
    state_summary = {
        "balance": llm_state["balance"],
        "positions": llm_state["positions"],
        "start_time": llm_state["start_time"],
        "invocation_count": llm_state["invocation_count"],
        "total_equity": current_equity,
        "total_margin": total_margin,
        "total_return_pct": total_return_pct,
        "net_unrealized_pnl": net_unrealized_pnl,
    }

    now = datetime.now(timezone.utc)
    minutes_running = int((now - llm_state["start_time"]).total_seconds() // 60)

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
        f"The current time is {now.isoformat()} and you've been invoked {llm_state['invocation_count']} times. ",
        "Below is a variety of state data, price data, and predictive signals so you can discover alpha.",
        "ALL PRICE OR SIGNAL SERIES BELOW ARE ORDERED OLDEST → NEWEST.",
        "Timeframe note: Intraday series use 3-minute intervals unless a different interval is explicitly mentioned.",
        "-" * 80,
        "CURRENT MARKET STATE FOR ALL COINS",
    ]

    # Add market data with same structure as trading_workflow
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
            f"- Total Return (%): {fmt(state_summary['total_return_pct'], 2)}",
            f"- Available Cash: {fmt(state_summary['balance'], 2)}",
            f"- Margin Allocated: {fmt(state_summary['total_margin'], 2)}",
            f"- Unrealized PnL: {fmt(state_summary['net_unrealized_pnl'], 2)}",
            f"- Current Account Value: {fmt(state_summary['total_equity'], 2)}",
            "Open positions and performance details:",
        ]
    )

    for coin, pos in llm_state["positions"].items():
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

    # Use the same instructions as trading_workflow
    prompt_lines.append(
        """
INSTRUCTIONS:
For each coin, provide a trading decision in JSON format. You can either:
1. "hold" - Keep current position (if you have one)
2. "entry" - Open a new position (if you don't have one)
3. "close" - Close current position

IMPORTANT RULES:
- If signal is "hold": side, justification MUST be empty strings ""
- If signal is "entry": side and justification MUST be filled
- If signal is "close": side MUST be empty string "", justification MUST be filled
- Always include all keys, but set irrelevant ones to empty string ""

FIELD DEFINITIONS:

- quantity: The amount of the coin to trade (not USD value, but actual coin units)
  * For ETH: quantity in ETH (e.g., 0.5 ETH)
  * For BTC: quantity in BTC (e.g., 0.02 BTC)
  * This is the BASE amount BEFORE leverage is applied
  * The actual position size = quantity × coin_price × leverage
  * Calculate quantity based on your risk tolerance and stop distance
  
  Example calculation:
  - If willing to risk $500, Entry: $3,200, Stop: $3,100, Leverage: 10x
  - Price difference: $100
  - Quantity = $500 / ($100 × 10) = 0.5 ETH
  - Position value = 0.5 × $3,200 × 10 = $16,000 exposure
  - If stopped out: loss = 0.5 × $100 × 10 = $500 ✓

- leverage: The multiplier applied to your position size
  * Common values: 1x (no leverage), 5x, 10x, 20x, 50x, 100x
  * Higher leverage = larger position with same capital BUT higher liquidation risk
  * 10x leverage means you control 10× the value with same margin
  * Liquidation occurs when loss reaches (100/leverage)%
    - 10x leverage → liquidated at ~10% adverse move
    - 20x leverage → liquidated at ~5% adverse move
    - 5x leverage → liquidated at ~20% adverse move
  * RECOMMENDED: Use 5-10x for crypto (volatile), 2-5x for conservative trades
  * Your stop_loss MUST be placed BEFORE liquidation price
  
  Liquidation price calculation:
  - For LONG: entry_price × (1 - 0.9/leverage)
  - For SHORT: entry_price × (1 + 0.9/leverage)
  
  Example with 10x leverage on LONG:
  - Entry: $3,200
  - Liquidation: $3,200 × (1 - 0.9/10) = $3,200 × 0.91 = $2,912
  - Your stop_loss at $3,100 is SAFE (above liquidation)

- profit_target: The price level where you will take profits and close the position. This should be based on:
  * Technical resistance/support levels
  * Fibonacci extensions
  * Risk-reward ratio (typically 2:1 or 3:1)
  * Previous high/low levels
  For LONG positions: profit_target > entry_price
  For SHORT positions: profit_target < entry_price

- stop_loss: The price level where you will cut losses and close the position to prevent further losses. This should be:
  * Below key support for LONG positions
  * Above key resistance for SHORT positions
  * Placed beyond recent swing high/low
  * Based on ATR (Average True Range) or volatility
  * CRITICAL: Must be placed BEFORE liquidation price (include safety buffer)
  * Never moved further away from entry (don't give losing trades more room)
  For LONG positions: stop_loss < entry_price (but > liquidation_price)
  For SHORT positions: stop_loss > entry_price (but < liquidation_price)

Return ONLY a valid JSON object with this structure:

EXAMPLE 1 - ENTRY signal (opening new long position with 10x leverage):
{
  "ETH": {
    "signal": "entry",
    "side": "long",
    "quantity": 0.5,
    "profit_target": 3250.0,
    "stop_loss": 3100.0,
    "leverage": 10,
    "confidence": 0.82,
    "invalidation_condition": "If price closes below 3150 on a 3-minute candle",
    "justification": "Bullish divergence on RSI with volume confirmation, breaking above key resistance at 3180. Entry at 3180, target at previous high 3250 (R:R 2.3:1), stop below support at 3100. Liquidation at ~2,898, stop safely above. Position: 0.5 ETH × $3,180 × 10x = $15,900 exposure"
  }
}

EXAMPLE 2 - HOLD signal (maintaining current position):
{
  "BTC": {
    "signal": "hold",
    "side": "",
    "quantity": 0.02,
    "profit_target": 68500.0,
    "stop_loss": 66000.0,
    "leverage": 10,
    "confidence": 0.75,
    "invalidation_condition": "If price closes below 66800 on a 3-minute candle",
    "justification": ""
  }
}

EXAMPLE 3 - CLOSE signal (closing existing position):
{
  "SOL": {
    "signal": "close",
    "side": "",
    "quantity": 5.0,
    "profit_target": 0.0,
    "stop_loss": 0.0,
    "leverage": 10,
    "confidence": 0.88,
    "invalidation_condition": "",
    "justification": "Target reached at 185, taking profit with +$1,250 gain. Bearish divergence forming on 15m chart, securing profits. Closed 5 SOL position opened at $160 with 10x leverage"
  }
}

EXAMPLE 4 - ENTRY signal (opening new short position with high leverage):
{
  "BNB": {
    "signal": "entry",
    "side": "short",
    "quantity": 2.5,
    "profit_target": 580.0,
    "stop_loss": 615.0,
    "leverage": 20,
    "confidence": 0.79,
    "invalidation_condition": "If price closes above 610 on a 3-minute candle",
    "justification": "Rejected at resistance 605, bearish engulfing pattern with increasing volume. Entry at 600, target at support 580 (R:R 1.4:1), stop above resistance at 615. Using 20x leverage for tighter stop. Liquidation at ~618, stop safely below. Position: 2.5 BNB × $600 × 20x = $30,000 exposure"
  }
}

EXAMPLE 5 - ENTRY signal with conservative leverage:
{
  "XRP": {
    "signal": "entry",
    "side": "long",
    "quantity": 800.0,
    "profit_target": 0.62,
    "stop_loss": 0.56,
    "leverage": 5,
    "confidence": 0.76,
    "invalidation_condition": "If price closes below 0.57 on a 3-minute candle or breaks down from ascending triangle",
    "justification": "Breaking out of ascending triangle at 0.585, entry at 0.59. Target at 0.62 (measured move). Using conservative 5x leverage for altcoin volatility. Initial stop at 0.56 (below triangle support), will trail stop to breakeven once price hits 0.605. Liquidation at ~0.484, stop safely above. Position: 800 XRP × $0.59 × 5x = $2,360 exposure"
  }
}

EXAMPLE 6 - ENTRY signal showing quantity calculation:
{
  "SOL": {
    "signal": "entry",
    "side": "long",
    "quantity": 3.0,
    "profit_target": 185.0,
    "stop_loss": 175.0,
    "leverage": 10,
    "confidence": 0.81,
    "invalidation_condition": "If price closes below 177 on a 3-minute candle",
    "justification": "Quantity calculation: for $500 risk, stop_distance $10 × leverage 10x = $100, so 500 / 100 = 5 units, rounded to 3 for conservative sizing. Breakout from consolidation at $178, targeting $185 resistance. Liquidation at ~162, stop at $175 provides safety buffer"
  }
}

LEVERAGE AND QUANTITY RELATIONSHIP:
- Higher leverage → smaller quantity needed for same risk exposure
- Lower leverage → larger quantity needed for same risk exposure
- Position value = quantity × price × leverage
- Always verify: stop_loss will be hit BEFORE liquidation price
- For volatile altcoins: use 5-10x leverage maximum
- For major coins (BTC/ETH): 10-20x can be acceptable with tight stops
- Never use max leverage (100x+) unless scalping with immediate stops

RISK MANAGEMENT RULES:
- profit_target and stop_loss MUST create a favorable risk-reward ratio (minimum 1.5:1, ideally 2:1 or higher)
- stop_loss placement is MORE important than profit_target - always protect capital first
- stop_loss MUST be placed with buffer BEFORE liquidation price (at least 10-20% away from liquidation)
- For ENTRY: both profit_target and stop_loss must be realistic price levels with clear reasoning
- For HOLD: maintain existing profit_target and stop_loss unless price action demands adjustment
- For CLOSE: set both to 0.0 as position is being exited
- quantity should be calculated to ensure your risk tolerance is respected while avoiding liquidation
- Never use percentage-based stops without considering market structure and volatility

FORMAT RULES:
- ALL keys must be present in every response
- Use empty string "" for text fields that don't apply (side, justification, invalidation_condition)
- Use 0.0 for numeric fields that don't apply (profit_target, stop_loss)
- signal: must be exactly "hold", "entry", or "close"
- side: "long", "short", or "" (empty only for hold/close)
- justification: required for entry/close, "" for hold
- profit_target and stop_loss: must be actual price levels, not percentages
- quantity: must be in coin units (not USD), calculated based on your risk tolerance and leverage
- leverage: must be realistic (5-20x recommended), always verify stop is before liquidation

IMPORTANT:
- Only suggest entries if you see strong opportunities
- Use proper risk management
- Provide clear invalidation conditions
- Return ONLY valid JSON, no other text
- DO NOT wrap in markdown code blocks (```json)
- DO NOT use escaped quotes (""), use regular quotes (")
- Return raw JSON only
""".strip()
    )

    return "\n".join(prompt_lines)


def calculate_llm_equity(
    model_key: str, llm_state: Dict, market_snapshots: Dict[str, Any]
) -> float:
    """Calculate current equity for an LLM."""
    # Calculate total margin (similar to trading workflow)
    total_margin = sum(
        float(pos.get("margin", 0.0)) for pos in llm_state["positions"].values()
    )

    # Start with balance + margin
    equity = llm_state["balance"] + total_margin

    # Add unrealized PnL from open positions
    for coin, pos in llm_state["positions"].items():
        current_price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
        pnl = calculate_position_pnl(pos, current_price)
        equity += pnl

    return equity


def calculate_position_pnl(position: Dict, current_price: float) -> float:
    """Calculate P&L for a position (without leverage in the calculation)."""
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
