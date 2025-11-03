#!/usr/bin/env python3
"""
Test script for Telegram message formatting and sending.
This allows you to preview and test the Telegram notification format.
"""

from bot import utils, config
from datetime import datetime, timezone

# Sample data to test the formatting
sample_new_trades = [
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coin": "BTC",
        "action": "ENTRY",
        "side": "long",
        "quantity": 0.05,
        "price": 43250.50,
        "profit_target": 44500.00,
        "stop_loss": 42800.00,
        "leverage": 10,
        "confidence": 0.75,
        "pnl": 0.0,
        "balance_after": 9500.00,
        "reason": "Strong bullish momentum with RSI showing oversold conditions. Price breaking above EMA resistance."
    },
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coin": "ETH",
        "action": "CLOSE",
        "side": "long",
        "quantity": 2.5,
        "price": 2280.75,
        "profit_target": 2350.00,
        "stop_loss": 2200.00,
        "leverage": 8,
        "confidence": 0.68,
        "pnl": 125.50,
        "balance_after": 9625.50,
        "reason": "Take profit target reached. Securing gains before potential reversal."
    }
]

sample_positions = {
    "BTC": {
        "side": "long",
        "quantity": 0.05,
        "entry_price": 43250.50,
        "current_price": 43350.00,
        "profit_target": 44500.00,
        "stop_loss": 42800.00,
        "leverage": 10,
        "margin": 216.25,
        "confidence": 0.75,
        "unrealized_pnl": 5.00
    },
    "SOL": {
        "side": "short",
        "quantity": 10.0,
        "entry_price": 98.50,
        "current_price": 97.80,
        "profit_target": 95.00,
        "stop_loss": 100.00,
        "leverage": 5,
        "margin": 197.00,
        "confidence": 0.62,
        "unrealized_pnl": 7.00
    }
}

sample_market_snapshots = {
    "BTC": {"price": 43350.00},
    "SOL": {"price": 97.80},
    "ETH": {"price": 2280.75}
}

def test_format_only():
    """Test the message formatting without sending."""
    print("\n" + "="*60)
    print("TESTING TELEGRAM MESSAGE FORMAT")
    print("="*60 + "\n")
    
    message = utils.format_trading_signal_message(
        new_trades=sample_new_trades,
        positions=sample_positions,
        market_snapshots=sample_market_snapshots,
        short_summary="Bullish momentum building across major alts. BTC looking strong above $43k support. Portfolio up 6.25% this week. 🚀",
        total_equity=10125.50,
        total_return_pct=1.26,
        net_unrealized_pnl=12.00,
    )
    
    print("Preview of Telegram message (HTML formatted):")
    print("-" * 60)
    print(message)
    print("-" * 60)
    print("\nNote: Telegram will render this with bold (<b>) and italic (<i>) formatting.\n")


def test_no_trades():
    """Test formatting when there are no new trades."""
    print("\n" + "="*60)
    print("TESTING WITH NO NEW TRADES")
    print("="*60 + "\n")
    
    message = utils.format_trading_signal_message(
        new_trades=[],  # No trades
        positions=sample_positions,
        market_snapshots=sample_market_snapshots,
        short_summary="Chill vibes only. Holding positions, waiting for the right moment. Patience = profits. 💎🙌",
        total_equity=10125.50,
        total_return_pct=1.26,
        net_unrealized_pnl=12.00,
    )
    
    print("Preview of Telegram message (HTML formatted):")
    print("-" * 60)
    print(message)
    print("-" * 60)


def test_no_positions():
    """Test formatting when there are no open positions."""
    print("\n" + "="*60)
    print("TESTING WITH NO POSITIONS")
    print("="*60 + "\n")
    
    message = utils.format_trading_signal_message(
        new_trades=[],
        positions={},  # No positions
        market_snapshots=sample_market_snapshots,
        short_summary="Cash gang for now. Market looking choppy, preserving capital. 💰",
        total_equity=10000.00,
        total_return_pct=0.00,
        net_unrealized_pnl=0.00,
    )
    
    print("Preview of Telegram message (HTML formatted):")
    print("-" * 60)
    print(message)
    print("-" * 60)


def test_send_to_telegram():
    """Actually send a test message to Telegram."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        print("\n" + "="*60)
        print("⚠️  TELEGRAM CREDENTIALS NOT CONFIGURED")
        print("="*60)
        print("\nPlease set the following environment variables in your .env file:")
        print("  - TELEGRAM_BOT_TOKEN")
        print("  - TELEGRAM_CHAT_ID")
        print("\nSee env.template for more information.")
        return
    
    print("\n" + "="*60)
    print("SENDING TEST MESSAGE TO TELEGRAM")
    print("="*60)
    print(f"\nBot Token: {config.TELEGRAM_BOT_TOKEN[:10]}...")
    print(f"Chat ID: {config.TELEGRAM_CHAT_ID}\n")
    
    message = utils.format_trading_signal_message(
        new_trades=sample_new_trades,
        positions=sample_positions,
        market_snapshots=sample_market_snapshots,
        short_summary="🧪 This is a TEST message from your trading bot! Looking good! 🚀",
        total_equity=10125.50,
        total_return_pct=1.26,
        net_unrealized_pnl=12.00,
    )
    
    try:
        utils.send_telegram_message(message, parse_mode="HTML")
        print("✅ Test message sent successfully!")
        print("\nCheck your Telegram to see how it looks.")
    except Exception as e:
        print(f"❌ Failed to send message: {e}")


if __name__ == "__main__":
    import sys
    
    print("\n🤖 Telegram Message Format Test Suite\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--send":
        # Send actual message to Telegram
        test_send_to_telegram()
    else:
        # Just preview the format
        test_format_only()
        print("\n")
        test_no_trades()
        print("\n")
        test_no_positions()
        
        print("\n" + "="*60)
        print("To actually send a test message to Telegram, run:")
        print(f"  python {sys.argv[0]} --send")
        print("="*60 + "\n")

