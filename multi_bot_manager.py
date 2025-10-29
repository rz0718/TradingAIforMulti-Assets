#!/usr/bin/env python3
"""
Multi-Bot Manager for Running and Comparing Multiple LLM Trading Bots
"""
from __future__ import annotations

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
import numpy as np

from llm_bot import LLMBot
from parameter import START_CAPITAL, CHECK_INTERVAL, DEFAULT_RISK_FREE_RATE

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | [%(name)s] | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


@dataclass
class BotPerformance:
    """Performance metrics for a single bot."""
    bot_id: str
    model: str
    start_time: datetime
    current_equity: float
    total_return_pct: float
    total_return_usd: float
    num_trades: int
    num_winning_trades: int
    num_losing_trades: int
    total_profit_usd: float
    total_loss_usd: float
    win_rate: float
    profit_factor: float  # total_profit / total_loss
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown_pct: float
    current_positions: int
    avg_holding_time_hours: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'last_updated': self.last_updated.isoformat(),
        }


class MultiBotManager:
    """Manages multiple LLM trading bot instances and tracks their performance."""
    
    def __init__(self, performance_history_file: Optional[Path] = None):
        """
        Initialize the multi-bot manager.
        
        Args:
            performance_history_file: Path to JSON file to store performance history.
                                     If None, uses DATA_DIR / "bot_performance_history.json"
        """
        self.bots: Dict[str, LLMBot] = {}
        self.bot_threads: Dict[str, threading.Thread] = {}
        self.bot_stop_flags: Dict[str, threading.Event] = {}
        self.performance_history_file = performance_history_file or (DATA_DIR / "bot_performance_history.json")
        
        # Performance tracking
        self.performance_snapshots: List[Dict[str, Any]] = []
        self.load_performance_history()
        
        logger.info(f"MultiBotManager initialized. Performance file: {self.performance_history_file}")
    
    def add_bot(self, model: str, bot_id: Optional[str] = None) -> str:
        """
        Add a new bot instance.
        
        Args:
            model: LLM model name to use (e.g., "deepseek-chat", "gpt-4", etc.)
            bot_id: Unique identifier for the bot. If None, auto-generates based on model.
        
        Returns:
            The bot_id (either provided or generated)
        """
        if bot_id is None:
            # Generate a unique bot_id based on model and timestamp
            base_id = model.replace(" ", "_").replace("/", "_").lower()
            existing_bots = [bid for bid in self.bots.keys() if base_id in bid]
            if existing_bots:
                bot_id = f"{base_id}_{len(existing_bots) + 1}"
            else:
                bot_id = base_id
        
        if bot_id in self.bots:
            logger.warning(f"Bot with id '{bot_id}' already exists. Skipping.")
            return bot_id
        
        logger.info(f"Adding bot: id='{bot_id}', model='{model}'")
        bot = LLMBot(model=model, bot_id=bot_id)
        self.bots[bot_id] = bot
        self.bot_stop_flags[bot_id] = threading.Event()
        
        return bot_id
    
    def start_bot(self, bot_id: str) -> bool:
        """
        Start a bot instance in a separate thread.
        
        Args:
            bot_id: The bot identifier
        
        Returns:
            True if started successfully, False otherwise
        """
        if bot_id not in self.bots:
            logger.error(f"Bot '{bot_id}' not found.")
            return False
        
        if bot_id in self.bot_threads and self.bot_threads[bot_id].is_alive():
            logger.warning(f"Bot '{bot_id}' is already running.")
            return False
        
        logger.info(f"Starting bot '{bot_id}' in background thread...")
        stop_event = self.bot_stop_flags[bot_id]
        bot = self.bots[bot_id]
        
        # Initialize bot before starting
        bot.init_csv_files()
        bot.load_equity_history()
        bot.load_state()
        
        def run_bot():
            try:
                # Monkey-patch the run method to respect stop event
                original_run = bot.run
                bot_is_running = True
                
                while bot_is_running and not stop_event.is_set():
                    try:
                        # Run one iteration
                        bot.iteration_counter += 1
                        bot.current_iteration_messages = []
                        
                        from market import get_binance_client
                        if not get_binance_client():
                            retry_delay = min(CHECK_INTERVAL, 60)
                            logger.warning(f"Bot '{bot_id}': Binance client unavailable; retrying in {retry_delay}s")
                            time.sleep(retry_delay)
                            continue
                        
                        # Check stop loss / take profit
                        bot.check_stop_loss_take_profit()
                        
                        # Get AI decisions
                        logger.info(f"Bot '{bot_id}': Requesting trading decisions from {bot.model}...")
                        prompt = bot.format_prompt_for_deepseek()
                        decisions = bot.call_llm_api(prompt)
                        
                        if decisions:
                            from config import SYMBOL_TO_COIN
                            from market import fetch_market_data
                            
                            # Process decisions for each coin
                            for coin in SYMBOL_TO_COIN.values():
                                if coin not in decisions:
                                    continue
                                
                                decision = decisions[coin]
                                signal = decision.get('signal', 'hold')
                                
                                # Log AI decision
                                bot.log_ai_decision(
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
                                    bot.execute_entry(coin, decision, current_price)
                                elif signal == 'close':
                                    bot.execute_close(coin, decision, current_price)
                                elif signal == 'hold':
                                    if coin in bot.positions:
                                        pos = bot.positions[coin]
                                        raw_reason = str(decision.get('justification', '')).strip()
                                        if raw_reason:
                                            pos['last_justification'] = " ".join(raw_reason.split())
                        
                        # Calculate and log portfolio summary
                        from parameter import START_CAPITAL, CHECK_INTERVAL, DEFAULT_RISK_FREE_RATE
                        from colorama import Fore, Style
                        
                        total_equity = bot.calculate_total_equity()
                        total_return = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100
                        bot.register_equity_snapshot(total_equity)
                        
                        sortino_ratio = bot.calculate_sortino_ratio(
                            bot.equity_history,
                            CHECK_INTERVAL,
                            DEFAULT_RISK_FREE_RATE,
                        )
                        
                        # Log portfolio state
                        bot.log_portfolio_state()
                        bot.save_state()
                        
                        # Sleep until next iteration
                        time.sleep(CHECK_INTERVAL)
                    except KeyboardInterrupt:
                        logger.info(f"Bot '{bot_id}': Received interrupt signal")
                        bot_is_running = False
                        break
                    except Exception as e:
                        logger.error(f"Bot '{bot_id}': Error in iteration: {e}", exc_info=True)
                        bot.save_state()
                        time.sleep(60)
                
                logger.info(f"Bot '{bot_id}': Stopped")
            except Exception as e:
                logger.error(f"Bot '{bot_id}': Fatal error: {e}", exc_info=True)
        
        thread = threading.Thread(target=run_bot, daemon=True, name=f"BotThread-{bot_id}")
        self.bot_threads[bot_id] = thread
        thread.start()
        
        return True
    
    def stop_bot(self, bot_id: str) -> bool:
        """
        Stop a running bot instance.
        
        Args:
            bot_id: The bot identifier
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if bot_id not in self.bot_stop_flags:
            logger.error(f"Bot '{bot_id}' not found.")
            return False
        
        logger.info(f"Stopping bot '{bot_id}'...")
        self.bot_stop_flags[bot_id].set()
        
        if bot_id in self.bot_threads:
            thread = self.bot_threads[bot_id]
            thread.join(timeout=30)
            if thread.is_alive():
                logger.warning(f"Bot '{bot_id}' thread did not stop within timeout.")
        
        if bot_id in self.bots:
            self.bots[bot_id].save_state()
        
        return True
    
    def start_all_bots(self) -> None:
        """Start all registered bot instances."""
        logger.info(f"Starting {len(self.bots)} bots...")
        for bot_id in self.bots.keys():
            self.start_bot(bot_id)
    
    def stop_all_bots(self) -> None:
        """Stop all running bot instances."""
        logger.info(f"Stopping {len(self.bot_threads)} bots...")
        for bot_id in list(self.bot_threads.keys()):
            self.stop_bot(bot_id)
    
    def get_bot_performance(self, bot_id: str) -> Optional[BotPerformance]:
        """
        Calculate and return performance metrics for a bot.
        
        Args:
            bot_id: The bot identifier
        
        Returns:
            BotPerformance object or None if bot not found
        """
        if bot_id not in self.bots:
            return None
        
        bot = self.bots[bot_id]
        
        # Load trade history
        trades_df = None
        if bot.TRADES_CSV.exists():
            try:
                trades_df = pd.read_csv(bot.TRADES_CSV)
            except Exception as e:
                logger.warning(f"Could not load trade history for '{bot_id}': {e}")
        
        # Load equity history
        equity_df = None
        if bot.STATE_CSV.exists():
            try:
                equity_df = pd.read_csv(bot.STATE_CSV)
                if 'total_equity' in equity_df.columns:
                    equity_df['total_equity'] = pd.to_numeric(equity_df['total_equity'], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not load equity history for '{bot_id}': {e}")
        
        # Calculate current equity
        total_equity = bot.calculate_total_equity()
        total_return_pct = ((total_equity - START_CAPITAL) / START_CAPITAL) * 100 if START_CAPITAL > 0 else 0.0
        total_return_usd = total_equity - START_CAPITAL
        
        # Trade statistics
        num_trades = 0
        num_winning_trades = 0
        num_losing_trades = 0
        total_profit_usd = 0.0
        total_loss_usd = 0.0
        
        if trades_df is not None and not trades_df.empty:
            closed_trades = trades_df[trades_df['action'] == 'CLOSE'].copy()
            if not closed_trades.empty and 'pnl' in closed_trades.columns:
                closed_trades['pnl'] = pd.to_numeric(closed_trades['pnl'], errors='coerce').fillna(0)
                num_trades = len(closed_trades)
                num_winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
                num_losing_trades = len(closed_trades[closed_trades['pnl'] < 0])
                total_profit_usd = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
                total_loss_usd = abs(closed_trades[closed_trades['pnl'] < 0]['pnl'].sum())
        
        win_rate = (num_winning_trades / num_trades * 100) if num_trades > 0 else 0.0
        profit_factor = (total_profit_usd / total_loss_usd) if total_loss_usd > 0 else (float('inf') if total_profit_usd > 0 else 0.0)
        
        # Risk metrics
        sharpe_ratio = None
        sortino_ratio = None
        max_drawdown_pct = 0.0
        
        if equity_df is not None and not equity_df.empty and 'total_equity' in equity_df.columns:
            equity_series = equity_df['total_equity'].dropna()
            if len(equity_series) > 1:
                # Calculate returns
                returns = equity_series.pct_change().dropna()
                if len(returns) > 0:
                    # Sharpe ratio (simplified)
                    if returns.std() > 0:
                        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(equity_series) / (CHECK_INTERVAL / 3600)) if len(equity_series) > 1 else None
                    
                    # Sortino ratio
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0 and downside_returns.std() > 0:
                        periods_per_year = (365 * 24 * 3600) / CHECK_INTERVAL
                        excess_return = returns.mean() - (DEFAULT_RISK_FREE_RATE / periods_per_year)
                        sortino_ratio = (excess_return / downside_returns.std()) * np.sqrt(periods_per_year) if excess_return and np.isfinite(excess_return) else None
                
                # Max drawdown
                peak = equity_series.expanding().max()
                drawdown = (equity_series - peak) / peak * 100
                max_drawdown_pct = drawdown.min() if not drawdown.empty else 0.0
        
        # Average holding time
        avg_holding_time_hours = 0.0
        if trades_df is not None and not trades_df.empty:
            # This is a simplified calculation - in reality, you'd need to match entry/exit pairs
            # For now, we'll estimate based on iteration count
            if bot.iteration_counter > 0:
                avg_holding_time_hours = (bot.iteration_counter * CHECK_INTERVAL) / 3600 / max(len(bot.positions), 1)
        
        return BotPerformance(
            bot_id=bot_id,
            model=bot.model,
            start_time=bot.BOT_START_TIME,
            current_equity=total_equity,
            total_return_pct=total_return_pct,
            total_return_usd=total_return_usd,
            num_trades=num_trades,
            num_winning_trades=num_winning_trades,
            num_losing_trades=num_losing_trades,
            total_profit_usd=total_profit_usd,
            total_loss_usd=total_loss_usd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            current_positions=len(bot.positions),
            avg_holding_time_hours=avg_holding_time_hours,
            last_updated=datetime.now(timezone.utc)
        )
    
    def update_performance_snapshot(self) -> None:
        """Update performance snapshot for all bots and save to history."""
        snapshot_time = datetime.now(timezone.utc)
        snapshot = {
            'timestamp': snapshot_time.isoformat(),
            'bots': {}
        }
        
        for bot_id in self.bots.keys():
            perf = self.get_bot_performance(bot_id)
            if perf:
                snapshot['bots'][bot_id] = perf.to_dict()
        
        self.performance_snapshots.append(snapshot)
        self.save_performance_history()
    
    def save_performance_history(self) -> None:
        """Save performance history to JSON file."""
        try:
            with open(self.performance_history_file, 'w') as f:
                json.dump(self.performance_snapshots, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
    
    def load_performance_history(self) -> None:
        """Load performance history from JSON file."""
        if not self.performance_history_file.exists():
            return
        
        try:
            with open(self.performance_history_file, 'r') as f:
                self.performance_snapshots = json.load(f)
            logger.info(f"Loaded {len(self.performance_snapshots)} performance snapshots")
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
            self.performance_snapshots = []
    
    def compare_bots(self, bot_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance metrics of multiple bots.
        
        Args:
            bot_ids: List of bot IDs to compare. If None, compares all bots.
        
        Returns:
            DataFrame with comparison metrics
        """
        if bot_ids is None:
            bot_ids = list(self.bots.keys())
        
        if not bot_ids:
            logger.warning("No bots to compare")
            return pd.DataFrame()
        
        comparison_data = []
        for bot_id in bot_ids:
            perf = self.get_bot_performance(bot_id)
            if perf:
                comparison_data.append(perf.to_dict())
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # Select and order columns for better readability
        display_columns = [
            'bot_id', 'model', 'total_return_pct', 'total_return_usd', 'current_equity',
            'num_trades', 'win_rate', 'profit_factor', 'max_drawdown_pct',
            'sharpe_ratio', 'sortino_ratio', 'current_positions', 'last_updated'
        ]
        
        available_columns = [c for c in display_columns if c in df.columns]
        return df[available_columns].sort_values('total_return_pct', ascending=False)
    
    def print_comparison(self, bot_ids: Optional[List[str]] = None) -> None:
        """Print a formatted comparison table of bot performance."""
        df = self.compare_bots(bot_ids)
        
        if df.empty:
            logger.warning("No performance data available for comparison")
            return
        
        print("\n" + "=" * 100)
        print("BOT PERFORMANCE COMPARISON")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")


def main():
    """Example usage of MultiBotManager."""
    manager = MultiBotManager()
    
    # Add multiple bots with different models
    bot_ids = [
        manager.add_bot("deepseek-chat", "deepseek-1"),
        manager.add_bot("anthropic/claude-3.5-sonnet", "claude-1"),
        manager.add_bot("openai/gpt-4", "gpt4-1"),
    ]
    
    print(f"Added {len(bot_ids)} bots: {', '.join(bot_ids)}")
    print("\nStarting all bots...")
    manager.start_all_bots()
    
    try:
        # Run for a while, periodically updating performance
        while True:
            time.sleep(180)  # Update every 5 minutes
            manager.update_performance_snapshot()
            manager.print_comparison()
    except KeyboardInterrupt:
        print("\nStopping all bots...")
        manager.stop_all_bots()
        manager.update_performance_snapshot()
        manager.print_comparison()


if __name__ == "__main__":
    main()

