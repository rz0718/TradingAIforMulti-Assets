#!/usr/bin/env python3
"""
Multi-LLM Trading Workflow for independent performance testing.
Each LLM trades independently with shared market data.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from . import config, data_processing, utils
from .multi_llm_manager import MultiLLMManager
from .performance_tracker import PerformanceTracker


class MultiTradingWorkflow:
    def __init__(self):
        self.llm_manager = MultiLLMManager()
        self.performance_tracker = PerformanceTracker()
        self.market_data = {}

    async def run_trading_loop(self):
        """Main trading loop for multi-LLM performance testing."""
        utils.setup_logging()
        utils.init_csv_files()

        logging.info("🚀 Initializing Multi-LLM Trading System...")
        logging.info(f"📊 Models: {list(config.LLM_MODELS.keys())}")
        logging.info(f"💰 Capital per LLM: ${config.CAPITAL_PER_LLM:.2f}")
        logging.info(f"📈 Symbols: {config.SYMBOLS}")

        iteration_count = 0

        while True:
            try:
                iteration_count += 1

                header = f"\n{'='*20} Iteration {iteration_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*20}"
                print(header)

                # 1. Fetch shared market data
                logging.info("📊 Fetching market data...")
                self.market_data = await self.fetch_market_data()

                if not self.market_data:
                    logging.warning("⚠️ No market data available, skipping iteration")
                    await asyncio.sleep(config.CHECK_INTERVAL)
                    continue

                # 2. Get decisions from all LLMs concurrently
                logging.info("🤖 Getting decisions from all LLMs...")
                all_decisions = await self.llm_manager.get_all_llm_decisions(
                    self.market_data
                )

                # 3. Execute trades for each LLM independently
                logging.info("⚡ Executing trades...")
                await self.execute_all_trades(all_decisions)

                # 4. Log performance and display summary
                all_performance = self.llm_manager.get_all_performance()
                self.performance_tracker.log_performance(
                    all_performance, self.market_data, self.llm_manager
                )
                self.performance_tracker.print_performance_summary(all_performance)

                # 5. Wait for next interval
                logging.info(f"⏰ Waiting {config.CHECK_INTERVAL} seconds...")
                await asyncio.sleep(config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                logging.info("🛑 Shutdown signal received.")
                self.print_final_performance()
                break
            except Exception as e:
                logging.error(f"❌ Error in trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data for all symbols."""
        market_snapshots = {}

        for symbol in config.SYMBOLS:
            snapshot = data_processing.collect_market_data(symbol)
            if snapshot:
                market_snapshots[snapshot["coin"]] = snapshot

        if len(market_snapshots) != len(config.SYMBOLS):
            logging.warning(
                f"⚠️ Only fetched data for {len(market_snapshots)}/{len(config.SYMBOLS)} symbols"
            )

        return market_snapshots

    async def execute_all_trades(self, all_decisions: Dict[str, Any]):
        """Execute trades for all LLMs independently."""
        for model_key, decisions in all_decisions.items():
            for decision in decisions:
                current_price = self.market_data.get(decision.coin, {}).get("price")
                if current_price:
                    self.llm_manager.execute_trade(
                        model_key, decision.coin, decision, current_price
                    )

    def print_final_performance(self):
        """Print final performance summary."""
        all_performance = self.llm_manager.get_all_performance()

        print("\n" + "🏆" * 20)
        print("FINAL PERFORMANCE SUMMARY")
        print("🏆" * 20)

        self.performance_tracker.print_performance_summary(all_performance)

        # Export final report
        report_path = self.performance_tracker.export_performance_report(
            all_performance
        )
        print(f"\n📊 Detailed report exported to: {report_path}")

        # Show top performer
        top_performer = self.performance_tracker.get_top_performer(all_performance)
        if top_performer:
            model_name = config.LLM_MODELS[top_performer]["name"]
            print(f"\n🥇 Top Performer: {model_name} ({top_performer})")
