#!/usr/bin/env python3
"""
Performance tracking system for comparing multiple LLM trading results.
"""
import csv
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from . import config, utils, prompts


class PerformanceTracker:
    def __init__(self):
        self.performance_csv = config.DATA_DIR / "llm_performance.csv"
        self.leaderboard_csv = config.DATA_DIR / "llm_leaderboard.csv"
        self.init_csv_files()

    def init_csv_files(self):
        """Initialize CSV files for performance tracking."""
        # LLM Performance CSV
        if not self.performance_csv.exists():
            with open(self.performance_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "model",
                        "balance",
                        "equity",
                        "total_return_pct",
                        "total_trades",
                        "winning_trades",
                        "win_rate",
                        "total_pnl",
                        "positions",
                        "best_trade",
                        "worst_trade",
                    ]
                )

        # Leaderboard CSV
        if not self.leaderboard_csv.exists():
            with open(self.leaderboard_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "rank",
                        "model",
                        "total_return_pct",
                        "win_rate",
                        "total_trades",
                        "total_pnl",
                        "sharpe_ratio",
                    ]
                )

    def log_performance(
        self,
        all_performance: Dict[str, Dict[str, Any]],
        market_snapshots: Dict[str, Any],
        llm_manager=None,
    ):
        """Log performance metrics for all LLMs."""
        timestamp = datetime.now().isoformat()

        # Calculate additional metrics
        enhanced_performance = {}
        for model_key, perf in all_performance.items():
            # Calculate current equity using LLM manager if available
            if llm_manager:
                llm_state = llm_manager.llm_states.get(model_key, {})
                equity = prompts.calculate_llm_equity(
                    model_key, llm_state, market_snapshots
                )
            else:
                # Fallback to balance only
                equity = perf["balance"]

            total_return_pct = (
                (equity - config.CAPITAL_PER_LLM) / config.CAPITAL_PER_LLM
            ) * 100

            enhanced_performance[model_key] = {
                **perf,
                "equity": equity,
                "total_return_pct": total_return_pct,
                "best_trade": 0.0,  # Placeholder
                "worst_trade": 0.0,  # Placeholder
            }

        # Log to CSV
        with open(self.performance_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for model_key, perf in enhanced_performance.items():
                writer.writerow(
                    [
                        timestamp,
                        model_key,
                        perf["balance"],
                        perf["equity"],
                        perf["total_return_pct"],
                        perf["total_trades"],
                        perf["winning_trades"],
                        perf["win_rate"],
                        perf["total_pnl"],
                        perf["positions"],
                        perf["best_trade"],
                        perf["worst_trade"],
                    ]
                )

        # Update leaderboard
        self.update_leaderboard(enhanced_performance)

    def update_leaderboard(self, performance: Dict[str, Dict[str, Any]]):
        """Update the leaderboard with current rankings."""
        # Sort by total return percentage
        sorted_models = sorted(
            performance.items(), key=lambda x: x[1]["total_return_pct"], reverse=True
        )

        timestamp = datetime.now().isoformat()

        with open(self.leaderboard_csv, "a", newline="") as f:
            writer = csv.writer(f)
            for rank, (model_key, perf) in enumerate(sorted_models, 1):
                # Calculate Sharpe ratio (simplified)
                sharpe_ratio = self.calculate_sharpe_ratio(model_key)

                writer.writerow(
                    [
                        timestamp,
                        rank,
                        model_key,
                        perf["total_return_pct"],
                        perf["win_rate"],
                        perf["total_trades"],
                        perf["total_pnl"],
                        sharpe_ratio,
                    ]
                )

    def calculate_sharpe_ratio(self, model_key: str) -> float:
        """Calculate simplified Sharpe ratio for a model."""
        # This is a placeholder - in practice, you'd calculate based on historical returns
        return 0.0

    def print_performance_summary(self, all_performance: Dict[str, Dict[str, Any]]):
        """Print a formatted performance summary."""
        print("\n" + "=" * 80)
        print("🤖 MULTI-LLM TRADING PERFORMANCE SUMMARY")
        print("=" * 80)

        # Sort by total return
        sorted_models = sorted(
            all_performance.items(),
            key=lambda x: x[1].get("total_pnl", 0),
            reverse=True,
        )

        print(
            f"{'Rank':<4} {'Model':<15} {'Return%':<8} {'Trades':<7} {'Win%':<6} {'P&L':<10} {'Positions':<9}"
        )
        print("-" * 80)

        for rank, (model_key, perf) in enumerate(sorted_models, 1):
            model_name = config.LLM_MODELS[model_key]["name"]
            total_return = (
                (perf["balance"] + perf["total_pnl"] - config.CAPITAL_PER_LLM)
                / config.CAPITAL_PER_LLM
            ) * 100

            print(
                f"{rank:<4} {model_name:<15} {total_return:<8.2f} {perf['total_trades']:<7} "
                f"{perf['win_rate']:<6.1f} ${perf['total_pnl']:<9.2f} {perf['positions']:<9}"
            )

        print("=" * 80)

    def get_top_performer(self, all_performance: Dict[str, Dict[str, Any]]) -> str:
        """Get the top performing model."""
        if not all_performance:
            return None

        return max(all_performance.items(), key=lambda x: x[1].get("total_pnl", 0))[0]

    def export_performance_report(
        self, all_performance: Dict[str, Dict[str, Any]]
    ) -> str:
        """Export a detailed performance report."""
        report_path = (
            config.DATA_DIR
            / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "models": all_performance,
            "summary": {
                "total_models": len(all_performance),
                "top_performer": self.get_top_performer(all_performance),
                "average_return": sum(
                    p.get("total_pnl", 0) for p in all_performance.values()
                )
                / len(all_performance),
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return str(report_path)
