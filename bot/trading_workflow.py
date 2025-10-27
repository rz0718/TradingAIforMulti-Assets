#!/usr/bin/env python3
"""
Main trading workflow, state management, and execution logic.
"""
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from colorama import Fore, Style

from . import clients, config, data_processing, prompts, utils


class TradingState:
    """Manages the full state of the trading bot."""

    def __init__(self):
        self.balance: float = config.START_CAPITAL
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.equity_history: list[float] = []
        self.start_time: datetime = datetime.now(timezone.utc)
        self.invocation_count: int = 0
        self.current_iteration_messages: list[str] = []

    def load_state(self):
        """Load persisted balance and positions if available."""
        if not utils.STATE_JSON.exists():
            logging.info("No existing state file found; starting fresh.")
            return
        try:
            with open(utils.STATE_JSON, "r") as f:
                data = json.load(f)
            self.balance = float(data.get("balance", config.START_CAPITAL))
            # ... (More robust state loading can be added here)
            logging.info("Loaded state from %s (balance: %.2f)", utils.STATE_JSON, self.balance)
        except Exception as e:
            logging.error("Failed to load state: %s", e, exc_info=True)

    def save_state(self):
        """Persist current balance and open positions."""
        try:
            with open(utils.STATE_JSON, "w") as f:
                json.dump({"balance": self.balance, "positions": self.positions}, f, indent=2)
        except Exception as e:
            logging.error("Failed to save state: %s", e, exc_info=True)

    def calculate_unrealized_pnl(self, coin: str, current_price: float) -> float:
        pos = self.positions[coin]
        if pos['side'] == 'long':
            return (current_price - pos['entry_price']) * pos['quantity']
        return (pos['entry_price'] - current_price) * pos['quantity']

    def get_summary(self, market_snapshots: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates a full summary of the current portfolio state."""
        total_margin = sum(float(p.get('margin', 0.0)) for p in self.positions.values())
        total_equity = self.balance + total_margin
        for coin, pos in self.positions.items():
            price = market_snapshots.get(coin, {}).get("price", pos["entry_price"])
            total_equity += self.calculate_unrealized_pnl(coin, price)
        
        total_return_pct = ((total_equity - config.START_CAPITAL) / config.START_CAPITAL) * 100
        net_unrealized_pnl = total_equity - self.balance - total_margin

        return {
            "balance": self.balance,
            "positions": self.positions,
            "start_time": self.start_time,
            "invocation_count": self.invocation_count,
            "total_equity": total_equity,
            "total_margin": total_margin,
            "total_return_pct": total_return_pct,
            "net_unrealized_pnl": net_unrealized_pnl,
        }

def get_llm_decisions(state: TradingState, market_snapshots: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Calls the LLM API and returns trading decisions."""
    llm_client = clients.get_llm_client()
    if not llm_client:
        logging.error("LLM client is not available.")
        return None

    prompt = prompts.create_trading_prompt(state.get_summary(market_snapshots), market_snapshots)
    
    try:
        utils.log_ai_message({
            'timestamp': datetime.now().isoformat(), 'direction': "sent", 'role': "system",
            'content': prompts.TRADING_RULES_PROMPT, 'metadata': {"model": config.LLM_MODEL_NAME}
        })
        utils.log_ai_message({
            'timestamp': datetime.now().isoformat(), 'direction': "sent", 'role': "user",
            'content': prompt, 'metadata': {"model": config.LLM_MODEL_NAME}
        })

        response = llm_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompts.TRADING_RULES_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        utils.log_ai_message({
            'timestamp': datetime.now().isoformat(), 'direction': "received", 'role': "assistant",
            'content': content, 'metadata': {"id": response.id}
        })

        if not content:
            logging.error("LLM API returned empty content.")
            return None
        
        return json.loads(content)

    except Exception as e:
        logging.error(f"Error calling LLM API: {e}", exc_info=True)
        return None

def execute_trade(state: TradingState, coin: str, decision: Dict[str, Any], price: float):
    """Executes a single trade based on an LLM decision."""
    # ... (Implementation of execute_entry and execute_close from original bot.py)
    # This function would be much larger in a real implementation.
    logging.info(f"Executing trade for {coin} based on signal: {decision.get('signal')}")
    # For now, we just log the intent.
    pass # Placeholder

def run_trading_loop():
    """The main event loop for the trading bot."""
    utils.setup_logging()
    utils.init_csv_files()
    
    state = TradingState()
    state.load_state()

    logging.info("Initializing clients...")
    if not clients.get_binance_client() or not clients.get_llm_client():
        logging.critical("Failed to initialize required API clients. Exiting.")
        return

    logging.info(f"Starting capital: ${config.START_CAPITAL:.2f}")
    logging.info(f"Monitoring: {list(config.SYMBOL_TO_COIN.values())}")

    while True:
        try:
            state.invocation_count += 1
            state.current_iteration_messages = []
            
            header = f"\n{Fore.CYAN}{'='*20} Iteration {state.invocation_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'='*20}{Style.RESET_ALL}"
            print(header)
            state.current_iteration_messages.append(utils.strip_ansi_codes(header))

            # 1. Fetch market data
            logging.info("Fetching market data for all symbols...")
            market_snapshots = {}
            for symbol in config.SYMBOLS:
                snapshot = data_processing.collect_market_data(symbol)
                if snapshot:
                    market_snapshots[snapshot["coin"]] = snapshot
            
            if len(market_snapshots) != len(config.SYMBOLS):
                logging.warning("Failed to fetch market data for one or more symbols.")

            # 2. Get LLM decisions
            logging.info("Requesting trading decisions from LLM...")
            decisions = get_llm_decisions(state, market_snapshots)

            # 3. Execute trades
            if decisions:
                for coin, decision in decisions.items():
                    if coin not in config.SYMBOL_TO_COIN.values(): continue
                    
                    utils.log_ai_decision({
                        'timestamp': datetime.now().isoformat(), 'coin': coin, **decision
                    })

                    current_price = market_snapshots.get(coin, {}).get('price')
                    if not current_price: continue

                    if decision.get('signal') in ['entry', 'close']:
                        execute_trade(state, coin, decision, current_price)
            else:
                logging.warning("No decisions received from LLM.")

            # 4. Log and display summary
            summary = state.get_summary(market_snapshots)
            utils.log_portfolio_state(summary)
            # ... (Display summary logic)

            # 5. Wait for next interval
            logging.info(f"Waiting {config.CHECK_INTERVAL} seconds...")
            time.sleep(config.CHECK_INTERVAL)

        except KeyboardInterrupt:
            logging.info("Shutdown signal received.")
            state.save_state()
            break
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
            state.save_state()
            time.sleep(60)
