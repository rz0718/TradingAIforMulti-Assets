#!/usr/bin/env python3
"""
Multi-LLM Manager for independent trading performance testing.
Each LLM trades independently with shared market data.
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE importing config
load_dotenv()

from . import config, prompts, utils


@dataclass
class LLMDecision:
    model_name: str
    coin: str
    signal: str
    side: Optional[str]
    quantity: float
    confidence: float
    reasoning: str
    timestamp: str
    risk_usd: float = 0.0
    profit_target: float = 0.0
    stop_loss: float = 0.0
    leverage: int = 1
    invalidation_condition: str = ""


class MultiLLMManager:
    def __init__(self):
        # Debug: Check if API key is loaded
        api_key = config.OPENROUTER_API_KEY
        if not api_key:
            logging.error("❌ OPENROUTER_API_KEY is not loaded!")
        else:
            logging.info(f"✅ OPENROUTER_API_KEY loaded (length: {len(api_key)})")

        self.openrouter_client = OpenAI(
            api_key=api_key, base_url=config.OPENROUTER_BASE_URL
        )
        self.llm_states = {}  # Independent state for each LLM
        self.initialize_llm_states()

    def initialize_llm_states(self):
        """Initialize independent trading state for each LLM."""
        for model_key in config.LLM_MODELS.keys():
            self.llm_states[model_key] = {
                "balance": config.CAPITAL_PER_LLM,  # Equal capital split
                "positions": {},
                "equity_history": [],
                "invocation_count": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
                "start_time": datetime.now(timezone.utc),
            }

    async def get_all_llm_decisions(
        self, market_snapshots: Dict[str, Any]
    ) -> Dict[str, List[LLMDecision]]:
        """Get trading decisions from all LLMs concurrently."""
        tasks = []

        for model_key, model_config in config.LLM_MODELS.items():
            task = self.get_llm_decision(model_key, model_config, market_snapshots)
            tasks.append(task)

        # Execute all LLM calls concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        all_decisions = {}
        for i, result in enumerate(results):
            model_key = list(config.LLM_MODELS.keys())[i]
            if isinstance(result, Exception):
                logging.error(f"Error getting decision from {model_key}: {result}")
                all_decisions[model_key] = []
            else:
                all_decisions[model_key] = result

        return all_decisions

    async def get_llm_decision(
        self, model_key: str, model_config: Dict, market_snapshots: Dict[str, Any]
    ) -> List[LLMDecision]:
        """Get trading decision from a single LLM."""
        try:
            # Get LLM-specific state
            llm_state = self.llm_states[model_key]

            # Create model-specific prompt
            prompt = prompts.create_llm_specific_prompt(
                model_key, llm_state, market_snapshots
            )

            # Log system prompt
            utils.log_ai_message(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "direction": "sent",
                    "role": "system",
                    "content": prompts.TRADING_RULES_PROMPT,
                    "metadata": {"model": model_key},
                }
            )

            # Log user prompt
            utils.log_ai_message(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "direction": "sent",
                    "role": "user",
                    "content": prompt,
                    "metadata": {"model": model_key},
                }
            )

            # Call OpenRouter API
            response = self.openrouter_client.chat.completions.create(
                model=model_config["model_id"],
                messages=[
                    {"role": "system", "content": prompts.TRADING_RULES_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            # Log the raw response for debugging
            utils.log_ai_message(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "direction": "received",
                    "role": "assistant",
                    "content": content,
                    "metadata": {"model": model_key},
                }
            )

            # Parse JSON with error handling and recovery
            decisions_data = self.parse_llm_response(content, model_key)

            # Convert to LLMDecision objects
            decisions = []
            for coin, decision_data in decisions_data.items():
                if coin in config.SYMBOL_TO_COIN.values():
                    decision = LLMDecision(
                        model_name=model_key,
                        coin=coin,
                        signal=decision_data.get("signal", "hold"),
                        side=decision_data.get("side"),
                        quantity=decision_data.get("quantity", 0.0),
                        confidence=decision_data.get("confidence", 0.5),
                        reasoning=decision_data.get("justification", ""),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        risk_usd=decision_data.get("risk_usd", 0.0),
                        profit_target=decision_data.get("profit_target", 0.0),
                        stop_loss=decision_data.get("stop_loss", 0.0),
                        leverage=decision_data.get("leverage", 1),
                        invalidation_condition=decision_data.get(
                            "invalidation_condition", ""
                        ),
                    )
                    decisions.append(decision)

            # Log the decision
            self.log_llm_decision(model_key, decisions)

            return decisions

        except Exception as e:
            logging.error(f"Error getting decision from {model_key}: {e}")
            return []

    def parse_llm_response(self, content: str, model_key: str) -> Dict[str, Any]:
        """Parse LLM response with robust error handling and recovery."""
        import re

        # First, try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse error for {model_key}: {e}")
            logging.debug(f"Content that failed to parse: {repr(content[:200])}...")

        # Handle Claude Sonnet's markdown code blocks
        try:
            # Look for ```json ... ``` pattern
            json_block_match = re.search(
                r"```json\s*(\{.*?\})\s*```", content, re.DOTALL
            )
            if json_block_match:
                json_str = json_block_match.group(1)
                # Fix escaped quotes
                json_str = json_str.replace('""', '"')
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Handle empty or whitespace-only responses
        if not content or not content.strip():
            logging.warning(f"Empty response from {model_key}")
            return self.create_default_decisions()

        # Try to extract JSON from the response using multiple strategies
        json_str = self.extract_json_from_response(content)
        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Handle incomplete JSON (like Gemini Pro truncation)
        try:
            # Try to complete incomplete JSON
            completed_json = self.complete_incomplete_json(content)
            if completed_json:
                return json.loads(completed_json)
        except json.JSONDecodeError:
            pass

        # Try to fix common JSON issues
        try:
            # Fix common issues: unquoted keys, trailing commas, etc.
            fixed_content = self.fix_json_issues(content)
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass

        # If all else fails, create default hold decisions
        logging.error(
            f"Failed to parse JSON for {model_key}, using default hold decisions"
        )
        return self.create_default_decisions()

    def extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response using multiple strategies."""
        import re

        # Strategy 1: Look for JSON object that ends before prompt instructions
        try:
            json_match = re.search(
                r"\{.*?\}(?=\n\nJSON FORMATTING RULES|IMPORTANT:|$)", content, re.DOTALL
            )
            if json_match:
                return json_match.group(0)
        except AttributeError:
            pass

        # Strategy 2: Look for JSON that ends with } followed by whitespace and prompt text
        try:
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}(?=\s*(?:JSON FORMATTING|IMPORTANT:|$))",
                content,
                re.DOTALL,
            )
            if json_match:
                return json_match.group(0)
        except AttributeError:
            pass

        # Strategy 3: Find the first complete JSON object (most aggressive)
        try:
            # Look for { followed by content until we find a complete JSON structure
            start_pos = content.find("{")
            if start_pos != -1:
                brace_count = 0
                end_pos = start_pos
                for i, char in enumerate(content[start_pos:], start_pos):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break

                if brace_count == 0:  # Found complete JSON
                    json_str = content[start_pos:end_pos]
                    # Check if it looks like valid JSON (has quotes and colons)
                    if '"' in json_str and ":" in json_str:
                        return json_str
        except (AttributeError, IndexError):
            pass

        # Strategy 3.5: Try to find JSON that might be followed by extra text
        try:
            # Look for JSON object that might have extra content after it
            json_match = re.search(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
                # Validate it looks like JSON
                if (
                    '"' in json_str
                    and ":" in json_str
                    and json_str.count("{") == json_str.count("}")
                ):
                    return json_str
        except AttributeError:
            pass

        # Strategy 4: Fallback - find any JSON object
        try:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json_match.group(0)
        except AttributeError:
            pass

        return None

    def fix_json_issues(self, content: str) -> str:
        """Fix common JSON formatting issues."""
        import re

        # Remove any text before the first {
        content = re.sub(r"^[^{]*", "", content)

        # Remove any text after the last }
        content = re.sub(r"}[^}]*$", "}", content)

        # Fix unquoted keys
        content = re.sub(r"(\w+):", r'"\1":', content)

        # Fix single quotes to double quotes
        content = content.replace("'", '"')

        # Remove trailing commas
        content = re.sub(r",\s*}", "}", content)
        content = re.sub(r",\s*]", "]", content)

        return content

    def complete_incomplete_json(self, content: str) -> str:
        """Complete incomplete JSON responses (like Gemini Pro truncation)."""
        import re

        # Find the JSON object
        json_match = re.search(r"\{.*", content, re.DOTALL)
        if not json_match:
            return None

        json_str = json_match.group(0)

        # Count braces to see if it's incomplete
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")

        if open_braces > close_braces:
            # Add missing closing braces
            missing_braces = open_braces - close_braces
            json_str += "}" * missing_braces

        # Check if we have incomplete coin entries
        # Look for the last complete coin entry
        lines = json_str.split("\n")
        completed_lines = []

        for line in lines:
            if line.strip():
                # Check if this line looks like an incomplete coin entry
                if re.match(r'\s*"[A-Z]+":\s*\{', line):
                    # This is a coin entry, check if it's complete
                    if not line.strip().endswith("}"):
                        # Incomplete coin entry, complete it
                        if '"signal"' in line:
                            # Has signal, complete the rest
                            completed_lines.append(line)
                            completed_lines.append('    "side": null,')
                            completed_lines.append('    "quantity": 0.0,')
                            completed_lines.append('    "confidence": 0.5,')
                            completed_lines.append(
                                '    "justification": "Incomplete response, defaulting to hold"'
                            )
                            completed_lines.append("  }")
                        else:
                            # Very incomplete, skip this coin
                            continue
                    else:
                        completed_lines.append(line)
                else:
                    completed_lines.append(line)

        # Reconstruct the JSON
        completed_json = "\n".join(completed_lines)

        # Ensure proper closing
        if not completed_json.strip().endswith("}"):
            completed_json += "}"

        return completed_json

    def create_default_decisions(self) -> Dict[str, Any]:
        """Create default hold decisions for all coins when parsing fails."""
        default_decisions = {}
        for coin in config.SYMBOL_TO_COIN.values():
            default_decisions[coin] = {
                "signal": "hold",
                "side": None,
                "quantity": 0.0,
                "confidence": 0.0,
                "justification": "JSON parsing failed, defaulting to hold",
            }
        return default_decisions

    def log_llm_decision(self, model_key: str, decisions: List[LLMDecision]):
        """Log decisions from a specific LLM."""
        for decision in decisions:
            utils.log_ai_decision(
                {
                    "timestamp": decision.timestamp,
                    "model": model_key,
                    "coin": decision.coin,
                    "signal": decision.signal,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                }
            )

    def execute_trade(
        self, model_key: str, coin: str, decision: LLMDecision, current_price: float
    ):
        """Execute a trade for a specific LLM."""
        llm_state = self.llm_states[model_key]

        if decision.signal == "entry":
            self.open_position(model_key, coin, decision, current_price)
        elif decision.signal == "close" and coin in llm_state["positions"]:
            self.close_position(model_key, coin, decision, current_price)

    def open_position(
        self, model_key: str, coin: str, decision: LLMDecision, current_price: float
    ):
        """Open a new position for an LLM."""
        llm_state = self.llm_states[model_key]

        # Use quantity directly from LLM decision
        quantity = decision.quantity
        position_value = quantity * current_price

        # Check if position value exceeds 10% of balance
        max_position_value = llm_state["balance"] * 0.1
        if position_value > max_position_value:
            # Scale down quantity to respect 10% limit
            quantity = max_position_value / current_price
            position_value = quantity * current_price
            logging.info(
                f"{model_key} scaled down position to respect 10% limit: {quantity:.4f} {coin}"
            )

        logging.info(
            f"{model_key} opening position: {quantity:.4f} {coin} @ ${current_price:.2f} = ${position_value:.2f}"
        )

        if quantity > 0:
            position = {
                "side": decision.side,
                "quantity": quantity,
                "entry_price": current_price,
                "leverage": decision.leverage,
                "stop_loss": decision.stop_loss,
                "profit_target": decision.profit_target,
                "timestamp": decision.timestamp,
            }

            llm_state["positions"][coin] = position
            # Reduce balance by the full position value (not leveraged)
            llm_state["balance"] -= position_value
            llm_state["total_trades"] += 1

            logging.info(
                f"{model_key} opened {decision.side} position in {coin}: {quantity:.4f} @ ${current_price:.2f} (leverage: {decision.leverage}x)"
            )

    def close_position(
        self, model_key: str, coin: str, decision: LLMDecision, current_price: float
    ):
        """Close an existing position for an LLM."""
        llm_state = self.llm_states[model_key]

        if coin in llm_state["positions"]:
            position = llm_state["positions"][coin]
            pnl = prompts.calculate_position_pnl(position, current_price)

            # Add back the current value of the position (includes PnL)
            current_value = position["quantity"] * current_price
            llm_state["balance"] += current_value

            if pnl > 0:
                llm_state["winning_trades"] += 1

            logging.info(
                f"{model_key} closed position in {coin}: P&L = ${pnl:.2f} (leverage: {position.get('leverage', 1)}x)"
            )
            del llm_state["positions"][coin]

    def get_llm_performance(self, model_key: str) -> Dict[str, Any]:
        """Get performance metrics for a specific LLM."""
        llm_state = self.llm_states[model_key]

        win_rate = (
            (llm_state["winning_trades"] / llm_state["total_trades"]) * 100
            if llm_state["total_trades"] > 0
            else 0
        )

        return {
            "model_name": model_key,
            "balance": llm_state["balance"],
            "total_trades": llm_state["total_trades"],
            "winning_trades": llm_state["winning_trades"],
            "win_rate": win_rate,
            "total_pnl": llm_state["total_pnl"],
            "positions": len(llm_state["positions"]),
            "start_time": llm_state["start_time"],
        }

    def get_all_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all LLMs."""
        performance = {}
        for model_key in config.LLM_MODELS.keys():
            performance[model_key] = self.get_llm_performance(model_key)
        return performance
