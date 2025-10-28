#!/usr/bin/env python3
"""
Main entry point for the multi-LLM trading bot.
"""

import asyncio
from dotenv import load_dotenv
from bot.multi_trading_workflow import MultiTradingWorkflow

# Load environment variables from .env file
load_dotenv()


async def main():
    """Main entry point for multi-LLM trading system."""
    workflow = MultiTradingWorkflow()
    await workflow.run_trading_loop()


if __name__ == "__main__":
    asyncio.run(main())
