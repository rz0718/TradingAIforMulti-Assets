#!/usr/bin/env python3
"""
Example script for running multiple LLM trading bots in parallel
"""
import time
import logging
from multi_bot_manager import MultiBotManager

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

def main():
    """Run multiple bots and compare their performance."""
    
    # Initialize the manager
    manager = MultiBotManager()
    
    # Add multiple bots with different models
    # You can use any models supported by OpenRouter
    print("Adding bots...")
    bot_ids = []
    
    # Example: Add multiple instances of the same model with different IDs
    bot_ids.append(manager.add_bot("deepseek/deepseek-chat-v3.1", "deepseek-1"))
    bot_ids.append(manager.add_bot("qwen/qwen3-max", "qwen-1"))
    
    # Or add different models to compare
    # bot_ids.append(manager.add_bot("anthropic/claude-3.5-sonnet", "claude-1"))
    # bot_ids.append(manager.add_bot("openai/gpt-4", "gpt4-1"))
    # bot_ids.append(manager.add_bot("meta-llama/llama-3.1-70b-instruct", "llama-1"))
    
    print(f"\n✓ Added {len(bot_ids)} bots:")
    for bot_id in bot_ids:
        bot = manager.bots[bot_id]
        print(f"  - {bot_id}: {bot.model}")
    
    print("\nStarting all bots in parallel...")
    manager.start_all_bots()
    
    print("\nBots are now running. They will trade independently.")
    print("Press Ctrl+C to stop all bots and see final comparison.\n")
    
    try:
        # Update performance and print comparison every 5 minutes
        update_interval = 180  # 5 minutes
        last_update = time.time()
        
        while True:
            time.sleep(10)  # Check every 10 seconds
            
            # Update performance snapshot periodically
            if time.time() - last_update >= update_interval:
                manager.update_performance_snapshot()
                manager.print_comparison()
                last_update = time.time()
            
    except KeyboardInterrupt:
        print("\n\nStopping all bots...")
        manager.stop_all_bots()
        
        print("\nGenerating final performance comparison...")
        manager.update_performance_snapshot()
        manager.print_comparison()
        
        print("\nPerformance history saved to:", manager.performance_history_file)
        print("\nDone!")


if __name__ == "__main__":
    main()

