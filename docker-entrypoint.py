#!/usr/bin/env python3
"""
Docker entrypoint script to run both trading bot and dashboard.
"""
import os
import sys
import signal
import subprocess
import time
from pathlib import Path

# Store process references
processes = []

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Received shutdown signal. Stopping all services...", flush=True)
    for proc in processes:
        if proc.poll() is None:  # Process is still running
            print(f"Stopping process {proc.pid}...", flush=True)
            proc.terminate()
    
    # Wait for processes to terminate
    time.sleep(2)
    
    # Force kill if still running
    for proc in processes:
        if proc.poll() is None:
            print(f"Force killing process {proc.pid}...", flush=True)
            proc.kill()
    
    print("✅ All services stopped", flush=True)
    sys.exit(0)

def main():
    """Start both trading bot and dashboard."""
    print("🚀 Starting AI Trading Bot System...", flush=True)
    print("=" * 50, flush=True)
    print("", flush=True)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start trading bot
    print("📈 Starting Trading Bot...", flush=True)
    bot_process = subprocess.Popen(
        [sys.executable, "-u", "main.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd="/app"
    )
    processes.append(bot_process)
    print(f"✅ Trading Bot started (PID: {bot_process.pid})", flush=True)
    print("", flush=True)
    
    # Wait a moment for bot to initialize
    time.sleep(2)
    
    # Start Streamlit dashboard
    print("📊 Starting Dashboard...", flush=True)
    dashboard_process = subprocess.Popen(
        [
            "streamlit", "run", "dashboard.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false"
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd="/app/front_end"
    )
    processes.append(dashboard_process)
    print(f"✅ Dashboard started (PID: {dashboard_process.pid})", flush=True)
    print("", flush=True)
    
    print("=" * 50, flush=True)
    print("✅ All services running!", flush=True)
    print("", flush=True)
    print("📊 Dashboard: http://localhost:8501", flush=True)
    print("📈 Trading Bot: Active", flush=True)
    print("=" * 50, flush=True)
    print("", flush=True)
    
    # Monitor processes
    try:
        while True:
            # Check if any process has died
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    print(f"⚠️  Process {proc.pid} exited with code {proc.returncode}", flush=True)
                    print("Shutting down all services...", flush=True)
                    signal_handler(None, None)
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()

