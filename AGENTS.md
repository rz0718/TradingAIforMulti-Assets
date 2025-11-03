# Repository Guidelines

## Project Structure & Module Organization
The core trading loop lives in `llm_bot.py`, which streams Alpaca market data, calls DeepSeek via OpenRouter, and writes CSV snapshots under `data/<bot_id>/`. Aggregated multi-bot orchestration is handled by `multi_bot_manager.py` and the CLI helper `run_multiple_bots.py`. Streamlit dashboards sit in `dashboard.py`, while indicators and market adapters live in `indicators.py`, `market_alpaca.py`, and `market_stock.py`. The `bot/` package hosts experimental multi-agent workflows and the `11.json` sample config. Prompts and constants reside in `prompt.py`, `parameter.py`, and `config_stock.py`; keep secrets in `.env`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create an isolated environment.
- `pip install -r requirements.txt` — install Binance, Streamlit, and helper libraries.
- `python llm_bot.py` — run a single DeepSeek-driven bot; ensure `.env` holds API keys.
- `python run_multiple_bots.py` — launch parallel agents defined in `multi_bot_manager.py`.
- `streamlit run dashboard.py` — view performance analytics at `http://localhost:8501`.
- `docker build -t tradebot .` and `docker run --env-file .env -v "$(pwd)/data:/app/data" tradebot` — containerized execution with persisted logs.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case identifiers, and UPPER_CASE constants in config modules. Prefer explicit imports, type hints, and structured `logging` calls; reserve `print` for CLI entry points.

## Testing Guidelines
Automated tests are not yet wired in; rely on targeted dry-runs. Before changes land, execute `python llm_bot.py` against a paper environment and confirm fresh entries in `data/<bot_id>/portfolio_state.csv`. For multi-agent features, run `python run_multiple_bots.py` until one iteration completes and review `bot_performance_history.json`. When modifying indicators, validate calculations with ad-hoc scripts under `examples/` and capture expected outputs in the PR description.

## Commit & Pull Request Guidelines
Git history favors short, present-tense summaries (`add yfinance`, `modify prompt`). Keep messages under ~60 characters, optionally adding a body for context. Pull requests should link issues, describe configuration changes, and attach logs or dashboard screenshots showing a successful iteration. Flag CSV schema changes so downstream tools remain in sync.

## Configuration & Security Tips
Never commit `.env` or API keys. Use `TRADEBOT_DATA_DIR` to redirect data exports when running multiple bots locally. When updating Telegram alerts, verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` with a disposable chat. Rotate OpenRouter keys after demos and scrub logs before sharing externally.
