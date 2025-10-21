# DeepSeek Paper Trading Bot

This repository contains a paper-trading bot that runs against the Binance REST API while leveraging DeepSeek for trade decision-making. The app persists its runtime data (portfolio state, AI messages, and trade history) inside a dedicated `data/` directory so it can be mounted as a volume when running in Docker.

## Prerequisites

- Docker 24+ (any engine capable of building Linux/AMD64 images)
- A `.env` file with the required API credentials:
  - `BN_API_KEY` / `BN_SECRET` for Binance access
  - `OPENROUTER_API_KEY` for DeepSeek requests

## Build the Image

```bash
docker build -t tradebot .
```

## Prepare Local Data Storage

Create a directory on the host that will receive the bot's CSV/JSON artifacts:

```bash
mkdir -p ./data
```

The container stores everything under `/app/data`. Mounting your host folder to that path keeps trade history and AI logs between runs.

## Run the Bot in Docker

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  tradebot
```

- `--env-file .env` injects API keys into the container.
- The volume mount keeps `portfolio_state.csv`, `portfolio_state.json`, `ai_messages.csv`, `ai_decisions.csv`, and `trade_history.csv` outside the container so you can inspect them locally.
- By default the app writes to `/app/data`. To override, set `TRADEBOT_DATA_DIR` and update the volume mount accordingly.

## Optional: Streamlit Dashboard

To launch the monitoring dashboard instead of the trading bot, run:

```bash
docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  -p 8501:8501 \
  tradebot \
  streamlit run dashboard.py
```

Then open <http://localhost:8501> to access the UI.

## Development Notes

- The Docker image sets `PYTHONDONTWRITEBYTECODE=1` and `PYTHONUNBUFFERED=1` for cleaner logging.
- When running locally without Docker, the bot still writes to the `data/` directory next to the source tree (or to `TRADEBOT_DATA_DIR` if set).
- Existing files inside `data/` are never overwritten automatically; if headers or columns change, migrate the files manually.
- The repository already includes sample CSV files in `data/` so you can explore the dashboard immediately. These files will be overwritten as the bot runs.
