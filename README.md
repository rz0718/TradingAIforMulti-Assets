# Modular LLM-based Quantitative Trading Bot

This is a cryptocurrency quantitative trading bot (paper trading) that uses Large Language Models (LLM) for decision-making.

The project has been refactored, breaking down the original single script into multiple independent, functionally clear modules to improve code readability, maintainability, and extensibility.

## Project Structure

The refactored project follows a modular structure:

```
.
├── bot/                  # Core bot logic modules
│   ├── __init__.py       # Marks bot directory as Python package
│   ├── config.py         # Configuration loading and management
│   ├── clients.py        # API client initialization (Binance, LLM)
│   ├── data_processing.py  # Market data fetching and technical indicator calculations
│   ├── prompts.py        # LLM prompt construction
│   ├── trading_workflow.py # Core trading workflow and state management
│   └── utils.py          # Common utility functions (logging, CSV operations)
├── main.py               # Project main entry point
├── dashboard.py          # Visualization monitoring dashboard
├── pyproject.toml        # Project configuration file (for configuring uv mirror sources)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables file (local configuration)
├── .env.example          # Environment variables example file
└── data/                 # Runtime data storage (CSV logs, etc.)
```

---

## Script Functionality Overview

#### `main.py`
The project's single entry point file. Its responsibility is very simple: import and call the main loop function from `trading_workflow` to start the bot.

#### `dashboard.py`
An interactive web dashboard built with Streamlit for visualizing the bot's real-time performance and historical data. See the "Visualization Dashboard" section below for details.

#### `pyproject.toml`
A configuration file following PEP 621 standards. Currently mainly used to configure PyPI mirror sources for modern package management tools like `uv` to speed up and stabilize the dependency installation process.

#### `bot/config.py`
Manages all project configurations. It loads sensitive information like API keys from the `.env` file and defines static configurations such as trading pairs and technical indicator parameters. All other modules should import configuration information from this file.

#### `bot/clients.py`
Manages all API connections to external services. It includes:
- **Binance Client (`get_binance_client`)**: Initializes and maintains a Binance client singleton for fetching market data.
- **LLM Client (`get_llm_client`)**: Initializes an OpenAI API-compatible client. This client is **adaptable** - you can easily switch between different model service providers (such as official OpenAI, Azure OpenAI, or any other compatible proxy service) by setting `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL_NAME` in the `.env` file.

#### `bot/data_processing.py`
All data processing and analysis logic is centralized here. It's responsible for fetching K-line data from Binance and calculating various technical indicators (such as EMA, RSI, MACD, ATR, etc.) to provide data support for decision-making.

#### `bot/prompts.py`
Specifically designed for constructing prompts sent to the large language model. It combines account status (balance, positions), detailed market data, and technical indicators into a structured, information-rich prompt that guides the LLM to make trading decisions.

#### `bot/trading_workflow.py`
This is the core of the bot. It includes:
- **State Management (`TradingState` class)**: Tracks all dynamic states of the bot, such as cash balance, current positions, historical net value, etc.
- **Main Trading Loop (`run_trading_loop`)**: An infinite loop that executes the complete "fetch data -> generate decision -> execute trade" process in each time cycle.
- **Trade Execution**: Contains functions for executing specific trading logic like buy, sell, take profit, stop loss, etc. (currently placeholders that need further implementation).

#### `bot/utils.py`
Stores project-wide common utility functions to keep other modules clean. Currently includes:
- Logging system initialization.
- CSV file creation and write operations (for recording trades, net value, and AI decisions).
- Telegram notification functionality.

---

## Installation and Usage

### 1. Environment Management (using uv)

This project recommends using `uv` for package management, which is an extremely fast Python package installer and virtual environment manager.

**a. Install uv** (if not already installed on your system)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**b. Create and activate virtual environment**

Run in the project root directory:

```bash
# 1. Create virtual environment
uv venv

# 2. Activate virtual environment
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install Dependencies

After activating the virtual environment, use `uv` to install all dependencies listed in `requirements.txt`. The project has configured the `pyproject.toml` file, and `uv` will automatically use domestic mirror sources to speed up downloads.

```bash
uv pip install -r requirements.txt
```

### 3. Environment Configuration

The project manages sensitive information and environment configuration through the `.env` file.

1. Copy the `.env.example` file and rename it to `.env`.
2. Open the `.env` file and fill in your personal information.

**Required fields:**

- `BN_API_KEY`: Your Binance API Key.
- `BN_SECRET`: Your Binance Secret Key.
- `LLM_API_KEY`: The API Key for the large language model service you're using (e.g., OpenAI's `sk-...`).

**Optional fields:**

- `LLM_BASE_URL`: If you're using a proxy or non-official OpenAI-compatible service, please fill in its base URL here.
- `LLM_MODEL_NAME`: Specify the model name to use, defaults to `gpt-4o`.
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`: If you want to receive Telegram notifications, please fill these in.

### 4. Run the Bot

After completing the configuration, simply run the `main.py` file to start the bot:

```bash
python main.py
```

After the bot starts, it will begin executing trading loops at the set time interval (default 3 minutes) and print detailed information to the console.

---

## Visualization Dashboard

`dashboard.py` provides an interactive web page for visualizing and monitoring the bot's performance.

### Main Features

- **Real-time Monitoring**: Displays key metrics like total assets, return rate, position status, floating P&L, etc.
- **Performance Analysis**: Plots asset net value curves and compares them with BTC buy-and-hold strategy.
- **Historical Tracking**: Clearly displays each historical trade and AI decision record in table format.

### How to Run

Ensure your virtual environment is activated, then run the following command in the project root directory:

```bash
streamlit run dashboard.py
```

After running, it will automatically open a local web page in your browser displaying the dashboard content.