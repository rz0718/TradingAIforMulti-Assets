# 模块化大模型量化交易机器人

这是一个基于大语言模型（LLM）进行决策的加密货币量化交易机器人（纸上交易）。

项目已经过重构，将原有的单一脚本拆分为多个独立的、功能明确的模块，以提高代码的可读性、可维护性和可扩展性。

## 项目结构

重构后的项目遵循模块化的结构：

```
.
├── bot/                  # 核心机器人逻辑模块
│   ├── __init__.py       # 将 bot 目录标记为 Python 包
│   ├── config.py         # 配置加载与管理
│   ├── clients.py        # API 客户端初始化 (币安, LLM)
│   ├── data_processing.py  # 市场数据获取与技术指标计算
│   ├── prompts.py        # LLM 的 Prompt 构建
│   ├── trading_workflow.py # 核心交易工作流与状态管理
│   └── utils.py          # 通用工具函数 (如日志, CSV 操作)
├── main.py               # 项目主入口点
├── dashboard.py          # 可视化监控仪表盘
├── pyproject.toml        # 项目配置文件 (用于配置 uv 镜像源)
├── requirements.txt      # Python 依赖项
├── .env                  # 环境变量文件 (本地配置)
├── .env.example          # 环境变量示例文件
└── data/                 # 存放运行时数据 (CSV 日志等)
```

---

## 各脚本功能介绍

#### `main.py`
项目的唯一入口文件。它的职责非常简单：导入并调用 `trading_workflow` 中的主循环函数，启动机器人。

#### `dashboard.py`
一个使用 Streamlit 构建的交互式网页仪表盘，用于可视化监控机器人的实时表现和历史数据。详情请见下文“可视化仪表盘”部分。

#### `pyproject.toml`
项目遵循 PEP 621 标准的配置文件。当前主要用于为 `uv` 等现代包管理工具配置 PyPI 镜像源，以加速和稳定依赖安装过程。

#### `bot/config.py`
负责管理项目的所有配置。它会从 `.env` 文件中加载 API 密钥等敏感信息，并定义了如交易对、技术指标参数等静态配置。所有其他模块需要配置信息时，都应从此文件导入。

#### `bot/clients.py`
管理与外部服务的所有 API 连接。它包含：
- **币安客户端 (`get_binance_client`)**: 初始化并维护一个币安客户端单例，用于获取市场数据。
- **LLM 客户端 (`get_llm_client`)**: 初始化一个与 OpenAI API 兼容的客户端。这个客户端是**可适配的**，您可以通过在 `.env` 文件中设置 `LLM_API_KEY`, `LLM_BASE_URL` 和 `LLM_MODEL_NAME` 来轻松切换不同的模型服务商（如官方 OpenAI, Azure OpenAI, 或任何其他兼容的代理服务）。

#### `bot/data_processing.py`
所有数据处理和分析的逻辑都集中在此。它负责从币安获取 K 线数据，并计算各种技术指标（如 EMA, RSI, MACD, ATR 等），为决策提供数据支持。

#### `bot/prompts.py`
专门用于构建发送给大模型的 Prompt。它将账户状态（余额、仓位）、详细的市场数据和技术指标组合成一个结构化的、信息丰富的 Prompt，引导 LLM 做出交易决策。

#### `bot/trading_workflow.py`
这是机器人的核心。它包含：
- **状态管理 (`TradingState` 类)**: 跟踪机器人的所有动态状态，如现金余额、当前持仓、历史净值等。
- **主交易循环 (`run_trading_loop`)**: 一个无限循环，在每个时间周期内按顺序执行“获取数据 -> 生成决策 -> 执行交易”的完整流程。
- **交易执行**: 包含执行买入、卖出、止盈、止损等具体交易逻辑的函数（当前为占位符，需要进一步实现）。

#### `bot/utils.py`
存放项目范围内的通用工具函数，以保持其他模块的整洁。目前包含：
- 日志系统初始化。
- CSV 文件的创建和写入操作（用于记录交易、净值和 AI 决策）。
- Telegram 通知功能。

---

## 安装与使用

### 1. 环境管理 (使用 uv)

本项目推荐使用 `uv` 进行包管理，它是一个极速的 Python 包安装器和虚拟环境管理器。

**a. 安装 uv** (如果您的系统中尚未安装)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**b. 创建并激活虚拟环境**

在项目根目录下运行：

```bash
# 1. 创建虚拟环境
uv venv

# 2. 激活虚拟环境
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. 安装依赖

在激活虚拟环境后，使用 `uv` 安装 `requirements.txt` 中列出的所有依赖。项目已配置 `pyproject.toml` 文件，`uv` 会自动使用国内镜像源来加速下载。

```bash
uv pip install -r requirements.txt
```

### 3. 环境配置

项目通过 `.env` 文件管理敏感信息和环境配置。

1.  将 `.env.example` 文件复制一份，并重命名为 `.env`。
2.  打开 `.env` 文件并填入您的个人信息。

**必填项:**

- `BN_API_KEY`: 您的币安 API Key。
- `BN_SECRET`: 您的币安 Secret Key。
- `LLM_API_KEY`: 您使用的大模型服务的 API Key (例如 OpenAI 的 `sk-...`)。

**可选项:**

- `LLM_BASE_URL`: 如果您使用代理或非官方的 OpenAI 兼容服务，请在此处填写其基础 URL。
- `LLM_MODEL_NAME`: 指定要使用的模型名称，默认为 `gpt-4o`。
- `TELEGRAM_BOT_TOKEN` 和 `TELEGRAM_CHAT_ID`: 如果您想接收 Telegram 通知，请填写。

### 4. 运行机器人

完成配置后，只需运行 `main.py` 文件即可启动机器人：

```bash
python main.py
```

机器人启动后，将开始按设定的时间间隔（默认为3分钟）执行交易循环，并在控制台打印详细信息。

---

## 可视化仪表盘 (Dashboard)

`dashboard.py` 提供了一个交互式网页，用于可视化监控机器人的表现。

### 主要功能

- **实时监控**: 展示总资产、回报率、持仓状态、浮动盈亏等关键指标。
- **性能分析**: 绘制资产净值曲线，并与 BTC 买入持有策略进行对比。
- **历史追溯**: 以表格形式清晰地展示每一笔历史交易和 AI 的决策记录。

### 如何运行

确保您的虚拟环境已激活，然后在项目根目录下运行以下命令：

```bash
streamlit run dashboard.py
```

运行后，它会在您的浏览器中自动打开一个本地网页，展示仪表盘内容。