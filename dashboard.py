#!/usr/bin/env python3
"""Streamlit dashboard for monitoring the DeepSeek trading bot."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from binance.client import Client
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DATA_DIR = Path(os.getenv("TRADEBOT_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = BASE_DIR / ".env"
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_SNAPSHOT_SECONDS = 180.0
BOT_DATA_FILENAMES = (
    "portfolio_state.csv",
    "trade_history.csv",
    "ai_decisions.csv",
    "ai_messages.csv",
)
PORTFOLIO_STATE_COLUMNS = [
    "total_balance",
    "total_equity",
    "total_return_pct",
    "num_positions",
    "position_details",
    "total_margin",
    "net_unrealized_pnl",
]

from config_stock import SYMBOL_TO_COIN
COIN_TO_SYMBOL = SYMBOL_TO_COIN
# COIN_TO_SYMBOL: Dict[str, str] = {
#     "ETH": "ETHUSDT",
#     "SOL": "SOLUSDT",
#     "XRP": "XRPUSDT",
#     "BTC": "BTCUSDT",
#     "DOGE": "DOGEUSDT",
#     "BNB": "BNBUSDT",
# }

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

BN_API_KEY = os.getenv("BN_API_KEY", "")
BN_SECRET = os.getenv("BN_SECRET", "")


def resolve_risk_free_rate() -> float:
    """Return annualized risk-free rate configured for Sortino ratio."""
    env_value = os.getenv("SORTINO_RISK_FREE_RATE") or os.getenv("RISK_FREE_RATE")
    if env_value is None:
        return DEFAULT_RISK_FREE_RATE
    try:
        return float(env_value)
    except (TypeError, ValueError):
        logging.warning(
            "Invalid SORTINO_RISK_FREE_RATE/RISK_FREE_RATE value '%s'; using default %.4f",
            env_value,
            DEFAULT_RISK_FREE_RATE,
        )
        return DEFAULT_RISK_FREE_RATE


RISK_FREE_RATE = resolve_risk_free_rate()

BINANCE_CLIENT: Client | None = None
if BN_API_KEY and BN_SECRET:
    try:
        BINANCE_CLIENT = Client(BN_API_KEY, BN_SECRET, testnet=False)
    except Exception as exc:
        logging.warning("Unable to initialize Binance client: %s", exc)
else:
    logging.info("Binance credentials not provided; live prices disabled.")


def discover_bot_runs(data_root: Path) -> List[tuple[str, Path]]:
    """Return available bot run directories (or legacy root files)."""
    runs: List[tuple[str, Path]] = []
    try:
        entries = sorted(data_root.iterdir())
    except FileNotFoundError:
        return runs

    for entry in entries:
        if not entry.is_dir():
            continue
        if any((entry / name).exists() for name in BOT_DATA_FILENAMES):
            runs.append((entry.name, entry))

    if any((data_root / name).exists() for name in BOT_DATA_FILENAMES):
        runs.insert(0, ("[root] data", data_root))

    return runs


def load_state_snapshot(json_path: str) -> dict[str, Any] | None:
    """Load latest portfolio snapshot from JSON if available."""
    path = Path(json_path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Unable to load portfolio snapshot %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logging.warning("Portfolio snapshot %s is not a JSON object.", path)
        return None
    return payload


def build_position_details_string(positions: dict[str, Any]) -> str:
    """Serialize structured position data to compact legacy string format."""
    if not positions:
        return "No positions"
    entries: List[str] = []
    for coin in sorted(positions):
        details = positions.get(coin) or {}
        side = details.get("side", "")
        quantity = details.get("quantity", 0.0)
        entry_price = details.get("entry_price", 0.0)
        try:
            qty_val = float(quantity)
        except (TypeError, ValueError):
            qty_val = 0.0
        try:
            price_val = float(entry_price)
        except (TypeError, ValueError):
            price_val = 0.0
        entries.append(f"{coin}:{side}:{qty_val:.4f}@{price_val:.4f}")
    return "; ".join(entries)


def build_snapshot_positions_df(positions: dict[str, Any]) -> pd.DataFrame:
    """Convert snapshot position dictionary into a DataFrame."""
    if not positions:
        return pd.DataFrame()
    rows: List[dict[str, Any]] = []
    for coin in sorted(positions):
        details = positions.get(coin) or {}
        row: dict[str, Any] = {"coin": coin}
        for field in (
            "side",
            "quantity",
            "entry_price",
            "profit_target",
            "stop_loss",
            "leverage",
            "confidence",
            "margin",
            "entry_justification",
            "last_justification",
        ):
            value = details.get(field)
            if field in {"quantity", "entry_price", "profit_target", "stop_loss", "leverage", "confidence", "margin"}:
                try:
                    row[field] = float(value) if value is not None else np.nan
                except (TypeError, ValueError):
                    row[field] = np.nan
            else:
                row[field] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def snapshot_to_series(snapshot: dict[str, Any]) -> tuple[pd.Series | None, pd.DataFrame]:
    """Convert snapshot payload into a portfolio state series and positions frame."""
    updated_at = pd.to_datetime(snapshot.get("updated_at"), errors="coerce")
    if pd.isna(updated_at):
        return None, pd.DataFrame()

    balance_raw = snapshot.get("balance")
    try:
        balance = float(balance_raw)
    except (TypeError, ValueError):
        return None, pd.DataFrame()

    positions = snapshot.get("positions") or {}
    positions_df = build_snapshot_positions_df(positions)
    total_margin = 0.0
    if not positions_df.empty and "margin" in positions_df.columns:
        total_margin = (
            pd.to_numeric(positions_df["margin"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
    net_unrealized_raw = snapshot.get("net_unrealized_pnl", 0.0)
    try:
        net_unrealized = float(net_unrealized_raw)
    except (TypeError, ValueError):
        net_unrealized = 0.0

    total_equity = balance + total_margin + net_unrealized
    position_details = build_position_details_string(positions)

    series = pd.Series(
        {
            "total_balance": balance,
            "total_equity": total_equity,
            "total_return_pct": np.nan,
            "num_positions": len(positions),
            "position_details": position_details,
            "total_margin": total_margin,
            "net_unrealized_pnl": net_unrealized,
        },
        name=updated_at,
    )
    return series, positions_df


def prepare_portfolio_state(
    state_df: pd.DataFrame, snapshot: dict[str, Any] | None
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """Merge historical CSV state with latest JSON snapshot for display."""
    if state_df.empty:
        timeline_df = pd.DataFrame(columns=PORTFOLIO_STATE_COLUMNS)
        timeline_df.index = pd.DatetimeIndex([], name="timestamp")
    else:
        timeline_df = state_df.copy()

    snapshot_positions = pd.DataFrame()
    if snapshot:
        snapshot_series, snapshot_positions = snapshot_to_series(snapshot)
        if snapshot_series is not None:
            timestamp = snapshot_series.name
            if timeline_df.empty or timestamp not in timeline_df.index:
                timeline_df = pd.concat([timeline_df, snapshot_series.to_frame().T])
            else:
                timeline_df.loc[timestamp] = snapshot_series

    if not timeline_df.empty:
        timeline_df.sort_index(inplace=True)
        equity_series = pd.to_numeric(
            timeline_df.get("total_equity", pd.Series(dtype="float64")), errors="coerce"
        )
        equity_series = equity_series.dropna()
        if not equity_series.empty:
            initial_equity = float(equity_series.iloc[0])
            if np.isfinite(initial_equity) and initial_equity != 0.0:
                timeline_df["total_return_pct"] = (
                    pd.to_numeric(
                        timeline_df["total_equity"], errors="coerce"
                    ).fillna(0.0)
                    / initial_equity
                    - 1.0
                ) * 100.0

    latest_series: pd.Series | None = None
    if not timeline_df.empty:
        latest_series = timeline_df.iloc[-1]

    positions_df = snapshot_positions
    if (positions_df is None or positions_df.empty) and latest_series is not None:
        positions_df = parse_positions(latest_series.get("position_details", ""))

    if positions_df is None:
        positions_df = pd.DataFrame()

    return timeline_df, latest_series, positions_df


def load_csv(path: Path, parse_dates: List[str] | None = None) -> pd.DataFrame:
    """Load a CSV into a DataFrame, returning empty frame when missing."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


@st.cache_data(ttl=15)
def get_portfolio_state(csv_path: str) -> pd.DataFrame:
    df = load_csv(Path(csv_path), parse_dates=["timestamp"])
    if df.empty:
        return df

    numeric_cols = [
        "total_balance",
        "total_equity",
        "total_return_pct",
        "num_positions",
        "total_margin",
        "net_unrealized_pnl",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    return df


@st.cache_data(ttl=15)
def get_trades(csv_path: str) -> pd.DataFrame:
    df = load_csv(Path(csv_path), parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    numeric_cols = [
        "quantity",
        "price",
        "profit_target",
        "stop_loss",
        "leverage",
        "confidence",
        "pnl",
        "balance_after",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=15)
def get_ai_decisions(csv_path: str) -> pd.DataFrame:
    df = load_csv(Path(csv_path), parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    return df


@st.cache_data(ttl=15)
def get_ai_messages(csv_path: str) -> pd.DataFrame:
    df = load_csv(Path(csv_path), parse_dates=["timestamp"])
    if df.empty:
        return df
    df.sort_values("timestamp", inplace=True, ascending=False)
    return df


def parse_positions(position_text: str | float) -> pd.DataFrame:
    """Split compact position text into structured rows."""
    if pd.isna(position_text) or not isinstance(position_text, str):
        return pd.DataFrame()
    if position_text.strip().lower() == "no positions":
        return pd.DataFrame()

    rows: List[Dict[str, str | float]] = []
    for chunk in position_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            symbol, side, rest = chunk.split(":")
            quantity, entry_price = rest.split("@")
            rows.append(
                {
                    "coin": symbol,
                    "side": side,
                    "quantity": float(quantity),
                    "entry_price": float(entry_price),
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


def fetch_current_prices(coins: List[str]) -> Dict[str, float | None]:
    """Fetch latest market prices for the provided coin tickers."""
    prices: Dict[str, float | None] = {coin: None for coin in coins}
    if not BINANCE_CLIENT:
        return prices

    for coin in coins:
        symbol = COIN_TO_SYMBOL.get(coin.upper(), f"{coin.upper()}USDT")
        try:
            ticker = BINANCE_CLIENT.get_symbol_ticker(symbol=symbol)
            prices[coin] = float(ticker["price"])
        except Exception as exc:
            logging.warning("Failed to fetch price for %s: %s", symbol, exc)
            prices[coin] = None
    return prices


def estimate_period_seconds(index: pd.Index, default: float = DEFAULT_SNAPSHOT_SECONDS) -> float:
    """Infer measurement cadence from a datetime-like index."""
    if index.size < 2:
        return default
    try:
        diffs = index.to_series().diff().dropna()
    except Exception:
        return default
    if diffs.empty:
        return default
    try:
        period_seconds = diffs.dt.total_seconds().median()
    except AttributeError:
        period_seconds = default
    if not period_seconds or not np.isfinite(period_seconds) or period_seconds <= 0:
        return default
    return float(period_seconds)


def compute_sharpe_ratio(trades_df: pd.DataFrame) -> float | None:
    """Compute annualized Sharpe ratio from realized (closed) trades."""
    if trades_df.empty or "action" not in trades_df.columns:
        return None

    actions = trades_df["action"].astype(str).str.upper()
    closes = trades_df.loc[actions == "CLOSE"].copy()
    if closes.empty or "balance_after" not in closes.columns:
        return None

    closes.sort_values("timestamp", inplace=True)
    closes = closes.set_index("timestamp")

    balances = pd.to_numeric(closes["balance_after"], errors="coerce").dropna()
    if balances.size < 2:
        return None

    returns = balances.pct_change().dropna()
    if returns.empty:
        return None

    std = returns.std()
    if std is None or np.isclose(std, 0.0):
        return None

    period_seconds = estimate_period_seconds(closes.index)

    periods_per_year = (365 * 24 * 60 * 60) / period_seconds
    sharpe = returns.mean() / std * np.sqrt(periods_per_year)
    return float(sharpe) if np.isfinite(sharpe) else None


def compute_sortino_ratio(state_df: pd.DataFrame, risk_free_rate: float) -> float | None:
    """Compute annualized Sortino ratio from total equity snapshots."""
    if state_df.empty or "total_equity" not in state_df.columns:
        return None

    equity = pd.to_numeric(state_df["total_equity"], errors="coerce").dropna()
    if equity.size < 2:
        return None

    returns = equity.pct_change().dropna()
    if returns.empty:
        return None

    period_seconds = estimate_period_seconds(equity.index)
    seconds_per_year = 365 * 24 * 60 * 60
    periods_per_year = seconds_per_year / period_seconds
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None

    per_period_rf = risk_free_rate / periods_per_year
    excess_return = returns.mean() - per_period_rf
    if not np.isfinite(excess_return):
        return None

    downside = np.minimum(returns - per_period_rf, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside)))
    if downside_deviation <= 0 or not np.isfinite(downside_deviation):
        return None

    sortino = (excess_return / downside_deviation) * np.sqrt(periods_per_year)
    return float(sortino) if np.isfinite(sortino) else None


def render_portfolio_tab(
    timeline_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    latest_state: pd.Series | None,
    positions_df: pd.DataFrame,
) -> None:
    if timeline_df.empty or latest_state is None:
        st.info("No portfolio data logged yet.")
        return

    latest = latest_state
    margin_allocated = latest.get("total_margin", 0.0)
    if pd.isna(margin_allocated):
        margin_allocated = 0.0
    margin_allocated = float(margin_allocated)
    unrealized_pnl = latest.get("net_unrealized_pnl", np.nan)
    if pd.isna(unrealized_pnl):
        unrealized_pnl = latest["total_equity"] - latest["total_balance"] - margin_allocated

    prev_unrealized = 0.0
    if len(timeline_df) > 1:
        prior = timeline_df.iloc[-2]
        prev_margin = prior.get("total_margin", 0.0)
        if pd.isna(prev_margin):
            prev_margin = 0.0
        prev_margin = float(prev_margin)
        prev_unrealized = prior.get("net_unrealized_pnl", np.nan)
        if pd.isna(prev_unrealized):
            prev_unrealized = prior["total_equity"] - prior["total_balance"] - prev_margin

    realized_pnl: float | None = None
    initial_equity_series = timeline_df["total_equity"].dropna()
    if not initial_equity_series.empty:
        initial_equity = float(initial_equity_series.iloc[0])
        realized_pnl = float(latest["total_equity"] - initial_equity)
        if np.isfinite(unrealized_pnl):
            realized_pnl -= float(unrealized_pnl)

    if realized_pnl is None or not np.isfinite(realized_pnl):
        realized_pnl = 0.0
        if not trades_df.empty and "action" in trades_df.columns and "pnl" in trades_df.columns:
            actions = trades_df["action"].fillna("").str.upper()
            realized_pnl = trades_df.loc[actions == "CLOSE", "pnl"].sum(skipna=True)
            if pd.isna(realized_pnl) or not np.isfinite(realized_pnl):
                realized_pnl = 0.0

    sharpe_ratio = compute_sharpe_ratio(trades_df)
    sortino_ratio = compute_sortino_ratio(timeline_df, RISK_FREE_RATE)

    col_a, col_b, col_c, col_d, col_e, col_f, col_g, col_h = st.columns(8)
    col_a.metric("Available Balance", f"${latest['total_balance']:.2f}")
    col_b.metric("Total Equity", f"${latest['total_equity']:.2f}")
    col_c.metric("Total Return %", f"{latest['total_return_pct']:.2f}%")
    col_d.metric("Margin Allocated", f"${margin_allocated:.2f}")
    col_e.metric(
        "Unrealized PnL",
        f"${unrealized_pnl:.2f}",
        delta=f"${unrealized_pnl - prev_unrealized:.2f}",
    )
    col_f.metric("Realized PnL", f"${realized_pnl:.2f}")
    col_g.metric(
        "Sharpe Ratio",
        f"{sharpe_ratio:.2f}" if sharpe_ratio is not None else "N/A",
    )
    col_h.metric(
        "Sortino Ratio",
                f"{sortino_ratio:.2f}" if sortino_ratio is not None else "N/A",
    )

    st.subheader("Equity Over Time")
    equity_values = pd.to_numeric(timeline_df["total_equity"], errors="coerce")
    equity_chart_df = (
        pd.DataFrame(
            {
                "timestamp": timeline_df.index,
                "Value": equity_values,
            }
        )
        .dropna(subset=["timestamp", "Value"])
        .sort_values("timestamp")
    )

    if equity_chart_df.empty:
        st.caption("Portfolio equity history will appear after the first snapshot.")
    else:
        lower = float(equity_chart_df["Value"].min())
        upper = float(equity_chart_df["Value"].max())
        span = upper - lower
        if span <= 0:
            span = max(abs(upper) * 0.02, 1.0)
        padding = span * 0.1
        lower_bound = lower - padding
        upper_bound = upper + padding

        equity_chart = (
            alt.Chart(equity_chart_df)
            .mark_line(interpolate="monotone", color="#f58518")
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y(
                    "Value:Q",
                    title="Equity ($)",
                    scale=alt.Scale(domain=[lower_bound, upper_bound]),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Timestamp"),
                    alt.Tooltip("Value:Q", title="Equity", format="$.2f"),
                ],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(equity_chart, use_container_width=True)  # type: ignore[arg-type]

    st.subheader("Open Positions")
    positions_display = positions_df.copy()
    if positions_display.empty:
        st.write("No open positions.")
    else:
        price_map = fetch_current_prices(positions_display["coin"].unique().tolist())
        positions_display["current_price"] = positions_display["coin"].map(price_map)

        def _row_unrealized(row: pd.Series) -> float | None:
            price = row.get("current_price")
            if price is None or pd.isna(price):
                return None
            diff = price - row["entry_price"]
            if str(row["side"]).lower() == "short":
                diff = row["entry_price"] - price
            return diff * row["quantity"]

        positions_display["unrealized_pnl"] = positions_display.apply(
            _row_unrealized, axis=1
        )  # type: ignore

        if positions_display["current_price"].isna().all():
            st.caption("Live price lookup unavailable; showing entry data only.")

        column_config: dict[str, Any] = {
            "quantity": st.column_config.NumberColumn(format="%.4f"),
            "entry_price": st.column_config.NumberColumn(format="$%.4f"),
            "current_price": st.column_config.NumberColumn(format="$%.4f"),
            "unrealized_pnl": st.column_config.NumberColumn(format="$%.2f"),
        }
        if "profit_target" in positions_display.columns:
            column_config["profit_target"] = st.column_config.NumberColumn(format="$%.4f")
        if "stop_loss" in positions_display.columns:
            column_config["stop_loss"] = st.column_config.NumberColumn(format="$%.4f")
        if "leverage" in positions_display.columns:
            column_config["leverage"] = st.column_config.NumberColumn(format="%.2f")
        if "confidence" in positions_display.columns:
            column_config["confidence"] = st.column_config.NumberColumn(format="%.2f")
        if "margin" in positions_display.columns:
            column_config["margin"] = st.column_config.NumberColumn(format="$%.2f")
        if "entry_justification" in positions_display.columns:
            column_config["entry_justification"] = st.column_config.TextColumn(
                "Entry justification", width="large"
            )
        if "last_justification" in positions_display.columns:
            column_config["last_justification"] = st.column_config.TextColumn(
                "Last justification", width="large"
            )

        st.dataframe(
            positions_display,
            column_config=column_config,
            use_container_width=True,
        )


def render_trades_tab(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        st.info("No trades recorded yet.")
        return

    st.dataframe(
        trades_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
            "quantity": st.column_config.NumberColumn(format="%.4f"),
            "price": st.column_config.NumberColumn(format="$%.4f"),
            "profit_target": st.column_config.NumberColumn(format="$%.4f"),
            "stop_loss": st.column_config.NumberColumn(format="$%.4f"),
            "pnl": st.column_config.NumberColumn(format="$%.2f"),
            "balance_after": st.column_config.NumberColumn(format="$%.2f"),
            "confidence": st.column_config.NumberColumn(format="%.2f"),
            "reason": st.column_config.TextColumn("Reason", width="large"),
        },
        use_container_width=True,
        height=420,
        hide_index=True,
    )


def render_ai_tab(decisions_df: pd.DataFrame, messages_df: pd.DataFrame) -> None:
    st.subheader("Latest Decisions per Symbol")
    if decisions_df.empty:
        st.info("No decisions yet.")
    else:
        latest_per_coin = (
            decisions_df.sort_values("timestamp")
            .drop_duplicates(subset=["coin"], keep="last")
            .sort_values("coin")
        )
        st.dataframe(
            latest_per_coin,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                "confidence": st.column_config.NumberColumn(format="%.2f"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Most recent LLM decision for each tracked symbol.")

    st.subheader("Decision History")
    if decisions_df.empty:
        st.caption("Decisions will appear once the bot logs them.")
    else:
        coin_options = ["All symbols"] + sorted(
            [coin for coin in decisions_df["coin"].dropna().unique()]
        )
        selected_coin = st.selectbox(
            "Filter by symbol",
            coin_options,
            index=0,
            key="decision_coin_filter",
        )
        if selected_coin == "All symbols":
            history_df = decisions_df.copy()
        else:
            history_df = decisions_df[decisions_df["coin"] == selected_coin].copy()
        st.dataframe(
            history_df,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                "confidence": st.column_config.NumberColumn(format="%.2f"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            },
            use_container_width=True,
            height=360,
            hide_index=True,
        )

    st.subheader("LLM Message Stream")
    if messages_df.empty:
        st.info("No messages logged yet.")
    else:
        direction_options = ["All directions"] + sorted(
            messages_df["direction"].dropna().astype(str).unique().tolist()
            if "direction" in messages_df.columns
            else []
        )
        role_options = ["All roles"] + sorted(
            messages_df["role"].dropna().astype(str).unique().tolist()
            if "role" in messages_df.columns
            else []
        )
        col_dir, col_role = st.columns(2)
        with col_dir:
            selected_direction = st.selectbox(
                "Direction",
                direction_options,
                index=0,
                key="ai_messages_direction_filter",
            )
        with col_role:
            selected_role = st.selectbox(
                "Role",
                role_options,
                index=0,
                key="ai_messages_role_filter",
            )

        filtered_messages = messages_df.copy()
        if selected_direction != "All directions" and "direction" in filtered_messages.columns:
            filtered_messages = filtered_messages[
                filtered_messages["direction"] == selected_direction
            ]
        if selected_role != "All roles" and "role" in filtered_messages.columns:
            filtered_messages = filtered_messages[
                filtered_messages["role"] == selected_role
            ]
        filtered_messages = filtered_messages.head(200)

        st.dataframe(
            filtered_messages,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                "direction": st.column_config.Column("Direction"),
                "role": st.column_config.Column("Role"),
                "metadata": st.column_config.Column("Metadata"),
                "content": st.column_config.TextColumn("Content", width="large"),
            },
            use_container_width=True,
            height=360,
            hide_index=True,
        )
        st.caption("Latest conversation snippets between orchestrator and LLM (max 200 rows).")


def main() -> None:
    st.set_page_config(page_title="LLM Bot Monitor", layout="wide")
    st.title("LLM Trading Bot Monitor")
    st.caption("Inspect portfolio state, trades, and LLM-generated reasoning for each bot run.")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    bot_runs = discover_bot_runs(DATA_DIR)
    if not bot_runs:
        st.warning(f"No bot data found under {DATA_DIR}.")
        return

    run_labels = [label for label, _ in bot_runs]
    selected_label = st.sidebar.selectbox("Bot / model run", run_labels)
    selected_path = dict(bot_runs)[selected_label]

    st.sidebar.caption(f"Data folder: {selected_path}")

    state_csv = selected_path / "portfolio_state.csv"
    trades_csv = selected_path / "trade_history.csv"
    decisions_csv = selected_path / "ai_decisions.csv"
    messages_csv = selected_path / "ai_messages.csv"

    state_df = get_portfolio_state(str(state_csv))
    trades_df = get_trades(str(trades_csv))
    decisions_df = get_ai_decisions(str(decisions_csv))
    messages_df = get_ai_messages(str(messages_csv))
    snapshot = load_state_snapshot(str(selected_path / "portfolio_state.json"))
    timeline_df, latest_state, positions_df = prepare_portfolio_state(state_df, snapshot)

    if latest_state is not None:
        positions_value = latest_state.get("num_positions", 0)
        try:
            positions = int(float(positions_value))
        except (TypeError, ValueError):
            positions = 0
        st.sidebar.metric("Open positions", positions)

        equity_raw = latest_state.get("total_equity", np.nan)
        try:
            equity_val = float(equity_raw)
        except (TypeError, ValueError):
            equity_val = np.nan
        if np.isfinite(equity_val):
            st.sidebar.metric("Total equity", f"${equity_val:,.2f}")

    if not decisions_df.empty and "timestamp" in decisions_df.columns:
        latest_decision_time = decisions_df.iloc[0]["timestamp"]
        if hasattr(latest_decision_time, "strftime"):
            rendered_ts = latest_decision_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            rendered_ts = str(latest_decision_time)
        st.sidebar.caption(f"Last decision: {rendered_ts}")

    st.markdown(f"**Selected run:** `{selected_label}`")

    portfolio_tab, trades_tab, ai_tab = st.tabs(["Portfolio", "Trades", "LLM Decisions"])

    with portfolio_tab:
        render_portfolio_tab(timeline_df, trades_df, latest_state, positions_df)

    with trades_tab:
        render_trades_tab(trades_df)

    with ai_tab:
        render_ai_tab(decisions_df, messages_df)


if __name__ == "__main__":
    main()
