from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize


TRADING_DAYS = 252
DEFAULT_TICKERS = (
    "GOLDBEES.NS, MON100.NS, EMBASSY.NS, ASTERDM.NS, HINDCOPPER.NS, "
    "DIXON.NS, POLYCAB.NS, LICI.NS, LT.NS, BAJFINANCE.NS, TCS.NS, "
    "RELIANCE.NS, HDFCBANK.NS, ITC.NS, HSCL.NS"
)


@dataclass(frozen=True)
class PortfolioResult:
    name: str
    target_daily_return: float
    daily_return: float
    daily_volatility: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    weights: pd.Series


def parse_tickers(raw: str) -> list[str]:
    tickers = []
    seen = set()
    for item in raw.replace("\n", ",").split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    return tickers


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.xs("Close", axis=1, level=1)
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(axis=1, how="all").sort_index()
    prices = prices.ffill().dropna(how="any")
    return prices


def portfolio_metrics(weights: np.ndarray, means: pd.Series, cov: pd.DataFrame, risk_free: float) -> dict[str, float]:
    daily_return = float(np.dot(weights, means.to_numpy()))
    daily_volatility = float(np.sqrt(weights.T @ cov.to_numpy() @ weights))
    annual_return = daily_return * TRADING_DAYS
    annual_volatility = daily_volatility * np.sqrt(TRADING_DAYS)
    sharpe = np.nan
    if annual_volatility > 0:
        sharpe = (annual_return - risk_free) / annual_volatility
    return {
        "daily_return": daily_return,
        "daily_volatility": daily_volatility,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
    }


def optimize_min_volatility(
    target_return: float,
    means: pd.Series,
    cov: pd.DataFrame,
    risk_free: float,
    allow_short: bool,
) -> PortfolioResult | None:
    n_assets = len(means)
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n_assets
    initial = np.repeat(1.0 / n_assets, n_assets)

    def objective(weights: np.ndarray) -> float:
        return float(weights.T @ cov.to_numpy() @ weights)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {"type": "eq", "fun": lambda weights: np.dot(weights, means.to_numpy()) - target_return},
    ]
    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return None

    weights = np.where(np.abs(result.x) < 1e-8, 0.0, result.x)
    metrics = portfolio_metrics(weights, means, cov, risk_free)
    return PortfolioResult(
        name=f"{target_return:.4%} daily target",
        target_daily_return=target_return,
        weights=pd.Series(weights, index=means.index),
        **metrics,
    )


def optimize_max_sharpe(
    means: pd.Series,
    cov: pd.DataFrame,
    risk_free: float,
    allow_short: bool,
) -> PortfolioResult | None:
    n_assets = len(means)
    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0)] * n_assets
    initial = np.repeat(1.0 / n_assets, n_assets)

    def objective(weights: np.ndarray) -> float:
        metrics = portfolio_metrics(weights, means, cov, risk_free)
        if np.isnan(metrics["sharpe"]):
            return 1e9
        return -metrics["sharpe"]

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return None

    weights = np.where(np.abs(result.x) < 1e-8, 0.0, result.x)
    metrics = portfolio_metrics(weights, means, cov, risk_free)
    return PortfolioResult(
        name="Max Sharpe",
        target_daily_return=metrics["daily_return"],
        weights=pd.Series(weights, index=means.index),
        **metrics,
    )


def build_frontier(
    means: pd.Series,
    cov: pd.DataFrame,
    risk_free: float,
    target_returns: Iterable[float],
    allow_short: bool,
) -> list[PortfolioResult]:
    portfolios = []
    for target in target_returns:
        result = optimize_min_volatility(target, means, cov, risk_free, allow_short)
        if result is not None:
            portfolios.append(result)
    return portfolios


def backtest_buy_and_hold(
    prices: pd.DataFrame,
    weights: pd.Series,
    initial_investment: float,
) -> pd.DataFrame:
    aligned_weights = weights.reindex(prices.columns).fillna(0.0)
    first_prices = prices.iloc[0]
    shares = (initial_investment * aligned_weights) / first_prices
    values = prices.mul(shares, axis=1).sum(axis=1)
    returns = values.pct_change().fillna(0.0)
    peak = values.cummax()
    drawdown = values / peak - 1
    return pd.DataFrame({"value": values, "daily_return": returns, "drawdown": drawdown})


def backtest_metrics(series: pd.DataFrame, risk_free: float) -> dict[str, float]:
    returns = series["daily_return"].iloc[1:]
    observations = int(returns.count())
    total_return = float(series["value"].iloc[-1] / series["value"].iloc[0] - 1)
    annual_return = float((1 + total_return) ** (TRADING_DAYS / observations) - 1) if observations else np.nan
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)) if observations else np.nan
    sharpe = (annual_return - risk_free) / annual_volatility if annual_volatility and annual_volatility > 0 else np.nan
    return {
        "Observations": observations,
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": float(series["drawdown"].min()),
    }


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def main() -> None:
    st.set_page_config(page_title="Portfolio Train/Test Optimizer", layout="wide")
    st.title("Portfolio Train/Test Optimizer")

    with st.sidebar:
        st.header("Inputs")
        tickers_raw = st.text_area("Yahoo Finance tickers", value=DEFAULT_TICKERS, height=150)
        start_date = st.date_input("Start date", value=pd.Timestamp("2023-06-30"))
        end_date = st.date_input("End date", value=pd.Timestamp.today().normalize())
        train_ratio = st.slider("Training split", 0.50, 0.90, 0.75, 0.05)
        risk_free = st.number_input("Annual risk-free rate", value=0.065, min_value=0.0, max_value=1.0, step=0.005, format="%.3f")
        initial_investment = st.number_input("Initial investment", value=100.0, min_value=1.0, step=100.0)
        allow_short = st.toggle("Allow short weights", value=False)
        target_count = st.slider("Efficient-frontier portfolios", 4, 25, 8)
        run = st.button("Run model", type="primary")

    tickers = parse_tickers(tickers_raw)
    if len(tickers) < 2:
        st.info("Enter at least two Yahoo Finance tickers.")
        return

    if not run:
        st.caption("Edit the ticker list and inputs, then run the model.")
        st.write(
            pd.DataFrame(
                {
                    "Current default securities": parse_tickers(DEFAULT_TICKERS),
                    "Example Yahoo format": ["NSE symbols usually use .NS"] * len(parse_tickers(DEFAULT_TICKERS)),
                }
            )
        )
        return

    with st.spinner("Fetching Yahoo Finance data and running the model..."):
        prices = load_prices(tuple(tickers), str(start_date), str(end_date + pd.Timedelta(days=1)))

    missing = sorted(set(tickers) - set(prices.columns))
    if prices.empty or prices.shape[1] < 2:
        st.error("Yahoo Finance did not return enough usable price series. Check ticker symbols and dates.")
        return
    if missing:
        st.warning(f"No usable data returned for: {', '.join(missing)}")

    returns = prices.pct_change().dropna(how="any")
    split_idx = int(len(returns) * train_ratio)
    if split_idx < 30 or len(returns) - split_idx < 10:
        st.error("Choose a wider date range or a split that leaves enough train and test observations.")
        return

    train_returns = returns.iloc[:split_idx]
    test_prices = prices.loc[returns.index[split_idx - 1] :]
    test_returns = returns.iloc[split_idx:]
    means = train_returns.mean()
    cov = train_returns.cov(ddof=0)
    corr = train_returns.corr()

    min_target = max(0.0, float(means.min()))
    max_target = float(means.max())
    targets = np.linspace(min_target, max_target, target_count)
    frontier = build_frontier(means, cov, risk_free, targets, allow_short)
    max_sharpe = optimize_max_sharpe(means, cov, risk_free, allow_short)
    equal_weights = pd.Series(np.repeat(1 / len(prices.columns), len(prices.columns)), index=prices.columns)

    if max_sharpe is None:
        st.error("The optimizer could not solve the max-Sharpe portfolio for these inputs.")
        return

    best_test = backtest_buy_and_hold(test_prices, max_sharpe.weights, initial_investment)
    equal_test = backtest_buy_and_hold(test_prices, equal_weights, initial_investment)
    best_metrics = backtest_metrics(best_test, risk_free)
    equal_metrics = backtest_metrics(equal_test, risk_free)

    st.subheader("Conclusion")
    winner = "optimized portfolio" if best_metrics["Sharpe Ratio"] > equal_metrics["Sharpe Ratio"] else "equal-weight portfolio"
    st.write(
        f"The {winner} has the higher out-of-sample Sharpe ratio. "
        f"Optimized Sharpe: {best_metrics['Sharpe Ratio']:.2f}; equal-weight Sharpe: {equal_metrics['Sharpe Ratio']:.2f}. "
        f"Optimized total return was {pct(best_metrics['Total Return'])} versus {pct(equal_metrics['Total Return'])}, "
        f"with maximum drawdown of {pct(best_metrics['Maximum Drawdown'])} versus {pct(equal_metrics['Maximum Drawdown'])}."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train observations", f"{len(train_returns):,}")
    c2.metric("Test observations", f"{len(test_returns):,}")
    c3.metric("Assets used", f"{prices.shape[1]:,}")
    c4.metric("Best train Sharpe", f"{max_sharpe.sharpe:.2f}")

    st.subheader("Test Backtest")
    comparison = pd.DataFrame(
        {
            "Optimized Portfolio": best_test["value"],
            "Equal Weight": equal_test["value"],
        }
    )
    st.plotly_chart(px.line(comparison, labels={"value": "Portfolio value", "index": "Date"}), use_container_width=True)

    metric_table = pd.DataFrame({"Optimized Portfolio": best_metrics, "Equal Weight": equal_metrics})
    st.dataframe(metric_table.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Train Optimization")
    rows = []
    for result in [*frontier, max_sharpe]:
        rows.append(
            {
                "Portfolio": result.name,
                "Daily Return": result.daily_return,
                "Daily Volatility": result.daily_volatility,
                "Annual Return": result.annual_return,
                "Annual Volatility": result.annual_volatility,
                "Sharpe Ratio": result.sharpe,
            }
        )
    portfolio_table = pd.DataFrame(rows).sort_values("Sharpe Ratio", ascending=False)
    numeric_portfolio_cols = portfolio_table.select_dtypes(include="number").columns
    st.dataframe(
        portfolio_table.style.format("{:.4f}", subset=numeric_portfolio_cols),
        use_container_width=True,
    )

    frontier_chart = go.Figure()
    if frontier:
        frontier_chart.add_trace(
            go.Scatter(
                x=[p.annual_volatility for p in frontier],
                y=[p.annual_return for p in frontier],
                mode="markers+lines",
                name="Target portfolios",
                text=[p.name for p in frontier],
            )
        )
    frontier_chart.add_trace(
        go.Scatter(
            x=[max_sharpe.annual_volatility],
            y=[max_sharpe.annual_return],
            mode="markers",
            name="Max Sharpe",
            marker={"size": 14},
        )
    )
    frontier_chart.update_layout(xaxis_title="Annual volatility", yaxis_title="Annual return")
    st.plotly_chart(frontier_chart, use_container_width=True)

    st.subheader("Weights")
    weights = pd.DataFrame(
        {
            "Max Sharpe Weight": max_sharpe.weights,
            "Equal Weight": equal_weights,
        }
    ).sort_values("Max Sharpe Weight", ascending=False)
    st.dataframe(weights.style.format("{:.2%}"), use_container_width=True)
    st.plotly_chart(px.bar(weights, y="Max Sharpe Weight"), use_container_width=True)

    st.subheader("Train Return Statistics")
    stats = pd.DataFrame(
        {
            "Mean Daily Return": means,
            "Daily Volatility": train_returns.std(ddof=0),
            "Coefficient of Variation": train_returns.std(ddof=0) / means.replace(0, np.nan),
            "Historical VaR 99%": train_returns.quantile(0.01),
            "Minimum Daily Return": train_returns.min(),
            "Maximum Daily Return": train_returns.max(),
        }
    )
    st.dataframe(stats.style.format("{:.4%}"), use_container_width=True)

    with st.expander("Correlation matrix"):
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    with st.expander("Variance-covariance matrix"):
        st.dataframe(cov.style.format("{:.6f}"), use_container_width=True)

    csv = pd.concat(
        {
            "prices": prices,
            "train_returns": train_returns,
            "test_returns": test_returns,
        },
        axis=1,
    ).to_csv()
    st.download_button("Download model data as CSV", csv, "portfolio_model_data.csv", "text/csv")


if __name__ == "__main__":
    main()
