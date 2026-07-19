# Portfolio Train/Test Optimizer

This Streamlit app recreates the workbook's portfolio workflow as a live, configurable project:

- Pulls adjusted close prices from Yahoo Finance through `yfinance`
- Lets you add or remove securities from the ticker list
- Splits the price history into train and test periods
- Computes daily returns, mean returns, volatility, covariance, correlation, historical VaR, and min/max daily returns
- Builds long-only target-return portfolios and a max-Sharpe portfolio
- Backtests the optimized weights against an equal-weight portfolio over the test period
- Displays conclusions, tables, charts, weights, and downloadable model data

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Create a new Streamlit app from the repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Notes on Netlify

Netlify is best for static frontend apps. This project uses Python and `yfinance` to fetch Yahoo Finance data server-side, so Streamlit Community Cloud is the simpler deployment target. To deploy on Netlify, you would need to rewrite the project as a static frontend plus a separate API/backend service.

## Ticker examples

Yahoo Finance uses exchange suffixes. Indian NSE tickers commonly end in `.NS`, such as:

- `RELIANCE.NS`
- `TCS.NS`
- `ITC.NS`
- `HDFCBANK.NS`
- `GOLDBEES.NS`
