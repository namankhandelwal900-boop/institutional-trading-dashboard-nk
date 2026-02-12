# Elite Multi-Timeframe Trading System

An institutional-grade trading dashboard built with Streamlit, Plotly, and Python. Features real-time data, multi-timeframe analysis, custom indicators (Quantum, SMC, Order Blocks), and a powerful backtester.

## Features
- **Multi-Timeframe Analysis**: Analyze stocks across 1m, 5m, 15m, 1h, 4h, 1d timeframes simultaneously.
- **Institutional Indicators**:
    - **Quantum Indicators**: Fibonacci ATR/ADR levels, Order Blocks, FVGs.
    - **Smart Money Concepts (SMC)**: Market Structure (BOS/CHoCH), Premium/Discount Zones.
- **Universal Search**: Support for NSE, BSE, Crypto (Binance/Coinbase), Forex, and US Stocks.
- **Backtesting Engine**: Test strategies with historical data.
- **Real-time Data**: Fetches live data using `yfinance`.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Institutional trading"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Run the application using Streamlit:

```bash
streamlit run app.py
```

## Deployment (Streamlit Cloud)

1. Push this code to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select the repository.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

## Dependencies
- streamlit
- pandas
- numpy
- plotly
- yfinance
- scipy
- streamlit-autorefresh

## Usage
- **Sidebar**: Select Asset Type or Search for specific symbols (e.g., `RELIANCE.NS`, `BTC-USD`).
- **Tabs**: Switch between Analysis, Charts, Backtester, and Details.
- **Refresh**: Use the "Refresh Now" button or enable Auto-Refresh for live updates.
