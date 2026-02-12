"""
Elite Multi-Timeframe Trading System
Professional trading dashboard for stocks, crypto, and forex
"""

# Check dependencies on startup
import sys

required_packages = {
    'streamlit': 'streamlit',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'plotly': 'plotly',
    'yfinance': 'yfinance'
}

missing_packages = []
for package_name, pip_name in required_packages.items():
    try:
        __import__(package_name)
    except ImportError:
        missing_packages.append(pip_name)

if missing_packages:
    print("\n" + "="*60)
    print("ERROR: Missing required packages!")
    print("="*60)
    print("\nPlease install missing packages:")
    print(f"\npip install {' '.join(missing_packages)}")
    print("\nOr install all requirements:")
    print("pip install -r requirements.txt")
    print("\n" + "="*60 + "\n")
    sys.exit(1)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import concurrent.futures
from datetime import datetime
import functools
from typing import Tuple, Dict

# Try to import auto-refresh, but make it optional
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False
    st.warning("streamlit-autorefresh not installed. Auto-refresh feature disabled. Install with: pip install streamlit-autorefresh")

# Import our modules
from data_fetcher import DataFetcher, LiveDataStream
from analyzer import MultiTimeframeAnalyzer
from risk_manager import RiskManager, PortfolioManager
from indicators import TechnicalIndicators, QuantumIndicators

from backtester import Backtester
from notifications import NotificationManager

# Page config
st.set_page_config(
    page_title="Elite Trading System by NK",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .buy-signal {
        color: #00ff00;
        font-weight: bold;
        font-size: 24px;
    }
    .sell-signal {
        color: #ff0000;
        font-weight: bold;
        font-size: 24px;
    }
    .neutral-signal {
        color: #ffa500;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_stream' not in st.session_state:
    st.session_state.data_stream = LiveDataStream(cache_seconds=60)

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = MultiTimeframeAnalyzer()

# Title
st.title("🚀 Elite Multi-Timeframe Trading System")
st.markdown("### 👨‍💻 Developed by Naman (NK)")
st.markdown("*Institutional-grade analysis for Stocks, Crypto & Forex*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Asset selection
    asset_type = st.selectbox(
        "Asset Type",
        ["NSE/BSE Stocks", "Crypto", "Forex", "Commodity", "US Stocks", "Custom"]
    )
    
    # Symbol input based on asset type
    
    # --- Symbol Selection Logic ---
    search_symbol = st.text_input("🔍 Search Any Symbol (NSE/BSE/Crypto)", placeholder="e.g. INFY.NS, TATAMOTORS.BO, BTC-USD")
    
    # Default category selection
    if asset_type == "NSE/BSE Stocks":
        col1, col2 = st.columns(2)
        with col1:
            quick_select = st.selectbox(
                "Quick Select",
                [""] + list(DataFetcher.NSE_STOCKS.keys())
            )
        with col2:
            if quick_select:
                symbol = DataFetcher.NSE_STOCKS[quick_select]
                st.text_input("Symbol", value=symbol, key="symbol_display", disabled=True)
            else:
                symbol = st.text_input("Symbol", "RELIANCE.NS")
    
    elif asset_type == "Crypto":
        col1, col2 = st.columns(2)
        with col1:
            quick_select = st.selectbox(
                "Quick Select",
                [""] + list(DataFetcher.CRYPTO_SYMBOLS.keys())
            )
        with col2:
            if quick_select:
                symbol = DataFetcher.CRYPTO_SYMBOLS[quick_select]
                st.text_input("Symbol", value=symbol, key="symbol_display", disabled=True)
            else:
                symbol = st.text_input("Symbol", "BTC-USD")
    
    elif asset_type == "Forex":
        col1, col2 = st.columns(2)
        with col1:
            quick_select = st.selectbox(
                "Quick Select",
                [""] + list(DataFetcher.FOREX_PAIRS.keys())
            )
        with col2:
            if quick_select:
                symbol = DataFetcher.FOREX_PAIRS[quick_select]
                st.text_input("Symbol", value=symbol, key="symbol_display", disabled=True)
            else:
                symbol = st.text_input("Symbol", "EURUSD=X")
    
    elif asset_type == "Commodity":
        col1, col2 = st.columns(2)
        with col1:
            quick_select = st.selectbox(
                "Quick Select",
                [""] + list(DataFetcher.COMMODITIES.keys())
            )
        with col2:
            if quick_select:
                symbol = DataFetcher.COMMODITIES[quick_select]
                st.text_input("Symbol", value=symbol, key="symbol_display", disabled=True)
            else:
                symbol = st.text_input("Symbol", "GC=F")
    
    else:
        symbol = st.text_input("Enter Symbol", "AAPL")
    
    # Override if search is used
    if search_symbol:
        symbol = search_symbol.strip().upper()
        
    symbol = symbol.upper()
    
    st.markdown("---")
    
    # Timeframe selection
    st.subheader("📊 Timeframes")
    
    available_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    selected_timeframes = st.multiselect(
        "Select Timeframes",
        available_timeframes,
        default=['1m', '5m', '15m', '1h']
    )
    
    if not selected_timeframes:
        st.warning("Please select at least one timeframe")
        selected_timeframes = ['5m', '15m']
    
    st.markdown("---")
    
    # Risk Management
    st.subheader("💰 Risk Management")
    
    account_size = st.number_input(
        "Account Size (₹)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    risk_per_trade = st.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1
    )
    
    st.markdown("---")
    
    # Auto-refresh (only if available)
    if AUTO_REFRESH_AVAILABLE:
        auto_refresh = st.checkbox("Auto Refresh (15s)", value=False)
        
        if auto_refresh:
            st_autorefresh(interval=15000, key="data_refresh")
    else:
        st.info("💡 Install streamlit-autorefresh for auto-refresh feature")
    
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        if 'data_stream' in st.session_state:
            st.session_state.data_stream.clear_cache()
        st.rerun()

    st.markdown("---")
    st.subheader("🔔 Active Alerts")
    
    # Notification Settings
    with st.expander("📱 Notification Settings"):
        st.session_state.telegram_token = st.text_input("Telegram Bot Token", value=st.session_state.get('telegram_token', ''), type="password")
        st.session_state.telegram_chat_id = st.text_input("Telegram Chat ID", value=st.session_state.get('telegram_chat_id', ''))
        st.caption("Get token from @BotFather and ID from @userinfobot")

    # Initialize Notifier
    notifier = NotificationManager()

    # Alert System
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    # Custom Alert Creator
    with st.form("alert_form"):
        col_a, col_b = st.columns(2)
        alert_cond = col_a.selectbox("Condition", ["Price Above", "Price Below"])
        alert_price = col_b.number_input("Target Price", value=float(current_price))
        
        if st.form_submit_button("➕ Add Alert"):
            st.session_state.alerts.append({
                'symbol': symbol,
                'type': 'above' if 'Above' in alert_cond else 'below',
                'target_price': alert_price,
                'time': datetime.now().strftime('%H:%M')
            })
            st.success(f"Alert set for {symbol} @ {alert_price}")

    # Display Alerts
    if st.session_state.alerts:
        st.write("---")
        for i, alert in enumerate(st.session_state.alerts):
            icon = "📈" if alert['type'] == 'above' else "📉"
            st.info(f"{icon} {alert['symbol']}: {alert['type'].upper()} {alert['target_price']}")
            if st.button("❌ Remove", key=f"del_alert_{i}"):
                st.session_state.alerts.pop(i)
                st.rerun()
    else:
        st.caption("No active alerts")

    # Run Check (Only on Auto-Refresh or Manual Refresh)
    # We pass the CURRENT fetched price to the checker
    try:
        notifier.check_and_send_alerts(current_price, symbol)
    except Exception:
        pass

    st.markdown("---")
    st.markdown("### 👨‍💻 Created by **Naman (NK)**")
    st.caption("© 2026 Institutional Trading Dashboard")

# --- DISCLAIMER FOOTER ---
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; margin_top: 50px; padding: 20px; border-top: 1px solid #333;'>
    <p><strong>⚠️ DISCLAIMER: FOR AI RESEARCH PURPOSES ONLY</strong></p>
    <p>This tool is designed for educational and AI research purposes only. It does not constitute financial advice.</p>
    <p>Trading in financial markets involves significant risk. Please consult with a certified financial advisor before making any investment decisions.</p>
    <p>The creator (Naman/NK) assumes no responsibility for any financial losses or trading decisions made based on this tool.</p>
</div>
""", unsafe_allow_html=True)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Analysis", "📊 Charts", "💼 Trade Plan", "🔙 Backtest", "📚 Details"])

@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_data_parallel(symbol: str, timeframes: list) -> Tuple[Dict[str, pd.DataFrame], float, Dict]:
    """Fetch data for all timeframes in parallel with caching"""
    data = {}
    
    # Function to fetch single timeframe
    def fetch_single(tf):
        return tf, DataFetcher.fetch_data(symbol, tf)
    
    # Ensure '1d' is fetched for Quantum Levels regardless of selection
    timeframes_to_fetch = set(timeframes)
    timeframes_to_fetch.add('1d')
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(timeframes_to_fetch), 10)) as executor:
        future_to_tf = {executor.submit(fetch_single, tf): tf for tf in timeframes_to_fetch}
        for future in concurrent.futures.as_completed(future_to_tf):
            tf, df = future.result()
            if df is not None and not df.empty:
                data[tf] = df
                
    # Get current price and market info (concurrently if possible, but fast enough)
    current_price = DataFetcher.get_current_price(symbol)
    market_info = DataFetcher.get_market_info(symbol)
    
    return data, current_price, market_info

# Fetch data for all timeframes
with st.spinner(f"⚡ Fetching market data for {symbol}..."):
    try:
        data_dict, current_price, market_info = fetch_all_data_parallel(symbol, selected_timeframes)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        data_dict = {}
        current_price = 0
        market_info = {}

    if not data_dict:
        st.error(f"❌ Unable to fetch data for {symbol}. Market might be closed or symbol is invalid.")
        st.info("Try checking the symbol format (e.g., RELIANCE.NS, BTC-USD, GC=F)")
        st.stop()

@st.cache_data(ttl=60, show_spinner=False)
def run_analysis_pipeline(data_dict: Dict[str, pd.DataFrame]) -> Dict:
    """Run the multi-timeframe analysis pipeline with caching"""
    # Instantiate analyzer locally to ensure thread safety and cache compatibility
    # This avoids pickling issues with session_state objects
    analyzer = MultiTimeframeAnalyzer()
    
    timeframe_analyses = {}
    weighted_scores = []
    
    # Helper for threading
    def analyze_tf_wrapper(df, tf):
        return tf, analyzer.analyze_single_timeframe(df, tf)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit analysis tasks
        future_to_tf = {
            executor.submit(analyze_tf_wrapper, df, tf): tf 
            for tf, df in data_dict.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_tf):
            tf, result = future.result()
            timeframe_analyses[tf] = result
            
            # Reconstruct weighted scores logic
            weight = analyzer.TIMEFRAME_WEIGHTS.get(tf, 0.10)
            weighted_scores.append(result['score'] * weight)
            
    # Calculate Consensus
    consensus_score = sum(weighted_scores)
    confidence = min(100, abs(consensus_score))
    
    if consensus_score >= 40: consensus_signal = 'STRONG BUY'
    elif consensus_score >= 20: consensus_signal = 'BUY'
    elif consensus_score >= -20: consensus_signal = 'NEUTRAL'
    elif consensus_score >= -40: consensus_signal = 'SELL'
    else: consensus_signal = 'STRONG SELL'
    
    bullish_count = sum(1 for a in timeframe_analyses.values() if 'BUY' in a['signal'])
    bearish_count = sum(1 for a in timeframe_analyses.values() if 'SELL' in a['signal'])
    neutral_count = len(timeframe_analyses) - bullish_count - bearish_count
    
    summary = f"{bullish_count}/{len(timeframe_analyses)} Bullish, {bearish_count}/{len(timeframe_analyses)} Bearish."
    
    return {
        'consensus_signal': consensus_signal,
        'consensus_score': consensus_score,
        'confidence': confidence,
        'timeframe_analyses': timeframe_analyses,
        'summary': summary,
        'counts': {'bullish': bullish_count, 'bearish': bearish_count, 'neutral': neutral_count, 'total': len(timeframe_analyses)}
    }

# Perform analysis
with st.spinner("🧠 Processing institutional analysis..."):
    try:
        analysis_result = run_analysis_pipeline(data_dict)
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        st.stop()

# TAB 1: Analysis Dashboard
with tab1:
    # Header with current price
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{current_price:.2f}" if current_price else "N/A"
        )
    
    with col2:
        signal = analysis_result['consensus_signal']
        signal_color = "buy-signal" if "BUY" in signal else "sell-signal" if "SELL" in signal else "neutral-signal"
        st.markdown(f"<p class='{signal_color}'>{signal}</p>", unsafe_allow_html=True)
    
    with col3:
        confidence = analysis_result['confidence']
        st.metric("Confidence", f"{confidence:.0f}%")
    
    with col4:
        st.metric("Timeframes", f"{analysis_result['counts']['total']}")
    
    st.markdown("---")
    
    # Consensus Summary
    st.subheader("🎯 Consensus Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Summary:** {analysis_result['summary']}")
        
        # --- SMC Dashboard ---
        st.markdown("---")
        st.subheader("🏦 Smart Money Concepts (SMC)")
        
        # Get data from primary timeframe (e.g., 5m or 15m)
        primary_tf = selected_timeframes[0]
        if primary_tf in analysis_result['timeframe_analyses']:
            smc_data = analysis_result['timeframe_analyses'][primary_tf]['data']
            
            # Display Zones
            if 'pd_zones' in smc_data:
                zones = smc_data['pd_zones']
                c1, c2, c3 = st.columns(3)
                c1.metric("Premium High", f"{zones.get('range_high', 0):.2f}")
                c2.metric("Equilibrium", f"{zones.get('equilibrium', 0):.2f}")
                c3.metric("Discount Low", f"{zones.get('range_low', 0):.2f}")
            
            # Display Structure
            if 'market_structure' in smc_data:
                ms = smc_data['market_structure']
                st.write(f"**Current Trend:** {ms.get('current_trend', 'N/A')}")
                
                # Show recent breaks
                breaks = ms.get('structure_breaks', [])
                if breaks:
                    latest_break = breaks[-1]
                    st.caption(f"Latest Break: {latest_break['type']} at {latest_break['price']:.2f}")
            
            # Liquidity
            if 'eq_highs_lows' in smc_data:
                 eq = smc_data['eq_highs_lows']
                 eqh_count = len(eq.get('eqh', []))
                 eql_count = len(eq.get('eql', []))
                 if eqh_count > 0:
                     st.warning(f"⚠️ {eqh_count} Equal Highs (Liquidity Targets) detected")
                 if eql_count > 0:
                     st.warning(f"⚠️ {eql_count} Equal Lows (Liquidity Targets) detected")
        
        # Score visualization
        score = analysis_result['consensus_score']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Consensus Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-100, -40], 'color': "rgba(255, 0, 0, 0.25)"},
                    {'range': [-40, -20], 'color': "rgba(255, 82, 82, 0.25)"},
                    {'range': [-20, 20], 'color': "rgba(255, 165, 0, 0.25)"},
                    {'range': [20, 40], 'color': "rgba(144, 238, 144, 0.25)"},
                    {'range': [40, 100], 'color': "rgba(0, 255, 0, 0.25)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Timeframe breakdown
        st.write("**Timeframe Breakdown:**")
        
        counts = analysis_result['counts']
        
        st.metric("🟢 Bullish", counts['bullish'])
        st.metric("🔴 Bearish", counts['bearish'])
        st.metric("🟡 Neutral", counts['neutral'])
    
    st.markdown("---")
    
    # Individual Timeframe Analysis
    st.subheader("📊 Timeframe Analysis")
    
    tf_analyses = analysis_result['timeframe_analyses']
    
    # Create columns for timeframes
    cols_per_row = 3
    tf_list = list(tf_analyses.keys())
    
    for i in range(0, len(tf_list), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(tf_list):
                tf = tf_list[i + j]
                tf_analysis = tf_analyses[tf]
                
                with col:
                    signal_emoji = "🟢" if "BUY" in tf_analysis['signal'] else "🔴" if "SELL" in tf_analysis['signal'] else "🟡"
                    
                    st.markdown(f"### {signal_emoji} {tf}")
                    st.write(f"**Signal:** {tf_analysis['signal']}")
                    st.write(f"**Confidence:** {tf_analysis['confidence']:.0f}%")
                    
                    # Show top reasons
                    if tf_analysis['reasons']:
                        st.write("**Key Signals:**")
                        for reason in tf_analysis['reasons'][:3]:
                            st.caption(f"• {reason}")

# TAB 2: Charts
with tab2:
    st.subheader(f"📈 {symbol} - Technical Chart")
    
    # Timeframe selector for chart
    chart_tf = st.selectbox("Chart Timeframe", selected_timeframes, index=0)
    
    if chart_tf in data_dict:
        df_chart = data_dict[chart_tf].copy()
        df_chart = TechnicalIndicators.calculate_all(df_chart)
        
        # Get analysis for this timeframe
        tf_analysis = analysis_result['timeframe_analyses'].get(chart_tf, {})
        tf_data = tf_analysis.get('data', {})
        
        # Create candlestick chart
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} - {chart_tf}', 'Volume', 'RSI'),
            vertical_spacing=0.05,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_chart.index,
            open=df_chart['Open'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name='Price'
        ), row=1, col=1)
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['EMA_9'],
            name='EMA 9', line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart['EMA_21'],
            name='EMA 21', line=dict(color='blue', width=1)
        ), row=1, col=1)
        
        # Add Order Blocks
        order_blocks = tf_data.get('order_blocks', {})
        
        for ob in order_blocks.get('bullish', [])[:3]:
            if not ob.get('breached'):
                fig.add_hrect(
                    y0=ob['bottom'], y1=ob['top'],
                    fillcolor="green", opacity=0.2,
                    line_width=0, row=1, col=1
                )
        
        for ob in order_blocks.get('bearish', [])[:3]:
            if not ob.get('breached'):
                fig.add_hrect(
                    y0=ob['bottom'], y1=ob['top'],
                    fillcolor="red", opacity=0.2,
                    line_width=0, row=1, col=1
                )
        
        # Add Support/Resistance
        sr_levels = tf_data.get('sr_levels', {})
        
        for support in sr_levels.get('support', [])[:2]:
            fig.add_hline(
                y=support['price'],
                line_dash="dash", line_color="green",
                annotation_text=f"Support: {support['price']:.2f}",
                row=1, col=1
            )
        
        for resistance in sr_levels.get('resistance', [])[:2]:
            fig.add_hline(
                y=resistance['price'],
                line_dash="dash", line_color="red",
                annotation_text=f"Resistance: {resistance['price']:.2f}",
                row=1, col=1
            )

        # Add Quantum Levels (from Daily Data)
        if '1d' in data_dict:
            daily_df = data_dict['1d']
            # Calculate levels
            q_levels = QuantumIndicators.calculate_levels(daily_df)
            
            if q_levels:
                # Plot Lines
                fig.add_hline(y=q_levels['buy_entry'], line_dash="dash", line_color="cyan", annotation_text="Buy Entry")
                fig.add_hline(y=q_levels['sell_entry'], line_dash="dash", line_color="orange", annotation_text="Sell Entry")
                fig.add_hline(y=q_levels['mid_line'], line_dash="dot", line_color="gray", annotation_text="Mid Line")
                
                # Targets
                fig.add_hline(y=q_levels['buy_target1'], line_dash="dot", line_color="green", annotation_text="Buy T1", opacity=0.5)
                fig.add_hline(y=q_levels['sell_target1'], line_dash="dot", line_color="red", annotation_text="Sell T1", opacity=0.5)
                fig.add_hline(y=q_levels['buy_sl'], line_dash="dot", line_color="red", annotation_text="Buy SL", opacity=0.3)
                fig.add_hline(y=q_levels['sell_sl'], line_dash="dot", line_color="red", annotation_text="Sell SL", opacity=0.3)
        
        # Volume
        colors = ['red' if df_chart['Close'].iloc[i] < df_chart['Open'].iloc[i] else 'green' 
                 for i in range(len(df_chart))]
        
        fig.add_trace(go.Bar(
            x=df_chart.index,
            y=df_chart['Volume'],
            name='Volume',
            marker_color=colors
        ), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df_chart.index,
            y=df_chart['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        
        # Chart Settings
        with st.expander("⚙️ Chart Overlays"):
            col_c1, col_c2, col_c3 = st.columns(3)
            show_vwap = col_c1.checkbox("Show VWAP", value=True)
            show_supertrend = col_c2.checkbox("Show Supertrend", value=True)
            show_ichimoku = col_c3.checkbox("Show Ichimoku Cloud", value=False)
            
            if show_vwap and 'VWAP' in df_chart.columns:
                    fig.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart['VWAP'],
                    name='VWAP', line=dict(color='orange', width=1.5, dash='dot')
                ), row=1, col=1)
            
            if show_supertrend and 'Supertrend' in df_chart.columns:
                # Color based on direction
                st_color = ['green' if d == 1 else 'red' for d in df_chart['Supertrend_Direction']]
                fig.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart['Supertrend'],
                    name='Supertrend', marker=dict(color=st_color, size=2), mode='markers'
                ), row=1, col=1)
                
            if show_ichimoku and 'Senkou_span_a' in df_chart.columns:
                # Add cloud
                fig.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart['Senkou_span_a'],
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_chart.index, y=df_chart['Senkou_span_b'],
                    fill='tonexty', fillcolor='rgba(0, 250, 154, 0.1)',
                    line=dict(color='rgba(0,0,0,0)'), name='Ichimoku Cloud',
                    hoverinfo='skip'
                ), row=1, col=1)
                
        st.plotly_chart(fig, use_container_width=True, key="main_chart")

# TAB 3: Trade Plan
with tab3:
    st.subheader("💼 Trade Plan & Risk Management")
    
    # Initialize Risk Manager
    risk_mgr = RiskManager(account_size=account_size, risk_per_trade_pct=risk_per_trade)
    
    # Generate trade plan
    signal = analysis_result['consensus_signal']
    
    if signal != 'NEUTRAL':
        # Use highest timeframe for trade planning
        highest_tf = selected_timeframes[-1]
        df_trade = data_dict[highest_tf]
        
        trade_plan = risk_mgr.generate_trade_plan(df_trade, signal, current_price)
        
        if trade_plan['valid']:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Trade Details
                st.markdown("### 📋 Trade Details")
                
                direction_emoji = "🟢" if trade_plan['direction'] == 'LONG' else "🔴"
                st.markdown(f"## {direction_emoji} {trade_plan['direction']} {trade_plan['signal']}")
                
                st.write(f"**Entry Price:** ₹{trade_plan['entry_price']:.2f}")
                st.write(f"**Stop Loss:** ₹{trade_plan['stop_loss']:.2f} ({trade_plan['stop_loss_pct']:.2f}% risk)")
                
                st.markdown("---")
                
                st.write("**Take Profit Levels:**")
                for tp_name, tp_data in trade_plan['take_profits'].items():
                    st.write(f"{tp_name}: ₹{tp_data['price']:.2f} (1:{tp_data['rr_ratio']}) - {tp_data['profit_pct']:.1f}% profit")
            
            with col2:
                # Position Sizing
                st.markdown("### 💰 Position Sizing")
                
                position = trade_plan['position']
                
                st.metric("Quantity", f"{position['quantity']:.4f}")
                st.metric("Position Value", f"₹{position['position_value']:,.2f}")
                st.metric("Risk Amount", f"₹{trade_plan['risk_amount']:,.2f}")
                st.metric("Risk %", f"{trade_plan['risk_pct']:.2f}%")
                
                if position.get('leverage', 1) > 1:
                    st.warning(f"⚠️ Leverage: {position['leverage']:.2f}x")
            
            st.markdown("---")
            
            # Trade Summary
            st.markdown("### 📝 Trade Summary")
            st.text(trade_plan['trade_summary'])
            
        else:
            st.info("No valid trade setup at the moment.")
    else:
        st.info("Consensus signal is NEUTRAL. Wait for clearer market direction.")
    
    st.markdown("---")
    
    # Risk Statistics
    st.subheader("📊 Risk Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Account Size", f"₹{account_size:,.0f}")
    
    with col2:
        st.metric("Max Risk per Trade", f"₹{account_size * risk_per_trade / 100:,.2f}")
    
    with col3:
        max_drawdown = risk_per_trade * 5  # Conservative estimate
        st.metric("Est. Max Drawdown", f"{max_drawdown:.1f}%")

# TAB 4: Backtest
with tab4:
    st.subheader("🔙 Strategy Backtest")
    
    st.info("Simulate the strategy performance on historical data (Last 500 candles)")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        bt_timeframe = st.selectbox("Timeframe", selected_timeframes, index=0, key="bt_tf")
    with col2:
        initial_capital = st.number_input("Initial Capital", value=100000, step=10000, key="bt_cap")
        
    if st.button("🚀 Run Backtest"):
        if bt_timeframe in data_dict:
            df_bt = data_dict[bt_timeframe].copy()
            
            with st.spinner("Running backtest simulation..."):
                backtester = Backtester(initial_capital=initial_capital)
                results = backtester.run_backtest(df_bt)
                
                if results:
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Return", f"{results['total_return']:.2f}%")
                    m2.metric("Win Rate", f"{results['win_rate']:.1f}%")
                    m3.metric("Total Trades", results['total_trades'])
                    m4.metric("Final Capital", f"₹{results['final_capital']:.2f}")
                    
                    # Equity Curve
                    st.line_chart(results['equity_curve'])
                    
                    st.success("Backtest completed!")
                else:
                    st.warning("Not enough data to run backtest.")
        else:
            st.error("Please ensure data is fetched for the selected timeframe.")

# TAB 5: Details
with tab5:
    st.subheader("📚 Market Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Symbol Information:**")
        st.write(f"Name: {market_info.get('name', symbol)}")
        st.write(f"Type: {market_info.get('asset_type', 'Unknown')}")
        if market_info.get('sector'):
            st.write(f"Sector: {market_info.get('sector')}")
        if market_info.get('industry'):
            st.write(f"Industry: {market_info.get('industry')}")
    
    with col2:
        st.write("**Market Data:**")
        if market_info.get('market_cap'):
            st.write(f"Market Cap: ₹{market_info['market_cap']/1e7:.2f} Cr")
        if market_info.get('avg_volume'):
            st.write(f"Avg Volume: {market_info['avg_volume']:,.0f}")
        if market_info.get('pe_ratio') != 'N/A':
            st.write(f"P/E Ratio: {market_info['pe_ratio']:.2f}")
    
    st.markdown("---")
    
    # Indicator explanations
    with st.expander("ℹ️ How This System Works"):
        st.markdown("""
        ### Elite Multi-Timeframe Trading System
        
        **What It Does:**
        - Analyzes 5-6 timeframes simultaneously
        - Uses institutional-grade indicators (Order Blocks, FVG, S/R, Fibonacci)
        - Generates consensus signals with confidence scores
        - Provides complete trade plans with risk management
        
        **Key Indicators:**
        
        1. **Order Blocks (OB):** Institutional buying/selling zones
        2. **Fair Value Gaps (FVG):** Inefficiencies in price that tend to get filled
        3. **Support/Resistance:** Key price levels where price tends to react
        4. **Fibonacci Levels:** Natural retracement zones (0.382, 0.5, 0.618, 0.786)
        5. **Technical Indicators:** RSI, MACD, Moving Averages for momentum
        
        **Signal Generation:**
        - Each timeframe gets a score based on all indicators
        - Scores are weighted by timeframe importance
        - Consensus signal requires majority agreement
        - Confidence shows strength of agreement
        
        **Risk Management:**
        - Position sizing based on account risk (1-2% per trade)
        - ATR-based stop loss placement
        - Multiple take-profit levels with R:R ratios (1.5:1, 2.5:1, 3.5:1)
        
        **Realistic Expectations:**
        - Target win rate: 60-70%
        - With 2:1+ R:R ratio = Profitable system
        - Always use stop losses
        - Never risk more than 1-2% per trade
        """)
    
    with st.expander("⚠️ Disclaimer"):
        st.warning("""
        **IMPORTANT DISCLAIMER:**
        
        This is an educational tool for technical analysis. 
        
        - NOT financial advice
        - Past performance doesn't guarantee future results
        - Trading involves substantial risk of loss
        - Never invest money you can't afford to lose
        - Always do your own research
        - Consult a financial advisor before trading
        
        The developers are not responsible for any financial losses incurred from using this system.
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.caption(f"Analyzing: {symbol}")

with col3:
    st.caption("Elite Trading System v1.0")
