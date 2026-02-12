"""
Elite Trading System - Backtester
Simulates trading strategy on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
from indicators import TechnicalIndicators

class Backtester:
    """Simple backtesting engine for the trading system"""
    
    def __init__(self, initial_capital: float = 10000, commission_pct: float = 0.1):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, df: pd.DataFrame, strategy_params: Dict = None) -> Dict:
        """
        Run backtest on a dataframe
        
        Args:
            df: DataFrame with OHLCV data
            strategy_params: Dictionary of strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        if strategy_params is None:
            strategy_params = {}
            
        if df is None or df.empty:
            return None
            
        df = df.copy()
        
        # Calculate indicators
        df = TechnicalIndicators.calculate_all(df)
        
        # Initialize variables
        capital = self.initial_capital
        position = 0  # 0: Flat, 1: Long, -1: Short
        entry_price = 0
        
        self.trades = []
        self.equity_curve = [capital]
        
        # Strategy Parameters
        rsi_oversold = strategy_params.get('rsi_oversold', 30)
        rsi_overbought = strategy_params.get('rsi_overbought', 70)
        
        # Loop through data
        for i in range(50, len(df)):
            current_date = df.index[i]
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            close = row['Close']
            
            # Simple Strategy Logic (can be expanded to use Analyzer)
            # Buy: Supertrend Bullish + RSI < 60 + MACD Bullish
            # Sell: Supertrend Bearish + RSI > 40 + MACD Bearish
            
            signal = 0
            
            # Check Supertrend
            st_bullish = row.get('Supertrend_Direction', 0) == 1
            st_bearish = row.get('Supertrend_Direction', 0) == -1
            
            # Check MACD
            macd_bullish = row['MACD'] > row['MACD_Signal']
            macd_bearish = row['MACD'] < row['MACD_Signal']
            
            # Check RSI
            rsi_bullish = row['RSI'] < 60  # Not overbought
            rsi_bearish = row['RSI'] > 40  # Not oversold
            
            # Generate Signal
            if st_bullish and macd_bullish and rsi_bullish:
                signal = 1
            elif st_bearish and macd_bearish and rsi_bearish:
                signal = -1
            
            # Execution Logic
            if position == 0:
                if signal == 1:
                    # Enter Long
                    position = 1
                    entry_price = close
                    self._record_trade('BUY', current_date, close, capital)
                elif signal == -1:
                    # Enter Short
                    position = -1
                    entry_price = close
                    self._record_trade('SELL', current_date, close, capital)
                    
            elif position == 1:
                # Exit Long if signal flips or Stop Loss / Take Profit
                if signal == -1:
                    # Close Long and Open Short (Reverse)
                    pnl = (close - entry_price) / entry_price * capital
                    capital += pnl
                    self._record_trade('SELL', current_date, close, capital, pnl)
                    
                    position = -1
                    entry_price = close
                    self._record_trade('SELL', current_date, close, capital)
                    
            elif position == -1:
                # Exit Short if signal flips
                if signal == 1:
                    # Close Short and Open Long (Reverse)
                    pnl = (entry_price - close) / entry_price * capital
                    capital += pnl
                    self._record_trade('BUY', current_date, close, capital, pnl)
                    
                    position = 1
                    entry_price = close
                    self._record_trade('BUY', current_date, close, capital)
            
            self.equity_curve.append(capital)
            
        # Close any open position at the end
        if position != 0:
            last_price = df.iloc[-1]['Close']
            if position == 1:
                pnl = (last_price - entry_price) / entry_price * capital
            else:
                pnl = (entry_price - last_price) / entry_price * capital
            capital += pnl
            self._record_trade('CLOSE', df.index[-1], last_price, capital, pnl)
            
        return self._calculate_metrics(capital, df)
    
    def _record_trade(self, action, date, price, equity, pnl=0):
        """Record a trade event"""
        self.trades.append({
            'action': action,
            'date': date,
            'price': price,
            'equity': equity,
            'pnl': pnl
        })
        
    def _calculate_metrics(self, final_capital, df):
        """Calculate backtest metrics"""
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]
        
        win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (wins or losses) else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_trades': len(wins) + len(losses),
            'win_rate': win_rate,
            'equity_curve': self.equity_curve
        }
