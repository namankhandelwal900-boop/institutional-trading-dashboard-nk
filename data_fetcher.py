"""
Elite Trading System - Multi-Asset Data Fetcher
Fetches real-time data for stocks, crypto, and forex
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import time

class DataFetcher:
    """Unified data fetcher for stocks, crypto, and forex"""
    
    # Asset type suffixes for yfinance
    CRYPTO_SUFFIX = "-USD"
    FOREX_SUFFIX = "=X"
    
    # Common symbols
    CRYPTO_SYMBOLS = {
        'Bitcoin': 'BTC-USD',
        'Ethereum': 'ETH-USD',
        'Binance Coin': 'BNB-USD',
        'Cardano': 'ADA-USD',
        'Solana': 'SOL-USD',
        'Ripple': 'XRP-USD',
        'Polkadot': 'DOT-USD',
        'Dogecoin': 'DOGE-USD',
        'Avalanche': 'AVAX-USD',
        'Polygon': 'MATIC-USD'
    }
    
    FOREX_PAIRS = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X',
        'USD/JPY': 'USDJPY=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CHF': 'USDCHF=X',
        'USD/CAD': 'USDCAD=X',
        'NZD/USD': 'NZDUSD=X',
        'EUR/GBP': 'EURGBP=X',
        'EUR/JPY': 'EURJPY=X',
        'GBP/JPY': 'GBPJPY=X'
    }
    
    NSE_STOCKS = {
        'Reliance': 'RELIANCE.NS',
        'TCS': 'TCS.NS',
        'HDFC Bank': 'HDFCBANK.NS',
        'Infosys': 'INFY.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'Hindustan Unilever': 'HINDUNILVR.NS',
        'ITC': 'ITC.NS',
        'Bharti Airtel': 'BHARTIARTL.NS',
        'State Bank': 'SBIN.NS',
        'Kotak Bank': 'KOTAKBANK.NS'
    }

    COMMODITIES = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude Oil': 'CL=F',
        'Natural Gas': 'NG=F',
        'Copper': 'HG=F'
    }
    
    @staticmethod
    def detect_asset_type(symbol: str) -> str:
        """Detect if symbol is stock, crypto, forex, or commodity"""
        symbol = symbol.upper()
        
        if symbol.endswith('-USD') or symbol.startswith('BTC') or symbol.startswith('ETH'):
            return 'crypto'
        elif symbol.endswith('=X') or '/' in symbol:
            return 'forex'
        elif symbol.endswith('=F'):
            return 'commodity'
        elif symbol.endswith('.NS') or symbol.endswith('.BO'):
            return 'stock_india'
        else:
            return 'stock_us'
    
    @staticmethod
    def fetch_data(symbol: str, timeframe: str = '1d', period: str = '1y') -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for any asset
        
        Args:
            symbol: Stock/Crypto/Forex symbol
            timeframe: 1m, 5m, 15m, 30m, 1h, 1d, 1wk
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Map timeframe to yfinance interval
            interval_map = {
                '1m': '1m',
                '3m': '2m',  # yfinance doesn't have 3m, use 2m or 5m. Let's use 2m as better approx or just 5m. Actually yfinance has 2m.
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '45m': '1h',  # yfinance doesn't have 45m, use 1h
                '1h': '1h',
                '2h': '1h',  # yfinance doesn't have 2h, use 1h
                '4h': '1h',  # yfinance doesn't have 4h, use 1h
                '1d': '1d',
                '1w': '1wk'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Adjust period based on timeframe for intraday data
            # Yahoo Finance limits: 1m = 7d, 1h = 730d
            if timeframe == '1m':
                period = '5d' # Safe limit for 1m
            elif timeframe in ['3m', '5m']:
                period = '5d'  
            elif timeframe in ['15m', '30m', '45m', '1h', '2h', '4h']:
                period = '59d'  # Max for hourly data is ~60d
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            # Standardize column names
            df.columns = [col.capitalize() for col in df.columns]
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    return None
            
            return df[required_cols]
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    @staticmethod
    def fetch_multiple_timeframes(symbol: str, timeframes: list) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes
        
        Args:
            symbol: Stock/Crypto/Forex symbol
            timeframes: List of timeframes ['5m', '15m', '1h', '1d']
        
        Returns:
            Dict with timeframe as key and DataFrame as value
        """
        data = {}
        
        for tf in timeframes:
            df = DataFetcher.fetch_data(symbol, tf)
            if df is not None:
                data[tf] = df
                time.sleep(0.5)  # Rate limiting
        
        return data
    
    @staticmethod
    def get_current_price(symbol: str) -> Optional[float]:
        """Get current/latest price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get current price
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if price:
                return float(price)
            
            # Fallback: get last close
            df = ticker.history(period='1d', interval='1m')
            if not df.empty:
                return float(df['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_market_info(symbol: str) -> Dict:
        """Get detailed market information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'asset_type': DataFetcher.detect_asset_type(symbol)
            }
        except:
            return {
                'name': symbol,
                'asset_type': DataFetcher.detect_asset_type(symbol)
            }
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Check if symbol is valid and has data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='5d')
            return not df.empty
        except:
            return False


class LiveDataStream:
    """Manage live data updates with caching"""
    
    def __init__(self, cache_seconds: int = 60):
        self.cache = {}
        self.cache_time = {}
        self.cache_seconds = cache_seconds
    
    def get_data(self, symbol: str, timeframe: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get data with caching
        
        Args:
            symbol: Stock/Crypto/Forex symbol
            timeframe: Timeframe string
            force_refresh: Force fetch new data
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        
        # Check if cache is valid
        if not force_refresh and cache_key in self.cache:
            cache_age = current_time - self.cache_time.get(cache_key, 0)
            
            if cache_age < self.cache_seconds:
                return self.cache[cache_key]
        
        # Fetch fresh data
        df = DataFetcher.fetch_data(symbol, timeframe)
        
        if df is not None:
            self.cache[cache_key] = df
            self.cache_time[cache_key] = current_time
        
        return df
    
    def clear_cache(self, symbol: str = None):
        """Clear cache for specific symbol or all"""
        if symbol:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self.cache[key]
                del self.cache_time[key]
        else:
            self.cache.clear()
            self.cache_time.clear()
