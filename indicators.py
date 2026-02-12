"""
Elite Trading System - Technical Indicators Engine
Implements institutional-grade indicators: Order Blocks, FVG, Support/Resistance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class OrderBlockDetector:
    """Detects institutional Order Blocks based on volume and price action"""
    
    def __init__(self, swing_length: int = 10, max_atr_mult: float = 3.5):
        self.swing_length = swing_length
        self.max_atr_mult = max_atr_mult
    
    def detect_order_blocks(self, df: pd.DataFrame) -> Dict:
        """
        Detect bullish and bearish order blocks
        Returns: Dict with bullish_obs and bearish_obs lists
        """
        df = df.copy()
        
        # Calculate ATR for filtering
        df['atr'] = self._calculate_atr(df, 14)
        
        bullish_obs = []
        bearish_obs = []
        
        # Find swing highs and lows
        for i in range(self.swing_length, len(df) - self.swing_length):
            # Bullish Order Block detection
            if self._is_swing_high(df, i):
                ob = self._create_bullish_ob(df, i)
                if ob and self._validate_ob_size(ob, df.iloc[i]['atr']):
                    bullish_obs.append(ob)
            
            # Bearish Order Block detection
            if self._is_swing_low(df, i):
                ob = self._create_bearish_ob(df, i)
                if ob and self._validate_ob_size(ob, df.iloc[i]['atr']):
                    bearish_obs.append(ob)
        
        # Filter to keep only most recent and relevant OBs
        bullish_obs = self._filter_order_blocks(bullish_obs, df, 'bullish')
        bearish_obs = self._filter_order_blocks(bearish_obs, df, 'bearish')
        
        return {
            'bullish': bullish_obs[-5:],  # Keep last 5
            'bearish': bearish_obs[-5:]
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _is_swing_high(self, df: pd.DataFrame, i: int) -> bool:
        """Check if index i is a swing high"""
        window = self.swing_length
        current_high = df.iloc[i]['High']
        
        # Check left side
        left_highs = df.iloc[i-window:i]['High']
        # Check right side
        right_highs = df.iloc[i+1:i+window+1]['High']
        
        return (current_high > left_highs.max()) and (current_high > right_highs.max())
    
    def _is_swing_low(self, df: pd.DataFrame, i: int) -> bool:
        """Check if index i is a swing low"""
        window = self.swing_length
        current_low = df.iloc[i]['Low']
        
        left_lows = df.iloc[i-window:i]['Low']
        right_lows = df.iloc[i+1:i+window+1]['Low']
        
        return (current_low < left_lows.min()) and (current_low < right_lows.min())
    
    def _create_bullish_ob(self, df: pd.DataFrame, swing_idx: int) -> Optional[Dict]:
        """Create bullish order block from swing high"""
        try:
            # Look back to find the last down candle before the move up
            for lookback in range(1, min(10, swing_idx)):
                candle_idx = swing_idx - lookback
                
                if df.iloc[candle_idx]['Close'] < df.iloc[candle_idx]['Open']:
                    # Found bearish candle - this is potential OB
                    ob_top = df.iloc[candle_idx]['High']
                    ob_bottom = df.iloc[candle_idx]['Low']
                    ob_volume = df.iloc[candle_idx]['Volume']
                    
                    return {
                        'type': 'bullish',
                        'top': ob_top,
                        'bottom': ob_bottom,
                        'volume': ob_volume,
                        'index': candle_idx,
                        'timestamp': df.index[candle_idx],
                        'breached': False
                    }
            return None
        except:
            return None
    
    def _create_bearish_ob(self, df: pd.DataFrame, swing_idx: int) -> Optional[Dict]:
        """Create bearish order block from swing low"""
        try:
            for lookback in range(1, min(10, swing_idx)):
                candle_idx = swing_idx - lookback
                
                if df.iloc[candle_idx]['Close'] > df.iloc[candle_idx]['Open']:
                    # Found bullish candle - this is potential OB
                    ob_top = df.iloc[candle_idx]['High']
                    ob_bottom = df.iloc[candle_idx]['Low']
                    ob_volume = df.iloc[candle_idx]['Volume']
                    
                    return {
                        'type': 'bearish',
                        'top': ob_top,
                        'bottom': ob_bottom,
                        'volume': ob_volume,
                        'index': candle_idx,
                        'timestamp': df.index[candle_idx],
                        'breached': False
                    }
            return None
        except:
            return None
    
    def _validate_ob_size(self, ob: Dict, atr: float) -> bool:
        """Validate order block size is not too large"""
        if pd.isna(atr) or atr == 0:
            return True
        
        ob_size = abs(ob['top'] - ob['bottom'])
        return ob_size <= (atr * self.max_atr_mult)
    
    def _filter_order_blocks(self, obs: List[Dict], df: pd.DataFrame, ob_type: str) -> List[Dict]:
        """Filter and mark breached order blocks"""
        current_price = df.iloc[-1]['Close']
        
        filtered_obs = []
        for ob in obs:
            # Check if OB has been breached
            if ob_type == 'bullish':
                ob['breached'] = current_price < ob['bottom']
            else:
                ob['breached'] = current_price > ob['top']
            
            filtered_obs.append(ob)
        
        return filtered_obs


class FVGDetector:
    """Fair Value Gap (FVG) and Inversion FVG detector"""
    
    def detect_fvg(self, df: pd.DataFrame) -> Dict:
        """
        Detect Fair Value Gaps
        Returns: Dict with bullish_fvg and bearish_fvg lists
        """
        bullish_fvg = []
        bearish_fvg = []
        
        for i in range(3, len(df)):
            # Bullish FVG: gap between candle[i-3] high and candle[i-1] low
            if df.iloc[i-3]['High'] < df.iloc[i-1]['Low']:
                fvg = {
                    'type': 'bullish',
                    'top': df.iloc[i-1]['Low'],
                    'bottom': df.iloc[i-3]['High'],
                    'index': i-2,
                    'timestamp': df.index[i-2],
                    'filled': False
                }
                
                # Check if filled
                for j in range(i, len(df)):
                    if df.iloc[j]['Low'] <= fvg['top']:
                        fvg['filled'] = True
                        break
                
                bullish_fvg.append(fvg)
            
            # Bearish FVG: gap between candle[i-3] low and candle[i-1] high
            if df.iloc[i-3]['Low'] > df.iloc[i-1]['High']:
                fvg = {
                    'type': 'bearish',
                    'top': df.iloc[i-3]['Low'],
                    'bottom': df.iloc[i-1]['High'],
                    'index': i-2,
                    'timestamp': df.index[i-2],
                    'filled': False
                }
                
                # Check if filled
                for j in range(i, len(df)):
                    if df.iloc[j]['High'] >= fvg['bottom']:
                        fvg['filled'] = True
                        break
                
                bearish_fvg.append(fvg)
        
        return {
            'bullish': bullish_fvg[-5:],
            'bearish': bearish_fvg[-5:]
        }


class SupportResistanceDetector:
    """Detect Support and Resistance levels based on pivot points"""
    
    def __init__(self, left_bars: int = 10, right_bars: int = 5):
        self.left_bars = left_bars
        self.right_bars = right_bars
    
    def detect_levels(self, df: pd.DataFrame) -> Dict:
        """
        Detect support and resistance levels
        Returns: Dict with support and resistance levels
        """
        resistance_levels = []
        support_levels = []
        
        # Find pivot highs and lows
        for i in range(self.left_bars, len(df) - self.right_bars):
            # Pivot High (Resistance)
            if self._is_pivot_high(df, i):
                resistance_levels.append({
                    'price': df.iloc[i]['High'],
                    'index': i,
                    'timestamp': df.index[i],
                    'touches': 1
                })
            
            # Pivot Low (Support)
            if self._is_pivot_low(df, i):
                support_levels.append({
                    'price': df.iloc[i]['Low'],
                    'index': i,
                    'timestamp': df.index[i],
                    'touches': 1
                })
        
        # Cluster similar levels
        resistance_levels = self._cluster_levels(resistance_levels)
        support_levels = self._cluster_levels(support_levels)
        
        # Sort by strength (touches)
        resistance_levels.sort(key=lambda x: x['touches'], reverse=True)
        support_levels.sort(key=lambda x: x['touches'], reverse=True)
        
        return {
            'resistance': resistance_levels[:5],
            'support': support_levels[:5]
        }
    
    def _is_pivot_high(self, df: pd.DataFrame, i: int) -> bool:
        """Check if index i is a pivot high"""
        high = df.iloc[i]['High']
        
        left_highs = df.iloc[i-self.left_bars:i]['High']
        right_highs = df.iloc[i+1:i+self.right_bars+1]['High']
        
        return (high >= left_highs.max()) and (high >= right_highs.max())
    
    def _is_pivot_low(self, df: pd.DataFrame, i: int) -> bool:
        """Check if index i is a pivot low"""
        low = df.iloc[i]['Low']
        
        left_lows = df.iloc[i-self.left_bars:i]['Low']
        right_lows = df.iloc[i+1:i+self.right_bars+1]['Low']
        
        return (low <= left_lows.min()) and (low <= right_lows.min())
    
    def _cluster_levels(self, levels: List[Dict], threshold: float = 0.002) -> List[Dict]:
        """Cluster similar price levels together"""
        if not levels:
            return []
        
        clustered = []
        used = set()
        
        for i, level in enumerate(levels):
            if i in used:
                continue
            
            cluster = [level]
            used.add(i)
            
            for j, other_level in enumerate(levels[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if prices are within threshold
                price_diff = abs(level['price'] - other_level['price']) / level['price']
                if price_diff < threshold:
                    cluster.append(other_level)
                    used.add(j)
            
            # Average the cluster
            avg_price = np.mean([l['price'] for l in cluster])
            touches = len(cluster)
            
            clustered.append({
                'price': avg_price,
                'touches': touches,
                'timestamp': cluster[0]['timestamp']
            })
        
        return clustered


class FibonacciCalculator:
    """Calculate Fibonacci retracement levels"""
    
    @staticmethod
    def calculate_levels(df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Calculate Fibonacci levels based on recent swing high/low
        """
        recent_data = df.tail(lookback)
        
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        
        diff = swing_high - swing_low
        
        # Determine trend
        current_price = df.iloc[-1]['Close']
        trend = 'uptrend' if current_price > (swing_high + swing_low) / 2 else 'downtrend'
        
        if trend == 'uptrend':
            # Retracement from swing low to swing high
            levels = {
                '0.0': swing_high,
                '0.236': swing_high - (diff * 0.236),
                '0.382': swing_high - (diff * 0.382),
                '0.5': swing_high - (diff * 0.5),
                '0.618': swing_high - (diff * 0.618),
                '0.786': swing_high - (diff * 0.786),
                '1.0': swing_low
            }
        else:
            # Retracement from swing high to swing low
            levels = {
                '0.0': swing_low,
                '0.236': swing_low + (diff * 0.236),
                '0.382': swing_low + (diff * 0.382),
                '0.5': swing_low + (diff * 0.5),
                '0.618': swing_low + (diff * 0.618),
                '0.786': swing_low + (diff * 0.786),
                '1.0': swing_high
            }
        
        return {
            'trend': trend,
            'swing_high': swing_high,
            'swing_low': swing_low,
            'levels': levels
        }


class TechnicalIndicators:
    """Calculate standard technical indicators"""
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # VWAP
        df['VWAP'] = TechnicalIndicators.calculate_vwap(df)
        
        # RSI
        df['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], 14)
        
        # MACD
        macd_data = TechnicalIndicators.calculate_macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Hist'] = macd_data['histogram']
        
        # Bollinger Bands
        bb = TechnicalIndicators.calculate_bollinger_bands(df['Close'], 20, 2)
        df['BB_Upper'] = bb['upper']
        df['BB_Middle'] = bb['middle']
        df['BB_Lower'] = bb['lower']
        
        # ATR
        df['ATR'] = TechnicalIndicators.calculate_atr(df, 14)
        
        # ADX
        adx_data = TechnicalIndicators.calculate_adx(df)
        df = pd.concat([df, adx_data], axis=1)
        
        # Stochastic
        stoch_data = TechnicalIndicators.calculate_stochastic(df)
        df = pd.concat([df, stoch_data], axis=1)
        
        # Supertrend
        st_data = TechnicalIndicators.calculate_supertrend(df)
        df = pd.concat([df, st_data], axis=1)
        
        # Ichimoku
        ichimoku_data = TechnicalIndicators.calculate_ichimoku(df)
        df = pd.concat([df, ichimoku_data], axis=1)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9) -> Dict:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)"""
        v = df['Volume']
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        return (tp * v).cumsum() / v.cumsum()
        
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        minus_dm = -minus_dm
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        diff = plus_dm - minus_dm
        plus_dm[diff < 0] = 0
        minus_dm[diff > 0] = 0
        
        atr = TechnicalIndicators.calculate_atr(df, period)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di
        })

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'Stoch_K': k,
            'Stoch_D': d
        })
        
    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate Supertrend"""
        atr = TechnicalIndicators.calculate_atr(df, period)
        
        hl2 = (df['High'] + df['Low']) / 2
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        # Initialize final bands
        final_upperband = pd.Series(index=df.index, dtype='float64')
        final_lowerband = pd.Series(index=df.index, dtype='float64')
        trend = pd.Series(index=df.index, dtype='int64')
        
        # We need to iterate to calculate supertrend correctly as it depends on previous values
        # Initialize with first values
        final_upperband.iloc[0] = basic_upperband.iloc[0]
        final_lowerband.iloc[0] = basic_lowerband.iloc[0]
        trend.iloc[0] = 1
        
        columns = ['Close']
        # Convert to numpy for faster iteration
        close = df['Close'].values
        bu = basic_upperband.values
        bl = basic_lowerband.values
        fu = np.zeros(len(df))
        fl = np.zeros(len(df))
        tr = np.zeros(len(df))
        
        # Initialize
        fu[0] = bu[0]
        fl[0] = bl[0]
        tr[0] = 1
        
        for i in range(1, len(df)):
            # Final Upper Band
            if bu[i] < fu[i-1] or close[i-1] > fu[i-1]:
                fu[i] = bu[i]
            else:
                fu[i] = fu[i-1]
                
            # Final Lower Band
            if bl[i] > fl[i-1] or close[i-1] < fl[i-1]:
                fl[i] = bl[i]
            else:
                fl[i] = fl[i-1]
                
            # Trend
            if tr[i-1] == 1:
                if close[i] <= fl[i]:
                    tr[i] = -1
                else:
                    tr[i] = 1
            else:
                if close[i] >= fu[i]:
                    tr[i] = 1
                else:
                    tr[i] = -1
                    
        return pd.DataFrame({
            'Supertrend': np.where(tr == 1, fl, fu),
            'Supertrend_Direction': tr
        }, index=df.index)

    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        nine_period_high = df['High'].rolling(window=9).max()
        nine_period_low = df['Low'].rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        twenty_six_period_high = df['High'].rolling(window=26).max()
        twenty_six_period_low = df['Low'].rolling(window=26).min()
        kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        fifty_two_period_high = df['High'].rolling(window=52).max()
        fifty_two_period_low = df['Low'].rolling(window=52).min()
        senkou_span_b = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou_span = df['Close'].shift(-26)
        
        return pd.DataFrame({
            'Tenkan_sen': tenkan_sen,
            'Kijun_sen': kijun_sen,
            'Senkou_span_a': senkou_span_a,
            'Senkou_span_b': senkou_span_b,
            'Chikou_span': chikou_span
        })


class SmartMoneyConcepts:
    """Smart Money Concepts Indicators (LuxAlgo style)"""
    
    @staticmethod
    def get_market_structure(df: pd.DataFrame, swing_length: int = 50) -> Dict:
        """
        Detect Swing Highs/Lows and Market Structure (BOS/CHoCH)
        """
        df = df.copy()
        highs = df['High'].values
        lows = df['Low'].values
        close = df['Close'].values
        
        structure = []
        swing_highs = []
        swing_lows = []
        
        # Simple pivot detection
        for i in range(swing_length, len(df) - swing_length):
            # Swing High
            if highs[i] == max(highs[i-swing_length:i+swing_length+1]):
                swing_highs.append({
                    'index': i,
                    'price': highs[i],
                    'type': 'HH' # Placeholder, will update based on trend
                })
                
            # Swing Low
            if lows[i] == min(lows[i-swing_length:i+swing_length+1]):
                swing_lows.append({
                    'index': i,
                    'price': lows[i],
                    'type': 'LL' # Placeholder
                })
        
        # Determine Trend and Structure Breaks
        trend = 0 # 1 Bullish, -1 Bearish
        
        # We need to iterate through time
        # This is a simplified version of BOS/CHoCH detection
        last_high = swing_highs[0] if swing_highs else None
        last_low = swing_lows[0] if swing_lows else None
        
        bos_choch = []
        
        # Find breaks
        if last_high and last_low:
            for i in range(max(last_high['index'], last_low['index']), len(df)):
                c = close[i]
                
                # Break of Structure (Bullish) - simplified
                if last_high and c > last_high['price']:
                    if trend == 1:
                        type_str = 'BOS'
                    else:
                        type_str = 'CHoCH'
                        trend = 1
                    
                    bos_choch.append({
                        'index': i,
                        'price': last_high['price'],
                        'type': 'Bullish ' + type_str,
                        'timestamp': df.index[i]
                    })
                    # Update high to avoid repeated signals (naive)
                    last_high = {'price': float('inf'), 'index': i} 

                # Break of Structure (Bearish)
                if last_low and c < last_low['price']:
                    if trend == -1:
                        type_str = 'BOS'
                    else:
                        type_str = 'CHoCH'
                        trend = -1
                        
                    bos_choch.append({
                        'index': i,
                        'price': last_low['price'],
                        'type': 'Bearish ' + type_str,
                        'timestamp': df.index[i]
                    })
                    last_low = {'price': float('-inf'), 'index': i}
        
        return {
            'swing_highs': swing_highs[-5:],
            'swing_lows': swing_lows[-5:],
            'structure_breaks': bos_choch[-5:],
            'current_trend': 'Bullish' if trend == 1 else 'Bearish' if trend == -1 else 'Neutral'
        }

    @staticmethod
    def detect_equal_highs_lows(df: pd.DataFrame, threshold: float = 0.05, len_bars: int = 3) -> Dict:
        """
        Detect Equal Highs (EQH) and Equal Lows (EQL)
        """
        eqh = []
        eql = []
        
        # Look at pivots
        for i in range(5, len(df) - 5):
            curr_high = df['High'].iloc[i]
            curr_low = df['Low'].iloc[i]
            
            # Check for previous similar highs
            for j in range(i - 20, i - len_bars):
                if j < 0: continue
                prev_high = df['High'].iloc[j]
                
                if abs(curr_high - prev_high) / curr_high * 100 < threshold:
                     # Check if they are pivots (local max)
                     if curr_high > df['High'].iloc[i-1] and curr_high > df['High'].iloc[i+1]:
                         eqh.append({'price': curr_high, 'index': i, 'timestamp': df.index[i]})
                         
            # Check for similar lows
            for j in range(i - 20, i - len_bars):
                if j < 0: continue
                prev_low = df['Low'].iloc[j]
                
                if abs(curr_low - prev_low) / curr_low * 100 < threshold:
                     if curr_low < df['Low'].iloc[i-1] and curr_low < df['Low'].iloc[i+1]:
                         eql.append({'price': curr_low, 'index': i, 'timestamp': df.index[i]})
                         
        return {'eqh': eqh[-3:], 'eql': eql[-3:]}

    @staticmethod
    def calculate_premium_discount_zones(df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Calculate Premium, Discount, and Equilibrium zones
        """
        recent = df.tail(lookback)
        high = recent['High'].max()
        low = recent['Low'].min()
        mid = (high + low) / 2
        
        return {
            'range_high': high,
            'range_low': low,
            'equilibrium': mid,
            'premium_zone': {'min': mid, 'max': high},
            'discount_zone': {'min': low, 'max': mid}
        }


class QuantumIndicators:
    """
    Quantum Fibonacci ATR/ADR Levels & Targets (Ported from Pine Script)
    """
    @staticmethod
    def calculate_levels(df_daily: pd.DataFrame, current_price: float = None) -> Dict:
        """
        Calculate Quantum Levels based on Daily Data
        
        Args:
            df_daily: DataFrame containing Daily OHLCV data
            current_price: Optional current price (unused for levels calculation but good for context)
            
        Returns:
            Dict containing level values
        """
        df = df_daily.copy()
        
        # Inputs
        length = 18
        entry_ratio = 0.618
        target_ratio_1 = 1.618
        target_ratio_2 = 2.618
        target_ratio_3 = 3.618
        
        # Ensure we have enough data
        if len(df) < length + 1:
            return {}
            
        # Metrics Calculation on Daily Data
        # EMA High/Low
        df['EMA_High'] = df['High'].ewm(span=length, adjust=False).mean()
        df['EMA_Low'] = df['Low'].ewm(span=length, adjust=False).mean()
        
        # ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        # RMA (Running Moving Average) for ATR in Pine Script is effectively EWM with alpha=1/length
        df['ATR'] = df['TR'].ewm(alpha=1/length, adjust=False).mean()
        
        # Get values for calculation
        # We need "Current Day" and "Previous Day"
        # Since we are likely looking at the latest incomplete day or the last complete day
        # Let's assume the last row is the "Current Day" (which might be forming)
        
        curr_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        cd_open = curr_row['Open']
        pd_close = prev_row['Close']
        
        # ADR (Average Daily Range)
        # Script: adr := high_req - low_req (where req is Daily EMA High/Low)
        # Note: Script uses getNonRepaintingSecurity.
        high_req = curr_row['EMA_High']
        low_req = curr_row['EMA_Low']
        adr = high_req - low_req
        
        atr = curr_row['ATR']
        
        # Level Calculations
        adr_resistance_avg = (cd_open + adr / 2 + pd_close + adr / 2) / 2
        adr_support_avg = (cd_open - adr / 2 + pd_close - adr / 2) / 2
        
        atr_resistance_avg = (cd_open + atr / 2 + pd_close + atr / 2) / 2
        atr_support_avg = (cd_open - atr / 2 + pd_close - atr / 2) / 2
        
        # Stop Loss Levels
        buy_stop_loss_avg = (atr_support_avg + adr_support_avg) / 2
        sell_stop_loss_avg = (atr_resistance_avg + adr_resistance_avg) / 2
        
        # Entry Levels
        # buy_entry_level := math.avg(...) + entryRatio * (diff)
        avg_sl = (buy_stop_loss_avg + sell_stop_loss_avg) / 2
        diff_sl = sell_stop_loss_avg - avg_sl
        
        buy_entry_level = avg_sl + entry_ratio * diff_sl
        # sell_entry_level := avg - entryRatio * (avg - buy_sl)
        # Note: (avg - buy_sl) is same as diff_sl if symmetrical
        sell_entry_level = avg_sl - entry_ratio * diff_sl
        
        # Targets
        buy_target1 = avg_sl + target_ratio_1 * diff_sl
        sell_target1 = avg_sl - target_ratio_1 * diff_sl
        
        buy_target2 = avg_sl + target_ratio_2 * diff_sl
        sell_target2 = avg_sl - target_ratio_2 * diff_sl
        
        buy_target3 = avg_sl + target_ratio_3 * diff_sl
        sell_target3 = avg_sl - target_ratio_3 * diff_sl
        
        mid_line = (buy_entry_level + sell_entry_level) / 2
        
        return {
            'buy_entry': buy_entry_level,
            'sell_entry': sell_entry_level,
            'buy_sl': buy_stop_loss_avg,
            'sell_sl': sell_stop_loss_avg,
            'buy_target1': buy_target1,
            'sell_target1': sell_target1,
            'buy_target2': buy_target2,
            'sell_target2': sell_target2,
            'buy_target3': buy_target3,
            'sell_target3': sell_target3,
            'mid_line': mid_line,
            'ema_high': curr_row['EMA_High'],
            'ema_low': curr_row['EMA_Low']
        }