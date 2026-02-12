"""
Elite Trading System - Multi-Timeframe Analyzer
Analyzes multiple timeframes and generates consensus signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from indicators import (
    OrderBlockDetector,
    FVGDetector,
    SupportResistanceDetector,
    FibonacciCalculator,
    SupportResistanceDetector,
    FibonacciCalculator,
    TechnicalIndicators,
    SmartMoneyConcepts
)

class MultiTimeframeAnalyzer:
    """Analyze multiple timeframes and generate consensus signals"""
    
    # Timeframe weights for consensus
    TIMEFRAME_WEIGHTS = {
        '1m': 0.05,
        '3m': 0.05,
        '5m': 0.10,
        '15m': 0.15,
        '30m': 0.15,
        '1h': 0.20,
        '4h': 0.20,
        '1d': 0.10
    }
    
    def __init__(self):
        self.ob_detector = OrderBlockDetector()
        self.fvg_detector = FVGDetector()
        self.sr_detector = SupportResistanceDetector()
    
    def analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Analyze a single timeframe
        
        Returns:
            Dict with analysis results
        """
        if df is None or df.empty or len(df) < 50:
            return {
                'signal': 'NEUTRAL',
                'score': 0,
                'confidence': 0,
                'reasons': ['Insufficient data']
            }
        
        # Calculate technical indicators
        df = TechnicalIndicators.calculate_all(df)
        
        # Detect patterns
        order_blocks = self.ob_detector.detect_order_blocks(df)
        fvgs = self.fvg_detector.detect_fvg(df)
        sr_levels = self.sr_detector.detect_levels(df)
        sr_levels = self.sr_detector.detect_levels(df)
        fib_levels = FibonacciCalculator.calculate_levels(df)
        
        # Smart Money Concepts
        market_structure = SmartMoneyConcepts.get_market_structure(df)
        pd_zones = SmartMoneyConcepts.calculate_premium_discount_zones(df)
        eq_highs_lows = SmartMoneyConcepts.detect_equal_highs_lows(df)
        
        # Current market state
        current_price = df.iloc[-1]['Close']
        latest = df.iloc[-1]
        
        # Generate signal components
        signals = []
        reasons = []
        
        # 1. Order Block Analysis
        ob_signal = self._analyze_order_blocks(order_blocks, current_price)
        signals.append(ob_signal['score'])
        if ob_signal['reason']:
            reasons.append(ob_signal['reason'])
        
        # 2. FVG Analysis
        fvg_signal = self._analyze_fvg(fvgs, current_price)
        signals.append(fvg_signal['score'])
        if fvg_signal['reason']:
            reasons.append(fvg_signal['reason'])
        
        # 3. Support/Resistance Analysis
        sr_signal = self._analyze_sr_levels(sr_levels, current_price)
        signals.append(sr_signal['score'])
        if sr_signal['reason']:
            reasons.append(sr_signal['reason'])
        
        # 4. Fibonacci Analysis
        fib_signal = self._analyze_fibonacci(fib_levels, current_price)
        signals.append(fib_signal['score'])
        if fib_signal['reason']:
            reasons.append(fib_signal['reason'])
            
        # 5. SMC Analysis
        smc_signal = self._analyze_smc(market_structure, pd_zones, eq_highs_lows, current_price)
        signals.append(smc_signal['score'])
        if smc_signal['reason']:
            reasons.append(smc_signal['reason'])
        
        # 6. Trend Analysis (Enhanced)
        trend_signal = self._analyze_trend(latest, df)
        signals.append(trend_signal['score'])
        if trend_signal['reason']:
            reasons.append(trend_signal['reason'])
        
        if trend_signal['reason']:
            reasons.append(trend_signal['reason'])
        
        # 7. Momentum Analysis (Enhanced)
        momentum_signal = self._analyze_momentum(latest)
        signals.append(momentum_signal['score'])
        if momentum_signal['reason']:
            reasons.append(momentum_signal['reason'])
            
        if momentum_signal['reason']:
            reasons.append(momentum_signal['reason'])
            
        # 8. Volume & VWAP Analysis
        volume_signal = self._analyze_volume_vwap(latest, df)
        signals.append(volume_signal['score'])
        if volume_signal['reason']:
            reasons.append(volume_signal['reason'])
        
        # Calculate overall score
        total_score = sum(signals)
        max_possible = len(signals) * 2
        normalized_score = (total_score / max_possible) * 100
        
        # Determine signal
        if normalized_score >= 40:
            signal = 'STRONG BUY'
        elif normalized_score >= 20:
            signal = 'BUY'
        elif normalized_score >= -20:
            signal = 'NEUTRAL'
        elif normalized_score >= -40:
            signal = 'SELL'
        else:
            signal = 'STRONG SELL'
        
        # Calculate confidence (0-100)
        confidence = min(100, abs(normalized_score))
        
        return {
            'timeframe': timeframe,
            'signal': signal,
            'score': normalized_score,
            'confidence': confidence,
            'reasons': reasons,
            'data': {
                'order_blocks': order_blocks,
                'fvgs': fvgs,
                'sr_levels': sr_levels,
                'sr_levels': sr_levels,
                'fib_levels': fib_levels,
                'market_structure': market_structure,
                'pd_zones': pd_zones,
                'eq_highs_lows': eq_highs_lows,
                'indicators': {
                    'rsi': latest.get('RSI'),
                    'macd': latest.get('MACD'),
                    'ema_9': latest.get('EMA_9'),
                    'ema_21': latest.get('EMA_21'),
                    'adx': latest.get('ADX'),
                    'stoch_k': latest.get('Stoch_K'),
                    'supertrend': latest.get('Supertrend'),
                    'vwap': latest.get('VWAP')
                }
            }
        }
    
    def _analyze_order_blocks(self, obs: Dict, current_price: float) -> Dict:
        """Analyze order block signals"""
        score = 0
        reason = None
        
        bullish_obs = [ob for ob in obs.get('bullish', []) if not ob['breached']]
        bearish_obs = [ob for ob in obs.get('bearish', []) if not ob['breached']]
        
        # Check if price is at order block
        for ob in bullish_obs:
            if ob['bottom'] <= current_price <= ob['top']:
                score += 2
                reason = f"Price at Bullish Order Block ({ob['bottom']:.2f}-{ob['top']:.2f})"
                break
            elif current_price > ob['top'] and (current_price - ob['top']) / current_price < 0.02:
                score += 1
                reason = f"Price near Bullish Order Block"
        
        for ob in bearish_obs:
            if ob['bottom'] <= current_price <= ob['top']:
                score -= 2
                reason = f"Price at Bearish Order Block ({ob['bottom']:.2f}-{ob['top']:.2f})"
                break
            elif current_price < ob['bottom'] and (ob['bottom'] - current_price) / current_price < 0.02:
                score -= 1
                reason = f"Price near Bearish Order Block"
        
        return {'score': score, 'reason': reason}
    
    def _analyze_fvg(self, fvgs: Dict, current_price: float) -> Dict:
        """Analyze Fair Value Gap signals"""
        score = 0
        reason = None
        
        bullish_fvgs = [fvg for fvg in fvgs.get('bullish', []) if not fvg['filled']]
        bearish_fvgs = [fvg for fvg in fvgs.get('bearish', []) if not fvg['filled']]
        
        # Check if price is in FVG
        for fvg in bullish_fvgs:
            if fvg['bottom'] <= current_price <= fvg['top']:
                score += 1.5
                reason = f"Price in Bullish FVG ({fvg['bottom']:.2f}-{fvg['top']:.2f})"
                break
        
        for fvg in bearish_fvgs:
            if fvg['bottom'] <= current_price <= fvg['top']:
                score -= 1.5
                reason = f"Price in Bearish FVG ({fvg['bottom']:.2f}-{fvg['top']:.2f})"
                break
        
        return {'score': score, 'reason': reason}
    
    def _analyze_sr_levels(self, levels: Dict, current_price: float) -> Dict:
        """Analyze Support/Resistance levels"""
        score = 0
        reason = None
        
        support_levels = levels.get('support', [])
        resistance_levels = levels.get('resistance', [])
        
        # Check proximity to S/R (within 1%)
        threshold = 0.01
        
        for support in support_levels[:3]:  # Top 3 support levels
            price_diff = (current_price - support['price']) / current_price
            if abs(price_diff) < threshold:
                score += 1
                reason = f"Price at Support ({support['price']:.2f})"
                break
        
        for resistance in resistance_levels[:3]:  # Top 3 resistance levels
            price_diff = (resistance['price'] - current_price) / current_price
            if abs(price_diff) < threshold:
                score -= 1
                reason = f"Price at Resistance ({resistance['price']:.2f})"
                break
        
        return {'score': score, 'reason': reason}
    
    def _analyze_fibonacci(self, fib: Dict, current_price: float) -> Dict:
        """Analyze Fibonacci retracement levels"""
        score = 0
        reason = None
        
        levels = fib.get('levels', {})
        trend = fib.get('trend', 'neutral')
        
        # Check if price is at key Fibonacci level (0.382, 0.5, 0.618, 0.786)
        threshold = 0.005  # 0.5% threshold
        
        for level_name, level_price in levels.items():
            if level_name in ['0.382', '0.5', '0.618', '0.786']:
                price_diff = abs(current_price - level_price) / current_price
                
                if price_diff < threshold:
                    if trend == 'uptrend':
                        score += 1
                        reason = f"Price at Fib {level_name} in uptrend ({level_price:.2f})"
                    elif trend == 'downtrend':
                        score -= 1
                        reason = f"Price at Fib {level_name} in downtrend ({level_price:.2f})"
                    break
        
        return {'score': score, 'reason': reason}
    
    def _analyze_smc(self, structure: Dict, zones: Dict, eq_hl: Dict, current_price: float) -> Dict:
        """Analyze Smart Money Concepts"""
        score = 0
        reasons = []
        
        # 1. Market Structure
        current_trend = structure.get('current_trend', 'Neutral')
        if current_trend == 'Bullish':
            score += 1.5
            reasons.append("Structure Bullish (BOS)")
        elif current_trend == 'Bearish':
            score -= 1.5
            reasons.append("Structure Bearish (BOS)")
        
        # 2. Premium/Discount Zones
        # Buy in Discount, Sell in Premium
        discount_zone = zones.get('discount_zone', {})
        premium_zone = zones.get('premium_zone', {})
        
        if discount_zone and discount_zone['min'] <= current_price <= discount_zone['max']:
            score += 1
            reasons.append("Price in Discount Zone")
        elif premium_zone and premium_zone['min'] <= current_price <= premium_zone['max']:
            score -= 1
            reasons.append("Price in Premium Zone")
            
        # 3. Equal Highs/Lows (Liquidity Targets)
        # If we are bullish and see Equal Highs above, that's a magnet (Bullish) factor? 
        # Actually LuxAlgo logic uses them as targets. 
        # Simple logic: EQH is resistance/magnet, EQL is support/magnet.
        # If price is approaching EQH from below -> Bullish Target.
        
        eqh = eq_hl.get('eqh', [])
        eql = eq_hl.get('eql', [])
        
        if eqh:
            # Check if any EQH is close above current price
            for h in eqh:
                if h['price'] > current_price and (h['price'] - current_price)/current_price < 0.02:
                     score += 0.5
                     reasons.append("Targeting Equal Highs (Liquidity)")
                     break
                     
        if eql:
            for l in eql:
                if l['price'] < current_price and (current_price - l['price'])/current_price < 0.02:
                    score -= 0.5
                    reasons.append("Targeting Equal Lows (Liquidity)")
                    break
        
        return {'score': score, 'reason': ', '.join(reasons) if reasons else None}
    
    def _analyze_trend(self, latest: pd.Series, df: pd.DataFrame = None) -> Dict:
        """Analyze trend using moving averages, Supertrend, and Ichimoku"""
        score = 0
        reasons = []
        
        close = latest.get('Close', 0)
        ema_9 = latest.get('EMA_9')
        ema_21 = latest.get('EMA_21')
        sma_200 = latest.get('SMA_200')
        supertrend_dir = latest.get('Supertrend_Direction')
        
        # Ichimoku
        tenkan = latest.get('Tenkan_sen')
        kijun = latest.get('Kijun_sen')
        cloud_top = max(latest.get('Senkou_span_a', 0), latest.get('Senkou_span_b', 0))
        cloud_bottom = min(latest.get('Senkou_span_a', 0), latest.get('Senkou_span_b', 0))
        
        # EMA alignment
        if pd.notna(ema_9) and pd.notna(ema_21):
            if ema_9 > ema_21 and close > ema_9:
                score += 1
                reasons.append("Bullish EMA alignment")
            elif ema_9 < ema_21 and close < ema_9:
                score -= 1
                reasons.append("Bearish EMA alignment")
        
        # Supertrend
        if pd.notna(supertrend_dir):
            if supertrend_dir == 1:
                score += 1
                reasons.append("Supertrend Bullish")
            elif supertrend_dir == -1:
                score -= 1
                reasons.append("Supertrend Bearish")
                
        # Ichimoku Cloud
        if pd.notna(cloud_top) and pd.notna(cloud_bottom):
            if close > cloud_top:
                score += 0.5
                reasons.append("Price above Cloud")
            elif close < cloud_bottom:
                score -= 0.5
                reasons.append("Price below Cloud")
                
        # TK Cross (Tenkan/Kijun)
        if pd.notna(tenkan) and pd.notna(kijun):
            if tenkan > kijun:
                score += 0.5
                reasons.append("TK Bullish Cross")
            elif tenkan < kijun:
                score -= 0.5
                reasons.append("TK Bearish Cross")
        
        # Long-term trend
        if pd.notna(sma_200):
            if close > sma_200:
                score += 0.5
                reasons.append("Above 200 SMA")
            else:
                score -= 0.5
                reasons.append("Below 200 SMA")
        
        return {'score': score, 'reason': ', '.join(reasons) if reasons else None}
    
    def _analyze_momentum(self, latest: pd.Series) -> Dict:
        """Analyze momentum using RSI, MACD, Stochastic, and ADX"""
        score = 0
        reasons = []
        
        rsi = latest.get('RSI')
        macd = latest.get('MACD')
        macd_signal = latest.get('MACD_Signal')
        stoch_k = latest.get('Stoch_K')
        stoch_d = latest.get('Stoch_D')
        adx = latest.get('ADX')
        
        # RSI analysis
        if pd.notna(rsi):
            if rsi < 30:
                score += 1.5
                reasons.append(f"RSI Oversold ({rsi:.1f})")
            elif rsi > 70:
                score -= 1.5
                reasons.append(f"RSI Overbought ({rsi:.1f})")
            elif 40 <= rsi <= 60:
                pass # Neutral
        
        # MACD analysis
        if pd.notna(macd) and pd.notna(macd_signal):
            if macd > macd_signal and macd > 0:
                score += 1
                reasons.append("MACD Bullish")
            elif macd < macd_signal and macd < 0:
                score -= 1
                reasons.append("MACD Bearish")
                
        # Stochastic
        if pd.notna(stoch_k) and pd.notna(stoch_d):
            if stoch_k < 20 and stoch_k > stoch_d:
                score += 1
                reasons.append("Stoch Oversold/Bullish")
            elif stoch_k > 80 and stoch_k < stoch_d:
                score -= 1
                reasons.append("Stoch Overbought/Bearish")
                
        # ADX (Trend Strength)
        if pd.notna(adx) and adx > 25:
            # ADX confirms the trend strength, multiply score slightly?
            # Or just add a small score if trend is strong
            reasons.append(f"Strong Trend (ADX {adx:.1f})")
        
        return {'score': score, 'reason': ', '.join(reasons) if reasons else None}

    def _analyze_volume_vwap(self, latest: pd.Series, df: pd.DataFrame) -> Dict:
        """Analyze Volume and VWAP"""
        score = 0
        reasons = []
        
        close = latest.get('Close')
        vwap = latest.get('VWAP')
        volume_ratio = latest.get('Volume_Ratio')
        
        # VWAP
        if pd.notna(vwap):
            if close > vwap:
                score += 1
                reasons.append("Price above VWAP")
            else:
                score -= 1
                reasons.append("Price below VWAP")
                
        # High Volume
        if pd.notna(volume_ratio) and volume_ratio > 1.5:
             # Volume typically confirms trend
             # But we need to know the candle color
             if close > latest.get('Open'):
                 score += 0.5
                 reasons.append("High Buying Volume")
             else:
                 score -= 0.5
                 reasons.append("High Selling Volume")
                 
        return {'score': score, 'reason': ', '.join(reasons) if reasons else None}
    
    def analyze_multiple_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze multiple timeframes and generate consensus
        
        Args:
            data_dict: Dict with timeframe as key, DataFrame as value
        
        Returns:
            Dict with consensus signal and individual timeframe analysis
        """
        timeframe_analyses = {}
        weighted_scores = []
        
        # Analyze each timeframe
        for timeframe, df in data_dict.items():
            analysis = self.analyze_single_timeframe(df, timeframe)
            timeframe_analyses[timeframe] = analysis
            
            # Add weighted score
            weight = self.TIMEFRAME_WEIGHTS.get(timeframe, 0.10)
            weighted_scores.append(analysis['score'] * weight)
        
        # Calculate consensus
        if not weighted_scores:
            return {
                'consensus_signal': 'NEUTRAL',
                'consensus_score': 0,
                'confidence': 0,
                'timeframe_analyses': {},
                'summary': 'No data available'
            }
        
        consensus_score = sum(weighted_scores)
        
        # Determine consensus signal
        if consensus_score >= 40:
            consensus_signal = 'STRONG BUY'
        elif consensus_score >= 20:
            consensus_signal = 'BUY'
        elif consensus_score >= -20:
            consensus_signal = 'NEUTRAL'
        elif consensus_score >= -40:
            consensus_signal = 'SELL'
        else:
            consensus_signal = 'STRONG SELL'
        
        # Calculate overall confidence
        confidence = min(100, abs(consensus_score))
        
        # Count agreement
        bullish_count = sum(1 for a in timeframe_analyses.values() if 'BUY' in a['signal'])
        bearish_count = sum(1 for a in timeframe_analyses.values() if 'SELL' in a['signal'])
        neutral_count = sum(1 for a in timeframe_analyses.values() if a['signal'] == 'NEUTRAL')
        total_timeframes = len(timeframe_analyses)
        
        # Generate summary
        agreement_pct = max(bullish_count, bearish_count, neutral_count) / total_timeframes * 100 if total_timeframes > 0 else 0
        
        summary = f"{bullish_count}/{total_timeframes} timeframes bullish, {bearish_count}/{total_timeframes} bearish. "
        summary += f"Agreement: {agreement_pct:.0f}%"
        
        return {
            'consensus_signal': consensus_signal,
            'consensus_score': consensus_score,
            'confidence': confidence,
            'timeframe_analyses': timeframe_analyses,
            'summary': summary,
            'counts': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count,
                'total': total_timeframes
            }
        }
