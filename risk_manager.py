"""
Elite Trading System - Risk Management
Calculates position sizes, stop losses, and take profit levels
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

class RiskManager:
    """Professional risk management system"""
    
    def __init__(self, account_size: float = 10000, risk_per_trade_pct: float = 1.0):
        """
        Initialize Risk Manager
        
        Args:
            account_size: Total account capital
            risk_per_trade_pct: Risk percentage per trade (default 1%)
        """
        self.account_size = account_size
        self.risk_per_trade_pct = risk_per_trade_pct
        self.risk_amount = account_size * (risk_per_trade_pct / 100)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> Dict:
        """
        Calculate position size based on risk
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Dict with position size details
        """
        if entry_price <= 0 or stop_loss <= 0:
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'error': 'Invalid prices'
            }
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return {
                'quantity': 0,
                'position_value': 0,
                'risk_amount': 0,
                'error': 'Stop loss too close to entry'
            }
        
        # Calculate quantity
        quantity = self.risk_amount / risk_per_unit
        position_value = quantity * entry_price
        
        # Validate position size (shouldn't exceed account size)
        if position_value > self.account_size:
            quantity = (self.account_size * 0.95) / entry_price  # Use 95% max
            position_value = quantity * entry_price
        
        return {
            'quantity': round(quantity, 4),
            'position_value': round(position_value, 2),
            'risk_amount': round(self.risk_amount, 2),
            'risk_per_unit': round(risk_per_unit, 2),
            'leverage': round(position_value / self.account_size, 2)
        }
    
    def calculate_stop_loss(self, df: pd.DataFrame, signal: str, multiplier: float = 2.0) -> float:
        """
        Calculate stop loss based on ATR
        
        Args:
            df: DataFrame with price data
            signal: BUY or SELL
            multiplier: ATR multiplier (default 2.0)
        
        Returns:
            Stop loss price
        """
        if df.empty:
            return 0
        
        latest = df.iloc[-1]
        current_price = latest['Close']
        
        # Calculate ATR if not already present
        if 'ATR' not in df.columns:
            from indicators import TechnicalIndicators
            df = TechnicalIndicators.calculate_all(df)
            latest = df.iloc[-1]
        
        atr = latest.get('ATR', current_price * 0.02)  # Fallback to 2% if ATR not available
        
        if pd.isna(atr):
            atr = current_price * 0.02
        
        # Calculate stop loss
        if 'BUY' in signal:
            stop_loss = current_price - (atr * multiplier)
        else:
            stop_loss = current_price + (atr * multiplier)
        
        return round(stop_loss, 2)
    
    def calculate_take_profit_levels(self, entry_price: float, stop_loss: float, 
                                     signal: str, risk_reward_ratios: list = [1.5, 2.5, 3.5]) -> Dict:
        """
        Calculate multiple take profit levels
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            signal: BUY or SELL
            risk_reward_ratios: List of R:R ratios
        
        Returns:
            Dict with TP levels
        """
        risk = abs(entry_price - stop_loss)
        
        take_profits = {}
        
        for i, rr_ratio in enumerate(risk_reward_ratios, 1):
            reward = risk * rr_ratio
            
            if 'BUY' in signal:
                tp_price = entry_price + reward
            else:
                tp_price = entry_price - reward
            
            take_profits[f'TP{i}'] = {
                'price': round(tp_price, 2),
                'rr_ratio': rr_ratio,
                'reward': round(reward, 2),
                'profit_pct': round((reward / entry_price) * 100, 2)
            }
        
        return take_profits
    
    def generate_trade_plan(self, df: pd.DataFrame, signal: str, 
                           current_price: float = None) -> Dict:
        """
        Generate complete trade plan with entry, SL, TP, and position size
        
        Args:
            df: DataFrame with price data
            signal: Trading signal (BUY/SELL)
            current_price: Optional current price (uses latest close if not provided)
        
        Returns:
            Complete trade plan
        """
        if df.empty or signal == 'NEUTRAL':
            return {
                'valid': False,
                'message': 'No valid trade setup'
            }
        
        # Get current price
        if current_price is None:
            current_price = df.iloc[-1]['Close']
        
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(df, signal)
        
        # Calculate take profit levels
        take_profits = self.calculate_take_profit_levels(current_price, stop_loss, signal)
        
        # Calculate position size
        position = self.calculate_position_size(current_price, stop_loss)
        
        # Calculate risk/reward metrics
        risk_amount = abs(current_price - stop_loss) * position['quantity']
        
        tp_rewards = {}
        for tp_name, tp_data in take_profits.items():
            reward_amount = abs(current_price - tp_data['price']) * position['quantity']
            tp_rewards[tp_name] = {
                **tp_data,
                'reward_amount': round(reward_amount, 2)
            }
        
        # Trade direction
        direction = 'LONG' if 'BUY' in signal else 'SHORT'
        
        return {
            'valid': True,
            'direction': direction,
            'signal': signal,
            'entry_price': round(current_price, 2),
            'stop_loss': stop_loss,
            'stop_loss_pct': round(abs(current_price - stop_loss) / current_price * 100, 2),
            'take_profits': tp_rewards,
            'position': position,
            'risk_amount': round(risk_amount, 2),
            'risk_pct': self.risk_per_trade_pct,
            'account_size': self.account_size,
            'trade_summary': self._generate_trade_summary(
                direction, current_price, stop_loss, tp_rewards, position
            )
        }
    
    def _generate_trade_summary(self, direction: str, entry: float, stop_loss: float, 
                                take_profits: Dict, position: Dict) -> str:
        """Generate human-readable trade summary"""
        
        summary = []
        summary.append(f"**{direction} Position**")
        summary.append(f"Entry: ₹{entry:.2f}")
        summary.append(f"Stop Loss: ₹{stop_loss:.2f} ({abs(entry-stop_loss)/entry*100:.1f}% risk)")
        summary.append("")
        summary.append("**Take Profit Levels:**")
        
        for tp_name, tp_data in sorted(take_profits.items()):
            summary.append(f"{tp_name}: ₹{tp_data['price']:.2f} (R:R 1:{tp_data['rr_ratio']}) - {tp_data['profit_pct']:.1f}% gain")
        
        summary.append("")
        summary.append("**Position Details:**")
        summary.append(f"Quantity: {position['quantity']:.4f}")
        summary.append(f"Position Value: ₹{position['position_value']:,.2f}")
        summary.append(f"Risk Amount: ₹{position['risk_amount']:,.2f} ({self.risk_per_trade_pct}% of account)")
        
        return "\n".join(summary)
    
    def calculate_win_rate_analysis(self, trades: list) -> Dict:
        """
        Analyze historical trade performance
        
        Args:
            trades: List of trade results (1 for win, 0 for loss)
        
        Returns:
            Performance metrics
        """
        if not trades:
            return {
                'win_rate': 0,
                'total_trades': 0,
                'wins': 0,
                'losses': 0
            }
        
        total_trades = len(trades)
        wins = sum(trades)
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100
        
        return {
            'win_rate': round(win_rate, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'expectancy': 'Positive' if win_rate > 50 else 'Negative'
        }
    
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Win rate as percentage (0-100)
            avg_win: Average win amount
            avg_loss: Average loss amount
        
        Returns:
            Optimal risk percentage (0-100)
        """
        if avg_loss == 0 or win_rate >= 100 or win_rate <= 0:
            return 1.0  # Default to 1% risk
        
        win_rate_decimal = win_rate / 100
        loss_rate = 1 - win_rate_decimal
        
        win_loss_ratio = avg_win / avg_loss
        
        kelly_pct = (win_rate_decimal * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Apply fractional Kelly (more conservative)
        fractional_kelly = kelly_pct * 0.5  # Use half Kelly
        
        # Limit between 0.5% and 3%
        optimal_risk = max(0.5, min(3.0, fractional_kelly * 100))
        
        return round(optimal_risk, 2)


class PortfolioManager:
    """Manage multiple positions and overall portfolio risk"""
    
    def __init__(self, account_size: float = 10000, max_positions: int = 5):
        self.account_size = account_size
        self.max_positions = max_positions
        self.positions = []
    
    def can_open_position(self) -> bool:
        """Check if new position can be opened"""
        return len(self.positions) < self.max_positions
    
    def add_position(self, trade_plan: Dict):
        """Add a new position"""
        if self.can_open_position():
            self.positions.append(trade_plan)
    
    def get_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        return sum(p['position']['position_value'] for p in self.positions)
    
    def get_total_risk(self) -> float:
        """Calculate total portfolio risk"""
        return sum(p['risk_amount'] for p in self.positions)
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_exposure = self.get_total_exposure()
        total_risk = self.get_total_risk()
        
        return {
            'total_positions': len(self.positions),
            'max_positions': self.max_positions,
            'total_exposure': round(total_exposure, 2),
            'total_risk': round(total_risk, 2),
            'exposure_pct': round((total_exposure / self.account_size) * 100, 2),
            'risk_pct': round((total_risk / self.account_size) * 100, 2)
        }
