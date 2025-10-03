import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(
        self,
        mt5_handler,
        max_risk_per_trade: float = 1.0,  # % of balance
        max_daily_loss: float = 5.0,      # % of balance
        max_trades_per_day: int = 10,
        max_open_positions: int = 3,
        max_correlation_exposure: float = 0.7
    ):
        self.mt5 = mt5_handler
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_day = max_trades_per_day
        self.max_open_positions = max_open_positions
        self.max_correlation_exposure = max_correlation_exposure
        
        self.daily_stats = {
            'trades_count': 0,
            'total_pnl': 0.0,
            'last_reset': datetime.now().date()
        }
        
        self.trade_history = []
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        today = datetime.now().date()
        if today != self.daily_stats['last_reset']:
            logger.info(f"Daily stats reset. Previous: {self.daily_stats}")
            self.daily_stats = {
                'trades_count': 0,
                'total_pnl': 0.0,
                'last_reset': today
            }
    
    def can_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if new trade is allowed
        
        Returns:
            (allowed: bool, reason: str)
        """
        self.reset_daily_stats()
        
        # Check max trades per day
        if self.daily_stats['trades_count'] >= self.max_trades_per_day:
            return False, f"Max daily trades reached ({self.max_trades_per_day})"
        
        # Check daily loss limit
        account = self.mt5.get_account_info()
        balance = account['balance']
        max_loss_amount = balance * (self.max_daily_loss / 100)
        
        if self.daily_stats['total_pnl'] <= -max_loss_amount:
            return False, f"Daily loss limit reached (-{self.max_daily_loss}%)"
        
        # Check max open positions
        positions = self.mt5.get_positions()
        if len(positions) >= self.max_open_positions:
            return False, f"Max open positions reached ({self.max_open_positions})"
        
        # Check margin
        free_margin_pct = account.get('free_margin_percent', 0)
        if free_margin_pct < 30:  # Less than 30% free margin
            return False, f"Low free margin ({free_margin_pct:.1f}%)"
        
        # Check correlation exposure
        if not self._check_correlation_limit(symbol, positions):
            return False, "Correlation exposure limit reached"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        symbol: str,
        stop_loss_pips: float,
        confidence: float = 0.7,
        news_impact: float = 0.0,
        trade_mode: str = "NORMAL"
    ) -> float:
        """
        Calculate optimal position size using multiple factors
        
        Args:
            symbol: Trading symbol
            stop_loss_pips: Stop loss distance in pips
            confidence: AI confidence score (0.0 - 1.0)
            news_impact: News impact score (0 - 10)
            trade_mode: NORMAL, AGGRESSIVE, SCALPING, LONG_HOLD
        """
        account = self.mt5.get_account_info()
        balance = account['balance']
        
        # Base risk amount
        risk_percent = self.max_risk_per_trade
        
        # Adjust based on trade mode
        mode_multipliers = {
            'NORMAL': 1.0,
            'AGGRESSIVE': 1.5,
            'SCALPING': 0.7,
            'LONG_HOLD': 0.8
        }
        risk_percent *= mode_multipliers.get(trade_mode, 1.0)
        
        # Adjust based on confidence (0.5x to 1.5x)
        confidence_multiplier = 0.5 + confidence
        risk_percent *= confidence_multiplier
        
        # Reduce risk during high impact news
        news_multiplier = max(0.3, 1.0 - (news_impact / 20))
        risk_percent *= news_multiplier
        
        # Adjust based on recent performance
        win_rate = self._calculate_recent_win_rate()
        if win_rate < 0.4:  # Less than 40% win rate
            risk_percent *= 0.7
        elif win_rate > 0.7:  # More than 70% win rate
            risk_percent *= 1.2
        
        # Calculate risk amount
        risk_amount = balance * (risk_percent / 100)
        
        # Calculate lot size
        lot_size = self.mt5.calculate_lot_size(symbol, risk_amount, stop_loss_pips)
        
        logger.info(f"Position size calculated: {lot_size:.2f} lots "
                   f"(Risk: {risk_percent:.2f}%, Confidence: {confidence:.2f})")
        
        return lot_size
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        direction: str,  # 'BUY' or 'SELL'
        atr: float,
        atr_multiplier: float = 2.0,
        min_pips: float = 10,
        max_pips: float = 100
    ) -> float:
        """Calculate dynamic stop loss based on ATR"""
        symbol_info = self.mt5.get_symbol_info(symbol)
        pip_value = symbol_info['pip_value']
        
        # ATR-based SL
        sl_distance = atr * atr_multiplier
        sl_pips = sl_distance / pip_value
        
        # Apply min/max limits
        sl_pips = max(min_pips, min(max_pips, sl_pips))
        
        # Calculate SL price
        if direction.upper() == 'BUY':
            sl_price = entry_price - (sl_pips * pip_value)
        else:
            sl_price = entry_price + (sl_pips * pip_value)
        
        return round(sl_price, symbol_info['digits'])
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0,
        symbol: Optional[str] = None
    ) -> float:
        """Calculate take profit based on risk-reward ratio"""
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * risk_reward_ratio
        
        if entry_price > stop_loss:  # BUY
            tp_price = entry_price + tp_distance
        else:  # SELL
            tp_price = entry_price - tp_distance
        
        if symbol:
            symbol_info = self.mt5.get_symbol_info(symbol)
            tp_price = round(tp_price, symbol_info['digits'])
        
        return tp_price
    
    def calculate_partial_close_levels(
        self,
        entry_price: float,
        stop_loss: float,
        levels: list = [0.5, 0.3, 0.2]  # Close 50%, 30%, 20%
    ) -> list:
        """
        Calculate multiple take profit levels for partial position closing
        
        Returns:
            List of (price, volume_percent) tuples
        """
        partial_levels = []
        
        for i, volume_pct in enumerate(levels):
            rr_ratio = (i + 1) * 1.5  # 1.5:1, 3:1, 4.5:1
            tp_price = self.calculate_take_profit(entry_price, stop_loss, rr_ratio)
            partial_levels.append((tp_price, volume_pct))
        
        return partial_levels
    
    def update_trade_stats(self, trade_result: Dict):
        """Update daily statistics after trade"""
        self.daily_stats['trades_count'] += 1
        self.daily_stats['total_pnl'] += trade_result.get('profit', 0.0)
        
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': trade_result.get('symbol'),
            'profit': trade_result.get('profit', 0.0),
            'pips': trade_result.get('profit_pips', 0.0),
            'type': trade_result.get('type'),
        })
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        account = self.mt5.get_account_info()
        positions = self.mt5.get_positions()
        
        # Calculate current exposure
        total_exposure = sum(abs(p['profit']) for p in positions)
        balance = account['balance']
        exposure_percent = (total_exposure / balance * 100) if balance > 0 else 0
        
        # Calculate current risk
        total_risk = 0
        for pos in positions:
            if pos['sl'] > 0:
                risk = abs(pos['price_open'] - pos['sl']) * pos['volume']
                total_risk += risk
        
        risk_percent = (total_risk / balance * 100) if balance > 0 else 0
        
        # Win rate
        win_rate = self._calculate_recent_win_rate()
        
        # Profit factor
        winning_trades = [t for t in self.trade_history if t['profit'] > 0]
        losing_trades = [t for t in self.trade_history if t['profit'] < 0]
        
        total_wins = sum(t['profit'] for t in winning_trades)
        total_losses = abs(sum(t['profit'] for t in losing_trades))
        
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        
        return {
            'balance': balance,
            'equity': account['equity'],
            'free_margin': account['margin_free'],
            'margin_level': account.get('margin_level', 0),
            'open_positions': len(positions),
            'exposure_percent': exposure_percent,
            'current_risk_percent': risk_percent,
            'daily_trades': self.daily_stats['trades_count'],
            'daily_pnl': self.daily_stats['total_pnl'],
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_daily_loss_remaining': self.max_daily_loss + (self.daily_stats['total_pnl'] / balance * 100)
        }
    
    def _check_correlation_limit(self, symbol: str, positions: list) -> bool:
        """Check if adding position would exceed correlation limit"""
        if not positions:
            return True
        
        # Simplified correlation check
        # In production, use actual correlation matrix
        correlated_symbols = {
            'XAUUSD': ['XAGUSD'],
            'EURUSD': ['GBPUSD', 'AUDUSD'],
            'USDJPY': ['EURJPY', 'GBPJPY'],
        }
        
        correlated = correlated_symbols.get(symbol, [])
        
        exposure_count = sum(1 for p in positions if p['symbol'] in correlated)
        
        return exposure_count < self.max_open_positions * self.max_correlation_exposure
    
    def _calculate_recent_win_rate(self, lookback: int = 20) -> float:
        """Calculate win rate from recent trades"""
        if not self.trade_history:
            return 0.5  # Default 50%
        
        recent_trades = self.trade_history[-lookback:]
        winning_trades = sum(1 for t in recent_trades if t['profit'] > 0)
        
        return winning_trades / len(recent_trades) if recent_trades else 0.5
    
    def check_drawdown(self, max_drawdown_percent: float = 20.0) -> Tuple[bool, float]:
        """
        Check if current drawdown exceeds limit
        
        Returns:
            (within_limit: bool, current_drawdown_percent: float)
        """
        account = self.mt5.get_account_info()
        balance = account['balance']
        equity = account['equity']
        
        drawdown = ((balance - equity) / balance * 100) if balance > 0 else 0
        
        within_limit = drawdown < max_drawdown_percent
        
        if not within_limit:
            logger.warning(f"⚠️ Drawdown limit exceeded: {drawdown:.2f}%")
        
        return within_limit, drawdown
    
    def should_stop_trading(self) -> Tuple[bool, str]:
        """Determine if trading should be stopped"""
        
        # Check drawdown
        within_limit, drawdown = self.check_drawdown()
        if not within_limit:
            return True, f"Max drawdown exceeded ({drawdown:.2f}%)"
        
        # Check daily loss
        account = self.mt5.get_account_info()
        daily_loss_pct = abs(self.daily_stats['total_pnl'] / account['balance'] * 100)
        if daily_loss_pct >= self.max_daily_loss:
            return True, f"Daily loss limit reached ({daily_loss_pct:.2f}%)"
        
        # Check consecutive losses
        recent = self.trade_history[-5:] if len(self.trade_history) >= 5 else self.trade_history
        if all(t['profit'] < 0 for t in recent) and len(recent) == 5:
            return True, "5 consecutive losses detected"
        
        # Check margin level
        margin_level = account.get('margin_level', 1000)
        if margin_level < 150:  # Below 150%
            return True, f"Low margin level ({margin_level:.1f}%)"
        
        return False, "OK"


# Kelly Criterion Calculator
class KellyCriterion:
    """Calculate optimal position size using Kelly Criterion"""
    
    @staticmethod
    def calculate(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly: float = 0.25  # Use max 25% of Kelly
    ) -> float:
        """
        Calculate Kelly percentage
        
        Formula: K% = W - [(1 - W) / R]
        Where:
            W = Win rate
            R = Win/Loss ratio
        """
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply safety factor
        kelly = max(0, min(kelly * max_kelly, max_kelly))
        
        return kelly


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    if mt5.initialize():
        risk_mgr = RiskManager(mt5)
        
        # Check if can trade
        can_trade, reason = risk_mgr.can_trade("XAUUSD")
        print(f"Can trade: {can_trade} ({reason})")
        
        # Calculate position size
        lot_size = risk_mgr.calculate_position_size(
            symbol="XAUUSD",
            stop_loss_pips=30,
            confidence=0.75,
            news_impact=3
        )
        print(f"Position size: {lot_size}")
        
        # Get risk metrics
        metrics = risk_mgr.get_risk_metrics()
        print(f"Risk metrics: {metrics}")
        
        mt5.shutdown()