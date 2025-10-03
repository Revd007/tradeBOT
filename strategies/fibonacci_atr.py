"""
Fibonacci + ATR Strategy - Medium Risk
Trades retracements to Fibonacci levels in trending markets
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class FibonacciATRStrategy(BaseStrategy):
    """Fibonacci retracement with ATR-based entries and risk management"""
    
    def __init__(self):
        super().__init__("Fibonacci-ATR", "MEDIUM")
        self.min_confidence = 0.60
        self.fib_levels = [0.382, 0.500, 0.618, 0.786]
    
    def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
        """
        Look for:
        - Clear trend direction
        - Price retracing to Fibonacci levels
        - Trend continuation signals
        - ATR-based risk management
        """
        if len(df) < 100:
            return None
        
        # Identify recent swing high/low
        swing_high, swing_low = self._find_swing_points(df, lookback=50)
        
        if swing_high is None or swing_low is None:
            return None
        
        # Calculate Fibonacci levels
        fib_levels = self.indicators.calculate_fibonacci_levels(
            swing_high, swing_low, 'retracement'
        )
        
        current = df.iloc[-1]
        close = current['close']
        atr = current['atr']
        
        # Determine trend
        trend = self.detect_trend(df)
        
        if trend == 'NEUTRAL':
            return None
        
        # Check if price is at a Fibonacci level
        at_fib_level = self._check_fib_level(close, fib_levels, atr)
        
        if not at_fib_level:
            return None
        
        level_name, level_price = at_fib_level
        
        # Bullish setup (uptrend, retracement to support)
        if trend == 'BULLISH' and close < swing_high * 0.95:
            bullish_conditions = {
                'at_fib_level': True,
                'uptrend': True,
                'rsi_not_overbought': current['rsi'] < 70,
                'macd_bullish': current['macd'] > current['macd_signal'],
                'price_above_ema21': close > current['ema_21'],
                'stoch_rising': current['stoch_k'] > current['stoch_d'],
                'candle_bullish': current['close'] > current['open'],
                'fib_bounce': self._detect_fib_bounce(df, level_price, 'BUY')
            }
            
            score = self.calculate_score(bullish_conditions)
            
            # Boost score for golden ratio (0.618)
            if level_name == '0.618':
                score += 0.10
            
            if score >= self.min_confidence:
                return self._create_fib_signal(
                    'BUY', df, symbol_info, score, at_fib_level, 
                    swing_high, swing_low, bullish_conditions
                )
        
        # Bearish setup (downtrend, retracement to resistance)
        elif trend == 'BEARISH' and close > swing_low * 1.05:
            bearish_conditions = {
                'at_fib_level': True,
                'downtrend': True,
                'rsi_not_oversold': current['rsi'] > 30,
                'macd_bearish': current['macd'] < current['macd_signal'],
                'price_below_ema21': close < current['ema_21'],
                'stoch_falling': current['stoch_k'] < current['stoch_d'],
                'candle_bearish': current['close'] < current['open'],
                'fib_bounce': self._detect_fib_bounce(df, level_price, 'SELL')
            }
            
            score = self.calculate_score(bearish_conditions)
            
            # Boost score for golden ratio (0.618)
            if level_name == '0.618':
                score += 0.10
            
            if score >= self.min_confidence:
                return self._create_fib_signal(
                    'SELL', df, symbol_info, score, at_fib_level,
                    swing_high, swing_low, bearish_conditions
                )
        
        return None
    
    def _find_swing_points(self, df: pd.DataFrame, lookback: int = 50) -> Tuple[Optional[float], Optional[float]]:
        """
        Find recent swing high and low
        
        Returns:
            (swing_high, swing_low) or (None, None)
        """
        if len(df) < lookback:
            return None, None
        
        recent = df.iloc[-lookback:]
        
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        
        # Ensure swing points are significant (at least 5x ATR)
        range_size = swing_high - swing_low
        atr = df['atr'].iloc[-1]
        
        if range_size < atr * 5:  # Too small range
            return None, None
        
        return swing_high, swing_low
    
    def _check_fib_level(
        self, 
        price: float, 
        fib_levels: Dict[str, float], 
        atr: float
    ) -> Optional[Tuple[str, float]]:
        """
        Check if price is at a Fibonacci level
        
        Returns:
            (level_name, level_price) or None
        """
        tolerance = atr * 0.5  # Within 0.5 ATR of Fibonacci level
        
        for level_name, level_price in fib_levels.items():
            if level_name in ['0.382', '0.500', '0.618', '0.786']:
                if abs(price - level_price) < tolerance:
                    return (level_name, level_price)
        
        return None
    
    def _detect_fib_bounce(
        self, 
        df: pd.DataFrame, 
        fib_level: float, 
        direction: str
    ) -> bool:
        """
        Detect if price is bouncing from Fibonacci level
        
        Args:
            df: Price data
            fib_level: Fibonacci level price
            direction: 'BUY' or 'SELL'
        """
        recent = df.iloc[-5:]  # Last 5 candles
        
        if direction == 'BUY':
            # Check if recent low touched level and price is moving up
            touched_level = any(abs(candle['low'] - fib_level) < recent['atr'].mean() * 0.3 
                               for _, candle in recent.iterrows())
            moving_up = recent['close'].iloc[-1] > recent['close'].iloc[-3]
            return touched_level and moving_up
        
        else:  # SELL
            # Check if recent high touched level and price is moving down
            touched_level = any(abs(candle['high'] - fib_level) < recent['atr'].mean() * 0.3 
                               for _, candle in recent.iterrows())
            moving_down = recent['close'].iloc[-1] < recent['close'].iloc[-3]
            return touched_level and moving_down
    
    def _create_fib_signal(
        self,
        action: str,
        df: pd.DataFrame,
        symbol_info: Dict,
        confidence: float,
        fib_level: Tuple[str, float],
        swing_high: float,
        swing_low: float,
        conditions: Dict
    ) -> Dict:
        """Create Fibonacci-based signal with optimal SL/TP"""
        current = df.iloc[-1]
        atr = current['atr']
        
        level_name, level_price = fib_level
        entry_price = current['close']
        
        # Calculate SL/TP based on swing points and ATR
        if action == 'BUY':
            # SL below Fibonacci level or swing low
            next_fib_distance = abs(level_price - swing_low) * 0.382  # Next Fib level down
            sl_option1 = level_price - (atr * 2)
            sl_option2 = swing_low - atr
            stop_loss = max(sl_option1, sl_option2)
            
            # TP at swing high or Fibonacci extension
            tp_option1 = swing_high + atr
            extension = swing_high + (swing_high - swing_low) * 0.618  # 618 extension
            take_profit = max(tp_option1, extension)
        
        else:  # SELL
            # SL above Fibonacci level or swing high
            next_fib_distance = abs(swing_high - level_price) * 0.382
            sl_option1 = level_price + (atr * 2)
            sl_option2 = swing_high + atr
            stop_loss = min(sl_option1, sl_option2)
            
            # TP at swing low or Fibonacci extension
            tp_option1 = swing_low - atr
            extension = swing_low - (swing_high - swing_low) * 0.618
            take_profit = min(tp_option1, extension)
        
        # Calculate actual risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'action': action,
            'entry_price': round(entry_price, symbol_info['digits']),
            'stop_loss': round(stop_loss, symbol_info['digits']),
            'take_profit': round(take_profit, symbol_info['digits']),
            'confidence': confidence,
            'reason': f"Fibonacci {level_name} retracement in {conditions.get('uptrend', False) and 'uptrend' or 'downtrend'}",
            'risk_level': 'MEDIUM',
            'metadata': {
                'conditions_met': [k for k, v in conditions.items() if v],
                'fib_level': level_name,
                'fib_price': round(level_price, symbol_info['digits']),
                'swing_high': round(swing_high, symbol_info['digits']),
                'swing_low': round(swing_low, symbol_info['digits']),
                'atr': round(atr, symbol_info['digits']),
                'rr_ratio': round(rr_ratio, 2),
                'distance_to_fib': abs(entry_price - level_price),
                'swing_range': round(swing_high - swing_low, symbol_info['digits'])
            }
        }
    
    def calculate_fibonacci_extensions(
        self,
        swing_high: float,
        swing_low: float
    ) -> Dict[str, float]:
        """Calculate Fibonacci extension levels for take profit targets"""
        diff = swing_high - swing_low
        
        extensions = {
            '1.0': swing_high + (1.0 * diff),
            '1.272': swing_high + (1.272 * diff),
            '1.618': swing_high + (1.618 * diff),
            '2.0': swing_high + (2.0 * diff),
            '2.618': swing_high + (2.618 * diff),
        }
        
        return extensions


if __name__ == "__main__":
    # Test Fibonacci-ATR strategy
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        strategy = FibonacciATRStrategy()
        
        df = mt5.get_candles("XAUUSD", "H1", count=200)
        df = strategy.add_all_indicators(df)
        
        signal = strategy.analyze(df, mt5.get_symbol_info("XAUUSD"))
        
        if signal:
            print(f"\n{'='*70}")
            print(f"FIBONACCI-ATR SIGNAL DETECTED")
            print(f"{'='*70}")
            print(f"Action: {signal['action']}")
            print(f"Entry: {signal['entry_price']}")
            print(f"Stop Loss: {signal['stop_loss']}")
            print(f"Take Profit: {signal['take_profit']}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print(f"Risk Level: {signal['risk_level']}")
            print(f"Reason: {signal['reason']}")
            
            meta = signal['metadata']
            print(f"\nFibonacci Analysis:")
            print(f"  Level: {meta['fib_level']} @ {meta['fib_price']}")
            print(f"  Swing High: {meta['swing_high']}")
            print(f"  Swing Low: {meta['swing_low']}")
            print(f"  Swing Range: {meta['swing_range']}")
            print(f"  Risk/Reward: 1:{meta['rr_ratio']:.2f}")
            print(f"  Distance to Fib: {meta['distance_to_fib']:.5f}")
            
            print(f"\nConditions Met:")
            for condition in meta['conditions_met']:
                print(f"  ✓ {condition}")
            
            # Calculate extensions
            extensions = strategy.calculate_fibonacci_extensions(
                meta['swing_high'], meta['swing_low']
            )
            print(f"\nFibonacci Extensions (Potential Targets):")
            for level, price in extensions.items():
                print(f"  {level}: {price:.2f}")
        
        else:
            print("No Fibonacci-ATR signal detected")
        
        mt5.shutdown()

