import pandas as pd
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy, CandlePatterns
import logging

logger = logging.getLogger(__name__)


class CounterTrendStrategy(BaseStrategy):
    """Counter-trend reversal strategy"""
    
    def __init__(self):
        super().__init__("Counter-Trend", "EXTREME")
        self.min_confidence = 0.55  # 🔥 LOWER (was 0.70) → more signals!
        self.patterns = CandlePatterns()
    
    def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
        """
        Look for reversal signals at extremes
        - Oversold/Overbought RSI
        - Price at Bollinger Bands extremes
        - Divergence
        - Reversal candle patterns
        """
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Get indicators
        rsi = current['rsi']
        bb_upper = current['bb_upper']
        bb_lower = current['bb_lower']
        close = current['close']
        
        # Bearish reversal conditions (SELL)
        bearish_conditions = {
            'rsi_overbought': rsi > 70,
            'price_at_bb_upper': close >= bb_upper * 0.998,
            'shooting_star': self.patterns.is_shooting_star(current),
            'bearish_engulfing': self.patterns.is_engulfing(df, len(df)-1, bullish=False),
            'macd_bearish_cross': (previous['macd'] > previous['macd_signal'] and
                                   current['macd'] < current['macd_signal']),
            'stoch_overbought': current['stoch_k'] > 80
        }
        
        # Bullish reversal conditions (BUY)
        bullish_conditions = {
            'rsi_oversold': rsi < 30,
            'price_at_bb_lower': close <= bb_lower * 1.002,
            'hammer': self.patterns.is_hammer(current),
            'bullish_engulfing': self.patterns.is_engulfing(df, len(df)-1, bullish=True),
            'macd_bullish_cross': (previous['macd'] < previous['macd_signal'] and
                                   current['macd'] > current['macd_signal']),
            'stoch_oversold': current['stoch_k'] < 20
        }
        
        bearish_score = self.calculate_score(bearish_conditions)
        bullish_score = self.calculate_score(bullish_conditions)
        
        # Detect divergence
        divergence = self._detect_divergence(df)
        if divergence == 'BULLISH':
            bullish_score += 0.15
        elif divergence == 'BEARISH':
            bearish_score += 0.15
        
        # Generate signal
        if bearish_score >= self.min_confidence:
            return self._create_signal(
                'SELL', df, symbol_info, bearish_score, 
                bearish_conditions, 'Counter-trend SELL'
            )
        
        elif bullish_score >= self.min_confidence:
            return self._create_signal(
                'BUY', df, symbol_info, bullish_score,
                bullish_conditions, 'Counter-trend BUY'
            )
        
        return None
    
    def _detect_divergence(self, df: pd.DataFrame, lookback: int = 20) -> Optional[str]:
        """Detect RSI divergence"""
        if len(df) < lookback:
            return None
        
        recent = df.iloc[-lookback:]
        
        # Find price peaks and troughs
        price_high_idx = recent['high'].idxmax()
        price_low_idx = recent['low'].idxmin()
        
        # Bullish divergence: Lower price low, Higher RSI low
        if price_low_idx > len(recent) - 10:
            prev_low_price = recent['low'].iloc[:-10].min()
            current_low_price = recent['low'].iloc[-1]
            
            prev_low_rsi = recent['rsi'].iloc[:-10].min()
            current_low_rsi = recent['rsi'].iloc[-1]
            
            if current_low_price < prev_low_price and current_low_rsi > prev_low_rsi:
                return 'BULLISH'
        
        # Bearish divergence: Higher price high, Lower RSI high
        if price_high_idx > len(recent) - 10:
            prev_high_price = recent['high'].iloc[:-10].max()
            current_high_price = recent['high'].iloc[-1]
            
            prev_high_rsi = recent['rsi'].iloc[:-10].max()
            current_high_rsi = recent['rsi'].iloc[-1]
            
            if current_high_price > prev_high_price and current_high_rsi < prev_high_rsi:
                return 'BEARISH'
        
        return None
    
    def _create_signal(
        self, 
        action: str, 
        df: pd.DataFrame, 
        symbol_info: Dict,
        confidence: float,
        conditions: Dict,
        reason: str
    ) -> Dict:
        """Create trading signal"""
        current = df.iloc[-1]
        atr = current['atr']
        pip_value = symbol_info['pip_value']
        
        entry_price = current['close']
        
        # Calculate SL based on ATR
        sl_distance = atr * 2.0
        
        if action == 'BUY':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * 3.0)  # 3:1 RR
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * 3.0)
        
        return {
            'action': action,
            'entry_price': round(entry_price, symbol_info['digits']),
            'stop_loss': round(stop_loss, symbol_info['digits']),
            'take_profit': round(take_profit, symbol_info['digits']),
            'confidence': confidence,
            'reason': reason,
            'risk_level': 'EXTREME',
            'metadata': {
                'conditions_met': [k for k, v in conditions.items() if v],
                'atr': atr,
                'rsi': current['rsi'],
                'entry_zone': f"{entry_price-pip_value*5:.5f} - {entry_price+pip_value*5:.5f}"
            }
        }


# strategies/breakout.py
"""
Breakout Strategy - Medium to High Risk
"""

class BreakoutStrategy(BaseStrategy):
    """Breakout from consolidation or key levels"""
    
    def __init__(self):
        super().__init__("Breakout", "HIGH")
        self.min_confidence = 0.65
    
    def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
        """
        Look for breakout signals:
        - Price breaking above/below key levels
        - Volume confirmation
        - Momentum confirmation
        """
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Detect support/resistance levels
        levels = self.indicators.detect_support_resistance(df)
        
        # Check for consolidation
        recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
        atr = current['atr']
        is_consolidating = recent_range < atr * 3
        
        # Bullish breakout conditions
        resistance_broken = False
        if levels['resistance']:
            nearest_resistance = min(levels['resistance'], key=lambda x: abs(x - current['close']))
            resistance_broken = (previous['close'] < nearest_resistance and 
                               current['close'] > nearest_resistance)
        
        bullish_conditions = {
            'resistance_broken': resistance_broken,
            'volume_increase': current['tick_volume'] > df['tick_volume'].iloc[-20:-1].mean() * 1.5,
            'macd_positive': current['macd_hist'] > 0,
            'rsi_momentum': 50 < current['rsi'] < 70,
            'consolidation_breakout': is_consolidating and current['close'] > df['high'].iloc[-20:-1].max(),
            'ema_alignment': current['ema_9'] > current['ema_21']
        }
        
        # Bearish breakout conditions
        support_broken = False
        if levels['support']:
            nearest_support = min(levels['support'], key=lambda x: abs(x - current['close']))
            support_broken = (previous['close'] > nearest_support and 
                            current['close'] < nearest_support)
        
        bearish_conditions = {
            'support_broken': support_broken,
            'volume_increase': current['tick_volume'] > df['tick_volume'].iloc[-20:-1].mean() * 1.5,
            'macd_negative': current['macd_hist'] < 0,
            'rsi_momentum': 30 < current['rsi'] < 50,
            'consolidation_breakdown': is_consolidating and current['close'] < df['low'].iloc[-20:-1].min(),
            'ema_alignment': current['ema_9'] < current['ema_21']
        }
        
        bullish_score = self.calculate_score(bullish_conditions)
        bearish_score = self.calculate_score(bearish_conditions)
        
        if bullish_score >= self.min_confidence:
            return self._create_breakout_signal('BUY', df, symbol_info, bullish_score, bullish_conditions)
        elif bearish_score >= self.min_confidence:
            return self._create_breakout_signal('SELL', df, symbol_info, bearish_score, bearish_conditions)
        
        return None
    
    def _create_breakout_signal(
        self, 
        action: str, 
        df: pd.DataFrame, 
        symbol_info: Dict,
        confidence: float,
        conditions: Dict
    ) -> Dict:
        """Create breakout signal with tighter stops"""
        current = df.iloc[-1]
        atr = current['atr']
        
        entry_price = current['close']
        
        # Tighter SL for breakouts (1.5x ATR)
        sl_distance = atr * 1.5
        
        if action == 'BUY':
            # SL below breakout level
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * 2.5)
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * 2.5)
        
        return {
            'action': action,
            'entry_price': round(entry_price, symbol_info['digits']),
            'stop_loss': round(stop_loss, symbol_info['digits']),
            'take_profit': round(take_profit, symbol_info['digits']),
            'confidence': confidence,
            'reason': f"Breakout {action}",
            'risk_level': 'HIGH',
            'metadata': {
                'conditions_met': [k for k, v in conditions.items() if v],
                'atr': atr,
                'breakout_type': 'consolidation' if conditions.get('consolidation_breakout') or conditions.get('consolidation_breakdown') else 'level'
            }
        }


# strategies/fibonacci_atr.py
"""
Fibonacci + ATR Strategy - Medium Risk
"""

class FibonacciATRStrategy(BaseStrategy):
    """Fibonacci retracement with ATR-based entries"""
    
    def __init__(self):
        super().__init__("Fibonacci-ATR", "MEDIUM")
        self.min_confidence = 0.60
        self.fib_levels = [0.382, 0.500, 0.618]
    
    def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
        """
        Look for:
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
        at_fib_level = None
        for level_name, level_price in fib_levels.items():
            if level_name in ['0.382', '0.500', '0.618']:
                if abs(close - level_price) < atr * 0.5:
                    at_fib_level = (level_name, level_price)
                    break
        
        if not at_fib_level:
            return None
        
        # Bullish setup (uptrend, retracement to support)
        if trend == 'BULLISH' and close < swing_high * 0.95:
            bullish_conditions = {
                'at_fib_level': True,
                'uptrend': True,
                'rsi_not_overbought': current['rsi'] < 70,
                'macd_bullish': current['macd'] > current['macd_signal'],
                'price_above_ema21': close > current['ema_21'],
                'stoch_rising': current['stoch_k'] > current['stoch_d'],
                'candle_bullish': current['close'] > current['open']
            }
            
            score = self.calculate_score(bullish_conditions)
            
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
                'candle_bearish': current['close'] < current['open']
            }
            
            score = self.calculate_score(bearish_conditions)
            
            if score >= self.min_confidence:
                return self._create_fib_signal(
                    'SELL', df, symbol_info, score, at_fib_level,
                    swing_high, swing_low, bearish_conditions
                )
        
        return None
    
    def _find_swing_points(self, df: pd.DataFrame, lookback: int = 50) -> tuple:
        """Find recent swing high and low"""
        recent = df.iloc[-lookback:]
        
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        
        # Ensure swing points are significant
        range_size = swing_high - swing_low
        atr = df['atr'].iloc[-1]
        
        if range_size < atr * 5:  # Too small range
            return None, None
        
        return swing_high, swing_low
    
    def _create_fib_signal(
        self,
        action: str,
        df: pd.DataFrame,
        symbol_info: Dict,
        confidence: float,
        fib_level: tuple,
        swing_high: float,
        swing_low: float,
        conditions: Dict
    ) -> Dict:
        """Create Fibonacci-based signal"""
        current = df.iloc[-1]
        atr = current['atr']
        
        level_name, level_price = fib_level
        entry_price = current['close']
        
        # Calculate SL/TP based on swing points and ATR
        if action == 'BUY':
            # SL below next Fibonacci level or swing low
            stop_loss = max(swing_low - atr, level_price - (atr * 2))
            # TP at swing high or extension
            take_profit = swing_high + atr
        else:
            # SL above next Fibonacci level or swing high
            stop_loss = min(swing_high + atr, level_price + (atr * 2))
            # TP at swing low or extension
            take_profit = swing_low - atr
        
        return {
            'action': action,
            'entry_price': round(entry_price, symbol_info['digits']),
            'stop_loss': round(stop_loss, symbol_info['digits']),
            'take_profit': round(take_profit, symbol_info['digits']),
            'confidence': confidence,
            'reason': f"Fibonacci {level_name} retracement",
            'risk_level': 'MEDIUM',
            'metadata': {
                'conditions_met': [k for k, v in conditions.items() if v],
                'fib_level': level_name,
                'fib_price': level_price,
                'swing_high': swing_high,
                'swing_low': swing_low,
                'atr': atr,
                'rr_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss)
            }
        }


# strategies/strategy_manager.py
"""
Strategy Manager - Coordinates multiple strategies
"""

class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self, mt5_handler, enabled_strategies: list = None):
        self.mt5 = mt5_handler
        
        # Initialize all strategies
        self.strategies = {
            'counter_trend': CounterTrendStrategy(),
            'breakout': BreakoutStrategy(),
            'fibonacci_atr': FibonacciATRStrategy()
        }
        
        # Filter enabled strategies
        if enabled_strategies:
            self.strategies = {
                k: v for k, v in self.strategies.items() 
                if k in enabled_strategies
            }
        
        logger.info(f"Initialized strategies: {list(self.strategies.keys())}")
    
    def analyze_all(self, symbol: str, timeframe: str = 'M5') -> Dict:
        """
        Run all strategies and return best signal
        
        Returns:
            {
                'best_signal': signal or None,
                'all_signals': {strategy_name: signal},
                'consensus': 'BUY', 'SELL', or 'NEUTRAL'
            }
        """
        df = self.mt5.get_candles(symbol, timeframe, count=200)
        symbol_info = self.mt5.get_symbol_info(symbol)
        
        signals = {}
        
        for name, strategy in self.strategies.items():
            try:
                df_copy = df.copy()
                df_copy = strategy.add_all_indicators(df_copy)
                signal = strategy.analyze(df_copy, symbol_info)
                signals[name] = signal
                
                if signal:
                    logger.info(f"✅ {name}: {signal['action']} (confidence: {signal['confidence']:.2f})")
            
            except Exception as e:
                logger.error(f"Error in {name} strategy: {str(e)}")
                signals[name] = None
        
        # Calculate consensus
        buy_signals = [s for s in signals.values() if s and s['action'] == 'BUY']
        sell_signals = [s for s in signals.values() if s and s['action'] == 'SELL']
        
        if len(buy_signals) > len(sell_signals):
            consensus = 'BUY'
        elif len(sell_signals) > len(buy_signals):
            consensus = 'SELL'
        else:
            consensus = 'NEUTRAL'
        
        # Select best signal (highest confidence)
        all_valid_signals = [s for s in signals.values() if s]
        best_signal = max(
            all_valid_signals,
            key=lambda s: s['confidence']
        ) if all_valid_signals else None
        
        return {
            'best_signal': best_signal,
            'all_signals': signals,
            'consensus': consensus,
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals)
        }
    
    def get_strategy_performance(self, strategy_name: str) -> Dict:
        """Get performance metrics for a specific strategy"""
        # This would connect to performance tracker
        # For now, return placeholder
        return {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'total_trades': 0
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        # Test individual strategy
        strategy = CounterTrendStrategy()
        
        df = mt5.get_candles("BTCUSDm", "M5", count=200)
        df = strategy.add_all_indicators(df)
        
        signal = strategy.analyze(df, mt5.get_symbol_info("BTCUSDm"))
        
        if signal:
            print(f"\nSignal detected:")
            print(f"Action: {signal['action']}")
            print(f"Entry: {signal['entry_price']}")
            print(f"SL: {signal['stop_loss']}")
            print(f"TP: {signal['take_profit']}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print(f"Reason: {signal['reason']}")
        else:
            print("No signal")
        
        # Test strategy manager
        manager = StrategyManager(mt5)
        result = manager.analyze_all("BTCUSDm", "M5")
        
        print(f"\n--- Strategy Manager Results ---")
        print(f"Consensus: {result['consensus']}")
        print(f"Buy signals: {result['buy_count']}")
        print(f"Sell signals: {result['sell_count']}")
        
        if result['best_signal']:
            print(f"\nBest Signal: {result['best_signal']['action']} "
                  f"(confidence: {result['best_signal']['confidence']:.2%})")
        
        mt5.shutdown()