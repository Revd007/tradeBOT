"""
Breakout Strategy - High Risk
Trades breakouts from consolidation or key support/resistance levels
"""

import pandas as pd
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """Breakout from consolidation or key levels"""
    
    def __init__(self):
        super().__init__("Breakout", "HIGH")
        self.min_confidence = 0.50  # 🔥 LOWER (was 0.65) → more signals!
    
    def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
        """
        🔥 IMPROVED: Look for breakout signals with:
        - Closing confirmation (candle MUST close beyond level)
        - Build-up filter (low volatility before breakout = stronger signal)
        - Volume confirmation
        - Momentum confirmation
        - Post-consolidation breakout
        """
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Detect support/resistance levels
        levels = self.indicators.detect_support_resistance(df)
        
        # Check for consolidation (tight range)
        recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
        atr = current['atr']
        is_consolidating = recent_range < atr * 3
        
        # 🔥 NEW: Build-up Filter (Low volatility before breakout)
        # Check if ATR in last 5-10 candles is significantly lower than normal
        atr_recent = df['atr'].iloc[-10:-1].mean()  # Last 10 candles (excluding current)
        atr_baseline = df['atr'].iloc[-30:-10].mean()  # Previous 20 candles
        
        # Build-up detected if recent ATR is < 70% of baseline
        has_buildup = (atr_recent / atr_baseline) < 0.7 if atr_baseline > 0 else False
        
        logger.debug(f"   Build-up check: Recent ATR={atr_recent:.5f}, Baseline={atr_baseline:.5f}, Ratio={atr_recent/atr_baseline:.2f}")
        if has_buildup:
            logger.debug(f"   ✅ Build-up detected! (Low volatility → potential explosive breakout)")
        
        # 🔥 IMPROVED: Bullish breakout with CLOSING CONFIRMATION
        resistance_broken = False
        closing_confirmed_bull = False
        
        if levels['resistance']:
            nearest_resistance = min(levels['resistance'], key=lambda x: abs(x - current['close']))
            
            # OLD: Check if price crossed level
            # NEW: Check if CANDLE CLOSED above level (confirmation!)
            resistance_broken = (previous['close'] < nearest_resistance and 
                              current['close'] > nearest_resistance)
            
            # 🔥 CLOSING CONFIRMATION: Candle must CLOSE above level, not just wick
            closing_confirmed_bull = (current['close'] > nearest_resistance and 
                                     current['open'] < nearest_resistance)
            
            if resistance_broken:
                logger.debug(f"   Resistance {nearest_resistance:.5f} broken. Close confirmed: {closing_confirmed_bull}")
        
        bullish_conditions = {
            'resistance_broken': resistance_broken,
            'closing_confirmed': closing_confirmed_bull,  # 🔥 NEW
            'buildup_detected': has_buildup,  # 🔥 NEW
            'volume_increase': current['tick_volume'] > df['tick_volume'].iloc[-20:-1].mean() * 1.5,
            'macd_positive': current['macd_hist'] > 0,
            'rsi_momentum': 50 < current['rsi'] < 70,
            'consolidation_breakout': is_consolidating and current['close'] > df['high'].iloc[-20:-1].max(),
            'ema_alignment': current['ema_9'] > current['ema_21'],
            'strong_candle': current['close'] > current['open'] and (current['close'] - current['open']) > atr * 0.5
        }
        
        # 🔥 IMPROVED: Bearish breakout with CLOSING CONFIRMATION
        support_broken = False
        closing_confirmed_bear = False
        
        if levels['support']:
            nearest_support = min(levels['support'], key=lambda x: abs(x - current['close']))
            
            support_broken = (previous['close'] > nearest_support and 
                            current['close'] < nearest_support)
            
            # 🔥 CLOSING CONFIRMATION: Candle must CLOSE below level
            closing_confirmed_bear = (current['close'] < nearest_support and 
                                     current['open'] > nearest_support)
            
            if support_broken:
                logger.debug(f"   Support {nearest_support:.5f} broken. Close confirmed: {closing_confirmed_bear}")
        
        bearish_conditions = {
            'support_broken': support_broken,
            'closing_confirmed': closing_confirmed_bear,  # 🔥 NEW
            'buildup_detected': has_buildup,  # 🔥 NEW
            'volume_increase': current['tick_volume'] > df['tick_volume'].iloc[-20:-1].mean() * 1.5,
            'macd_negative': current['macd_hist'] < 0,
            'rsi_momentum': 30 < current['rsi'] < 50,
            'consolidation_breakdown': is_consolidating and current['close'] < df['low'].iloc[-20:-1].min(),
            'ema_alignment': current['ema_9'] < current['ema_21'],
            'strong_candle': current['close'] < current['open'] and (current['open'] - current['close']) > atr * 0.5
        }
        
        bullish_score = self.calculate_score(bullish_conditions)
        bearish_score = self.calculate_score(bearish_conditions)
        
        # 🔥 IMPROVED: Stronger boosts for high-quality breakouts
        # Boost for confirmed close + volume
        if closing_confirmed_bull and bullish_conditions['volume_increase']:
            bullish_score += 0.15  # Increased from 0.10
            logger.debug(f"   📈 Bullish boost: Closing confirmed + volume (+0.15)")
        
        # Extra boost if build-up detected (indicates pressure)
        if has_buildup and resistance_broken:
            bullish_score += 0.10
            logger.debug(f"   📈 Bullish boost: Build-up detected (+0.10)")
        
        if closing_confirmed_bear and bearish_conditions['volume_increase']:
            bearish_score += 0.15  # Increased from 0.10
            logger.debug(f"   📉 Bearish boost: Closing confirmed + volume (+0.15)")
        
        if has_buildup and support_broken:
            bearish_score += 0.10
            logger.debug(f"   📉 Bearish boost: Build-up detected (+0.10)")
        
        # 🔥 FILTER: Require closing confirmation for level breakouts
        # (Consolidation breakouts don't need this strict requirement)
        if bullish_score >= self.min_confidence:
            # If it's a level breakout, require closing confirmation
            if resistance_broken and not closing_confirmed_bull:
                logger.debug(f"   🚫 Bullish signal REJECTED: No closing confirmation")
                return None
            return self._create_breakout_signal('BUY', df, symbol_info, bullish_score, bullish_conditions)
        
        elif bearish_score >= self.min_confidence:
            # If it's a level breakout, require closing confirmation
            if support_broken and not closing_confirmed_bear:
                logger.debug(f"   🚫 Bearish signal REJECTED: No closing confirmation")
                return None
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
            take_profit = entry_price + (sl_distance * 2.5)  # 2.5:1 RR
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * 2.5)
        
        # Determine breakout type
        breakout_type = 'level'
        if conditions.get('consolidation_breakout') or conditions.get('consolidation_breakdown'):
            breakout_type = 'consolidation'
        elif conditions.get('resistance_broken') or conditions.get('support_broken'):
            breakout_type = 'support_resistance'
        
        return {
            'action': action,
            'entry_price': round(entry_price, symbol_info['digits']),
            'stop_loss': round(stop_loss, symbol_info['digits']),
            'take_profit': round(take_profit, symbol_info['digits']),
            'confidence': confidence,
            'reason': f"Breakout {action} - {breakout_type}",
            'risk_level': 'HIGH',
            'metadata': {
                'conditions_met': [k for k, v in conditions.items() if v],
                'atr': atr,
                'breakout_type': breakout_type,
                'volume_ratio': current['tick_volume'] / df['tick_volume'].iloc[-20:-1].mean(),
                'risk_reward_ratio': 2.5
            }
        }
    
    def detect_false_breakout(self, df: pd.DataFrame, lookback: int = 5) -> bool:
        """
        Detect potential false breakout
        Returns True if recent breakout failed
        """
        if len(df) < lookback + 20:
            return False
        
        recent = df.iloc[-lookback:]
        previous_range = df.iloc[-lookback-20:-lookback]
        
        # Check if price broke out then reversed
        recent_high = recent['high'].max()
        recent_low = recent['low'].min()
        prev_high = previous_range['high'].max()
        prev_low = previous_range['low'].min()
        
        # Bullish false breakout
        if recent_high > prev_high and recent.iloc[-1]['close'] < prev_high:
            return True
        
        # Bearish false breakout
        if recent_low < prev_low and recent.iloc[-1]['close'] > prev_low:
            return True
        
        return False


if __name__ == "__main__":
    # Test breakout strategy
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        strategy = BreakoutStrategy()
        
        df = mt5.get_candles("XAUUSDm", "M5", count=200)
        df = strategy.add_all_indicators(df)
        
        signal = strategy.analyze(df, mt5.get_symbol_info("XAUUSDm"))
        
        if signal:
            print(f"\n{'='*60}")
            print(f"BREAKOUT SIGNAL DETECTED")
            print(f"{'='*60}")
            print(f"Action: {signal['action']}")
            print(f"Entry: {signal['entry_price']}")
            print(f"Stop Loss: {signal['stop_loss']}")
            print(f"Take Profit: {signal['take_profit']}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print(f"Risk Level: {signal['risk_level']}")
            print(f"Reason: {signal['reason']}")
            print(f"\nBreakout Type: {signal['metadata']['breakout_type']}")
            print(f"Volume Ratio: {signal['metadata']['volume_ratio']:.2f}x")
            print(f"\nConditions Met:")
            for condition in signal['metadata']['conditions_met']:
                print(f"  ✓ {condition}")
        else:
            print("No breakout signal detected")
        
        # Check for false breakout
        if strategy.detect_false_breakout(df):
            print("\n⚠️ WARNING: Potential false breakout detected")
        
        mt5.shutdown()

