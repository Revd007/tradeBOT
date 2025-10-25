"""
Mean Reversion Strategy - Specialized for overextended market conditions
Designed to catch reversals when price is too far from moving averages
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    🔥 NEW: Specialized strategy for mean reversion trading in overextended markets.
    Only called when market condition is OVEREXTENDED_BULLISH or OVEREXTENDED_BEARISH.
    """
    
    def __init__(self):
        super().__init__("Mean-Reversion", "HIGH")
        self.min_confidence = 0.45  # 🔥 LOWERED from 0.65 → 0.45 for more signals!
    
    def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
        """
        Look for reversal signals after overextended conditions.
        
        SELL conditions (after overextended bullish):
        - RSI exits overbought zone (70+ → below 70)
        - MACD histogram turning down
        - Bearish confirmation candle
        
        BUY conditions (after overextended bearish):
        - RSI exits oversold zone (below 30 → above 30)
        - MACD histogram turning up
        - Bullish confirmation candle
        """
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check if required indicators exist
        required_cols = ['rsi', 'macd_hist', 'atr', 'ema_21', 'open', 'close', 'high', 'low', 'tick_volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning("⚠️  Missing required indicators for mean reversion analysis")
            return None
        
        # Get 3 recent candles for analysis
        recent_3_candles = df.iloc[-3:]
        
        # --- SELL Conditions (after overextended bullish) ---
        bearish_conditions = {
            # 🔥 LOGIKA BARU: Apakah RSI baru saja keluar dari overbought?
            'rsi_exit_overbought': (recent_3_candles['rsi'].max() > 70) and (current['rsi'] < previous['rsi']),
            'macd_turning_down': current['macd_hist'] < previous['macd_hist'],
            'bearish_candle': current['close'] < current['open'],
            # 🔥 KONDISI BARU: Apakah ada lonjakan volume pada candle pembalikan?
            'volume_spike': current['tick_volume'] > (df['tick_volume'].iloc[-20:-1].mean() * 1.5)
        }
        
        # --- BUY Conditions (after overextended bearish) ---
        bullish_conditions = {
            # 🔥 LOGIKA BARU: Apakah RSI baru saja keluar dari oversold?
            'rsi_exit_oversold': (recent_3_candles['rsi'].min() < 30) and (current['rsi'] > previous['rsi']),
            'macd_turning_up': current['macd_hist'] > previous['macd_hist'],
            'bullish_candle': current['close'] > current['open'],
            # 🔥 KONDISI BARU: Apakah ada lonjakan volume pada candle pembalikan?
            'volume_spike': current['tick_volume'] > (df['tick_volume'].iloc[-20:-1].mean() * 1.5)
        }
        
        # 🔥 NEW: Use weighted scoring system
        bearish_score = self._calculate_score_weighted(bearish_conditions)
        bullish_score = self._calculate_score_weighted(bullish_conditions)
        
        # 🔥 DETAILED LOGGING for debugging
        logger.info(f"🔍 Mean Reversion Analysis (Weighted): Bearish={bearish_score:.2f}, Bullish={bullish_score:.2f}")
        logger.debug(f"   RSI: {current['rsi']:.1f} (prev: {previous['rsi']:.1f})")
        logger.debug(f"   MACD Hist: {current['macd_hist']:.4f} (prev: {previous['macd_hist']:.4f})")
        logger.debug(f"   Volume: {current['tick_volume']} (avg: {df['tick_volume'].iloc[-20:-1].mean():.0f})")
        logger.debug(f"   Bearish conditions: {bearish_conditions}")
        logger.debug(f"   Bullish conditions: {bullish_conditions}")
        
        # 🔥 NEW: Show which conditions contributed to the score
        if bearish_score > 0:
            logger.debug(f"   Bearish score breakdown:")
            if bearish_conditions.get('rsi_exit_overbought'):
                logger.debug(f"     - RSI exit overbought: +0.40")
            if bearish_conditions.get('macd_turning_down'):
                logger.debug(f"     - MACD turning down: +0.25")
            if bearish_conditions.get('bearish_candle'):
                logger.debug(f"     - Bearish candle: +0.20")
            if bearish_conditions.get('volume_spike'):
                logger.debug(f"     - Volume spike: +0.15")
        
        if bullish_score > 0:
            logger.debug(f"   Bullish score breakdown:")
            if bullish_conditions.get('rsi_exit_oversold'):
                logger.debug(f"     - RSI exit oversold: +0.40")
            if bullish_conditions.get('macd_turning_up'):
                logger.debug(f"     - MACD turning up: +0.25")
            if bullish_conditions.get('bullish_candle'):
                logger.debug(f"     - Bullish candle: +0.20")
            if bullish_conditions.get('volume_spike'):
                logger.debug(f"     - Volume spike: +0.15")
        
        if bearish_score >= self.min_confidence:
            return self._create_reversion_signal('SELL', df, symbol_info, bearish_score)
        elif bullish_score >= self.min_confidence:
            return self._create_reversion_signal('BUY', df, symbol_info, bullish_score)
        
        return None
    
    def _calculate_score(self, conditions: Dict[str, bool]) -> float:
        """Calculate confidence score based on condition matches (legacy method)"""
        total_conditions = len(conditions)
        matched_conditions = sum(conditions.values())
        
        if total_conditions == 0:
            return 0.0
        
        base_score = matched_conditions / total_conditions
        
        # Boost score if multiple conditions align
        if matched_conditions >= 3:
            base_score *= 1.2  # 20% boost for strong alignment
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def _calculate_score_weighted(self, conditions: Dict[str, bool]) -> float:
        """
        🔥 NEW: Sistem skor berbobot untuk sinyal mean reversion.
        Memberikan bobot berbeda pada setiap kondisi berdasarkan pentingnya.
        """
        score = 0.0
        
        # Beri bobot lebih pada pemicu utama (momentum)
        if conditions.get('rsi_exit_overbought') or conditions.get('rsi_exit_oversold'):
            score += 0.40  # Pemicu utama - RSI keluar dari zona jenuh
        
        if conditions.get('macd_turning_down') or conditions.get('macd_turning_up'):
            score += 0.25  # Konfirmasi momentum sekunder - MACD berbalik
        
        # Bobot lebih rendah untuk konfirmasi candle & volume
        if conditions.get('bearish_candle') or conditions.get('bullish_candle'):
            score += 0.20  # Konfirmasi harga - Candle pembalikan
        
        if conditions.get('volume_spike'):
            score += 0.15  # Konfirmasi partisipasi pasar - Volume spike
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _create_reversion_signal(self, action: str, df: pd.DataFrame, symbol_info: Dict, confidence: float) -> Dict:
        """Create mean reversion signal with dynamic targets"""
        current = df.iloc[-1]
        entry_price = current['close']
        atr = current['atr']
        
        # Target TP is EMA 21 (the "mean" we're reverting to)
        take_profit = current['ema_21']
        
        if action == 'SELL':
            # SL placed above recent high
            stop_loss = df['high'].iloc[-5:].max() + (atr * 0.5)
            # Ensure TP is below entry price
            if take_profit >= entry_price:
                take_profit = entry_price - (atr * 2.0)  # Fallback TP
        else:  # BUY
            # SL placed below recent low
            stop_loss = df['low'].iloc[-5:].min() - (atr * 0.5)
            # Ensure TP is above entry price
            if take_profit <= entry_price:
                take_profit = entry_price + (atr * 2.0)  # Fallback TP
        
        # Validate Risk/Reward ratio (minimum 1:1 for mean reversion)
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward < risk:
            logger.debug(f"Mean reversion signal ignored: poor R/R ratio ({reward/risk:.2f}:1)")
            return None
        
        logger.info(f"✅ Mean Reversion {action} signal: R/R = {reward/risk:.2f}:1, Target: EMA21")
        
        return {
            'action': action,
            'entry_price': round(entry_price, symbol_info['digits']),
            'stop_loss': round(stop_loss, symbol_info['digits']),
            'take_profit': round(take_profit, symbol_info['digits']),
            'confidence': confidence,
            'reason': f"Mean Reversion {action} - targeting EMA21 (R/R: {reward/risk:.1f}:1)",
            'risk_level': 'HIGH',
            'strategy_type': 'mean_reversion'
        }


if __name__ == "__main__":
    # Test mean reversion strategy
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        strategy = MeanReversionStrategy()
        
        # Get test data
        df = mt5.get_candles("XAUUSDm", "M5", count=200)
        df = strategy.add_all_indicators(df)
        symbol_info = mt5.get_symbol_info("XAUUSDm")
        
        # Test analysis
        signal = strategy.analyze(df, symbol_info)
        
        if signal:
            print(f"\n✅ Mean Reversion Signal Found:")
            print(f"Action: {signal['action']}")
            print(f"Entry: {signal['entry_price']}")
            print(f"SL: {signal['stop_loss']}")
            print(f"TP: {signal['take_profit']}")
            print(f"Confidence: {signal['confidence']:.2%}")
            print(f"Reason: {signal['reason']}")
        else:
            print("\n⚪ No mean reversion signal found")
        
        mt5.shutdown()
