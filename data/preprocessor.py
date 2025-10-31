"""
Data Preprocessing for ML Models
🔥 FINAL VERSION: Lean, Mean, High-Impact Features for Cost-Sensitive Learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Helper class untuk kalkulasi indikator teknikal"""
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate EMA dengan error handling"""
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR (Average True Range)"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi


class DataPreprocessor:
    """Preprocesses market data for ML models - Specialist Architecture"""
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.indicators = TechnicalIndicators()
    
    def create_features_scalper(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔥 SIMPLIFIED SCALPER BRAIN (M5/M15): Price Action + Core Indicators (ATR, RSI, EMA)"""
        logger.info("   🔥 Activating SIMPLIFIED CANDLE-FOCUSED brain: Less indicators, more signal quality...")
        
        df = df.copy()
        
        # --- CORE PRICE ACTION (Lightweight) ---
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        body_range = (df['high'] - df['low']).replace(0, np.nan)
        df['body_to_range_ratio'] = df['body_size'] / body_range
        df['upper_wick_to_range_ratio'] = df['upper_wick'] / body_range
        df['lower_wick_to_range_ratio'] = df['lower_wick'] / body_range
        
        # Basic candle types
        df['is_doji'] = (df['body_to_range_ratio'] < 0.1).astype(int)
        df['is_hammer'] = ((df['lower_wick_to_range_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.3)).astype(int)
        df['is_shooting_star'] = ((df['upper_wick_to_range_ratio'] > 0.6) & (df['body_to_range_ratio'] < 0.3)).astype(int)
        
        # Close strength
        df['close_position_in_candle'] = (df['close'] - df['low']) / body_range
        df['is_strong_close'] = (
            ((df['close'] > df['open']) & (df['close_position_in_candle'] > 0.8)) | 
            ((df['close'] < df['open']) & (df['close_position_in_candle'] < 0.2))
        ).astype(int)
        
        # Volume (if available)
        if 'tick_volume' in df.columns:
            vol_ma = df['tick_volume'].rolling(10).mean().replace(0, np.nan)
            df['volume_spike'] = (df['tick_volume'] > vol_ma * 1.5).astype(int)
        else:
            df['volume_spike'] = 0
        
        # Consecutive candles
        is_bullish_mask = (df['close'] > df['open'])
        is_bearish_mask = (df['close'] < df['open'])
        df['consecutive_bullish'] = is_bullish_mask.groupby((~is_bullish_mask).cumsum()).cumsum()
        df['consecutive_bearish'] = is_bearish_mask.groupby((~is_bearish_mask).cumsum()).cumsum()
        
        # Simple patterns
        df['bullish_pattern'] = (
            df['is_strong_close'] * 2.0 +
            df['is_hammer'] * 1.5 +
            df['volume_spike'] * 1.0 -
            df['is_doji'] * 1.0
        ).clip(lower=0)
        
        df['bearish_pattern'] = (
            (df['close'] < df['open']).astype(int) * 2.0 +
            df['is_shooting_star'] * 1.5 +
            df['volume_spike'] * 1.0 -
            df['is_doji'] * 1.0
        ).clip(lower=0)
        
        # --- CORE INDICATORS ONLY (reduce noise) ---
        try:
            df['atr'] = self.indicators.calculate_atr(df)
        except Exception:
            df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        try:
            df['rsi'] = self.indicators.calculate_rsi(df)
        except Exception:
            # Fallback quick RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
        df['ema_9'] = self.indicators.calculate_ema(df, 9)
        df['ema_21'] = self.indicators.calculate_ema(df, 21)
        df['ema_cross_bull'] = (df['ema_9'] > df['ema_21']).astype(int)

        # Minimal trend/overextension flags
        df['overbought'] = (df['rsi'] > 70).astype(int)
        df['oversold'] = (df['rsi'] < 30).astype(int)

        # Legacy compatibility (keep columns downstream expect)
        df['bullish_confluence_score'] = df['bullish_pattern']
        df['candle_quality_score'] = df['bullish_pattern']
        
        return df
    
    def create_features_swing(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔥 PURE CANDLE SWING BRAIN (H1/H4): NO INDICATORS!"""
        logger.info("   🔥 Activating PURE CANDLE SWING brain: NO INDICATORS...")
        
        df = df.copy()
        
        # --- SIMPLE TREND FROM PRICE MOMENTUM (NO EMA!) ---
        df['price_momentum_10'] = df['close'].pct_change(10) * 100
        df['price_momentum_20'] = df['close'].pct_change(20) * 100
        
        df['is_uptrend'] = (df['price_momentum_10'] > 0.5).astype(int)
        df['is_downtrend'] = (df['price_momentum_10'] < -0.5).astype(int)
        df['trend_strength'] = abs(df['price_momentum_20'])
        
        # Simple candle patterns (same as scalper)
        df['body_size'] = abs(df['close'] - df['open'])
        body_range = (df['high'] - df['low']).replace(0, np.nan)
        df['body_to_range_ratio'] = df['body_size'] / body_range
        df['is_doji'] = (df['body_to_range_ratio'] < 0.1).astype(int)
        df['is_strong_close'] = (
            ((df['close'] > df['open']) & ((df['close'] - df['low']) / body_range > 0.8)) | 
            ((df['close'] < df['open']) & ((df['close'] - df['low']) / body_range < 0.2))
        ).astype(int)
        
        # Simple volume
        if 'tick_volume' in df.columns:
            vol_ma = df['tick_volume'].rolling(20).mean().replace(0, np.nan)
            df['volume_spike'] = (df['tick_volume'] > vol_ma * 1.3).astype(int)
        else:
            df['volume_spike'] = 0
        
        # Simple patterns (NO EMA!)
        df['bullish_pattern'] = (
            df['is_uptrend'] * 2.0 +
            df['is_strong_close'] * 1.5 +
            df['volume_spike'] * 1.0 -
            df['is_doji'] * 1.0
        ).clip(lower=0)
        
        df['bearish_pattern'] = (
            df['is_downtrend'] * 2.0 +
            (df['close'] < df['open']).astype(int) * 1.5 +
            df['volume_spike'] * 1.0 -
            df['is_doji'] * 1.0
        ).clip(lower=0)
        
        # Legacy compatibility
        df['bullish_confluence_score'] = df['bullish_pattern']
        df['candle_quality_score'] = df['bullish_pattern']
        
        return df
    
    def create_features(self, df: pd.DataFrame, timeframe: str = 'M5') -> pd.DataFrame:
        """
        🔥 FITUR DNA PASAR: Fitur universal yang menangkap perilaku, bukan hanya pola.
        """
        df = df.copy()
        
        # =====================================================
        # 1. VOLATILITAS (Volatility) - Seberapa "liar" pasar?
        # =====================================================
        df['candle_range'] = df['high'] - df['low']
        
        # Calculate ATR (True Range) for more accurate volatility
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volatilitas relatif terhadap harga (ATR Percentage)
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # Akselerasi Volatilitas: Apakah volatilitas meningkat atau menurun?
        df['volatility_accel'] = df['atr_pct'].diff(periods=5)
        
        # =====================================================
        # 2. MOMENTUM - Kekuatan pergerakan harga saat ini
        # =====================================================
        df['momentum_5'] = df['close'].pct_change(periods=5) * 100
        df['momentum_14'] = df['close'].pct_change(periods=14) * 100
        
        # =====================================================
        # 3. KARAKTER CANDLE - Siapa yang mengontrol?
        # =====================================================
        full_range = (df['high'] - df['low']).replace(0, np.nan)
        df['body_vs_range'] = (abs(df['close'] - df['open']) / full_range).fillna(0)
        df['close_pos_in_range'] = ((df['close'] - df['low']) / full_range).fillna(0.5)
        
        # =====================================================
        # 4. KEKUATAN TREN - Higher highs atau lower lows?
        # =====================================================
        rolling_max = df['high'].rolling(window=20).max()
        rolling_min = df['low'].rolling(window=20).min()
        df['trend_strength'] = ((df['close'] - rolling_min) / (rolling_max - rolling_min)).fillna(0.5) * 2 - 1
        
        # =====================================================
        # 🔥🔥🔥 FITUR SPESIFIK BEARISH (Critical for SELL)
        # =====================================================
        # 1. Exhaustion Wick
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        df['exhaustion_wick_ratio'] = upper_wick / full_range

        # 2. Bearish Engulfing
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        is_prev_bullish = prev_close > prev_open
        is_current_bearish = df['close'] < df['open']
        engulfs_body = (df['open'] > prev_close) & (df['close'] < prev_open)
        df['is_bearish_engulfing'] = (is_prev_bullish & is_current_bearish & engulfs_body).astype(int)

        # 3. Failed Breakout
        high_20 = df['high'].rolling(20).max().shift(1)
        df['failed_breakout'] = (
            (df['high'] > high_20 * 0.99) & (df['close'] < df['open'])
        ).astype(int)
        
        # 4. Momentum Divergence (slowing upward momentum)
        df['momentum_divergence'] = (
            (df['momentum_5'] > 0) & (df['momentum_5'] < df['momentum_14'] * 0.5)
        ).astype(int)
        
        # =====================================================
        # 🔥🔥🔥 FITUR SPESIFIK BULLISH (Critical for BUY)
        # =====================================================
        # 1. Lower Wick (buying pressure)
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        df['buy_pressure_wick_ratio'] = lower_wick / full_range
        
        # 2. Bullish Engulfing
        is_prev_bearish = prev_close < prev_open
        is_current_bullish = df['close'] > df['open']
        engulfs_body_bullish = (df['open'] < prev_close) & (df['close'] > prev_open)
        df['is_bullish_engulfing'] = (is_prev_bearish & is_current_bullish & engulfs_body_bullish).astype(int)
        
        # 3. Breakout Success
        low_20 = df['low'].rolling(20).min().shift(1)
        df['successful_breakout'] = (
            (df['low'] < low_20 * 1.01) & (df['close'] > df['open'])
        ).astype(int)
        
        # 4. Momentum Acceleration (increasing upward momentum)
        df['momentum_acceleration'] = (
            (df['momentum_5'] > 0) & (df['momentum_5'] > df['momentum_14'] * 1.5)
        ).astype(int)
        
        # =====================================================
        # 🔥 NEWS-ENHANCED FEATURES (if available)
        # =====================================================
        if 'news_sentiment_score' in df.columns:
            # News Momentum: Is sentiment changing?
            df['news_sentiment_change'] = df['news_sentiment_score'].diff(5)
            
            # Sentiment-Price Divergence
            if 'close' in df.columns:
                price_change = df['close'].pct_change(5)
                df['sentiment_price_divergence'] = (
                    (price_change > 0) & (df['news_sentiment_change'] < 0)
                ).astype(int) - (
                    (price_change < 0) & (df['news_sentiment_change'] > 0)
                ).astype(int)
        
        if 'news_bullish_ratio' in df.columns and 'news_bearish_ratio' in df.columns:
            df['news_sentiment_balance'] = df['news_bullish_ratio'] - df['news_bearish_ratio']
            
            df['strong_bullish_news'] = (
                (df['news_bullish_ratio'] > 0.6) & (df.get('news_sentiment_score', 0) > 0.2)
            ).astype(int)
            
            df['strong_bearish_news'] = (
                (df['news_bearish_ratio'] > 0.6) & (df.get('news_sentiment_score', 0) < -0.2)
            ).astype(int)
        
        if 'calendar_impact_score' in df.columns:
            df['high_calendar_impact'] = (df['calendar_impact_score'] > 7.0).astype(int)
            
            if 'atr_pct' in df.columns:
                df['calendar_vol_synergy'] = df['calendar_impact_score'] * df['atr_pct']
        
        # =====================================================
        # 🔥🔥🔥 ADVANCED MARKET FEATURES (Proven for Trading)
        # =====================================================
        
        # 1. Price Acceleration (change in momentum)
        df['price_acceleration_5'] = df['momentum_5'].diff(5)
        df['price_acceleration_10'] = df['momentum_14'].diff(7)
        
        # 2. Volume-Price Confirmation
        if 'tick_volume' in df.columns:
            price_change_3 = df['close'].pct_change(3)
            volume_change_3 = df['tick_volume'].pct_change(3)
            df['volume_price_sync'] = (
                (price_change_3 * volume_change_3) > 0
            ).astype(int)
            
            # Volume strength
            vol_ma_20 = df['tick_volume'].rolling(20).mean()
            df['volume_strength'] = (df['tick_volume'] / vol_ma_20).fillna(1.0)
        else:
            df['volume_price_sync'] = 0
            df['volume_strength'] = 1.0
        
        # 3. Support/Resistance Levels (simplified)
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_level'] = df['low'].rolling(20).min()
        
        # Distance to S/R levels (normalized)
        range_sr = (df['resistance_level'] - df['support_level']).replace(0, np.nan)
        df['distance_to_resistance'] = ((df['resistance_level'] - df['close']) / range_sr).fillna(0.5)
        df['distance_to_support'] = ((df['close'] - df['support_level']) / range_sr).fillna(0.5)
        
        # Near S/R flags
        df['near_resistance'] = (df['close'] > df['resistance_level'] * 0.98).astype(int)
        df['near_support'] = (df['close'] < df['support_level'] * 1.02).astype(int)
        
        # 4. Market Regime Detection
        if 'atr_pct' in df.columns:
            # Volatility regime (0=low, 1=medium, 2=high, 3=extreme)
            try:
                df['volatility_regime'] = pd.qcut(
                    df['atr_pct'], 
                    q=4, 
                    labels=[0, 1, 2, 3], 
                    duplicates='drop'
                ).astype(int)
            except:
                df['volatility_regime'] = 1  # Default to medium
        
        # Trend regime (0=ranging, 1=weak trend, 2=strong trend)
        try:
            df['trend_regime'] = pd.qcut(
                abs(df['momentum_14']), 
                q=3, 
                labels=[0, 1, 2], 
                duplicates='drop'
            ).astype(int)
        except:
            df['trend_regime'] = 1  # Default to weak trend
        
        # =====================================================
        # 🔥🔥🔥 MARKET DNA: ADX & REGIME FILTER (NEW!)
        # =====================================================
        # ADX adalah "GURU" yang menentukan strategi mana yang cocok
        
        # Calculate Directional Movement
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        # Only keep positive movements
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth with EMA (Wilder's smoothing)
        alpha = 1/14
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
        
        # True Range (already have ATR, but need smooth TR)
        tr_smooth = true_range.ewm(alpha=alpha, adjust=False).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        # ADX calculation
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # 🔥 REGIME FEATURES: Model akan belajar kapan menggunakan strategi apa
        df['market_regime_trending'] = (df['adx'] > 25).astype(int)
        df['market_regime_ranging'] = (df['adx'] < 20).astype(int)
        df['market_regime_transition'] = ((df['adx'] >= 20) & (df['adx'] <= 25)).astype(int)
        
        # Trend Direction (from DI)
        df['trend_direction_bullish'] = (plus_di > minus_di + 5).astype(int)
        df['trend_direction_bearish'] = (minus_di > plus_di + 5).astype(int)
        
        # 🔥 NEW: Context interaction - momentum quality gated by trend strength
        # Model learns that momentum matters more when trend is strong
        try:
            df['trend_x_momentum'] = (df['adx'].fillna(0) * df['momentum_14'].fillna(0))
        except Exception:
            df['trend_x_momentum'] = 0.0

        # Volatility spike detection (extreme moves)
        atr_baseline = df['atr'].rolling(20).mean()
        df['volatility_spike'] = (df['atr'] > atr_baseline * 1.5).astype(int)
        
        # =====================================================
        # 🔥🔥🔥 CONFIRMED BREAKOUT FEATURES (NEW!)
        # =====================================================
        # Model akan belajar bahwa breakout dengan konfirmasi >> breakout biasa
        
        # 1. Bullish Breakout: Close above resistance (not just wick)
        resistance_20 = df['high'].rolling(20).max().shift(1)
        df['confirmed_bullish_breakout'] = (
            (df['close'].shift(1) < resistance_20) &  # Was below
            (df['close'] > resistance_20) &           # Now closed above
            (df['close'] > df['open']) &              # Bullish candle
            (df['volume_strength'] > 1.3)             # Volume confirmation
        ).astype(int)
        
        # 2. Bearish Breakout: Close below support (not just wick)
        support_20 = df['low'].rolling(20).min().shift(1)
        df['confirmed_bearish_breakout'] = (
            (df['close'].shift(1) > support_20) &     # Was above
            (df['close'] < support_20) &              # Now closed below
            (df['close'] < df['open']) &              # Bearish candle
            (df['volume_strength'] > 1.3)             # Volume confirmation
        ).astype(int)
        
        # 3. Build-up before breakout (low volatility = pressure building)
        atr_recent = df['atr'].rolling(10).mean()
        atr_baseline_buildup = df['atr'].rolling(30).mean()
        df['buildup_detected'] = (
            (atr_recent / atr_baseline_buildup) < 0.7
        ).astype(int)
        
        # Breakout WITH build-up (very powerful!)
        df['explosive_bullish_breakout'] = (
            df['confirmed_bullish_breakout'] & df['buildup_detected']
        ).astype(int)
        
        df['explosive_bearish_breakout'] = (
            df['confirmed_bearish_breakout'] & df['buildup_detected']
        ).astype(int)
        
        # =====================================================
        # 🔥🔥🔥 DIVERGENCE FEATURES (NEW!)
        # =====================================================
        # RSI Divergence adalah sinyal counter-trend terkuat
        
        # Simple RSI calculation (if not exists)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Price peaks and troughs
        price_high_20 = df['high'].rolling(20).max()
        price_low_20 = df['low'].rolling(20).min()
        
        # RSI peaks and troughs
        rsi_high_20 = df['rsi'].rolling(20).max()
        rsi_low_20 = df['rsi'].rolling(20).min()
        
        # Bullish Divergence: Price makes lower low, RSI makes higher low
        price_making_lower_low = df['low'] < price_low_20.shift(10)
        rsi_making_higher_low = df['rsi'] > rsi_low_20.shift(10)
        df['bullish_divergence'] = (
            price_making_lower_low & rsi_making_higher_low & (df['rsi'] < 40)
        ).astype(int)
        
        # Bearish Divergence: Price makes higher high, RSI makes lower high
        price_making_higher_high = df['high'] > price_high_20.shift(10)
        rsi_making_lower_high = df['rsi'] < rsi_high_20.shift(10)
        df['bearish_divergence'] = (
            price_making_higher_high & rsi_making_lower_high & (df['rsi'] > 60)
        ).astype(int)
        
        # =====================================================
        # 🔥🔥🔥 OVEREXTENSION FEATURES (NEW!)
        # =====================================================
        # Mengukur seberapa jauh harga dari "nilai wajar" (EMA)
        
        # Calculate EMAs (if not exists)
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Distance from EMA in ATR units (quantitative!)
        df['overextension_ema21_atr'] = (df['close'] - df['ema_21']) / df['atr']
        df['overextension_ema50_atr'] = (df['close'] - df['ema_50']) / df['atr']
        
        # Overextended flags (price too far from mean)
        df['overextended_bullish'] = (df['overextension_ema21_atr'] > 1.5).astype(int)
        df['overextended_bearish'] = (df['overextension_ema21_atr'] < -1.5).astype(int)
        
        # Extreme overextension (very strong reversion signal)
        df['extreme_overextended_bullish'] = (df['overextension_ema21_atr'] > 2.5).astype(int)
        df['extreme_overextended_bearish'] = (df['overextension_ema21_atr'] < -2.5).astype(int)
        
        # 5. Price Position in Range (Stochastic-like)
        rolling_high_50 = df['high'].rolling(50).max()
        rolling_low_50 = df['low'].rolling(50).min()
        range_50 = (rolling_high_50 - rolling_low_50).replace(0, np.nan)
        df['price_position_50'] = ((df['close'] - rolling_low_50) / range_50).fillna(0.5)
        
        # 6. Momentum Quality (consistent momentum)
        # Count consecutive bullish/bearish momentum
        momentum_sign = np.sign(df['momentum_5'])
        df['momentum_consistency'] = (
            momentum_sign.groupby((momentum_sign != momentum_sign.shift()).cumsum()).cumcount() + 1
        )
        
        # =====================================================
        # FINAL CLEANUP
        # =====================================================
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Count new DNA features
        dna_features = [
            'adx', 'plus_di', 'minus_di',
            'market_regime_trending', 'market_regime_ranging', 
            'confirmed_bullish_breakout', 'confirmed_bearish_breakout',
            'bullish_divergence', 'bearish_divergence',
            'overextension_ema21_atr', 'overextended_bullish', 'overextended_bearish'
        ]
        existing_dna = sum(1 for f in dna_features if f in df.columns)
        
        logger.info(f"   ✅ Market DNA + Advanced features created ({existing_dna}/{len(dna_features)} DNA features active)")
        logger.info(f"      🎯 ADX-based regime filter, confirmed breakouts, divergence, overextension")
        return df
    
    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        median_spread: float,
        tp_multiplier: float = 1.5,
        sl_multiplier: float = 1.0,
        inv_freq_mult: float = 1.0,
        use_dynamic_barriers: bool = True
    ) -> Tuple[pd.Series, pd.Series]:
        """
        🔥🔥🔥 TRIPLE-BARRIER LABELING (STANDAR INDUSTRI) 🔥🔥🔥
        Melabeli data berdasarkan hasil simulasi trade (TP/SL/Waktu).
        
        Args:
            df: DataFrame with OHLC data
            horizon: Lookforward period (time barrier)
            median_spread: Median spread for friction calculation
            tp_multiplier: Take Profit distance in ATR multiples
            sl_multiplier: Stop Loss distance in ATR multiples
            inv_freq_mult: Inverse frequency weight multiplier
            use_dynamic_barriers: Adaptive TP/SL based on market volatility
        """
        logger.info(f"🎯 Applying Triple-Barrier Labeling: horizon={horizon}, TP={tp_multiplier}x, SL={sl_multiplier}x")
        if use_dynamic_barriers:
            logger.info("   🔥 Using DYNAMIC barriers (volatility-adaptive)")

        labels = pd.Series(-1, index=df.index, dtype=np.int8)
        weights = pd.Series(1.0, index=df.index, dtype=np.float32)
        
        # Calculate ATR if not exists (untuk menentukan jarak TP/SL)
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean().fillna(method='bfill')
        
        close_prices = df['close'].to_numpy()
        high_prices = df['high'].to_numpy()
        low_prices = df['low'].to_numpy()
        atr_values = df['atr'].to_numpy()
        
        # Get timeframe for context
        current_tf = 'M15'  # Default
        if hasattr(df, 'attrs') and 'timeframe' in df.attrs:
            current_tf = df.attrs['timeframe']
        
        buy_count = 0
        sell_count = 0
        hold_count = 0
        
        # 🔥 DYNAMIC BARRIERS: Calculate volatility factor for adaptive TP/SL
        if use_dynamic_barriers and 'atr_pct' in df.columns:
            # Volatility factor: current ATR vs median ATR
            atr_pct_median = df['atr_pct'].median()
            atr_pct_values = df['atr_pct'].to_numpy()
        else:
            atr_pct_values = None

        for i in range(len(df) - horizon):
            entry_price = close_prices[i]
            atr = atr_values[i]
            
            # Skip jika ATR sangat kecil (avoid unrealistic targets)
            if np.isnan(atr) or atr < entry_price * 0.0001:
                hold_count += 1
                continue

            # 🔥 DYNAMIC MULTIPLIERS based on current volatility
            current_tp_mult = tp_multiplier
            current_sl_mult = sl_multiplier
            
            if use_dynamic_barriers and atr_pct_values is not None:
                # Adaptive multiplier: widen targets in high vol, tighten in low vol
                vol_factor = atr_pct_values[i] / atr_pct_median if atr_pct_median > 0 else 1.0
                
                # Scale TP/SL: higher vol = wider targets
                current_tp_mult = tp_multiplier * np.clip(vol_factor, 0.8, 1.3)
                current_sl_mult = sl_multiplier * np.clip(vol_factor, 0.8, 1.3)
            
            # 🔥 TRIPLE BARRIER: Tentukan 3 barrier (with dynamic multipliers)
            take_profit_buy = entry_price + (atr * current_tp_mult)
            stop_loss_buy = entry_price - (atr * current_sl_mult)
            
            take_profit_sell = entry_price - (atr * current_tp_mult)
            stop_loss_sell = entry_price + (atr * current_sl_mult)
            
            # Lihat ke masa depan (dalam rentang horizon)
            tp_hit_buy = False
            sl_hit_buy = False
            tp_hit_sell = False
            sl_hit_sell = False
            
            tp_hit_time_buy = float('inf')
            sl_hit_time_buy = float('inf')
            tp_hit_time_sell = float('inf')
            sl_hit_time_sell = float('inf')
            
            for j in range(1, min(horizon + 1, len(df) - i)):
                future_high = high_prices[i + j]
                future_low = low_prices[i + j]
                
                # Check BUY scenario barriers
                if not tp_hit_buy and future_high >= take_profit_buy:
                    tp_hit_buy = True
                    tp_hit_time_buy = j
                
                if not sl_hit_buy and future_low <= stop_loss_buy:
                    sl_hit_buy = True
                    sl_hit_time_buy = j
                
                # Check SELL scenario barriers
                if not tp_hit_sell and future_low <= take_profit_sell:
                    tp_hit_sell = True
                    tp_hit_time_sell = j
                
                if not sl_hit_sell and future_high >= stop_loss_sell:
                    sl_hit_sell = True
                    sl_hit_time_sell = j
                
                # Early exit if both scenarios resolved
                if (tp_hit_buy or sl_hit_buy) and (tp_hit_sell or sl_hit_sell):
                    break
            
            # 🔥 LABELING LOGIC: Pilih scenario terbaik (BUY atau SELL)
            # BUY scenario: TP tercapai sebelum SL
            buy_profitable = tp_hit_buy and (tp_hit_time_buy < sl_hit_time_buy)
            # SELL scenario: TP tercapai sebelum SL
            sell_profitable = tp_hit_sell and (tp_hit_time_sell < sl_hit_time_sell)
            
            # Assign label based on which scenario is better
            if buy_profitable and not sell_profitable:
                labels.iloc[i] = 1  # BUY
                # Weight by how fast TP was hit
                weights.iloc[i] = 1.0 + (horizon / max(1, tp_hit_time_buy))
                buy_count += 1
                
            elif sell_profitable and not buy_profitable:
                labels.iloc[i] = 0  # SELL
                # Weight by how fast TP was hit
                weights.iloc[i] = 1.0 + (horizon / max(1, tp_hit_time_sell))
                sell_count += 1
                
            elif buy_profitable and sell_profitable:
                # Both profitable: choose the one that hits TP faster
                if tp_hit_time_buy < tp_hit_time_sell:
                    labels.iloc[i] = 1  # BUY
                    weights.iloc[i] = 1.0 + (horizon / max(1, tp_hit_time_buy))
                    buy_count += 1
                else:
                    labels.iloc[i] = 0  # SELL
                    weights.iloc[i] = 1.0 + (horizon / max(1, tp_hit_time_sell))
                    sell_count += 1
            else:
                # Neither profitable or both hit SL: HOLD
                hold_count += 1

        counts = labels.value_counts().sort_index()
        logger.info(f"   Triple-Barrier Label Dist: SELL={counts.get(0, 0)}, BUY={counts.get(1, 0)}, HOLD={counts.get(-1, 0)}")
        
        # 🔥 INVERSE FREQUENCY WEIGHTING: Balance classes
        n_buy = counts.get(1, 0)
        n_sell = counts.get(0, 0)
        total = max(1, n_buy + n_sell)
        
        if n_buy > 0 and n_sell > 0:
            inv_freq = {
                1: (total / (2 * n_buy)) * inv_freq_mult,
                0: (total / (2 * n_sell)) * inv_freq_mult
            }
        else:
            inv_freq = {
                1: 1.0 * inv_freq_mult,
                0: 1.0 * inv_freq_mult
        }
        
        for i in range(len(labels)):
            if labels.iloc[i] != -1:
                weights.iloc[i] = weights.iloc[i] * inv_freq[labels.iloc[i]]
        
        logger.info(f"   Applied inverse-frequency weights (mult={inv_freq_mult}): BUY x{inv_freq[1]:.2f}, SELL x{inv_freq[0]:.2f}")
        
        return labels, weights
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'target',
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        df_clean = df.dropna()
        
        if feature_columns is None:
            exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume',
                           target_column, 'spread', 'real_volume', 'spread_price', 'weights']
            feature_columns = [col for col in df_clean.columns if col not in exclude_cols]
        
        self.feature_names = feature_columns
        
        X = df_clean[feature_columns].values
        y = df_clean[target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training data prepared: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        logger.info(f"Features: {len(feature_columns)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_importance(
        self,
        model,
        top_n: int = 20
    ) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importances = model.feature_importances_
        
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        feature_imp = feature_imp.sort_values('importance', ascending=False)
        
        return feature_imp.head(top_n)
