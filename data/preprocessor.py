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


class DataPreprocessor:
    """Preprocesses market data for ML models - Specialist Architecture"""
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.indicators = TechnicalIndicators()
    
    def create_features_scalper(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔥 CANDLE-FOCUSED BRAIN (M5/M15): Pure Price Action, No Indicators!"""
        logger.info("   🔥 Activating CANDLE-FOCUSED brain: Pure Price Action...")
        
        df = df.copy()
        
        # --- CANDLE PATTERNS ONLY (No Indicators!) ---
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
        
        # Simple volume (if available)
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
        
        # Legacy compatibility
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
        🔥 ARSITEKTUR ROUTER: Panggil otak yang tepat untuk TF yang tepat
        """
        df = df.copy()
        
        # --- PURE CANDLE FEATURES ONLY (NO INDICATORS!) ---
        df['price_change_1'] = df['close'].pct_change(1) * 100
        df['price_change_3'] = df['close'].pct_change(3) * 100
        
        # Simple volatility from price range (NO ATR!)
        df['candle_range'] = df['high'] - df['low']
        df['range_ma_5'] = df['candle_range'].rolling(5).mean()
        df['is_high_vol'] = (df['candle_range'] > df['range_ma_5'] * 1.2).astype(int)

        # Simple trend from price momentum (NO SMA!)
        df['price_momentum'] = df['close'].pct_change(5) * 100
        df['is_strong_trend'] = (abs(df['price_momentum']) > 1.0).astype(int)
        df['is_choppy'] = (abs(df['price_momentum']) < 0.3).astype(int)

        # --- ROUTER: Panggil specialist brain ---
        if timeframe in ['M5', 'M15']:
            df = self.create_features_scalper(df)
        elif timeframe in ['H1', 'H4']:
            df = self.create_features_swing(df)
        
        # --- SIMPLE COMMON FEATURES ---
        # Simple market regime filter
        df['market_regime_filter'] = (
            df.get('is_strong_trend', 0) * 0.5 +
            (1 - df.get('is_choppy', 0)) * 0.3 +
            df.get('is_high_vol', 0) * 0.2
        ).clip(0, 1)

        # 🔥🔥🔥 FITUR BARU: Tambahkan Fitur Spesifik Bearish 🔥🔥🔥
        # 1. Exhaustion Wick: Wick atas yang panjang menunjukkan tekanan jual.
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        candle_range = df['high'] - df['low']
        df['exhaustion_wick_ratio'] = upper_wick / candle_range.replace(0, np.nan)

        # 2. Bearish Engulfing Sederhana (Vectorized)
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        is_prev_bullish = prev_close > prev_open
        is_current_bearish = df['close'] < df['open']
        engulfs_body = (df['open'] > prev_close) & (df['close'] < prev_open)
        df['is_bearish_engulfing'] = (is_prev_bullish & is_current_bearish & engulfs_body).astype(int)

        # 3. Momentum Positif yang Melambat (Tanda Kelelahan Pembeli)
        price_change_1 = df['close'].diff(1)
        price_change_3 = df['close'].diff(3)
        df['slowing_momentum_up'] = ((price_change_3 > 0) & (price_change_1 < (price_change_3 / 3))).astype(int)
        
        # 4. Volume Divergence Bearish: Harga naik tapi volume turun
        if 'tick_volume' in df.columns:
            vol_ma_5 = df['tick_volume'].rolling(5).mean()
            price_momentum_3 = df['close'].pct_change(3)
            vol_momentum_3 = df['tick_volume'].pct_change(3)
            df['bearish_volume_divergence'] = ((price_momentum_3 > 0) & (vol_momentum_3 < -0.1)).astype(int)
        else:
            df['bearish_volume_divergence'] = 0
        
        # 5. Resistance Rejection: Harga mendekati high sebelumnya tapi gagal menembus
        high_10 = df['high'].rolling(10).max().shift(1)
        df['resistance_rejection'] = ((df['high'] > high_10 * 0.99) & (df['close'] < df['open'])).astype(int)
        
        # 6. 🔥 BARU: Weakness Detection - Harga naik tapi momentum melemah
        df['price_weakness'] = (
            (df['close'] > df['open']) &  # Candle hijau
            (df['close'].pct_change(1) < df['close'].pct_change(3) * 0.5)  # Momentum melemah
        ).astype(int)
        
        # 7. 🔥 BARU: Failed Breakout - Harga mencoba breakout tapi gagal
        df['failed_breakout'] = (
            (df['high'] > df['high'].rolling(5).max().shift(1)) &  # Break high
            (df['close'] < df['open'])  # Tapi close bearish
        ).astype(int)

        # 🔥🔥🔥-------------------------------------------🔥🔥🔥
        
        # 🔥🔥🔥 NEWS-ENHANCED FEATURES (if available) 🔥🔥🔥
        if 'news_sentiment_score' in df.columns:
            # News Momentum: Is sentiment changing?
            df['news_sentiment_change'] = df['news_sentiment_score'].diff(5)
            
            # Sentiment-Price Divergence: Price up but sentiment down = potential reversal
            if 'close' in df.columns:
                price_change = df['close'].pct_change(5)
                df['sentiment_price_divergence'] = (
                    (price_change > 0) & (df['news_sentiment_change'] < 0)
                ).astype(int) - (
                    (price_change < 0) & (df['news_sentiment_change'] > 0)
                ).astype(int)
            
            # Sentiment-Volatility Interaction: High sentiment + low vol = compression
            if 'atr_pct' in df.columns and 'regime_high_vol' in df.columns:
                df['sentiment_vol_interaction'] = (
                    abs(df['news_sentiment_score']) * (1 - df['regime_high_vol'])
                )
        
        if 'news_bullish_ratio' in df.columns and 'news_bearish_ratio' in df.columns:
            # News Sentiment Balance: Bullish vs Bearish ratio
            df['news_sentiment_balance'] = df['news_bullish_ratio'] - df['news_bearish_ratio']
            
            # Strong Bullish News: High bullish ratio + positive sentiment
            df['strong_bullish_news'] = (
                (df['news_bullish_ratio'] > 0.6) & (df['news_sentiment_score'] > 0.2)
            ).astype(int)
            
            # Strong Bearish News: High bearish ratio + negative sentiment
            df['strong_bearish_news'] = (
                (df['news_bearish_ratio'] > 0.6) & (df['news_sentiment_score'] < -0.2)
            ).astype(int)
        
        if 'calendar_impact_score' in df.columns:
            # High-Impact Event Proximity: Is a major event near?
            df['high_calendar_impact'] = (df['calendar_impact_score'] > 7.0).astype(int)
            
            # Calendar-Volatility Interaction: High impact events often = high vol
            if 'atr_pct' in df.columns:
                df['calendar_vol_synergy'] = df['calendar_impact_score'] * df['atr_pct']
        
        # 🔥🔥🔥 FINAL CLEANUP: Ganti semua nilai infinity atau NaN yang mungkin tercipta
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        median_spread: float,
        slippage_factor: float = 0.25,
        inv_freq_mult: float = 1.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        🔥 PURE CANDLE LABELING: NO INDICATORS!
        - Hanya melabeli pergerakan yang signifikan berdasarkan candle patterns
        """
        # NO ATR REQUIRED - Use price range instead!

        labels = pd.Series(-1, index=df.index, dtype=np.int8)
        weights = pd.Series(1.0, index=df.index, dtype=np.float32)
        
        friction_cost = (slippage_factor * median_spread) + median_spread
        logger.info(f"🎯 Smart Labeling: horizon={horizon}, total_friction={friction_cost:.5f}")
        
        close, high, low = df['close'].to_numpy(), df['high'].to_numpy(), df['low'].to_numpy()
        
        # Calculate price range (NO ATR!)
        price_range = high - low
        range_ma = pd.Series(price_range).rolling(20).mean().to_numpy()

        # 🔥 PURE CANDLE LABELING PARAMETERS
        current_tf = 'M15'  # Default
        if hasattr(df, 'attrs') and 'timeframe' in df.attrs:
            current_tf = df.attrs['timeframe']
        
        # 🔥🔥🔥 ULTRA LOOSE LABELING - CATCH MORE SIGNALS!
        tf_labeling_params = {
            'M5':  {'min_profit_range': 0.3, 'dead_zone_range': 0.05, 'min_volume_pct': 0.05},  # 🔥 ULTRA LONGGAR!
            'M15': {'min_profit_range': 0.4, 'dead_zone_range': 0.08, 'min_volume_pct': 0.08},  # 🔥 SANGAT LONGGAR!
            'H1':  {'min_profit_range': 0.6, 'dead_zone_range': 0.12, 'min_volume_pct': 0.15},  # 🔥 LONGGAR!
            'H4':  {'min_profit_range': 0.8, 'dead_zone_range': 0.15, 'min_volume_pct': 0.25}   # 🔥 LONGGAR!
        }
        
        params = tf_labeling_params.get(current_tf, tf_labeling_params['M15'])
        MIN_PROFIT_TARGET_RANGE = params['min_profit_range']
        DEAD_ZONE_RANGE = params['dead_zone_range']
        MIN_VOLUME_PCT = params['min_volume_pct']

        for i in range(len(df) - horizon):
            range_at_entry = range_ma[i] if not np.isnan(range_ma[i]) else price_range[i]
            if range_at_entry <= 0:
                continue

            entry_price = close[i]
            
            # 1. Stop Loss (SL) berdasarkan price range
            sl_dist = range_at_entry * 1.5  # SL 1.5x range
            lower_barrier = entry_price - sl_dist

            # 2. Take Profit (TP) yang signifikan
            min_tp_dist = max(friction_cost * 2, range_at_entry * MIN_PROFIT_TARGET_RANGE)
            upper_barrier = entry_price + min_tp_dist

            # 3. "Zona Mati" (Dead Zone)
            dead_zone_upper = entry_price + (range_at_entry * DEAD_ZONE_RANGE)
            dead_zone_lower = entry_price - (range_at_entry * DEAD_ZONE_RANGE)
            
            future_highs, future_lows = high[i+1:i+1+horizon], low[i+1:i+1+horizon]
            
            tp_hit_time, sl_hit_time = float('inf'), float('inf')

            # Cari kapan TP atau SL pertama kali disentuh
            for t in range(len(future_highs)):
                if future_lows[t] <= lower_barrier:
                    sl_hit_time = t + 1
                    break
                if future_highs[t] >= upper_barrier:
                    tp_hit_time = t + 1
                    break

            # 🔥🔥🔥 TIERED LABELING v2.0: BALANCED DIET (Quality + Quantity)
            # Beri model lebih banyak data, tapi dengan PRIORITAS JELAS pada setup terbaik
            
            # Get quality metrics
            current_candle_quality = df['candle_quality_score'].iloc[i] if 'candle_quality_score' in df.columns else 0
            market_regime = df['market_regime_filter'].iloc[i] if 'market_regime_filter' in df.columns else 0.5
            
            # --- Tentukan Tier Kualitas (LOOSENED!) ---
            # Tier 1: Kualitas Bintang 5 (High-Quality Setup) - PRIORITAS UTAMA
            is_high_quality = (current_candle_quality > 0.5 and market_regime > 0.3)  # 🔥 LOOSENED!
            # Tier 2: Kualitas Bintang 4 (Medium-Quality Setup) - TETAP DIPAKAI
            is_medium_quality = (current_candle_quality > 0.2 and market_regime > 0.2)  # 🔥 LOOSENED!
            # Tier 3: Kualitas Bintang 3 (Low-Quality Setup) - CATCH MORE!
            is_low_quality = (current_candle_quality > 0.0 and market_regime > 0.1)  # 🔥 CATCH ALL!
            
            # Volume filter (tetap ada)
            skip_low_volume = False
            if 'tick_volume' in df.columns:
                vol_ma = df['tick_volume'].rolling(50).mean()
                if df['tick_volume'].iloc[i] < vol_ma.iloc[i] * MIN_VOLUME_PCT:
                    skip_low_volume = True
            
            # --- Logika Pelabelan Bertingkat (INCLUDE TIER 3!) ---
            if not skip_low_volume:
                if is_high_quality:
                    # Tier 1: SUPER BOOST
                    if tp_hit_time < sl_hit_time:
                        labels.iloc[i] = 1  # BUY
                        weights.iloc[i] = 4.0 + (horizon / max(1, tp_hit_time))
                    elif sl_hit_time < tp_hit_time:
                        labels.iloc[i] = 0  # SELL
                        weights.iloc[i] = 2.0
                        
                elif is_medium_quality:
                    # Tier 2: NORMAL BOOST
                    if tp_hit_time < sl_hit_time:
                        labels.iloc[i] = 1  # BUY
                        weights.iloc[i] = 2.0 + (horizon / max(1, tp_hit_time))
                    elif sl_hit_time < tp_hit_time:
                        labels.iloc[i] = 0  # SELL
                        weights.iloc[i] = 1.0
                        
                elif is_low_quality:
                    # Tier 3: LOW BOOST - CATCH MORE SIGNALS!
                    if tp_hit_time < sl_hit_time:
                        labels.iloc[i] = 1  # BUY
                        weights.iloc[i] = 1.0 + (horizon / max(1, tp_hit_time)) * 0.5
                    elif sl_hit_time < tp_hit_time:
                        labels.iloc[i] = 0  # SELL
                        weights.iloc[i] = 0.5
            # else: Very low quality setup = HOLD (-1)

        counts = labels.value_counts().sort_index()
        logger.info(f"   Smart Label Dist: SELL={counts.get(0, 0)}, BUY={counts.get(1, 0)}, HOLD={counts.get(-1, 0)}")
        
        # Inverse frequency weighting
        n_buy = counts.get(1, 0)
        n_sell = counts.get(0, 0)
        total = max(1, n_buy + n_sell)
        
        inv_freq = {
            1: (total / (2 * max(1, n_buy))) * inv_freq_mult,
            0: (total / (2 * max(1, n_sell))) * inv_freq_mult
        }
        
        for i in range(len(labels)):
            if labels.iloc[i] != -1:
                weights.iloc[i] = weights.iloc[i] * inv_freq[labels.iloc[i]]
        
        logger.info(f"   Applied inverse-frequency sample weights (mult={inv_freq_mult}): BUY x{inv_freq[1]:.2f}, SELL x{inv_freq[0]:.2f}")
        
        # 🔥🔥🔥 REGIME-BASED WEIGHT BOOST (TIMEFRAME-SPECIFIC) 🔥🔥🔥
        # M5/M15 butuh boost lebih besar karena noise tinggi
        if 'market_regime_filter' in df.columns:
            regime_boost_mult = {
                'M5': 1.5,   # 🔥 +150% untuk sinyal terbaik di M5
                'M15': 1.3,  # 🔥 +130% untuk M15
                'H1': 1.0,   # +100% untuk H1
                'H4': 0.8    # +80% untuk H4 (sudah stabil, jangan terlalu agresif)
            }
            boost_mult = regime_boost_mult.get(current_tf, 1.0)
            regime_boost = 1 + (df['market_regime_filter'] * boost_mult)
            weights *= regime_boost
            logger.info(f"   Applied regime-based weight boost (up to +{int(boost_mult*100)}% for {current_tf})")
        # 🔥🔥🔥-----------------------------🔥🔥🔥
        
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
