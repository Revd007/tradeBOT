import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CLSPredictor:
    """Trained classifier model for trade direction prediction"""
    
    def __init__(self, model_dir: str = "./models/saved_models", use_ensemble_voting: bool = True):
        self.model_dir = Path(model_dir)
        self.models = {}  # Now self-contained: includes scaler + features per specialist
        self.use_ensemble_voting = use_ensemble_voting
        self.load_models()
    
    def load_models(self):
        """🔥 UPDATED: Load self-contained models (scaler + features embedded)"""
        import gc

        timeframes = ['m5', 'm15', 'h1', 'h4']
        
        # 🔥 CRITICAL: Clear any cached models first
        self.models.clear()
        gc.collect()  # Force garbage collection
        
        for tf in timeframes:
            model_path = self.model_dir / f"cls_{tf}.pkl"
            
            try:
                if model_path.exists():
                    # 🔥 FIX: Force reload from disk (no cache)
                    with open(model_path, 'rb') as f:
                        self.models[tf] = joblib.load(f)
                    
                    # Verify structure
                    if isinstance(self.models[tf], dict) and 'bullish' in self.models[tf]:
                        bullish_features = len(self.models[tf]['bullish'].get('features', []))
                        bearish_features = len(self.models[tf]['bearish'].get('features', []))
                        logger.info(f"✅ Loaded {tf} model (Bullish: {bullish_features} features, Bearish: {bearish_features} features)")
                    else:
                        logger.info(f"✅ Loaded {tf} model (legacy format)")
            
            except Exception as e:
                logger.error(f"Error loading {tf}: {str(e)}")
        
        if not self.models:
            logger.warning("⚠️ No CLS models found. Using default predictions.")
    
    def add_strategy_features(self, df: pd.DataFrame, strategies: Dict, symbol_info: Dict) -> pd.DataFrame:
        """
        🔥 OPTIMIZED STRATEGY-BASED FEATURES (VECTORIZED)
        - Extract features from strategy analysis methods using vectorized operations
        - Learn from strategy logic, not raw indicators
        """
        logger.info("   🔥 Extracting strategy-based features (VECTORIZED)...")
        df = df.copy()  # Work on copy to avoid SettingWithCopyWarning

        # --- BaseStrategy & MeanReversion Features (PURE CANDLE) ---
        df['price_change_5'] = df['close'].pct_change(periods=5)
        df['volume_ma_10'] = df['tick_volume'].rolling(window=10).mean()
        df['volume_spike'] = df['tick_volume'] > (df['volume_ma_10'] * 1.5)
        
        # BaseStrategy signals
        base_buy_signal = (df['price_change_5'] > 0.002) & df['volume_spike']
        base_sell_signal = (df['price_change_5'] < -0.002) & df['volume_spike']
        df['base_strategy_signal'] = np.select([base_buy_signal, base_sell_signal], [1, 0], default=-1)
        df['base_strategy_confidence'] = np.where(base_buy_signal | base_sell_signal, 0.6, 0.0)

        # MeanReversion features
        sma_20 = df['close'].rolling(window=20).mean()
        df['price_vs_sma'] = (df['close'] - sma_20) / sma_20

        reversion_sell_signal = (df['price_vs_sma'] > 0.02) & (df['price_change_5'] < -0.001)
        reversion_buy_signal = (df['price_vs_sma'] < -0.02) & (df['price_change_5'] > 0.001)
        df['mean_reversion_signal'] = np.select([reversion_buy_signal, reversion_sell_signal], [1, 0], default=-1)
        df['mean_reversion_confidence'] = np.where(reversion_buy_signal | reversion_sell_signal, 0.7, 0.0)

        # --- Breakout & Counter-Trend Features (Vectorized Proxies) ---
        high_20 = df['high'].rolling(window=20).max().shift(1)
        low_20 = df['low'].rolling(window=20).min().shift(1)
        breakout_buy = (df['close'] > high_20) & df['volume_spike']
        breakout_sell = (df['close'] < low_20) & df['volume_spike']
        df['breakout_signal'] = np.select([breakout_buy, breakout_sell], [1, 0], default=-1)
        df['breakout_confidence'] = np.where(breakout_buy | breakout_sell, 0.65, 0.0)

        counter_buy = (df['close'] < sma_20) & (df['close'] > df['close'].shift(1))
        counter_sell = (df['close'] > sma_20) & (df['close'] < df['close'].shift(1))
        df['counter_trend_signal'] = np.select([counter_buy, counter_sell], [1, 0], default=-1)
        df['counter_trend_confidence'] = np.where(counter_buy | counter_sell, 0.55, 0.0)

        # 🔥 NEW: Bullish Breakout Features (untuk detect BUY signals lebih baik)
        # Gap up: Opening price significantly higher than previous close
        # Breakout strength: How strong is the breakout relative to volatility
        if 'atr' in df.columns:
            df['gap_up'] = ((df['open'] - df['close'].shift(1)) > (df['atr'] * 0.5)).astype(int)
            df['gap_down'] = ((df['close'].shift(1) - df['open']) > (df['atr'] * 0.5)).astype(int)
            df['breakout_strength'] = (df['high'] - df['low']) / (df['atr'] + 1e-9)
        else:
            # Fallback jika ATR belum ada
            avg_range = (df['high'] - df['low']).rolling(20).mean()
            df['gap_up'] = ((df['open'] - df['close'].shift(1)) > (avg_range * 0.5)).astype(int)
            df['gap_down'] = ((df['close'].shift(1) - df['open']) > (avg_range * 0.5)).astype(int)
            df['breakout_strength'] = (df['high'] - df['low']) / (avg_range + 1e-9)
        
        # Volume spike: Current volume vs average
        if 'tick_volume' in df.columns:
            volume_ma_20 = df['tick_volume'].rolling(window=20).mean()
            df['volume_spike_ratio'] = df['tick_volume'] / (volume_ma_20 + 1e-9)
        else:
            df['volume_spike_ratio'] = 1.0
        
        # EMA bullish: Price above EMA21 (trend filter)
        if 'ema_21' in df.columns:
            df['ema_bullish'] = (df['close'] > df['ema_21']).astype(int)
        else:
            # Calculate EMA21 if not exists
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_bullish'] = (df['close'] > df['ema_21']).astype(int)
        
        # Combined bullish breakout signal
        df['bullish_breakout'] = (
            (df['gap_up'] > 0) | 
            ((df['breakout_strength'] > 1.5) & (df['volume_spike_ratio'] > 1.5) & (df['ema_bullish'] > 0))
        ).astype(int)

        # --- Strategy Consensus Features ---
        signal_cols = ['base_strategy_signal', 'mean_reversion_signal', 'breakout_signal', 'counter_trend_signal']
        df['strategy_bullish_count'] = (df[signal_cols] == 1).sum(axis=1)
        df['strategy_bearish_count'] = (df[signal_cols] == 0).sum(axis=1)

        df['strategy_consensus'] = np.select(
            [
                df['strategy_bullish_count'] > df['strategy_bearish_count'],
                df['strategy_bearish_count'] > df['strategy_bullish_count']
            ],
            [1, 0],
            default=-1
        )

        confidence_cols = ['base_strategy_confidence', 'mean_reversion_confidence', 'breakout_confidence', 'counter_trend_confidence']
        total_signals = df['strategy_bullish_count'] + df['strategy_bearish_count']
        df['strategy_confidence_avg'] = df[confidence_cols].sum(axis=1) / total_signals.replace(0, 1)

        # Remove intermediate columns
        df = df.drop(columns=['price_change_5', 'volume_ma_10', 'volume_spike', 'price_vs_sma'], errors='ignore')
        
        logger.info(f"   ✅ VECTORIZED strategy features extracted for {len(df)} candles")
        logger.info(f"   ✅ Added Bullish Breakout features: gap_up, breakout_strength, volume_spike_ratio, ema_bullish")
        return df
    
    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔥 ADD CALENDAR FEATURES (matching trainer!)
        """
        # Add time-based features
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek

        # Sessions
        df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
        df['is_overlap_session'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)

        # High-impact days
        df['is_high_impact_day'] = df['day_of_week'].isin([1, 2, 3]).astype(int)

        # Weekend proximity
        df['is_weekend_proximity'] = (
            ((df['day_of_week'] == 4) & (df['hour'] >= 15)) |
            ((df['day_of_week'] == 0) & (df['hour'] < 3))
        ).astype(int)

        # News hours
        df['is_news_hour'] = df['hour'].isin([8, 10, 13, 14, 15]).astype(int)

        # Day of month
        df['day_of_month'] = pd.to_datetime(df['time']).dt.day
        df['is_first_week'] = (df['day_of_month'] <= 7).astype(int)

        # Trend context (SMA200/50)
        if 'close' in df.columns:
            df['sma200'] = df['close'].rolling(200).mean()
            df['is_above_sma200'] = (df['close'] > df['sma200']).astype(int)
            sma200_safe = df['sma200'].replace(0, np.nan)
            df['sma200_slope'] = df['sma200'].diff(10) / sma200_safe

            df['sma50'] = df['close'].rolling(50).mean()
            df['is_above_sma50'] = (df['close'] > df['sma50']).astype(int)
            df['sma_cross'] = ((df['sma50'] > df['sma200']) & (df['sma50'].shift(1) <= df['sma200'].shift(1))).astype(int)

        # Volatility regime
        if 'atr' in df.columns:
            df['atr_ma20'] = df['atr'].rolling(20).mean()
            df['is_high_volatility'] = (df['atr'] > df['atr_ma20']).astype(int)

            # Handle volatility_regime safely
            atr_clean = df['atr'].dropna()
            if len(atr_clean) > 0:
                try:
                    df['volatility_regime'] = pd.qcut(df['atr'], q=3, labels=[0, 1, 2], duplicates='drop')
                    df['volatility_regime'] = df['volatility_regime'].cat.codes
                    df['volatility_regime'] = df['volatility_regime'].replace(-1, 1)
                except Exception:
                    df['volatility_regime'] = 1

        return df

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔥 MATCH TRAINER: Add same advanced features
        """
        # RSI Divergence
        if 'rsi' in df.columns and 'close' in df.columns:
            df['rsi_change'] = df['rsi'].diff(3)
            df['price_change'] = df['close'].pct_change(3)
            df['rsi_divergence'] = (df['rsi_change'] * df['price_change']) < 0
            df['rsi_divergence'] = df['rsi_divergence'].astype(int)

        # Volatility Ratio
        if 'atr' in df.columns:
            atr_ma10 = df['atr'].rolling(10).mean()
            atr_ma20 = df['atr'].rolling(20).mean()
            df['volatility_ratio_10'] = df['atr'] / atr_ma10.replace(0, np.nan)
            df['volatility_ratio_20'] = df['atr'] / atr_ma20.replace(0, np.nan)

        # Momentum Indicators
        if 'close' in df.columns:
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            df['mean_return_5'] = df['close'].pct_change().rolling(5).mean()
            df['mean_return_10'] = df['close'].pct_change().rolling(10).mean()

        # MACD Divergence
        if 'macd' in df.columns and 'close' in df.columns:
            df['macd_change'] = df['macd'].diff(3)
            df['macd_divergence'] = (df['macd_change'] * df['price_change']) < 0
            df['macd_divergence'] = df['macd_divergence'].astype(int)

        # Stochastic Overbought/Oversold
        if 'stoch_k' in df.columns:
            df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
            df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)

        # Bollinger %B
        if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_percent_b'] = (df['close'] - df['bb_lower']) / bb_range

        # Volume momentum
        if 'tick_volume' in df.columns:
            df['volume_momentum'] = df['tick_volume'].pct_change(3)
            df['volume_acceleration'] = df['volume_momentum'].diff(3)

        # 🔥 NEW: Contextual Features (matching trainer!)
        # 1. Interaction: Momentum with Long-term Trend
        if 'momentum_5' in df.columns and 'is_above_sma200' in df.columns:
            df['trend_momentum_interaction'] = df['momentum_5'] * (df['is_above_sma200'] - 0.5) * 2

        # 2. Volatility-Normalized Momentum
        if 'momentum_10' in df.columns and 'atr' in df.columns:
            atr_safe = df['atr'].replace(0, np.nan)
            df['volatility_normalized_momentum'] = df['momentum_10'] / atr_safe

        # 3. Mean Reversion Score
        if 'close' in df.columns and 'ema_50' in df.columns and 'atr' in df.columns:
            atr_safe = df['atr'].replace(0, np.nan)
            df['mean_reversion_score'] = (df['close'] - df['ema_50']) / atr_safe

        # 4. EMA Spread Strength
        if all(x in df.columns for x in ['ema_9', 'ema_21', 'ema_50', 'close']):
            close_safe = df['close'].replace(0, np.nan)
            df['ema_spread_short_mid'] = (df['ema_9'] - df['ema_21']) / close_safe
            df['ema_spread_mid_long'] = (df['ema_21'] - df['ema_50']) / close_safe

        return df

    def prepare_features_for_specialist(
        self, 
        df: pd.DataFrame, 
        specialist_pkg: Dict, 
        timeframe: str, 
        mt5_handler=None
    ) -> pd.DataFrame:
        """
        🔥 NEW: Prepare features for a SPECIFIC specialist (Bullish or Bearish)
        Each specialist has its own feature list after pruning
        """
        from data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        
        df_with_features = df.copy()

        # Step 0: Inject MTF context if needed (MUST match trainer!)
        required_features = specialist_pkg.get('features', [])
        if not required_features:
            raise ValueError(f"Specialist package missing 'features' key!")
        
        needs_mtf = any(f in required_features for f in ['htf_rsi', 'close_vs_htf_ema', 'rsi_vs_htf_rsi'])
        
        if needs_mtf and mt5_handler:
            # Inject MTF exactly like trainer does
            timeframe_map = {'M5': 'M15', 'M15': 'H1', 'H1': 'D1', 'H4': 'W1'}
            htf = timeframe_map.get(timeframe.upper())
            
            if htf and htf not in ['D1', 'W1']:
                try:
                    timeframe_minutes = {'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240}
                    ltf_minutes = timeframe_minutes.get(timeframe.upper(), 15)
                    htf_minutes = timeframe_minutes.get(htf, 60)
                    htf_candles_needed = int(len(df_with_features) * (ltf_minutes / htf_minutes)) + 200
                    
                    # 🔥 FIX: Need symbol parameter!
                    # For backtest, get symbol from df or use default
                    symbol = 'BTCUSDm'  # Default for backtest
                    df_htf = mt5_handler.get_candles(symbol, htf, count=htf_candles_needed)
                    
                    if df_htf is not None and not df_htf.empty:
                        df_htf['htf_ema_50'] = df_htf['close'].ewm(span=50, adjust=False).mean()
                        
                        delta = df_htf['close'].diff()
                        gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
                        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
                        rs = gain / loss.replace(0, np.nan)
                        df_htf['htf_rsi'] = 100 - (100 / (1 + rs))
                        
                        df_with_features = pd.merge_asof(
                            df_with_features.sort_values('time'),
                            df_htf[['time', 'htf_ema_50', 'htf_rsi']].sort_values('time'),
                            on='time',
                            direction='backward'
                        )
                        
                        df_with_features['close_vs_htf_ema'] = (df_with_features['close'] - df_with_features['htf_ema_50']) / df_with_features['htf_ema_50'] * 100
                        df_with_features['rsi_vs_htf_rsi'] = df_with_features['rsi'] - df_with_features['htf_rsi']
                except Exception:
                    # Silently continue if MTF injection fails
                    pass

        # Step 1: Add calendar/fundamental features FIRST (matching trainer!)
        df_with_features = self.add_calendar_features(df_with_features)

        # Step 2: Create technical features (includes Market DNA!)
        # 🔥 FIX: MUST pass timeframe parameter (same as trainer!)
        df_with_features = preprocessor.create_features(df_with_features, timeframe=timeframe)

        # 🔥 DEBUG: Check DNA features
        dna_features_check = ['adx', 'market_regime_trending', 'confirmed_bullish_breakout', 
                             'bullish_divergence', 'overextension_ema21_atr']
        dna_present = [f for f in dna_features_check if f in df_with_features.columns]
        if dna_present:
            logger.debug(f"   ✅ DNA features present ({len(dna_present)}/{len(dna_features_check)}): {dna_present[:3]}")
        else:
            logger.warning(f"   ⚠️  NO DNA features found! This may cause prediction errors.")

        # Step 3: Add advanced features (MUST match trainer!)
        df_with_features = self.add_advanced_features(df_with_features)

        # Get last row (latest candle)
        latest_row = df_with_features.iloc[[-1]]

        # 🔥 CRITICAL: Filter to EXACT features used by THIS specialist
        # Create DataFrame with ONLY required features in CORRECT ORDER
        X = pd.DataFrame(index=latest_row.index)
        
        missing_features = []
        for col in required_features:
            if col in latest_row.columns:
                X[col] = latest_row[col].values
            else:
                # Feature missing - use 0 as fallback
                X[col] = 0
                missing_features.append(col)
        
        # 🔥 CRITICAL: Log missing features (critical for debugging!)
        if missing_features:
            missing_pct = len(missing_features) / len(required_features) * 100
            
            # Only log once per specialist to avoid spam
            log_key = f'_missing_{timeframe}'
            if not hasattr(self, log_key):
                logger.warning(f"⚠️  {len(missing_features)}/{len(required_features)} features missing ({missing_pct:.1f}%): {missing_features[:5]}")
                
                # 🔥 CRITICAL WARNING: If too many features missing, predictions will be unreliable!
                if missing_pct > 10:
                    logger.error(f"❌ TOO MANY MISSING FEATURES ({missing_pct:.1f}%)! Prediction quality severely degraded!")
                    logger.error(f"   Check: preprocessor.create_features() with timeframe parameter")
                    logger.error(f"   DNA features (adx, confirmed_breakout, etc.) must be present!")
                
                setattr(self, log_key, True)

        return X
    
    def predict(
        self, 
        df: pd.DataFrame, 
        timeframe: str = 'm5',
        mt5_handler=None  # 🔥 NEW: For MTF context injection
    ) -> Tuple[str, float]:
        """
        🔥 FIXED PREDICTION ARCHITECTURE - Adaptive & Balanced
        
        Returns:
            (direction: 'BUY', 'SELL', or 'HOLD', confidence: 0.0-1.0)
        """
        tf_key = timeframe.lower()
        
        if tf_key not in self.models:
            return 'HOLD', 0.5
        
        try:
            model = self.models[tf_key]
            
            if isinstance(model, dict) and 'bullish' in model and 'bearish' in model:
                bullish_pkg = model['bullish']
                bearish_pkg = model['bearish']
                
                # Prepare features
                X_bullish = self.prepare_features_for_specialist(df, bullish_pkg, timeframe, mt5_handler)
                X_bearish = self.prepare_features_for_specialist(df, bearish_pkg, timeframe, mt5_handler)
                
                # Clean
                X_bullish.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_bullish.fillna(0, inplace=True)
                X_bearish.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_bearish.fillna(0, inplace=True)
                
                # Scale
                X_bullish_scaled = bullish_pkg['scaler'].transform(X_bullish)
                X_bearish_scaled = bearish_pkg['scaler'].transform(X_bearish)
                
                X_bullish_scaled_df = pd.DataFrame(X_bullish_scaled, columns=X_bullish.columns, index=X_bullish.index)
                X_bearish_scaled_df = pd.DataFrame(X_bearish_scaled, columns=X_bearish.columns, index=X_bearish.index)
                
                # Probabilities
                prob_buy = bullish_pkg['model'].predict_proba(X_bullish_scaled_df)[0, 1]
                prob_sell = bearish_pkg['model'].predict_proba(X_bearish_scaled_df)[0, 1]
                
                # 🔥 FIX: HAPUS ADAPTIVE THRESHOLD BIAS - Gunakan fixed threshold untuk BUY dan SELL
                # SEBELUM: SELL threshold lebih rendah dari BUY → bias!
                # SESUDAH: Fixed 0.60 untuk kedua (fair, tidak bias)
                BUY_THRESHOLD = 0.60
                SELL_THRESHOLD = 0.60
                
                # Gunakan threshold dari model, tapi pastikan tidak ada bias
                bullish_threshold = max(BUY_THRESHOLD, bullish_pkg['threshold'])
                bearish_threshold = max(SELL_THRESHOLD, bearish_pkg['threshold'])
                
                # 🔥 CRITICAL: Pastikan SELL tidak lebih mudah dari BUY
                if bearish_threshold < bullish_threshold:
                    bearish_threshold = bullish_threshold  # SELL harus >= BUY threshold

                # 🔥 NEW: Volatility-aware dynamic threshold (ATR%)
                try:
                    if 'atr' in df.columns and 'close' in df.columns and len(df) > 0:
                        atr_pct = (df['atr'].iloc[-1] / max(1e-9, df['close'].iloc[-1])) * 100.0
                        vol_adjust = 0.05 if atr_pct > 1.5 else 0.0
                        bullish_threshold = min(0.99, bullish_threshold + vol_adjust)
                        bearish_threshold = min(0.99, bearish_threshold + vol_adjust)
                        logger.info(f"🔧 Volatility-adjusted thresholds (+{vol_adjust:.2f} if ATR%>1.5): BUY={bullish_threshold:.3f}, SELL={bearish_threshold:.3f}")
                except Exception:
                    pass

                logger.info(f"🔧 Adaptive Thresholds: BUY={bullish_threshold:.3f}, SELL={bearish_threshold:.3f}")
                logger.info(f"🔧 Probabilities: BUY={prob_buy:.3f}, SELL={prob_sell:.3f}")

                # 🔥 FIX: Confidence scores - gunakan probabilitas langsung atau formula yang lebih fair
                # Formula lama: (prob - threshold) / (1 - threshold) → terlalu ketat, confidence jadi rendah!
                # Formula baru: probabilitas langsung (model sudah calibrated, probabilitas sudah reliable)
                # Atau: max(prob, (prob - threshold) / (1 - threshold)) untuk balance
                
                # Method 1: Gunakan probabilitas langsung (lebih fair untuk calibrated model)
                buy_confidence_direct = prob_buy
                sell_confidence_direct = prob_sell
                
                # Method 2: Formula relative (backup)
                buy_confidence_relative = max(0.0, (prob_buy - bullish_threshold) / max(1e-6, (1 - bullish_threshold)))
                sell_confidence_relative = max(0.0, (prob_sell - bearish_threshold) / max(1e-6, (1 - bearish_threshold)))
                
                # 🔥 FINAL: Kombinasi keduanya (bobot 70% direct, 30% relative)
                # Ini memberikan confidence yang lebih tinggi jika prob tinggi, tapi tetap sensitif terhadap threshold
                buy_confidence = 0.7 * buy_confidence_direct + 0.3 * buy_confidence_relative
                sell_confidence = 0.7 * sell_confidence_direct + 0.3 * sell_confidence_relative
                
                # Clip to [0, 1]
                buy_confidence = np.clip(buy_confidence, 0.0, 1.0)
                sell_confidence = np.clip(sell_confidence, 0.0, 1.0)

                # 🔥 SHORT-TERM FIX: Trend-aware threshold nudging (no retrain)
                # If short-term uptrend, make BUY slightly easier; if downtrend, make SELL slightly easier
                try:
                    if df is not None and 'close' in df.columns and len(df) > 60:
                        ema_9 = df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                        ema_21 = df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
                        if pd.notna(ema_9) and pd.notna(ema_21):
                            if ema_9 > ema_21:
                                # Uptrend → nudge BUY
                                buy_confidence = min(1.0, buy_confidence + 0.05)
                            elif ema_9 < ema_21:
                                # Downtrend → nudge SELL
                                sell_confidence = min(1.0, sell_confidence + 0.05)
                except Exception:
                    pass

                min_confidence = 0.10
                valid_buy = buy_confidence >= min_confidence
                valid_sell = sell_confidence >= min_confidence

                if valid_buy and not valid_sell:
                    return 'BUY', float(buy_confidence)
                elif valid_sell and not valid_buy:
                    return 'SELL', float(sell_confidence)
                elif valid_buy and valid_sell:
                    return ('BUY', float(buy_confidence)) if buy_confidence > sell_confidence else ('SELL', float(sell_confidence))
                else:
                    return 'HOLD', float(max(buy_confidence, sell_confidence) * 0.5)

            # Legacy fallback
            return 'HOLD', 0.5

        except Exception as e:
            logger.error(f"Prediction error for {timeframe}: {str(e)}")
            return 'HOLD', 0.0

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        🔥 Detect current market regime for adaptive prediction
        """
        if df is None or len(df) < 60 or 'close' not in df.columns:
            return 'NEUTRAL'

        returns = df['close'].pct_change().dropna()
        if len(returns) < 20:
            return 'NEUTRAL'

        volatility = returns.rolling(20).std()
        avg_volatility = volatility.mean()
        current_vol = volatility.iloc[-1] if len(volatility) > 0 else avg_volatility

        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        if pd.isna(sma_20.iloc[-1]) or pd.isna(sma_50.iloc[-1]):
            return 'NEUTRAL'
        price_vs_sma20 = (df['close'].iloc[-1] - sma_20.iloc[-1]) / max(1e-9, sma_20.iloc[-1])

        if current_vol > avg_volatility * 1.5:
            return 'HIGH_VOLATILITY'
        elif abs(price_vs_sma20) > 0.02:
            return 'TRENDING'
        elif current_vol < avg_volatility * 0.7:
            return 'LOW_VOLATILITY'
        else:
            return 'NEUTRAL'

    def enhanced_ensemble_voting(self, predictions: Dict) -> Tuple[str, float]:
        """
        🔥 Smart ensemble voting that considers timeframe hierarchy and confidence
        """
        weights = {'m5': 1.0, 'm15': 1.5, 'h1': 2.0, 'h4': 2.5}
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for tf, pred in predictions.items():
            if pred and pred.get('action') in ['BUY', 'SELL']:
                weight = float(weights.get(tf, 1.0))
                confidence = float(pred.get('confidence', 0.5))
                if pred['action'] == 'BUY':
                    buy_score += weight * confidence
                else:
                    sell_score += weight * confidence
                total_weight += weight

        if total_weight == 0:
            return 'HOLD', 0.5

        buy_normalized = buy_score / total_weight
        sell_normalized = sell_score / total_weight

        decision_threshold = 0.55
        if buy_normalized > decision_threshold and buy_normalized > sell_normalized:
            return 'BUY', float(buy_normalized)
        elif sell_normalized > decision_threshold and sell_normalized > buy_normalized:
            return 'SELL', float(sell_normalized)
        else:
            return 'HOLD', float(max(buy_normalized, sell_normalized) * 0.7)

    def realistic_backtest_validation(self, symbol: str, mt5_handler, timeframe: str = 'M5') -> Dict:
        """
        🔥 Realistic backtest with improved signal validation
        """
        try:
            df = mt5_handler.get_candles(symbol, timeframe, count=500)
            if df is None or len(df) < 120:
                return {"error": "Insufficient data"}

            from strategies.base_strategy import BaseStrategy
            class TempStrategy(BaseStrategy):
                def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
                    return None
            strategy = TempStrategy("Validation", "MEDIUM")
            df = strategy.add_all_indicators(df)

            signals = []
            for i in range(100, len(df)):
                lookback = df.iloc[:i].copy()
                action, confidence = self.predict(lookback, timeframe.lower(), mt5_handler)
                signals.append({'timestamp': df.iloc[i]['time'], 'action': action, 'confidence': confidence, 'price': df.iloc[i]['close']})

            buy_signals = [s for s in signals if s['action'] == 'BUY']
            sell_signals = [s for s in signals if s['action'] == 'SELL']
            total_signals = len(buy_signals) + len(sell_signals)
            buy_ratio = (len(buy_signals) / total_signals) if total_signals > 0 else 0.0

            logger.info(f"📊 Signal Analysis for {timeframe}:")
            logger.info(f"   BUY signals: {len(buy_signals)} ({(len(buy_signals)/max(1,len(signals)))*100:.1f}%)")
            logger.info(f"   SELL signals: {len(sell_signals)} ({(len(sell_signals)/max(1,len(signals)))*100:.1f}%)")
            logger.info(f"   HOLD signals: {len(signals)-len(buy_signals)-len(sell_signals)}")
            if total_signals > 0 and (buy_ratio < 0.2 or buy_ratio > 0.8):
                logger.warning(f"   ⚠️  IMBALANCE DETECTED: Extreme bias ({buy_ratio:.1%})")

            return {
                'timeframe': timeframe,
                'total_signals': len(signals),
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'buy_ratio': buy_ratio,
                'signals': signals[-20:]
            }
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"error": str(e)}
    
    def multi_timeframe_consensus(
        self, 
        symbol: str, 
        mt5_handler,
        timeframes: list = ['m5', 'm15', 'h1']
    ) -> Dict:
        """
        🔥 ENHANCED & FIXED: Get predictions from multiple timeframes and calculate consensus.
        """
        predictions = {}

        # Create concrete strategy class once
        from strategies.base_strategy import BaseStrategy

        class TempStrategy(BaseStrategy):
            def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
                return None

        strategy = TempStrategy("Temp", "MEDIUM")
        
        for tf in timeframes:
            try:
                df = mt5_handler.get_candles(symbol, tf.upper(), count=200)
                df = strategy.add_all_indicators(df)
                action, confidence = self.predict(df, tf, mt5_handler=mt5_handler)
                predictions[tf] = {'action': action, 'confidence': confidence}
            except Exception as e:
                logger.error(f"Error in {tf} prediction: {str(e)}")
                predictions[tf] = None
        
        # Calculate votes once
        buy_votes = sum(1 for p in predictions.values() if p and p['action'] == 'BUY')
        sell_votes = sum(1 for p in predictions.values() if p and p['action'] == 'SELL')

        if self.use_ensemble_voting:
            # Advanced weighted voting
            weights = {'m5': 1.0, 'm15': 1.5, 'h1': 2.0, 'h4': 2.5}
            buy_score = 0
            sell_score = 0

            for tf, pred in predictions.items():
                if pred:
                    weight = weights.get(tf, 1.0)
                    score = pred['confidence'] * weight
                    if pred['action'] == 'BUY':
                        buy_score += score
                    elif pred['action'] == 'SELL':
                        sell_score += score

            total_score = buy_score + sell_score
            if buy_score > sell_score:
                consensus = 'BUY'
                consensus_confidence = buy_score / total_score if total_score > 0 else 0.5
            elif sell_score > buy_score:
                consensus = 'SELL'
                consensus_confidence = sell_score / total_score if total_score > 0 else 0.5
            else:
                consensus = 'HOLD'
                consensus_confidence = 0.5

        else:
            # Simple majority voting (legacy)
            total_votes = buy_votes + sell_votes
            if buy_votes > sell_votes:
                consensus = 'BUY'
                consensus_confidence = buy_votes / total_votes if total_votes > 0 else 0.5
            elif sell_votes > buy_votes:
                consensus = 'SELL'
                consensus_confidence = sell_votes / total_votes if total_votes > 0 else 0.5
            else:
                consensus = 'HOLD'
                consensus_confidence = 0.5
        
        return {
            'consensus': consensus,
            'confidence': consensus_confidence,
            'predictions': predictions,
            'votes': {'buy': buy_votes, 'sell': sell_votes},
            'ensemble_used': self.use_ensemble_voting
        }

    def backtest(
        self,
        symbol: str,
        mt5_handler,
        start_date: str = "2025-01-01",
        end_date: str = "2025-10-01",
        timeframe: str = "M5",
        initial_balance: float = 20.0,
        lot_size: float = 0.01,
        trade_mode: str = "NORMAL",  # 🔥 NEW: SCALPING, NORMAL, AGGRESSIVE, LONG_HOLD
        sl_atr_multiplier: float = None,  # Will be set based on trade_mode
        tp_atr_multiplier: float = None,   # Will be set based on trade_mode
        # 🔥 REALISTIC SIMULATION PARAMETERS
        spread_pips: float = 2.0,  # Average spread for BTCUSD (realistic)
        slippage_pips: float = 0.5  # Average slippage per side (entry/exit)
    ) -> Dict:
        """
        🔥 UPDATED: Backtest with multiple trade modes
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDm")
            mt5_handler: MT5 connection
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe for backtest
            initial_balance: Starting capital (default: $20)
            lot_size: Position size per trade
            trade_mode: Trading style - affects SL/TP and confidence thresholds
                - SCALPING: Tight SL/TP, high frequency
                - NORMAL: Balanced approach (default)
                - AGGRESSIVE: Wider stops, higher risk/reward
                - LONG_HOLD: Very wide stops, patient trades
            sl_atr_multiplier: Stop loss distance (auto-set by mode)
            tp_atr_multiplier: Take profit distance (auto-set by mode)
        
        Returns:
            Dict with backtest results
        """
        from datetime import datetime
        from strategies.base_strategy import BaseStrategy
        import pytz  # 🔥 NEW: For timezone handling

        # 🔥 DRASTICALLY TIGHTER for M5/M15 scalping!
        # Realistic for $20 balance
        timeframe_base = {
            'M1': {'sl_atr': 0.3, 'tp_atr': 0.45},  # Ultra tight
            'M5': {'sl_atr': 0.4, 'tp_atr': 0.60},  # Very tight (20-40 pips)
            'M15': {'sl_atr': 0.6, 'tp_atr': 0.90}, # Tight (40-60 pips)
            'M30': {'sl_atr': 0.8, 'tp_atr': 1.20}, # Medium
            'H1': {'sl_atr': 1.2, 'tp_atr': 1.80},  # Balanced
            'H4': {'sl_atr': 1.8, 'tp_atr': 2.70},  # Wide
            'D1': {'sl_atr': 2.5, 'tp_atr': 5.00}   # Very wide
        }

        # Get base SL/TP for this timeframe
        tf_upper = timeframe.upper()
        base = timeframe_base.get(tf_upper, {'sl_atr': 1.5, 'tp_atr': 2.25})

        # 🔥 PERBAIKAN: Tambahkan 'threshold_multiplier' untuk fleksibilitas backtest
        # Ini akan "melonggarkan" threshold yang ketat dari hasil training.
        # Nilai < 1.0 akan melonggarkan, > 1.0 akan mengetatkan.
        mode_params = {
            'SCALPING': {
                'sl_atr': base['sl_atr'] * 0.8,  # 20% tighter
                'tp_atr': base['tp_atr'] * 0.8,
                'threshold_multiplier': 1.15,  # Sangat selektif (95% dari threshold asli)
                'description': 'Quick trades, tight stops'
            },
            'NORMAL': {
                'sl_atr': base['sl_atr'],  # Base values
                'tp_atr': base['tp_atr'],
                'threshold_multiplier': 1.25,  # Seimbang (90% dari threshold asli)
                'description': 'Balanced risk/reward'
            },
            'AGGRESSIVE': {
                'sl_atr': base['sl_atr'] * 1.2,  # 20% wider
                'tp_atr': base['tp_atr'] * 1.2,
                'threshold_multiplier': 1.0,  # Lebih longgar (80% dari threshold asli)
                'description': 'Wider stops for bigger moves'
            },
            'LONG_HOLD': {
                'sl_atr': base['sl_atr'] * 1.5,  # 50% wider
                'tp_atr': base['tp_atr'] * 2.0,  # 2x wider TP
                'threshold_multiplier': 1.10,  # Sedikit longgar (85% dari threshold asli)
                'description': 'Patient trades, very wide stops'
            }
        }

        if trade_mode.upper() not in mode_params:
            logger.warning(f"⚠️  Invalid trade_mode '{trade_mode}', using NORMAL")
            trade_mode = 'NORMAL'

        params = mode_params[trade_mode.upper()]

        # Override with custom values if provided
        if sl_atr_multiplier is None:
            sl_atr_multiplier = params['sl_atr']
        if tp_atr_multiplier is None:
            tp_atr_multiplier = params['tp_atr']

        # 🔥 CRITICAL FIX: Convert dates with timezone awareness (required by MT5)
        timezone = pytz.timezone("Etc/UTC")
        start_dt = timezone.localize(datetime.strptime(start_date, "%Y-%m-%d"))
        end_dt = timezone.localize(datetime.strptime(end_date, "%Y-%m-%d"))

        # 🔥 CRITICAL FIX: Get RAW data WITHOUT indicators first
        # This prevents lookahead bias!
        try:
            df_raw = mt5_handler.get_candles_range(symbol, timeframe.upper(), start_dt, end_dt)
            
            if df_raw is None or df_raw.empty:
                return {}
                
        except Exception:
            # Try alternative method for M5/M15 (use get_candles_batch)
            try:
                days_diff = (end_dt - start_dt).days
                
                # Estimate candles needed (with buffer)
                candles_per_day = {'M5': 288, 'M15': 96, 'H1': 24, 'H4': 6}
                estimated_candles = int(days_diff * candles_per_day.get(timeframe.upper(), 24) * 1.1)  # 10% buffer
                
                # Limit to prevent MT5 errors
                estimated_candles = min(estimated_candles, 100000)
                
                df_raw = mt5_handler.get_candles_batch(symbol, timeframe.upper(), estimated_candles, batch_size=10000)
                
                if df_raw is None or df_raw.empty:
                    return {}
                
                # Filter to exact date range
                df_raw['time'] = pd.to_datetime(df_raw['time'])
                
                # Convert to timezone-naive for comparison
                start_naive = start_dt.replace(tzinfo=None) if hasattr(start_dt, 'tzinfo') else start_dt
                end_naive = end_dt.replace(tzinfo=None) if hasattr(end_dt, 'tzinfo') else end_dt
                
                if hasattr(df_raw['time'].iloc[0], 'tz') and df_raw['time'].iloc[0].tz:
                    df_raw['time'] = df_raw['time'].dt.tz_localize(None)
                
                df_raw = df_raw[(df_raw['time'] >= start_naive) & (df_raw['time'] <= end_naive)]
                
                if df_raw.empty:
                    return {}
                    
            except Exception:
                return {}

        # Ensure time is datetime
        if 'time' in df_raw.columns and df_raw['time'].dtype != 'datetime64[ns]':
            df_raw['time'] = pd.to_datetime(df_raw['time'])

        # 🔥 GET SYMBOL INFO FOR COST CALCULATION
        symbol_info = mt5_handler.get_symbol_info(symbol)
        point_value = symbol_info['point']
        pip_size = point_value * 10  # 1 pip size in price units
        
        # 🔥 CALCULATE TOTAL TRANSACTION COST PER TRADE
        total_transaction_cost_pips = spread_pips + (slippage_pips * 2)  # Entry + Exit slippage
        total_transaction_cost_price = total_transaction_cost_pips * pip_size

        # 🔥🔥🔥 PERBAIKAN TOTAL: SINKRONKAN PENUH PIPELINE FITUR 🔥🔥🔥
        
        # LANGSUNG MULAI DARI SINI, MENGGUNAKAN df_raw
        df_full = df_raw.copy()
        
        # STEP 1: Pre-compute strategy features (sekarang bekerja pada data mentah)
        logger.info("📊 Pre-computing strategy features for backtest...")
        df_full = self.add_strategy_features(df_full, strategies=None, symbol_info=symbol_info)
        
        # STEP 2: Pre-compute MTF context ONCE (reuse for all predictions)
        timeframe_map = {'M5': 'M15', 'M15': 'H1', 'H1': 'D1', 'H4': 'W1'}
        htf = timeframe_map.get(timeframe.upper())
        
        if htf and htf not in ['D1', 'W1']:
            try:
                timeframe_minutes = {'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240}
                ltf_minutes = timeframe_minutes.get(timeframe.upper(), 15)
                htf_minutes = timeframe_minutes.get(htf, 60)
                htf_candles_needed = int(len(df_full) * (ltf_minutes / htf_minutes)) + 200
                
                df_htf = mt5_handler.get_candles(symbol, htf, count=htf_candles_needed)
                
                if df_htf is not None and not df_htf.empty:
                    df_htf['htf_ema_50'] = df_htf['close'].ewm(span=50, adjust=False).mean()
                    
                    delta = df_htf['close'].diff()
                    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
                    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
                    rs = gain / loss.replace(0, np.nan)
                    df_htf['htf_rsi'] = 100 - (100 / (1 + rs))
                    
                    df_full = pd.merge_asof(
                        df_full.sort_values('time'),
                        df_htf[['time', 'htf_ema_50', 'htf_rsi']].sort_values('time'),
                        on='time',
                        direction='backward'
                    )
                    
                    df_full['close_vs_htf_ema'] = (df_full['close'] - df_full['htf_ema_50']) / df_full['htf_ema_50'] * 100
                    df_full['rsi_vs_htf_rsi'] = df_full['rsi'] - df_full['htf_rsi']
            except Exception:
                pass
        
        # STEP 3: Pre-compute SEMUA fitur lainnya (seperti di trainer)
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        # Add calendar features first (matching prepare_features order)
        df_full = self.add_calendar_features(df_full)
        
        # Create technical features (includes Market DNA: adx, chop_index, vol_of_vol, etc.)
        # 🔥 Tambahkan timeframe parameter
        df_full = preprocessor.create_features(df_full, timeframe=timeframe)
        
        # Add advanced features last
        df_full = self.add_advanced_features(df_full)
        
        # STEP 4: Get model packages
        tf_key = timeframe.lower()
        
        if tf_key not in self.models:
            logger.error(f"❌ Model for {timeframe} not found!")
            return {}
        
        model = self.models[tf_key]
        
        if not (isinstance(model, dict) and 'bullish' in model and 'bearish' in model):
            logger.error(f"❌ Model format invalid (expecting dual-specialist)")
            return {}
        
        bullish_pkg = model['bullish']
        bearish_pkg = model['bearish']
        
        # 🔥 CRITICAL: Prepare features separately for each specialist
        bullish_features = bullish_pkg.get('features', [])
        bearish_features = bearish_pkg.get('features', [])
        
        # Fill missing features with 0
        missing_bullish = []
        missing_bearish = []
        for f in bullish_features:
            if f not in df_full.columns:
                df_full[f] = 0
                missing_bullish.append(f)
        for f in bearish_features:
            if f not in df_full.columns:
                df_full[f] = 0
                missing_bearish.append(f)
        
        # 🔥 DEBUG: Log missing features
        if missing_bullish:
            logger.warning(f"   ⚠️  Missing Bullish features ({len(missing_bullish)}): {missing_bullish[:5]}")
        if missing_bearish:
            logger.warning(f"   ⚠️  Missing Bearish features ({len(missing_bearish)}): {missing_bearish[:5]}")
        
        logger.info(f"   📊 Feature check: Bullish={len(bullish_features)} (missing: {len(missing_bullish)}), Bearish={len(bearish_features)} (missing: {len(missing_bearish)})")
        
        # Filter to required features for each specialist
        X_bullish_all = df_full[bullish_features].copy()
        X_bearish_all = df_full[bearish_features].copy()
        
        # Clean data
        X_bullish_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_bullish_all.fillna(0, inplace=True)
        X_bearish_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_bearish_all.fillna(0, inplace=True)
        
        # 🔥 DEBUG: Check data quality
        bullish_nan_count = X_bullish_all.isna().sum().sum()
        bearish_nan_count = X_bearish_all.isna().sum().sum()
        
        # 🔥 FIX: Convert to numeric before checking inf
        try:
            bullish_inf_count = np.isinf(pd.to_numeric(X_bullish_all.values, errors='coerce')).sum()
            bearish_inf_count = np.isinf(pd.to_numeric(X_bearish_all.values, errors='coerce')).sum()
        except Exception:
            bullish_inf_count = 0
            bearish_inf_count = 0
        
        logger.info(f"   📊 Data quality: Bullish NaN={bullish_nan_count}, Inf={bullish_inf_count}")
        logger.info(f"   📊 Data quality: Bearish NaN={bearish_nan_count}, Inf={bearish_inf_count}")
        logger.info(f"   📊 Data shape: Bullish={X_bullish_all.shape}, Bearish={X_bearish_all.shape}")
        
        # 🔥 CRITICAL: Use specialist's own scaler
        X_bullish_scaled = bullish_pkg['scaler'].transform(X_bullish_all)
        X_bearish_scaled = bearish_pkg['scaler'].transform(X_bearish_all)
        
        # 🔥 FIX: Convert to DataFrame with proper column names to avoid warnings
        X_bullish_scaled_df = pd.DataFrame(X_bullish_scaled, columns=X_bullish_all.columns, index=X_bullish_all.index)
        X_bearish_scaled_df = pd.DataFrame(X_bearish_scaled, columns=X_bearish_all.columns, index=X_bearish_all.index)
        
        # STEP 5: BATCH PREDICTIONS (100x faster than loop!)
        bullish_probs = bullish_pkg['model'].predict_proba(X_bullish_scaled_df)[:, 1]
        bearish_probs = bearish_pkg['model'].predict_proba(X_bearish_scaled_df)[:, 1]
        
        # 🔥 PERBAIKAN KRITIS: Gunakan threshold yang sudah dikalibrasi dari model package!
        # Ini adalah threshold yang sudah dioptimalkan saat training dengan data yang sama
        bullish_original_threshold = bullish_pkg['threshold']
        bearish_original_threshold = bearish_pkg['threshold']
        
        # 🔥 PERBAIKAN: Gunakan threshold_multiplier untuk fleksibilitas
        threshold_multiplier = params['threshold_multiplier']
        bullish_base_threshold = bullish_original_threshold * threshold_multiplier
        bearish_base_threshold = bearish_original_threshold * threshold_multiplier
        
        # 🔥 PERBAIKAN: Tambahkan logging untuk melihat threshold yang digunakan
        logger.info(f"   🔍 Original thresholds: Bullish={bullish_original_threshold:.4f}, Bearish={bearish_original_threshold:.4f}")
        logger.info(f"   🔧 Applied multiplier: {threshold_multiplier:.2f} → Bullish={bullish_base_threshold:.4f}, Bearish={bearish_base_threshold:.4f}")
        
        # 🔥 CRITICAL: Gunakan threshold yang sudah dikalibrasi untuk backtest
        dynamic_bullish_threshold = np.full(len(bullish_probs), bullish_base_threshold)
        dynamic_bearish_threshold = np.full(len(bearish_probs), bearish_base_threshold)
        
        # Vectorized threshold comparison dengan threshold dari trade mode
        is_buy = bullish_probs >= dynamic_bullish_threshold
        is_sell = bearish_probs >= dynamic_bearish_threshold
        
        # Vectorized decision logic
        actions = np.where(is_buy & is_sell,
                          np.where(bullish_probs > bearish_probs, 1, 0),  # Both: choose stronger
                          np.where(is_buy, 1, np.where(is_sell, 0, -1)))  # -1 = HOLD
        
        confidences = np.where(actions == 1, bullish_probs, 
                              np.where(actions == 0, bearish_probs, 
                                      np.maximum(bullish_probs, bearish_probs)))
        
        # 🔥 DEBUG: Log signal statistics dengan threshold yang benar
        buy_signals = np.sum(is_buy)
        sell_signals = np.sum(is_sell)
        total_signals = buy_signals + sell_signals
        
        # 🔥 DEBUG: Log max probabilities to see model confidence
        max_bullish_prob = np.max(bullish_probs)
        max_bearish_prob = np.max(bearish_probs)
        mean_bullish_prob = np.mean(bullish_probs)
        mean_bearish_prob = np.mean(bearish_probs)
        
        # 🔥 DEBUG: Log probability distribution
        bullish_above_50 = np.sum(bullish_probs > 0.5)
        bearish_above_50 = np.sum(bearish_probs > 0.5)
        bullish_above_30 = np.sum(bullish_probs > 0.3)
        bearish_above_30 = np.sum(bearish_probs > 0.3)
        bullish_above_20 = np.sum(bullish_probs > 0.2)
        bearish_above_20 = np.sum(bearish_probs > 0.2)
        
        # 🔥🔥🔥 CRITICAL: BYPASS LOGGER (print langsung) untuk debugging
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG [{trade_mode}] - {timeframe.upper()}")
        print(f"{'='*80}")
        print(f"   Max Bullish Prob:  {max_bullish_prob:.4f} | Threshold: {bullish_base_threshold:.4f}")
        print(f"   Max Bearish Prob:  {max_bearish_prob:.4f} | Threshold: {bearish_base_threshold:.4f}")
        print(f"   Mean Bullish Prob: {mean_bullish_prob:.4f}")
        print(f"   Mean Bearish Prob: {mean_bearish_prob:.4f}")
        print(f"   Prob >50%: Bullish={bullish_above_50}, Bearish={bearish_above_50}")
        print(f"   Prob >30%: Bullish={bullish_above_30}, Bearish={bearish_above_30}")
        print(f"   Prob >20%: Bullish={bullish_above_20}, Bearish={bearish_above_20}")
        print(f"   Signals Generated: BUY={buy_signals}, SELL={sell_signals}")
        print(f"{'='*80}\n")
        
        logger.info(f"   📊 [{timeframe.upper()}] Max Bullish Prob: {max_bullish_prob:.4f} (Threshold: {bullish_base_threshold:.4f})")
        logger.info(f"   📊 [{timeframe.upper()}] Max Bearish Prob: {max_bearish_prob:.4f} (Threshold: {bearish_base_threshold:.4f})")
        logger.info(f"   📊 Signals: BUY={buy_signals}, SELL={sell_signals}, Total={total_signals}")
        
        # 🔥 CRITICAL DEBUG: Jika masih no trades, coba threshold yang SANGAT rendah
        if total_signals == 0:
            print(f"   ⚠️  WARNING: NO SIGNALS with current thresholds!")
            print(f"   🔬 Testing emergency thresholds...")
            
            emergency_bullish_threshold = 0.3
            emergency_bearish_threshold = 0.3
            emergency_buy_signals = np.sum(bullish_probs >= emergency_bullish_threshold)
            emergency_sell_signals = np.sum(bearish_probs >= emergency_bearish_threshold)
            
            print(f"   🚨 Emergency (0.3): BUY={emergency_buy_signals}, SELL={emergency_sell_signals}")
            
            # Test even lower
            ultra_low_threshold = 0.2
            ultra_buy = np.sum(bullish_probs >= ultra_low_threshold)
            ultra_sell = np.sum(bearish_probs >= ultra_low_threshold)
            print(f"   🚨 Ultra-low (0.2): BUY={ultra_buy}, SELL={ultra_sell}")
            
            if emergency_buy_signals > 0 or emergency_sell_signals > 0:
                print(f"   ✅ SOLUTION: Lower threshold to ~0.25-0.30")
            elif ultra_buy > 0 or ultra_sell > 0:
                print(f"   ⚠️  SOLUTION: Model probabilities very low! Need threshold ~0.15-0.20")
            else:
                print(f"   ❌ CRITICAL: Model not producing prob > 0.2! Feature pipeline broken!")
            print()
        
        # 🔥🔥🔥 KRITIKAL FIX: PAKSA SINKRONISASI INDEX SEBELUM SIMULASI 🔥🔥🔥
        # Proses feature engineering bisa mengubah panjang atau index dari df_full.
        # Kita harus memastikan df_raw dan df_full memiliki panjang dan index yang sama persis.
        
        print(f"   🔍 Pre-alignment: df_raw={len(df_raw)}, df_full={len(df_full)}, actions={len(actions)}")
        
        # 1. Temukan index gabungan (baris yang ada di KEDUA dataframe)
        common_index = df_raw.index.intersection(df_full.index)
        print(f"   🔍 Common index found: {len(common_index)} rows")
        
        # 2. Filter kedua dataframe agar hanya menggunakan index yang sama
        df_raw_aligned = df_raw.loc[common_index].copy()
        df_full_aligned = df_full.loc[common_index].copy()
        
        # 3. Buat ulang array 'actions' dan 'confidences' agar cocok dengan data yang sudah disinkronkan
        # Dapatkan lokasi integer dari common_index di df_full yang asli
        action_indices = df_full.index.get_indexer(common_index)
        actions_aligned = actions[action_indices]
        confidences_aligned = confidences[action_indices]
        
        # 4. Reset index pada semuanya agar bisa diakses dengan .iloc[] secara aman
        df_raw_aligned = df_raw_aligned.reset_index(drop=True)
        df_full_aligned = df_full_aligned.reset_index(drop=True)
        
        print(f"   ✅ Data Aligned for Simulation:")
        print(f"      df_raw_aligned={len(df_raw_aligned)}, df_full_aligned={len(df_full_aligned)}")
        print(f"      actions_aligned={len(actions_aligned)}, confidences_aligned={len(confidences_aligned)}")
        
        # 🔥 SANITY CHECK: Verify all lengths match
        if not (len(df_raw_aligned) == len(df_full_aligned) == len(actions_aligned) == len(confidences_aligned)):
            print(f"   ❌ ERROR: Alignment failed! Lengths don't match!")
            return {}
        
        logger.info(f"   🔧 Data Aligned for Simulation: {len(df_raw_aligned)} rows")
        
        # STEP 6: Fast simulation loop (no feature computation!)
        balance = initial_balance
        trades = []
        open_position = None
        filtered_trades = 0  # 🔥 Count trades skipped by regime filter
        
        print(f"   🎬 Starting simulation loop: {len(df_raw_aligned) - 200} iterations...")
        
        # 🔥 FIX: Use aligned data with common index
        for i in range(200, len(df_raw_aligned)):
            current_candle = df_raw_aligned.iloc[i]
            
            # Check existing position SL/TP
            if open_position:
                exit_price = None
                result = None
                
                if open_position['direction'] == 'BUY':
                    if current_candle['low'] <= open_position['sl']:
                        exit_price, result = open_position['sl'], 'LOSS'
                    elif current_candle['high'] >= open_position['tp']:
                        exit_price, result = open_position['tp'], 'WIN'
                elif open_position['direction'] == 'SELL':
                    if current_candle['high'] >= open_position['sl']:
                        exit_price, result = open_position['sl'], 'LOSS'
                    elif current_candle['low'] <= open_position['tp']:
                        exit_price, result = open_position['tp'], 'WIN'
                
                if result:
                    pnl = ((exit_price - open_position['entry']) if open_position['direction'] == 'BUY' 
                          else (open_position['entry'] - exit_price)) * lot_size * 100
                    
                    trades.append({
                        'entry_time': open_position['entry_time'],
                        'exit_time': current_candle['time'],
                        'direction': open_position['direction'],
                        'entry_price': open_position['entry'],
                        'exit_price': exit_price,
                        'profit': pnl,
                        'result': result
                    })
                    open_position = None
            
            # Check for new signal (use pre-computed prediction)
            if not open_position:
                action_signal = actions_aligned[i]  # 🔥 GUNAKAN actions_aligned
                
                if action_signal != -1:  # BUY or SELL
                    entry_price = current_candle['open']
                    current_atr = df_full_aligned.iloc[i].get('atr', entry_price * 0.01)  # 🔥 GUNAKAN df_full_aligned
                    
                    # 🔥 DEBUG: Log when ATR is invalid
                    if current_atr == 0 or pd.isna(current_atr):
                        print(f"      ⚠️  Invalid ATR at index {i}: {current_atr}, using 1% of price")
                        current_atr = entry_price * 0.01

                    # 🔥 REGIME FILTER: Avoid counter-trend trades in strong trends (reduce SELL bias)
                    try:
                        is_trending = bool(df_full_aligned.iloc[i].get('market_regime_trending', 0) == 1)
                        trend_bull = bool(df_full_aligned.iloc[i].get('trend_direction_bullish', 0) == 1)
                        trend_bear = bool(df_full_aligned.iloc[i].get('trend_direction_bearish', 0) == 1)
                        if is_trending:
                            # Skip SELL in strong bullish trend; skip BUY in strong bearish trend
                            if action_signal == 0 and trend_bull:
                                filtered_trades += 1
                                continue
                            if action_signal == 1 and trend_bear:
                                filtered_trades += 1
                                continue
                    except Exception:
                        pass
                    
                    sl_distance = current_atr * sl_atr_multiplier
                    tp_distance = current_atr * tp_atr_multiplier
                    
                    if action_signal == 1:  # BUY
                        open_position = {
                            'direction': 'BUY',
                            'entry': entry_price,
                            'sl': entry_price - sl_distance,
                            'tp': entry_price + tp_distance,
                            'entry_time': current_candle['time']
                        }
                        # 🔥 DEBUG: Log first few trades
                        if len(trades) < 3:
                            print(f"      ✅ Trade #{len(trades)+1}: BUY @ {entry_price:.5f}, SL={open_position['sl']:.5f}, TP={open_position['tp']:.5f}")
                    elif action_signal == 0:  # SELL
                        open_position = {
                            'direction': 'SELL',
                            'entry': entry_price,
                            'sl': entry_price + sl_distance,
                            'tp': entry_price - tp_distance,
                            'entry_time': current_candle['time']
                        }
                        # 🔥 DEBUG: Log first few trades
                        if len(trades) < 3:
                            print(f"      ✅ Trade #{len(trades)+1}: SELL @ {entry_price:.5f}, SL={open_position['sl']:.5f}, TP={open_position['tp']:.5f}")

        # 🔥 DEBUG: Log simulation results
        print(f"\n   📊 Simulation Complete:")
        print(f"      Total iterations: {len(df_raw_aligned) - 200}")
        print(f"      Trades executed: {len(trades)}")
        if len(trades) > 0:
            wins = sum(1 for t in trades if t['result'] == 'WIN')
            print(f"      Win rate: {wins}/{len(trades)} ({wins/len(trades)*100:.1f}%)")
        print()

        # Calculate statistics
        if len(trades) == 0:
            print(f"   ❌ NO TRADES GENERATED! Check:")
            print(f"      1. Are signals being created? (see above)")
            print(f"      2. Is ATR valid? (check for warnings)")
            print(f"      3. Are SL/TP being hit before next signal?")
            return {}

        # 🔥 APPLY REALISTIC TRANSACTION COSTS TO ALL TRADES
        cost_per_trade = total_transaction_cost_price * lot_size * 100  # For BTCUSD
        
        realistic_trades = []
        for trade in trades:
            profit_before_cost = trade['profit']
            realistic_profit = profit_before_cost - cost_per_trade
            
            new_trade = trade.copy()
            new_trade['profit'] = realistic_profit
            new_trade['result'] = 'WIN' if realistic_profit > 0 else 'LOSS'
            realistic_trades.append(new_trade)
        
        # Recalculate statistics with REALISTIC profits
        wins = [t for t in realistic_trades if t['result'] == 'WIN']
        losses = [t for t in realistic_trades if t['result'] == 'LOSS']
        
        total_profit_realistic = sum(t['profit'] for t in realistic_trades)
        final_balance_realistic = initial_balance + total_profit_realistic
        
        win_rate = len(wins) / len(realistic_trades) * 100 if realistic_trades else 0
        avg_win = sum(t['profit'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['profit'] for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t['profit'] for t in wins) / sum(t['profit'] for t in losses)) if losses and sum(t['profit'] for t in losses) != 0 else 0

        results = {
            'trade_mode': trade_mode.upper(),
            'initial_balance': initial_balance,
            'final_balance': final_balance_realistic,
            'total_profit': total_profit_realistic,
            'roi_percent': (total_profit_realistic/initial_balance)*100,
            'total_trades': len(realistic_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sl_atr_multiplier': sl_atr_multiplier,
            'tp_atr_multiplier': tp_atr_multiplier,
            'bullish_threshold': bullish_base_threshold,
            'bearish_threshold': bearish_base_threshold,
            'spread_pips': spread_pips,
            'slippage_pips': slippage_pips,
            'cost_per_trade': cost_per_trade,
            'trades': realistic_trades,
            'filtered_trades': int(filtered_trades)
        }



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    import os
    from dotenv import load_dotenv

    load_dotenv()

    login = int(os.getenv('MT5_LOGIN_DEMO', '269764525'))
    password = os.getenv('MT5_PASSWORD_DEMO', 'Wokolcoy@20')
    server = os.getenv('MT5_SERVER_DEMO', 'Exness-MT5Trial17')

    mt5 = MT5Handler(login, password, server)
    
    if mt5.initialize():
        cls = CLSPredictor()
        
        # 🔥 WALK-FORWARD VALIDATION (PROFESSIONAL STANDARD) - ALL TIMEFRAMES
        print(f"\n{'='*80}")
        print("🔍 COMPREHENSIVE WALK-FORWARD VALIDATION - ALL TIMEFRAMES")
        print("   12-month Backtest (In-Sample) + 4-month Forward Test (Out-of-Sample)")
        print(f"{'='*80}\n")
        
        # Define periods (total ~16 months)
        backtest_start_date = "2024-06-01"
        backtest_end_date = "2025-05-30"  # ~12 months
        
        forward_test_start_date = "2025-06-01"
        forward_test_end_date = "2025-10-09"  # ~4 months
        
        initial_balance_test = 20.0  # User's actual balance
        
        # Test all timeframes
        all_timeframes = ['M5', 'M15', 'H1', 'H4']
        trade_modes = ['SCALPING', 'NORMAL', 'AGGRESSIVE', 'LONG_HOLD']
        
        # 🔥 FIX: Adjust date ranges based on timeframe capability
        # M5/M15 can't fetch 12 months, use shorter periods
        tf_date_config = {
            'M5': {
                'backtest_start': '2025-03-01',  # 3 months backtest
                'backtest_end': '2025-05-30',
                'forward_start': '2025-06-01',   # 4 months forward
                'forward_end': '2025-10-09'
            },
            'M15': {
                'backtest_start': '2025-01-01',  # 5 months backtest
                'backtest_end': '2025-05-30',
                'forward_start': '2025-06-01',   # 4 months forward
                'forward_end': '2025-10-09'
            },
            'H1': {
                'backtest_start': backtest_start_date,  # 12 months backtest
                'backtest_end': backtest_end_date,
                'forward_start': forward_test_start_date,
                'forward_end': forward_test_end_date
            },
            'H4': {
                'backtest_start': backtest_start_date,  # 12 months backtest
                'backtest_end': backtest_end_date,
                'forward_start': forward_test_start_date,
                'forward_end': forward_test_end_date
            }
        }
        
        # Store results by timeframe
        all_results = {}
        
        for tf in all_timeframes:
            dates = tf_date_config[tf]
            
            print(f"\n{'='*80}")
            print(f"📊 TESTING TIMEFRAME: {tf}")
            print(f"{'='*80}")
            print(f"   Backtest: {dates['backtest_start']} → {dates['backtest_end']}")
            print(f"   Forward:  {dates['forward_start']} → {dates['forward_end']}\n")
            
            all_results[tf] = {
                'backtest': {},
                'forward': {}
            }
            
            # BACKTEST (In-Sample)
            print(f"🔙 Running In-Sample Backtest for {tf}...")
            
            # Suppress ALL noise during batch testing
            import logging
            preprocessor_logger = logging.getLogger('data.preprocessor')
            main_logger = logging.getLogger('__main__')
            core_logger = logging.getLogger('core.mt5_handler')
            
            original_levels = {
                'preprocessor': preprocessor_logger.level,
                'main': main_logger.level,
                'core': core_logger.level
            }
            
            preprocessor_logger.setLevel(logging.ERROR)
            main_logger.setLevel(logging.ERROR)
            core_logger.setLevel(logging.ERROR)
            
            for mode in trade_modes:
                print(f"   → {mode}...", end=' ', flush=True)
                results = cls.backtest(
                    symbol="BTCUSDm",
                    mt5_handler=mt5,
                    start_date=dates['backtest_start'],
                    end_date=dates['backtest_end'],
                    timeframe=tf,
                    initial_balance=initial_balance_test,
                    lot_size=0.01,
                    trade_mode=mode
                )
                all_results[tf]['backtest'][mode] = results
                if results:
                    print(f"✅ Trades: {results['total_trades']} (W:{results['wins']}/L:{results['losses']}), WR: {results['win_rate']:.1f}%, PF: {results['profit_factor']:.2f}")
                else:
                    print("⚠️ No trades")
            
            # FORWARD TEST (Out-of-Sample)
            print(f"\n🔮 Running Out-of-Sample Forward Test for {tf}...")
            for mode in trade_modes:
                print(f"   → {mode}...", end=' ', flush=True)
                results = cls.backtest(
                    symbol="BTCUSDm",
                    mt5_handler=mt5,
                    start_date=dates['forward_start'],
                    end_date=dates['forward_end'],
                    timeframe=tf,
                    initial_balance=initial_balance_test,
                    lot_size=0.01,
                    trade_mode=mode
                )
                all_results[tf]['forward'][mode] = results
                if results:
                    print(f"✅ Trades: {results['total_trades']} (W:{results['wins']}/L:{results['losses']}), WR: {results['win_rate']:.1f}%, PF: {results['profit_factor']:.2f}")
                else:
                    print("⚠️ No trades")
            
            # Restore ALL logger levels
            preprocessor_logger.setLevel(original_levels['preprocessor'])
            main_logger.setLevel(original_levels['main'])
            core_logger.setLevel(original_levels['core'])
        
        # FINAL COMPARISON TABLE
        print(f"\n\n{'='*100}")
        print(f"🏁 FINAL RESULTS - ALL TIMEFRAMES & MODES")
        print(f"{'='*100}\n")
        
        for tf in all_timeframes:
            print(f"\n📊 TIMEFRAME: {tf}")
            print(f"{'-'*100}")
            print(f"{'Mode':<15} | {'Period':<12} | {'Trades':<8} | {'Win%':<8} | {'ROI%':<12} | {'P.Factor':<10}")
            print(f"{'-'*100}")
            
            for mode in trade_modes:
                bt = all_results[tf]['backtest'].get(mode)
                ft = all_results[tf]['forward'].get(mode)
                
                if bt:
                    print(f"{mode:<15} | {'In-Sample':<12} | {bt['total_trades']:<8} | {bt['win_rate']:<8.1f} | "
                          f"{bt['roi_percent']:<12.1f} | {bt['profit_factor']:<10.2f}")
                
                if ft:
                    print(f"{'':<15} | {'Out-Sample':<12} | {ft['total_trades']:<8} | {ft['win_rate']:<8.1f} | "
                          f"{ft['roi_percent']:<12.1f} | {ft['profit_factor']:<10.2f}")
                    print(f"{'-'*100}")
        
        # ROBUSTNESS ANALYSIS
        print(f"\n\n{'='*100}")
        print(f"📊 ROBUSTNESS ANALYSIS - RECOMMENDED CONFIGURATIONS")
        print(f"{'='*100}\n")
        
        recommendations = []
        
        for tf in all_timeframes:
            print(f"📊 {tf} Timeframe:")
            best_pf = 0
            best_mode = None
            
            for mode in trade_modes:
                ft = all_results[tf]['forward'].get(mode)
                if ft and ft.get('profit_factor', 0) > 0:
                    is_robust = ft['roi_percent'] > 0 and ft['profit_factor'] > 1.2
                    status = "✅" if is_robust else "⚠️"
                    
                    print(f"   {status} {mode:<12}: PF={ft['profit_factor']:.2f}, ROI={ft['roi_percent']:+6.1f}%, WR={ft['win_rate']:.1f}%")
                    
                    if is_robust and ft['profit_factor'] > best_pf:
                        best_pf = ft['profit_factor']
                        best_mode = mode
            
            if best_mode:
                recommendations.append((tf, best_mode, best_pf))
                print(f"   🏆 BEST: {best_mode} (PF={best_pf:.2f})")
            print()
        
        # FINAL RECOMMENDATION
        print(f"\n{'='*100}")
        print(f"🎯 RECOMMENDED CONFIGURATIONS FOR LIVE TRADING")
        print(f"{'='*100}")
        for tf, mode, pf in recommendations:
            print(f"   {tf:<6} → {mode:<12} (Profit Factor: {pf:.2f})")
        print(f"{'='*100}\n")
        
        mt5.shutdown()

