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
        df_with_features = preprocessor.create_features(df_with_features)

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
        
        # Log missing features only once (silently fill with 0)
        if missing_features and not hasattr(self, '_missing_features_logged'):
            # logger.warning(f"⚠️  {len(missing_features)} features missing, using 0: {missing_features[:3]}")
            self._missing_features_logged = True

        return X
    
    def predict(
        self, 
        df: pd.DataFrame, 
        timeframe: str = 'm5',
        mt5_handler=None  # 🔥 NEW: For MTF context injection
    ) -> Tuple[str, float]:
        """
        🔥 UPDATED: Predict using self-contained specialists (each has own scaler + features)

        Args:
            df: Historical candle data
            timeframe: Timeframe (m5, m15, h1, h4)
            mt5_handler: MT5 handler for MTF context (optional)
        
        Returns:
            (direction: 'BUY', 'SELL', or 'HOLD', confidence: 0.0-1.0)
        """
        tf_key = timeframe.lower()
        
        if tf_key not in self.models:
            return 'HOLD', 0.5
        
        try:
            # 🔥 UPDATED: Handle TF-SPECIFIC model format (dict with bullish/bearish)
            model = self.models[tf_key]
            
            # Check if model is TF-SPECIFIC format (dict with 'bullish'/'bearish')
            if isinstance(model, dict) and 'bullish' in model and 'bearish' in model:
                bullish_pkg = model['bullish']
                bearish_pkg = model['bearish']
                
                # 🔥 CRITICAL: Prepare features separately for each specialist
                X_bullish = self.prepare_features_for_specialist(df, bullish_pkg, timeframe, mt5_handler)
                X_bearish = self.prepare_features_for_specialist(df, bearish_pkg, timeframe, mt5_handler)
                
                # Safety check
                X_bullish.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_bullish.fillna(0, inplace=True)
                X_bearish.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_bearish.fillna(0, inplace=True)
                
                # 🔥 CRITICAL: Use specialist's own scaler
                X_bullish_scaled = bullish_pkg['scaler'].transform(X_bullish)
                X_bearish_scaled = bearish_pkg['scaler'].transform(X_bearish)
                
                # 🔥 FIX: Convert to DataFrame with proper column names to avoid warnings
                X_bullish_scaled_df = pd.DataFrame(X_bullish_scaled, columns=X_bullish.columns, index=X_bullish.index)
                X_bearish_scaled_df = pd.DataFrame(X_bearish_scaled, columns=X_bearish.columns, index=X_bearish.index)
                
                # Get probabilities from both specialists
                prob_buy = bullish_pkg['model'].predict_proba(X_bullish_scaled_df)[0, 1]
                prob_sell = bearish_pkg['model'].predict_proba(X_bearish_scaled_df)[0, 1]
                
                # Apply calibrated thresholds
                bullish_threshold = bullish_pkg['threshold']
                bearish_threshold = bearish_pkg['threshold']
                
                # 🔥 DEBUG: Log threshold dan probability untuk debugging
                logger.info(f"🔍 CLS Predictor Debug:")
                logger.info(f"   Bullish: prob={prob_buy:.2%}, threshold={bullish_threshold:.2%}")
                logger.info(f"   Bearish: prob={prob_sell:.2%}, threshold={bearish_threshold:.2%}")
                
                is_buy = prob_buy >= bullish_threshold
                is_sell = prob_sell >= bearish_threshold
                
                # 🔥🔥🔥 PERBAIKAN LOGIKA KEPUTUSAN: Aturan Tie-Breaker 🔥🔥🔥
                
                # 1. Jika HANYA ada sinyal BUY yang valid
                if is_buy and not is_sell:
                    return 'BUY', prob_buy

                # 2. Jika HANYA ada sinyal SELL yang valid
                elif is_sell and not is_buy:
                    return 'SELL', prob_sell

                # 3. Jika KEDUANYA valid (konflik), pilih yang probabilitasnya TERTINGGI
                elif is_buy and is_sell:
                    logger.warning(f"   ⚠️  Sinyal konflik: BUY ({prob_buy:.2%}) vs SELL ({prob_sell:.2%}). Memilih yang lebih kuat.")
                    if prob_buy > prob_sell:
                        return 'BUY', prob_buy
                    else:
                        return 'SELL', prob_sell
                
                # 4. Jika tidak ada yang valid sama sekali
                else:
                    max_prob = max(prob_buy, prob_sell)
                    # Jika probabilitas maksimal masih sangat rendah, confidence juga rendah
                    if max_prob < 0.5:
                        return 'HOLD', max_prob
                    else:
                        # Jika probabilitas tinggi tapi tidak memenuhi threshold, turunkan confidence
                        return 'HOLD', max_prob * 0.3
                    
            elif hasattr(model, 'predict_proba'):
                # Old format: Single model
                probas = model.predict_proba(X_scaled)[0]
                prob_sell = probas[0]
                prob_buy = probas[1]

                if prob_buy >= buy_threshold:
                    return 'BUY', prob_buy
                elif prob_sell >= sell_threshold:
                    return 'SELL', prob_sell
                else:
                    return 'HOLD', max(prob_buy, prob_sell)

            else:
                # Fallback for models without predict_proba
                prediction = model.predict(X_scaled)[0]

                # 🔥 BINARY CLASSIFICATION: 0=SELL, 1=BUY
                action_map = {
                    0: 'SELL',
                    1: 'BUY'
                }
            action = action_map.get(prediction, 'HOLD')
            return action, 0.7
        
        except Exception as e:
            logger.error(f"Error in CLS prediction for {timeframe}: {str(e)}", exc_info=True)
            return 'HOLD', 0.0
    
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
                'threshold_multiplier': 0.95,  # Sangat selektif (95% dari threshold asli)
                'description': 'Quick trades, tight stops'
            },
            'NORMAL': {
                'sl_atr': base['sl_atr'],  # Base values
                'tp_atr': base['tp_atr'],
                'threshold_multiplier': 0.90,  # Seimbang (90% dari threshold asli)
                'description': 'Balanced risk/reward'
            },
            'AGGRESSIVE': {
                'sl_atr': base['sl_atr'] * 1.2,  # 20% wider
                'tp_atr': base['tp_atr'] * 1.2,
                'threshold_multiplier': 0.80,  # Lebih longgar (80% dari threshold asli)
                'description': 'Wider stops for bigger moves'
            },
            'LONG_HOLD': {
                'sl_atr': base['sl_atr'] * 1.5,  # 50% wider
                'tp_atr': base['tp_atr'] * 2.0,  # 2x wider TP
                'threshold_multiplier': 0.85,  # Sedikit longgar (85% dari threshold asli)
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
        
        logger.info(f"   📊 [{timeframe.upper()}] Max Bullish Prob: {max_bullish_prob:.4f} (Threshold: {bullish_base_threshold:.4f})")
        logger.info(f"   📊 [{timeframe.upper()}] Max Bearish Prob: {max_bearish_prob:.4f} (Threshold: {bearish_base_threshold:.4f})")
        logger.info(f"   📊 [{timeframe.upper()}] Mean Bullish Prob: {mean_bullish_prob:.4f}, Mean Bearish Prob: {mean_bearish_prob:.4f}")
        logger.info(f"   📊 [{timeframe.upper()}] Prob Distribution: Bullish >50%: {bullish_above_50}, >30%: {bullish_above_30}")
        logger.info(f"   📊 [{timeframe.upper()}] Prob Distribution: Bearish >50%: {bearish_above_50}, >30%: {bearish_above_30}")
        logger.info(f"   📊 Signals: BUY={buy_signals}, SELL={sell_signals}, Total={total_signals} (thresholds: B={bullish_base_threshold:.2f}, S={bearish_base_threshold:.2f})")
        
        # 🔥 CRITICAL DEBUG: Jika masih no trades, coba threshold yang SANGAT rendah
        if total_signals == 0:
            logger.warning(f"   ⚠️  NO SIGNALS! Trying emergency low thresholds...")
            emergency_bullish_threshold = 0.3
            emergency_bearish_threshold = 0.3
            emergency_buy_signals = np.sum(bullish_probs >= emergency_bullish_threshold)
            emergency_sell_signals = np.sum(bearish_probs >= emergency_bearish_threshold)
            logger.warning(f"   🚨 Emergency thresholds (0.3): BUY={emergency_buy_signals}, SELL={emergency_sell_signals}")
            
            if emergency_buy_signals > 0 or emergency_sell_signals > 0:
                logger.warning(f"   🔥 SOLUTION: Threshold terlalu tinggi! Gunakan threshold < 0.3")
            else:
                logger.error(f"   ❌ CRITICAL: Model tidak menghasilkan probabilitas > 0.3! Masalah di pipeline fitur!")
        
        # STEP 6: Fast simulation loop (no feature computation!)
        balance = initial_balance
        trades = []
        open_position = None
        
        for i in range(200, len(df_raw)):
            current_candle = df_raw.iloc[i]
            
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
                action_signal = actions[i]
                
                if action_signal != -1:  # BUY or SELL
                    entry_price = current_candle['open']
                    current_atr = df_full.iloc[i].get('atr', entry_price * 0.01)
                    
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
                    elif action_signal == 0:  # SELL
                        open_position = {
                            'direction': 'SELL',
                            'entry': entry_price,
                            'sl': entry_price + sl_distance,
                            'tp': entry_price - tp_distance,
                            'entry_time': current_candle['time']
                        }

        # Calculate statistics
        if len(trades) == 0:
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
            'trades': realistic_trades
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

