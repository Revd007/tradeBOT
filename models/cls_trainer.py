"""
CLS Model Trainer with DeepSeek LLM Integration
Train classifier models for different timeframes to predict trade direction
Features:
- GPU-accelerated DeepSeek LLM for feature analysis
- SMOTE for handling class imbalance
- XGBoost support
- 5 years of training data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# XGBoost support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

# LightGBM support
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not installed. Install with: pip install lightgbm")

# Optuna for hyperparameter tuning
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not installed. Install with: pip install optuna")

# LLM Integration with GPU support
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("llama_cpp_python not installed. LLM features disabled.")

logger = logging.getLogger(__name__)


class CLSModelTrainer:
    """Train CLS classifier models for trade direction prediction with LLM integration"""
    
    def __init__(
        self, 
        output_dir: str = "./models/saved_models",
        llm_model_path: str = "./models/Llama-3.2-3B-Instruct-BF16.gguf",
        use_gpu: bool = True,
        use_smote: bool = False,  # 🔥 DISABLED: Ganti dengan scale_pos_weight
        use_optuna: bool = False,  # 🔥 DISABLED: causes overfitting
        use_ensemble: bool = True,
        use_lstm: bool = True       # 🔥 NEW: Enable LSTM for temporal patterns
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timeframes = ['M5', 'M15', 'H1', 'H4']
        self.models = {}
        self.scalers = {}
        self.lstm_models = {}  # 🔥 NEW: Store LSTM models
        self.use_smote = True  # 🔥 AKTIFKAN KEMBALI, KITA AKAN GUNAKAN SMOTETomek
        self.use_optuna = use_optuna
        self.use_ensemble = use_ensemble
        self.use_lstm = False  # 🔥 DISABLED: LSTM failing & not contributing
        
        # Initialize LLM with GPU
        self.llm = None
        if LLAMA_AVAILABLE and Path(llm_model_path).exists():
            try:
                model_name = Path(llm_model_path).name
                logger.info(f"🚀 Loading {model_name}...")
                self.llm = Llama(
                    model_path=llm_model_path,
                    n_gpu_layers=-1,  # Load all layers to GPU (RTX 3070)
                    n_ctx=2048,       # Smaller for Llama-3.2-3B (was 4096)
                    n_batch=256,      # Smaller batch (was 512)
                    n_threads=6,      # Less threads (was 8)
                    verbose=False
                )
                logger.info(f"✅ {model_name} loaded on GPU!")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load LLM: {e}")
                self.llm = None
        else:
            if not LLAMA_AVAILABLE:
                logger.warning("⚠️  llama_cpp_python not available")
            if not Path(llm_model_path).exists():
                logger.warning(f"⚠️  LLM model not found: {llm_model_path}")
    
    def add_calendar_features(self, df: pd.DataFrame, mt5_handler) -> pd.DataFrame:
        """
        🔥 ENHANCED: Add calendar + fundamental context features
        """
        logger.info("📅 Adding calendar/fundamental features...")
        
        try:
            from data.calendar_scraper import EconomicCalendarScraper
            import os
            
            # Initialize calendar scraper
            api_key = os.getenv('TRADING_ECONOMICS_KEY', 'cd83fd1ed69545d:ivc5icl7su21s5m')
            calendar = EconomicCalendarScraper(api_key=api_key)
            
            # Add time-based features for calendar events
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
            
            # High-impact sessions (London: 7-16, NY: 12-21 UTC)
            df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
            df['is_overlap_session'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
            
            # High-impact days (Tue-Thu usually have more volatility)
            df['is_high_impact_day'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
            
            # Weekend proximity (Friday afternoon, Monday morning = low liquidity)
            df['is_weekend_proximity'] = (
                ((df['day_of_week'] == 4) & (df['hour'] >= 15)) |  # Friday PM
                ((df['day_of_week'] == 0) & (df['hour'] < 3))      # Monday AM
            ).astype(int)
            
            # News volatility proxy (based on historical patterns)
            # High-impact news usually at: 8:30, 10:00, 13:30, 14:00, 15:00 UTC
            df['is_news_hour'] = df['hour'].isin([8, 10, 13, 14, 15]).astype(int)
            
            # Day of month (NFP = 1st Friday, FOMC = varies)
            df['day_of_month'] = pd.to_datetime(df['time']).dt.day
            df['is_first_week'] = (df['day_of_month'] <= 7).astype(int)
            
            # 🔥 NEW: Trend Context (SMA200 crossing & slope)
            if 'close' in df.columns:
                df['sma200'] = df['close'].rolling(200).mean()
                df['is_above_sma200'] = (df['close'] > df['sma200']).astype(int)
                df['sma200_slope'] = df['sma200'].diff(10) / df['sma200']  # 10-period slope
                
                # SMA50 for medium-term trend
                df['sma50'] = df['close'].rolling(50).mean()
                df['is_above_sma50'] = (df['close'] > df['sma50']).astype(int)
                df['sma_cross'] = ((df['sma50'] > df['sma200']) & (df['sma50'].shift(1) <= df['sma200'].shift(1))).astype(int)
            
            # 🔥 NEW: Volatility Filtering (avoid low volatility periods)
            if 'atr' in df.columns:
                df['atr_ma20'] = df['atr'].rolling(20).mean()
                df['is_high_volatility'] = (df['atr'] > df['atr_ma20']).astype(int)
                
                # Handle NaN values before qcut
                atr_clean = df['atr'].dropna()
                if len(atr_clean) > 0:
                    try:
                        df['volatility_regime'] = pd.qcut(df['atr'], q=3, labels=[0, 1, 2], duplicates='drop')
                        df['volatility_regime'] = df['volatility_regime'].cat.codes  # Convert to int (handles NaN as -1)
                        df['volatility_regime'] = df['volatility_regime'].replace(-1, 1)  # Replace NaN (-1) with medium (1)
                    except Exception as e:
                        logger.warning(f"⚠️  volatility_regime failed: {e}, using fallback")
                        df['volatility_regime'] = 1  # Default to medium volatility
            
            logger.info(f"✅ Added 18 calendar + fundamental features")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to add calendar features: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def collect_training_data(
        self,
        mt5_handler,
        symbol: str,
        timeframe: str,
        candles: int = 500000
    ) -> pd.DataFrame:
        """
        🔥 ENHANCED: Collect historical data with calendar/news features
        
        Args:
            mt5_handler: MT5 connection
            symbol: Trading symbol (e.g., XAUUSDm)
            timeframe: Timeframe (M5, M15, H1, H4)
            candles: Number of candles to collect
        """
        logger.info(f"Collecting {candles:,} candles for {timeframe} data...")
        
        # 🔥 FIX: Always use batch collection for reliability (10K batch size safe for MT5)
        df = mt5_handler.get_candles_batch(symbol, timeframe, candles, batch_size=10000)
        
        # 🔥 PERBAIKAN URUTAN OPERASI #1: HITUNG INDIKATOR DASAR TERLEBIH DAHULU
        from strategies.base_strategy import BaseStrategy
        
        class TrainingStrategy(BaseStrategy):
            def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
                return None
        
        strategy = TrainingStrategy("Trainer", "MEDIUM")
        df = strategy.add_all_indicators(df)
        logger.info("   ✅ Indikator dasar (RSI, MACD, dll.) telah dihitung.")
        
        # 🔥 PERBAIKAN URUTAN OPERASI #2: SEKARANG LAKUKAN INJEKSI MTF
        logger.info("📊 Injecting Timeframe-Relative MTF Context...")
        timeframe_map = {'M5': 'M15', 'M15': 'H1', 'H1': 'D1', 'H4': 'W1'}
        htf = timeframe_map.get(timeframe)
        
        if htf and htf not in ['D1', 'W1']:
            try:
                timeframe_minutes = {'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440}
                ltf_minutes = timeframe_minutes.get(timeframe, 15)
                htf_minutes = timeframe_minutes.get(htf, 60)
                htf_candles_needed = int(len(df) * (ltf_minutes / htf_minutes)) + 200
                
                logger.info(f"   Fetching {htf_candles_needed} candles from {htf}...")
                df_htf = mt5_handler.get_candles_batch(symbol, htf, htf_candles_needed, batch_size=5000)
                
                # Hitung indikator HTF (RSI menggunakan EWM - robust)
                df_htf['htf_ema_50'] = df_htf['close'].ewm(span=50, adjust=False).mean()
                
                # RSI calculation (robust method)
                delta = df_htf['close'].diff()
                gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
                rs = gain / loss.replace(0, np.nan)
                df_htf['htf_rsi'] = 100 - (100 / (1 + rs))
                
                # Merge dengan merge_asof
                df = pd.merge_asof(
                    df.sort_values('time'),
                    df_htf[['time', 'htf_ema_50', 'htf_rsi']].sort_values('time'),
                    on='time',
                    direction='backward'
                ).dropna(subset=['htf_ema_50', 'htf_rsi'])
                
                # Fitur perbandingan
                df['close_vs_htf_ema'] = (df['close'] - df['htf_ema_50']) / df['htf_ema_50'] * 100
                df['rsi_vs_htf_rsi'] = df['rsi'] - df['htf_rsi']
                
                logger.info(f"   ✅ MTF context from {htf} successfully injected")
            except Exception as e:
                logger.error(f"   ❌ Gagal total menginjeksi MTF context: {e}", exc_info=True)
        
        # Add calendar/news features FIRST (fundamental)
        df = self.add_calendar_features(df, mt5_handler)
        
        # Add features
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        df = preprocessor.create_features(df)
        
        # 🔥 CRITICAL FIX: BALANCED thresholds (not too low, not too high)
        # Too low (0.3) = model bias to SELL (BUY recall 39%)
        # Too high (0.9) = model miss signals (SELL recall 43%)
        # SWEET SPOT: 0.4-0.7 range
        atr_multipliers = {
            'M5': 0.42,  # 42% of ATR (balanced)
            'M15': 0.48, # 48% of ATR
            'H1': 0.55,  # 55% of ATR
            'H4': 0.70   # 70% of ATR
        }
        multiplier = atr_multipliers.get(timeframe, 0.45)
        
        # Calculate median ATR from data
        if 'atr' in df.columns and df['atr'].notna().sum() > 100:
            median_atr = df['atr'].median()
            median_close = df['close'].median()
            threshold = multiplier * (median_atr / median_close)
            logger.info(f"📊 ATR-based threshold: {threshold*100:.3f}% (ATR: ${median_atr:.1f}, Multiplier: {multiplier}x) - {timeframe}")
        else:
            # Fallback if ATR not ready
            threshold_map = {
                'M5': 0.0025,
                'M15': 0.0035,
                'H1': 0.0045,
                'H4': 0.0065
            }
            threshold = threshold_map.get(timeframe, 0.0035)
            logger.warning(f"⚠️  ATR not available, using fixed threshold: {threshold*100:.3f}% - {timeframe}")
        
        # 🔥 STRATEGI 1: TRIPLE BARRIER dengan Dynamic Sample Weighting
        df['target'], df['weights'] = preprocessor.create_labels(
            df,
            method='triple_barrier',
            horizon=12 if timeframe == 'M5' else (10 if timeframe == 'M15' else (8 if timeframe == 'H1' else 6)),
            risk_reward_ratio=1.5,
            atr_multiplier_sl=1.5,
            binary=True
        )
        
        # Remove HOLD samples (label=-1 in binary mode)
        initial_count = len(df)
        df = df[df['target'] != -1].copy()
        df = df.dropna(subset=['target', 'weights'])  # 🔥 Pastikan weights ikut terfilter
        removed_count = initial_count - len(df)
        
        logger.info(f"🎯 TRIPLE BARRIER: Removed {removed_count:,} HOLD samples ({removed_count/initial_count*100:.1f}%)")
        logger.info(f"✅ Collected {len(df)} candles with features")
        
        return df
    
    def analyze_features_with_llm(self, feature_importance: pd.DataFrame, timeframe: str) -> str:
        """
        Use DeepSeek LLM to analyze feature importance and provide insights
        
        Args:
            feature_importance: DataFrame with feature names and importance scores
            timeframe: Trading timeframe
            
        Returns:
            LLM analysis text
        """
        if self.llm is None:
            return "LLM not available"
        
        # Prepare prompt for LLM
        top_features = feature_importance.head(10)
        features_text = "\n".join([
            f"{i+1}. {row['feature']}: {row['importance']:.4f}"
            for i, (_, row) in enumerate(top_features.iterrows())
        ])
        
        prompt = f"""<｜begin▁of▁sentence｜>As a trading AI expert, analyze these top 10 most important features for {timeframe} timeframe trading prediction:

{features_text}

Provide concise insights about:
1. What these features tell us about market behavior
2. Trading recommendations based on feature importance
3. Potential risks or biases

Keep response under 150 words.<｜end▁of▁sentence｜>"""
        
        try:
            logger.info(f"🤖 Analyzing features with LLM...")
            response = self.llm(
                prompt,
                max_tokens=300,
                temperature=0.7,
                top_p=0.9,
                stop=["<｜end▁of▁sentence｜>"],
                echo=False
            )
            
            analysis = response['choices'][0]['text'].strip()
            logger.info(f"\n{'='*60}")
            logger.info(f"🤖 LLM Analysis for {timeframe}:")
            logger.info(f"{'='*60}")
            logger.info(analysis)
            logger.info(f"{'='*60}\n")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ LLM analysis failed: {e}")
            return f"LLM analysis failed: {str(e)}"
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔥 NEW: Add advanced features for better prediction
        - RSI Divergence
        - Volatility Ratio
        - Momentum Indicators
        - Rolling Statistics
        """
        # RSI Divergence (3-period lookback)
        if 'rsi' in df.columns and 'close' in df.columns:
            df['rsi_change'] = df['rsi'].diff(3)
            df['price_change'] = df['close'].pct_change(3)
            df['rsi_divergence'] = (df['rsi_change'] * df['price_change']) < 0  # Divergence detected
            df['rsi_divergence'] = df['rsi_divergence'].astype(int)
        
        # Volatility Ratio (current vs historical)
        if 'atr' in df.columns:
            df['volatility_ratio_10'] = df['atr'] / df['atr'].rolling(10).mean()
            df['volatility_ratio_20'] = df['atr'] / df['atr'].rolling(20).mean()
        
        # Momentum Indicators
        if 'close' in df.columns:
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            
            # Rolling mean return
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
        
        # 🔥 NEW: Contextual Features to improve BUY signal quality
        
        # 1. Interaction: Momentum with Long-term Trend
        # Is the upward momentum happening above SMA200 (trend confirmation)?
        if 'momentum_5' in df.columns and 'is_above_sma200' in df.columns:
            df['trend_momentum_interaction'] = df['momentum_5'] * (df['is_above_sma200'] - 0.5) * 2
        
        # 2. Volatility-Normalized Momentum
        # How strong is this momentum compared to normal volatility?
        if 'momentum_10' in df.columns and 'atr' in df.columns:
            df['volatility_normalized_momentum'] = df['momentum_10'] / df['atr'].replace(0, np.nan)
        
        # 3. Mean Reversion Score
        # How far is price from its 'home' (EMA 50)?
        if 'close' in df.columns and 'ema_50' in df.columns and 'atr' in df.columns:
            df['mean_reversion_score'] = (df['close'] - df['ema_50']) / df['atr'].replace(0, np.nan)
        
        # 4. EMA Spread Strength
        # Measuring the 'fanning out' or spread of moving averages
        if all(x in df.columns for x in ['ema_9', 'ema_21', 'ema_50', 'close']):
            df['ema_spread_short_mid'] = (df['ema_9'] - df['ema_21']) / df['close']
            df['ema_spread_mid_long'] = (df['ema_21'] - df['ema_50']) / df['close']
        
        logger.info(f"✅ Added advanced + contextual features")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, timeframe: str = 'M5') -> Tuple[pd.DataFrame, pd.Series]:
        """
        🔥 FIX #2: Feature selection - reduce curse of dimensionality
        
        Returns:
            X (features), y (target)
        """
        # Add advanced features
        df = self.add_advanced_features(df)
        
        # Select feature columns (exclude metadata and target)
        exclude_cols = [
            'time', 'open', 'high', 'low', 'close', 'tick_volume',
            'target', 'spread', 'real_volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN
        df_clean = df.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        logger.info(f"Initial features: {len(feature_cols)}")
        
        # 🔥 FEATURE SELECTION: Use RandomForest to find top 50 features
        if len(feature_cols) > 50 and len(df_clean) > 1000:
            logger.info("🔍 Performing feature selection (top 50 features)...")
            from sklearn.ensemble import RandomForestClassifier
            
            # Quick RF to get feature importance
            rf_selector = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            rf_selector.fit(X, y)
            
            # Get top 50 features
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf_selector.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(50)['feature'].tolist()
            
            logger.info(f"✅ Selected top 50 features (from {len(feature_cols)})")
            logger.info(f"Top 5: {', '.join(top_features[:5])}")
            
            X = X[top_features]
        
        logger.info(f"Final features: {X.shape[1]}")
        logger.info(f"Samples after cleaning: {len(df_clean)}")
        logger.info(f"Label distribution:\n{y.value_counts()}")
        
        # Check class balance
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        if imbalance_ratio > 3:
            logger.warning(f"⚠️  High class imbalance detected! Ratio: {imbalance_ratio:.1f}")
        else:
            logger.info(f"✅ Good class balance! Ratio: {imbalance_ratio:.1f}")
        
        return X, y
    
    def train_lstm_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        timeframe: str
    ) -> Optional[object]:
        """
        🔥 NEW: Train LSTM model for temporal patterns
        """
        try:
            from models.lstm_model import LSTMTrader, TF_AVAILABLE
            
            if not TF_AVAILABLE:
                logger.warning("⚠️  TensorFlow not available, skipping LSTM")
                return None
            
            logger.info(f"🧠 Training LSTM model for {timeframe}...")
            
            # 🔥 FIX: Lower sequence length to prevent OOM and errors
            seq_lengths = {'M5': 10, 'M15': 8, 'H1': 6, 'H4': 5}
            seq_length = seq_lengths.get(timeframe, 8)
            
            # Create LSTM model with smaller architecture
            lstm = LSTMTrader(
                sequence_length=seq_length,
                lstm_units=32,      # Reduced from 64
                dropout_rate=0.4,   # Higher dropout
                learning_rate=0.002  # Slightly higher LR
            )
            
            # Train LSTM with smaller batch size
            history = lstm.train(
                X_train, y_train,
                X_test, y_test,
                epochs=30,          # Reduced from 50
                batch_size=32,      # Reduced from 64
                verbose=0  # Silent training
            )
            
            return lstm
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            return None
    
    def optuna_tune_lightgbm(self, X_train, y_train, X_test, y_test, n_trials: int = 30) -> Dict:
        """
        🔥 NEW: Auto-tune LightGBM hyperparameters with Optuna
        """
        if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
            logger.warning("Optuna or LightGBM not available, skipping tuning")
            return {}
        
        logger.info(f"🔍 Starting Optuna hyperparameter tuning ({n_trials} trials)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Evaluate on test set (F1 score for binary)
            from sklearn.metrics import f1_score
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            return f1
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"✅ Optuna tuning complete!")
        logger.info(f"   Best F1 Score: {study.best_value:.2%}")
        logger.info(f"   Best params: {study.best_params}")
        
        return study.best_params
    
    def train_specialist_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series,
        specialty: str,  # 'BULLISH' or 'BEARISH'
        timeframe: str
    ) -> object:
        """
        🔥 DUAL-SPECIALIST ARCHITECTURE dengan Parameter Asimetris
        """
        logger.info(f"🔥 Melatih {specialty} Specialist Model untuk {timeframe}...")
        
        if specialty == 'BULLISH':
            y_specialist = (y == 1).astype(int)
            target_class_name = 'BUY'
            # 🔥 PARAMETER KHUSUS UNTUK BULLISH MODEL (lebih kompleks, lebih regularisasi)
            params = {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.04,
                'num_leaves': 35,
                'colsample_bytree': 0.7,
                'subsample': 0.7,
                'reg_alpha': 0.2,
                'reg_lambda': 0.2
            }
        else:  # BEARISH
            y_specialist = (y == 0).astype(int)
            target_class_name = 'SELL'
            # 🔥 PARAMETER STANDAR (SUDAH TERBUKTI BAGUS) UNTUK BEARISH MODEL
            params = {
                'n_estimators': 250,
                'max_depth': 5,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            }
        
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X, y_specialist, weights, test_size=0.2, random_state=42, stratify=y_specialist
        )
        
        logger.info(f"🔄 Menerapkan SMOTETomek untuk {specialty} model...")
        smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42, n_jobs=-1)
        X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
        
        model = lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=-1,
            **params
        )
        model.fit(X_res, y_res)
        
        y_pred = model.predict(X_test)
        logger.info(f"\n--- LAPORAN KINERJA FINAL: {specialty} Specialist ({timeframe}) ---")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=['NON-' + target_class_name, target_class_name]))
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        logger.info(f" M E T R I K   K U N C I   U N T U K  {target_class_name.upper()} ".center(60, '='))
        logger.info(f" Precision: {precision:.2%}")
        logger.info(f" Recall:    {recall:.2%}")
        logger.info(f" F1-Score:  {f1:.2%}")
        logger.info("=" * 60)
        
        return model
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'lightgbm',
        timeframe: str = 'M5'
    ) -> Tuple[object, StandardScaler, List[str]]:
        """
        🔥 ARSITEKTUR HIBRIDA + 3 STRATEGI LANJUTAN:
        1. Dynamic Sample Weighting
        2. Multi-Timeframe Context (sudah di fitur)
        3. Probabilistic Meta-Model
        
        Returns:
            (trained_model_tuple, scaler, feature_names)
            trained_model_tuple = (primary_model, meta_model, risk_features)
        """
        # 🔥 STRATEGI 1: Pisahkan weights dari features
        if 'weights' in X.columns:
            weights = X['weights'].copy()
            X = X.drop(columns=['weights'])
            feature_names = X.columns.tolist()
        else:
            weights = pd.Series(1.0, index=X.index)
        
        # 1. PEMBAGIAN DATA (termasuk weights)
        X_train_pri, X_test_meta, y_train_pri, y_test_meta, w_train_pri, w_test_meta = train_test_split(
            X, y, weights, test_size=0.4, random_state=42, stratify=y
        )
        X_train_meta, X_test_final, y_train_meta, y_test_final, w_train_meta, w_test_final = train_test_split(
            X_test_meta, y_test_meta, w_test_meta, test_size=0.5, random_state=42, stratify=y_test_meta
        )
        
        logger.info(f"Data Split: Primary Train={len(X_train_pri)}, Meta Train={len(X_train_meta)}, Final Test={len(X_test_final)}")
        
        scaler = StandardScaler()
        X_train_pri_scaled = scaler.fit_transform(X_train_pri)
        
        # =================================================================
        # TAHAP 1: Latih Primary Model (Fokus Recall dengan SMOTETomek)
        # =================================================================
        logger.info("🔥 [Tahap 1] Melatih Primary Model (Signal Generator)...")
        
        X_train_resampled, y_train_resampled, w_train_resampled = X_train_pri_scaled, y_train_pri.values, w_train_pri.values
        if self.use_smote:
            logger.info("🔄 Menerapkan SMOTETomek untuk menyeimbangkan data training Primary...")
            smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
            try:
                X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_pri_scaled, y_train_pri)
                logger.info(f"Ukuran data setelah SMOTETomek: {X_train_resampled.shape}")
                # 🔥 Sesuaikan weights untuk data resampled
                w_train_resampled = np.ones(len(y_train_resampled))
            except Exception as e:
                logger.error(f"❌ SMOTETomek gagal: {e}. Melanjutkan tanpa resampling.")
        
        primary_model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            colsample_bytree=0.7,
            subsample=0.7,
            class_weight='balanced',
            verbose=-1
        )
        # 🔥 STRATEGI 1: Gunakan sample_weight
        primary_model.fit(X_train_resampled, y_train_resampled, sample_weight=w_train_resampled)
        
        # Evaluasi Primary Model pada data OOS (Out-of-Sample)
        X_train_meta_scaled = scaler.transform(X_train_meta)
        y_pred_pri_on_meta = primary_model.predict(X_train_meta_scaled)
        logger.info("\n--- Laporan Kinerja Primary Model (pada Data OOS) ---")
        logger.info("\n" + classification_report(y_train_meta, y_pred_pri_on_meta, target_names=['SELL', 'BUY']))
        
        # =================================================================
        # TAHAP 2: Latih Meta Model (Risk Confirmation Filter)
        # =================================================================
        logger.info("🔥 [Tahap 2] Membangun set fitur & melatih Meta Model (Risk Filter)...")
        
        # Dapatkan probabilitas dari Primary Model
        primary_probas_on_meta = primary_model.predict_proba(X_train_meta_scaled)[:, 1]
        
        # Buat DataFrame fitur untuk Meta Model
        meta_features_df = pd.DataFrame(index=X_train_meta.index)
        meta_features_df['primary_proba_buy'] = primary_probas_on_meta
        
        # Gabungkan dengan fitur-fitur sadar-risiko dari data asli
        risk_features = ['adx', 'atr_pct', 'regime_trending', 'regime_ranging', 'regime_high_vol']
        for feat in risk_features:
            if feat in X_train_meta.columns:
                meta_features_df[feat] = X_train_meta[feat].values
        
        # Target untuk Meta Model: Apakah Primary Model benar?
        meta_target = (y_pred_pri_on_meta == y_train_meta.values).astype(int)
        
        logger.info(f"Distribusi Target Meta: Sinyal Benar={meta_target.sum()}, Sinyal Salah={len(meta_target) - meta_target.sum()}")
        
        meta_model = lgb.LGBMClassifier(
            n_estimators=75,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=8,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            reg_alpha=0.5,
            reg_lambda=0.5,
            verbose=-1
        )
        meta_model.fit(meta_features_df, meta_target)
        
        # =================================================================
        # TAHAP 3: Evaluasi Arsitektur Hibrida pada Final Test Set
        # =================================================================
        logger.info("\n--- Laporan Akhir (Arsitektur Hibrida pada Final Test Set) ---")
        
        # 1. Dapatkan prediksi Primary Model
        X_test_final_scaled = scaler.transform(X_test_final)
        final_pred_pri = primary_model.predict(X_test_final_scaled)
        final_probas_pri = primary_model.predict_proba(X_test_final_scaled)[:, 1]
        
        # 2. Bangun fitur untuk Meta Model pada test set
        final_meta_features_df = pd.DataFrame(index=X_test_final.index)
        final_meta_features_df['primary_proba_buy'] = final_probas_pri
        for feat in risk_features:
            if feat in X_test_final.columns:
                final_meta_features_df[feat] = X_test_final[feat].values
        
        # 3. 🔥 STRATEGI 3: Dapatkan PREDIKSI PROBABILITAS dari Meta Model
        meta_probas = meta_model.predict_proba(final_meta_features_df)[:, 1]
        
        # 4. 🔥 Terapkan Threshold Keyakinan yang Tinggi
        CONFIDENCE_THRESHOLD = 0.65  # Hanya percaya jika Meta-Model >65% yakin
        meta_verdict = (meta_probas > CONFIDENCE_THRESHOLD).astype(int)
        logger.info(f"   Confidence threshold: {CONFIDENCE_THRESHOLD:.0%} (signals approved: {meta_verdict.sum()}/{len(meta_verdict)})")
        
        # 5. Buat keputusan akhir: sinyal = sinyal primary HANYA JIKA meta_verdict = 1 (percaya)
        final_pred_hybrid = np.where(meta_verdict == 1, final_pred_pri, -1)  # -1 untuk HOLD/FILTERED
        # Kita hanya peduli pada sinyal yang diloloskan
        final_pred_hybrid[final_pred_pri == 0] = 0  # Loloskan semua sinyal SELL
        
        # Hapus sinyal yang difilter untuk evaluasi
        eval_indices = np.where(final_pred_hybrid != -1)[0]
        
        logger.info(f"Total sinyal setelah filter Meta: {len(eval_indices)} dari {len(X_test_final)}")
        logger.info("\n🔥 Laporan Klasifikasi Setelah Filter Meta Model:")
        logger.info("\n" + classification_report(y_test_final.iloc[eval_indices], final_pred_hybrid[eval_indices], target_names=['SELL', 'BUY'], zero_division=0))
        logger.info("\n🔥 Konfusi Matriks Setelah Filter:")
        logger.info(str(confusion_matrix(y_test_final.iloc[eval_indices], final_pred_hybrid[eval_indices])))
        
        # Calculate final metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_final.iloc[eval_indices], final_pred_hybrid[eval_indices], labels=[0, 1], zero_division=0
        )
        
        logger.info("\n📊 Hybrid Architecture Performance (FINAL SIGNALS):")
        logger.info("="*60)
        class_names = ['SELL', 'BUY']
        for i, class_name in enumerate(class_names):
            recall_emoji = "✅" if recall[i] >= 0.55 else ("⚠️" if recall[i] >= 0.50 else "❌")
            prec_emoji = "✅" if precision[i] >= 0.55 else ("⚠️" if precision[i] >= 0.50 else "❌")
            logger.info(f"  {class_name:6} | Prec: {precision[i]:>6.2%} {prec_emoji} | Recall: {recall[i]:>6.2%} {recall_emoji} | F1: {f1[i]:>6.2%}")
        logger.info("="*60)
        
        macro_f1 = f1.mean()
        logger.info(f"\n🎯 Overall Macro F1-Score: {macro_f1:.2%} (TARGET: > 60%)")
        
        if precision[1] >= 0.65:
            logger.info(f"✅ EXCELLENT! BUY Precision {precision[1]:.2%} >= 65% target!")
        
        final_model = (primary_model, meta_model, risk_features)
        
        return final_model, scaler, feature_names
    
    
    def train_all_timeframes(
        self,
        mt5_handler,
        symbol: str = 'XAUUSDm',
        model_type: str = 'lightgbm'
    ):
        """
        🔥 DUAL-SPECIALIST ARCHITECTURE
        Train two independent specialist models for each timeframe
        """
        logger.info(f"{'='*60}")
        logger.info(f"🚀 Starting DUAL-SPECIALIST Training for {symbol}")
        logger.info(f"{'='*60}\n")
        
        if self.llm:
            logger.info("✅ LLM Analysis: ENABLED (GPU)")
        else:
            logger.info("⚠️  LLM Analysis: DISABLED")
        
        logger.info("✅ Architecture: DUAL-SPECIALIST (Bullish + Bearish)")
        logger.info(f"✅ Model Engine: {model_type.upper()}")
        
        # Data config
        candles_config = {
            'M5': 100000,
            'M15': 100000,
            'H1': 60000,
            'H4': 15000
        }
        
        for timeframe in self.timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {timeframe} Model (Dual-Specialist Architecture)")
            logger.info(f"{'='*60}\n")
            
            try:
                # Collect data
                candles_needed = candles_config[timeframe]
                logger.info(f"Collecting {candles_needed:,} candles for {timeframe} data...")
                
                df = self.collect_training_data(
                    mt5_handler,
                    symbol,
                    timeframe,
                    candles=candles_needed
                )
                
                # Prepare features
                X, y = self.prepare_features(df, timeframe)
                
                # Pisahkan weights
                weights = X.pop('weights') if 'weights' in X.columns else pd.Series(1.0, index=X.index)
                
                # 🔥🔥 STRATEGI ASYMMETRIC FEATURE SET 🔥🔥
                # Fitur-fitur khusus bullish yang akan ditambahkan (yang belum ada di base)
                bullish_special = [
                    'bull_squeeze', 'bull_structure_hl', 'bull_pullback_depth_pct',
                    'bull_momentum_acceleration', 'bull_reversal_confirmation', 'bullish_volume_spike',
                    'bull_hl_low_chop'
                ]
                
                # Fitur-fitur khusus bearish
                bearish_special = ['bear_expansion', 'bear_structure_lh']
                
                # Base features (sudah include chop_index, ema_9_slope, atr_std_14, vol_of_vol dari preprocessor)
                base_features = [col for col in X.columns if col not in bullish_special + bearish_special]
                
                # Build final feature sets (NO DUPLICATES)
                bullish_features = base_features + [f for f in bullish_special if f in X.columns]
                bearish_features = base_features + [f for f in bearish_special if f in X.columns]
                
                bull_alpha_count = len([f for f in bullish_features if f.startswith('bull_') or f in ['chop_index', 'ema_9_slope', 'bull_hl_low_chop', 'vol_of_vol', 'atr_std_14']])
                logger.info(f"   Bullish Specialist: {len(bullish_features)} fitur (termasuk {bull_alpha_count} fitur alpha/kesehatan)")
                logger.info(f"   Bearish Specialist: {len(bearish_features)} fitur")
                
                # Standarisasi data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                
                # 🔥 Latih dua model spesialis dengan set fitur masing-masing
                logger.info(f"\n🎯 Training TWO independent specialists for {timeframe}...")
                bullish_model = self.train_specialist_model(X_scaled[bullish_features], y, weights, 'BULLISH', timeframe)
                bearish_model = self.train_specialist_model(X_scaled[bearish_features], y, weights, 'BEARISH', timeframe)
                
                # Save feature names untuk masing-masing model
                feature_names = {'bullish': bullish_features, 'bearish': bearish_features}
                
                # Simpan model gabungan
                final_model = {
                    'bullish': bullish_model,
                    'bearish': bearish_model
                }
                
                tf_key = timeframe.lower()
                model_path = self.output_dir / f"cls_{tf_key}.pkl"
                scaler_path = self.output_dir / f"scaler_{tf_key}.pkl"
                features_path = self.output_dir / f"features_{tf_key}.json"
                
                joblib.dump(final_model, model_path)
                joblib.dump(scaler, scaler_path)
                
                import json
                with open(features_path, 'w') as f:
                    json.dump(feature_names, f, indent=2)
                
                logger.info(f"✅ Saved Dual-Specialist models to {model_path}")
                logger.info(f"✅ Saved scaler to {scaler_path}")
                logger.info(f"✅ Saved {len(feature_names)} features to {features_path}")
                
                self.models[tf_key] = final_model
                self.scalers[tf_key] = scaler
                
            except Exception as e:
                logger.error(f"❌ Error training {timeframe} model: {str(e)}", exc_info=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 TRAINING COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Trained models: {list(self.models.keys())}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"\n📊 NEXT STEPS:")
        logger.info(f"1. Review per-class recall and F1-scores above.")
        logger.info(f"2. Ensure Macro F1 is above 58% for reliable performance.")
        logger.info(f"3. If recall is still low, consider adjusting ATR multipliers in `collect_training_data`.")
        logger.info(f"4. Test models using `cls_predictor.py` before live deployment.")
        logger.info(f"{'='*60}")
    
    def retrain_single_timeframe(
        self,
        mt5_handler,
        timeframe: str,
        symbol: str = 'XAUUSDm',
        model_type: str = 'random_forest'
    ):
        """Retrain a single timeframe model"""
        logger.info(f"Retraining {timeframe} model for {symbol}...")
        
        # Collect data
        df = self.collect_training_data(mt5_handler, symbol, timeframe, candles=5000)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Train model
        model, scaler = self.train_model(X, y, model_type)
        
        # Save
        tf_key = timeframe.lower()
        model_path = self.output_dir / f"cls_{tf_key}.pkl"
        scaler_path = self.output_dir / f"scaler_{tf_key}.pkl"
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"✅ Model retrained and saved")


if __name__ == "__main__":
    # Train CLS models
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║           CLS MODEL TRAINER                               ║
║  Train classifier models for trade direction prediction   ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    from core.mt5_handler import MT5Handler
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Connect to MT5
    login = int(os.getenv('MT5_LOGIN_DEMO', '269764525'))
    password = os.getenv('MT5_PASSWORD_DEMO', 'Wokolcoy@20')
    server = os.getenv('MT5_SERVER_DEMO', 'Exness-MT5Trial17')
    
    mt5 = MT5Handler(login, password, server)
    
    if not mt5.initialize():
        print("❌ Failed to connect to MT5")
        exit(1)
    
    # Create trainer
    print("\n🔧 Initializing DUAL-SPECIALIST trainer...")
    print("\n🚀 Starting DUAL-SPECIALIST ARCHITECTURE training...")
    print("⏱️  Training time: 30-40 minutes (Full pipeline)...")
    print("📊 Collecting up to 100K candles per timeframe.")
    print("\n🎯 PARADIGM SHIFT: DUAL-SPECIALIST ARCHITECTURE")
    print("   🔥 STRATEGY #1: Independent Bullish & Bearish Specialists")
    print("      → Each model trained ONLY for its specialty")
    print("      → Bullish Specialist: BUY vs NON-BUY")
    print("      → Bearish Specialist: SELL vs NON-SELL")
    print("      → NO MORE internal class conflict!")
    print("   🔥 STRATEGY #2: Asymmetric Feature Engineering")
    print("      → Bull-specific: Volatility Squeeze, Higher Lows")
    print("      → Bear-specific: Volatility Expansion, Lower Highs")
    print("   🔥 STRATEGY #3: Multi-Timeframe Context")
    print("      → M5 gets H1 context, M15 gets H4 context")
    print("      → HTF trend confirmation built into features")
    print("   🔥 STRATEGY #4: Dynamic Sample Weighting")
    print("      → Fast winners get higher weight")
    print("      → Model learns from quality, not just quantity")
    print("\n🎯 TARGET HASIL:")
    print("   📈 PARITAS METRIK: Bull == Bear performance")
    print("   📈 Bullish Specialist: Prec/Recall/F1 ~75-80%")
    print("   📈 Bearish Specialist: Prec/Recall/F1 ~75-80%")
    print("   📈 Overall: Two EQUAL experts, no more bias!\n")
    
    trainer = CLSModelTrainer(
        output_dir="./models/saved_models",
        llm_model_path="./models/Llama-3.2-3B-Instruct-BF16.gguf",
        use_gpu=True,
        use_smote=True,     # 🔥 AKTIFKAN: SMOTETomek untuk Primary Model
        use_optuna=False,   # 🔥 DISABLED: causes overfitting
        use_ensemble=True,  # 🔥 Enable DUAL ensemble (LightGBM + XGBoost)
        use_lstm=False      # 🔥 DISABLED: LSTM was failing & not helping
    )
    
    # Train all timeframes
    trainer.train_all_timeframes(
        mt5_handler=mt5,
        symbol='XAUUSDm',
        model_type='lightgbm'  # Options: 'lightgbm', 'xgboost', 'random_forest', 'gradient_boosting'
    )
    
    mt5.shutdown()
    
    print("\n✅ Training complete! Models saved to ./models/saved_models/")
    print("\nYou can now use these models with CLSPredictor for live trading.")
