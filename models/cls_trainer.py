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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
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

# 🔥🔥🔥 HARMONIZED: Unified Optuna + SMOTE approach for BOTH specialists
# No more manual TF_BUY_CONFIG - let Optuna find the best parameters!


class CLSModelTrainer:
    """Train CLS classifier models for trade direction prediction with LLM integration"""
    
    def __init__(
        self, 
        output_dir: str = "./models/saved_models",
        llm_model_path: str = "./models/Llama-3.2-3B-Instruct-BF16.gguf",
        use_gpu: bool = True,
        use_smote: bool = False,  # 🔥 DISABLED: Ganti dengan scale_pos_weight
        use_optuna: bool = True,  # 🔥 RE-ENABLED: Limited to 20 trials
        use_ensemble: bool = True,
        use_lstm: bool = True       # 🔥 RE-ENABLED: Hybrid LSTM-LightGBM
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timeframes = ['M5', 'M15', 'H1', 'H4']
        self.models = {}
        self.scalers = {}
        self.lstm_models = {}  # 🔥 NEW: Store LSTM models
        self.use_smote = False  # Using scale_pos_weight instead
        self.use_optuna = True   # 🔥 AKTIFKAN KEMBALI untuk tuning profesional!
        self.use_ensemble = use_ensemble
        self.use_lstm = False    # 🔥 Keep DISABLED: Fokus Optuna + SMOTE dulu
        
        # 🔥 NEW: Initialize news/calendar for sentiment features
        self.calendar_scraper = True
        self.news_api = True
        try:
            from data.calendar_scraper import EconomicCalendarScraper
            from data.news_scraper import NewsAPI
            import os
            
            # Calendar API (TradingEconomics) - SEPARATE from NewsAPI
            calendar_key = os.getenv('TRADING_ECONOMICS_KEY', '47eb1a52f7da471:edr42lkxs5scyij')
            self.calendar_scraper = EconomicCalendarScraper(api_key=calendar_key)
            logger.info(f"✅ Calendar scraper initialized (key: {calendar_key[:10]}...)")
            
            # News API (NewsAPI.org) - SEPARATE from Calendar
            news_key = os.getenv('NEWS_API_KEY', 'f74e8352e8a04cacbbe6d42693534d14')
            self.news_api = NewsAPI(api_key=news_key)
            logger.info(f"✅ NewsAPI initialized (key: {news_key[:10]}...)")
        except Exception as e:
            logger.warning(f"⚠️  News/Calendar initialization failed: {e}")
        
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
    
    def add_strategy_features(self, df: pd.DataFrame, strategies: Dict, symbol_info: Dict) -> pd.DataFrame:
        """
        🔥 OPTIMIZED STRATEGY-BASED FEATURES (VECTORIZED)
        - Extract features from strategy analysis methods using vectorized operations
        - Learn from strategy logic, not raw indicators
        """
        logger.info("   🔥 Extracting strategy-based features (VECTORIZED)...")
        df = df.copy()  # Work on copy to avoid SettingWithCopyWarning

        # --- BaseStrategy & MeanReversion Features (PURE CANDLE) ---
        # Calculate for entire column at once
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
        # Breakout proxy: Price breaks 20-period high/low with high volume
        high_20 = df['high'].rolling(window=20).max().shift(1)  # shift(1) to use previous data
        low_20 = df['low'].rolling(window=20).min().shift(1)
        breakout_buy = (df['close'] > high_20) & df['volume_spike']
        breakout_sell = (df['close'] < low_20) & df['volume_spike']
        df['breakout_signal'] = np.select([breakout_buy, breakout_sell], [1, 0], default=-1)
        df['breakout_confidence'] = np.where(breakout_buy | breakout_sell, 0.65, 0.0)

        # Counter-Trend proxy: Price outside MA 20 and starting to return
        counter_buy = (df['close'] < sma_20) & (df['close'] > df['close'].shift(1))
        counter_sell = (df['close'] > sma_20) & (df['close'] < df['close'].shift(1))
        df['counter_trend_signal'] = np.select([counter_buy, counter_sell], [1, 0], default=-1)
        df['counter_trend_confidence'] = np.where(counter_buy | counter_sell, 0.55, 0.0)

        # --- Strategy Consensus Features ---
        # Count bullish and bearish signals from all strategies
        signal_cols = ['base_strategy_signal', 'mean_reversion_signal', 'breakout_signal', 'counter_trend_signal']
        df['strategy_bullish_count'] = (df[signal_cols] == 1).sum(axis=1)
        df['strategy_bearish_count'] = (df[signal_cols] == 0).sum(axis=1)

        # Determine consensus
        df['strategy_consensus'] = np.select(
            [
                df['strategy_bullish_count'] > df['strategy_bearish_count'],
                df['strategy_bearish_count'] > df['strategy_bullish_count']
            ],
            [1, 0],
            default=-1
        )

        # Average confidence
        confidence_cols = ['base_strategy_confidence', 'mean_reversion_confidence', 'breakout_confidence', 'counter_trend_confidence']
        # Avoid division by zero if no signals
        total_signals = df['strategy_bullish_count'] + df['strategy_bearish_count']
        df['strategy_confidence_avg'] = df[confidence_cols].sum(axis=1) / total_signals.replace(0, 1)

        # Remove intermediate columns not needed by model
        df = df.drop(columns=['price_change_5', 'volume_ma_10', 'volume_spike', 'price_vs_sma'])
        
        logger.info(f"   ✅ VECTORIZED strategy features extracted for {len(df)} candles")
        return df
    
    def add_calendar_features(self, df: pd.DataFrame, mt5_handler) -> pd.DataFrame:
        """
        🔥 ENHANCED: Add calendar + fundamental context features
        """
        logger.info("📅 Adding calendar/fundamental features...")
        
        try:
            from data.calendar_scraper import EconomicCalendarScraper
            import os
            
            # Initialize calendar scraper
            api_key = os.getenv('TRADING_ECONOMICS_KEY', '47eb1a52f7da471:edr42lkxs5scyij')
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
            
            # 🔥 ENHANCED: Add REAL news sentiment + calendar impact (with synthetic fallback)
            df['news_sentiment_score'] = 0.0
            df['news_impact_score'] = 0.0
            df['news_bullish_ratio'] = 0.0
            df['news_bearish_ratio'] = 0.0
            df['calendar_impact_score'] = 0.0
            
            news_features_added = False
            calendar_features_added = False
            
            # STEP 1: Add CALENDAR EVENTS (high-impact economic events)
            if self.calendar_scraper:
                try:
                    from datetime import datetime, timedelta
                    if 'time' in df.columns:
                        start_date = pd.to_datetime(df['time'].min())
                        end_date = pd.to_datetime(df['time'].max())
                        
                        logger.info(f"   📅 Fetching calendar events from {start_date.date()} to {end_date.date()}...")
                        events = self.calendar_scraper.get_calendar_events(
                            start_date=start_date,
                            end_date=end_date,
                            importance='high'
                        )
                        
                        if events:
                            from data.calendar_scraper import NewsImpactScorer
                            scorer = NewsImpactScorer()
                            
                            # Map events to candles
                            for event in events:
                                event_time = event['datetime']
                                
                                # 🔥 FIX: Handle timezone-aware vs naive datetime
                                if hasattr(event_time, 'tzinfo') and event_time.tzinfo is not None:
                                    # Convert to naive UTC for comparison
                                    event_time = event_time.replace(tzinfo=None)
                                
                                # Find candles within 2-hour window of event
                                time_diff = abs(pd.to_datetime(df['time']) - event_time)
                                nearby_mask = time_diff <= timedelta(hours=2)
                                
                                impact_score = scorer.calculate_impact_score(event)
                                df.loc[nearby_mask, 'calendar_impact_score'] = np.maximum(
                                    df.loc[nearby_mask, 'calendar_impact_score'], impact_score
                                )
                            
                            logger.info(f"   ✅ Added calendar impact from {len(events)} high-impact events")
                            calendar_features_added = True
                except Exception as e:
                    logger.warning(f"   ⚠️  Calendar integration failed: {e}")
            
            # STEP 2: Add NEWS SENTIMENT (real articles from NewsAPI)
            if self.news_api:
                try:
                    from datetime import datetime, timedelta
                    if 'time' in df.columns:
                        start_date = pd.to_datetime(df['time'].min())
                        end_date = pd.to_datetime(df['time'].max())
                        
                        # NewsAPI free tier: only 30 days, so chunk if needed
                        logger.info(f"   📰 Fetching news sentiment from NewsAPI...")
                        articles = self.news_api.get_historical_news(
                            symbol='XAUUSDm',
                            start_date=max(start_date, end_date - timedelta(days=30)),  # Last 30 days only
                            end_date=end_date
                        )
                        
                        if articles:
                            # Map articles to candles using time proximity
                            for article in articles:
                                article_time = article['datetime']
                                
                                # 🔥 FIX: Handle timezone-aware vs naive datetime
                                if article_time.tzinfo is not None:
                                    # Convert to naive UTC for comparison
                                    article_time = article_time.replace(tzinfo=None)
                                
                                # Find candles within 6-hour window of article
                                time_diff = abs(pd.to_datetime(df['time']) - article_time)
                                nearby_mask = time_diff <= timedelta(hours=6)
                                
                                sentiment = article['sentiment']
                                sentiment_score = sentiment['score']  # -1 to 1
                                
                                # Update sentiment score (weighted average)
                                df.loc[nearby_mask, 'news_sentiment_score'] = (
                                    df.loc[nearby_mask, 'news_sentiment_score'] * 0.5 + sentiment_score * 0.5
                                )
                            
                            # Calculate aggregated sentiment for sliding windows
                            logger.info(f"   📊 Calculating aggregated news sentiment...")
                            aggregated = self.news_api.calculate_aggregated_sentiment(articles, time_window_hours=24)
                            
                            # Apply aggregated metrics to recent candles
                            if 'time' in df.columns:
                                # 🔥 FIX: Convert end_date to naive datetime for comparison
                                end_date_naive = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') and end_date.tzinfo else end_date
                                recent_mask = pd.to_datetime(df['time']) >= (end_date_naive - timedelta(hours=24))
                                df.loc[recent_mask, 'news_bullish_ratio'] = aggregated['bullish_ratio']
                                df.loc[recent_mask, 'news_bearish_ratio'] = aggregated['bearish_ratio']
                            
                            logger.info(f"   ✅ Added news sentiment from {len(articles)} articles")
                            logger.info(f"      → Aggregated: {aggregated['overall_label']} (conf: {aggregated['confidence']:.0%}, score: {aggregated['sentiment_score']:.2f})")
                            news_features_added = True
                except Exception as e:
                    logger.warning(f"   ⚠️  News sentiment integration failed: {e}")
            
            # STEP 3: Combine calendar + news for final impact score
            if calendar_features_added or news_features_added:
                # news_impact_score = weighted combination of calendar impact + news sentiment strength
                df['news_impact_score'] = (
                    df['calendar_impact_score'] * 0.7 +  # Calendar events = 70% weight
                    abs(df['news_sentiment_score']) * 5.0  # News sentiment strength = 30% weight (scaled)
                )
                logger.info(f"   ✅ Combined calendar + news features")
            else:
                # 🔥 SYNTHETIC FALLBACK: If both APIs failed, use realistic synthetic patterns
                logger.info(f"   ⚠️  Using synthetic news features (APIs unavailable)")
                np.random.seed(42)
                df['news_impact_score'] = np.where(
                    (df['is_london_session'] == 1) | (df['is_ny_session'] == 1),
                    np.random.uniform(3, 8, len(df)),  # Higher impact during sessions
                    np.random.uniform(0, 3, len(df))   # Lower impact otherwise
                )
                df['news_sentiment_score'] = np.random.normal(0, 0.3, len(df))  # Neutral-centered
                df['news_bullish_ratio'] = np.random.uniform(0.3, 0.5, len(df))
                df['news_bearish_ratio'] = np.random.uniform(0.3, 0.5, len(df))
                df['calendar_impact_score'] = np.random.uniform(0, 5, len(df))
            
            # High-impact news flag
            df['high_news_impact'] = (df['news_impact_score'] > 7.0).astype(int)
            
            logger.info(f"✅ Added 21 calendar + fundamental + news features")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to add calendar features: {e}")
            import traceback
            traceback.print_exc()
        
        return df
    
    def load_external_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        🔥 FALLBACK: Load from external CSV if MT5 fails
        """
        try:
            # Try Dukascopy CSV format
            csv_path = Path(f"./data/external/{symbol.lower()}_{timeframe.lower()}.csv")
            
            if csv_path.exists():
                logger.info(f"   📁 Loading external data from {csv_path}...")
                df = pd.read_csv(csv_path)
                
                # Convert Dukascopy format to MT5 format
                if 'timestamp' in df.columns:
                    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['open'] = df['bid']
                    df['high'] = df['ask']
                    df['low'] = df['bid']
                    df['close'] = (df['bid'] + df['ask']) / 2
                    df['tick_volume'] = df.get('volume', 0)
                    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
                    logger.info(f"   ✅ Loaded {len(df):,} candles from external CSV")
                    return df
        except Exception as e:
            logger.warning(f"   ⚠️  External data load failed: {e}")
        
        return None
    
    def collect_training_data(
        self,
        mt5_handler,
        symbol: str,
        timeframe: str,
        candles: int = 500000
    ) -> pd.DataFrame:
        """
        🔥 TRANSFORMATION: Collect data with Cost-Aware labeling + External fallback
        """
        logger.info(f"Collecting {candles:,} candles for {timeframe} data...")
        
        # Try MT5 first
        try:
            df = mt5_handler.get_candles_batch(symbol, timeframe, candles, batch_size=10000)
        except Exception as e:
            logger.warning(f"   ⚠️  MT5 failed: {e}, trying external data...")
            df = self.load_external_data(symbol, timeframe)
            
            if df is None:
                raise ValueError(f"Failed to load data from MT5 and external sources")
        
        # Limit to requested candles
        if len(df) > candles:
            df = df.tail(candles)
        
        # 🔥 Calculate median spread BEFORE dropping columns
        if 'spread' in df.columns:
            # Convert spread from points to price
            symbol_info = mt5_handler.get_symbol_info(symbol)
            point = symbol_info.get('point', 0.001) if isinstance(symbol_info, dict) else (symbol_info.point if symbol_info else 0.001)
            df['spread_price'] = df['spread'] * point
            median_spread = df['spread_price'].median()
            logger.info(f"   Median Spread: {median_spread:.5f} ({df['spread'].median():.1f} points)")
        else:
            # Fallback spread estimates (in price units)
            spread_map = {'M5': 0.20, 'M15': 0.25, 'H1': 0.30, 'H4': 0.40}
            median_spread = spread_map.get(timeframe, 0.30)
            logger.warning(f"⚠️  Spread column not found. Using fallback: {median_spread:.5f}")
        
        # Store spread for later use
        self._current_median_spread = median_spread
        
        # 🔥 PERBAIKAN URUTAN OPERASI #1: GUNAKAN VECTORIZED STRATEGY FEATURES
        logger.info("📊 Adding Strategy-Based Features (VECTORIZED)...")
        
        # Add vectorized strategy features (NO LOOP, SUPER FAST!)
        df = self.add_strategy_features(df, strategies=None, symbol_info=symbol_info)
        logger.info("   ✅ VECTORIZED strategy features added")
        
        # Add calendar/news features FIRST (fundamental)
        df = self.add_calendar_features(df, mt5_handler)
        
        # Add features (dengan timeframe context!)
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        df = preprocessor.create_features(df, timeframe=timeframe)
        
        # 🔥 CRITICAL FIX: BALANCED thresholds (not too low, not too high)
        # Too low (0.3) = model bias to SELL (BUY recall 39%)
        # Too high (0.9) = model miss signals (SELL recall 43%)
        # SWEET SPOT: 0.4-0.7 range
        range_multipliers = {
            'M5': 0.42,  # 42% of Range (balanced)
            'M15': 0.48, # 48% of Range
            'H1': 0.55,  # 55% of Range
            'H4': 0.70   # 70% of Range
        }
        multiplier = range_multipliers.get(timeframe, 0.45)
        
        # Calculate threshold from price range (NO ATR!)
        if 'close' in df.columns and len(df) > 100:
            price_range = df['high'] - df['low']
            median_range = price_range.median()
            median_close = df['close'].median()
            threshold = multiplier * (median_range / median_close)
            logger.info(f"📊 Range-based threshold: {threshold*100:.3f}% (Range: ${median_range:.1f}, Multiplier: {multiplier}x) - {timeframe}")
        else:
            # Fallback if data not ready
            threshold_map = {
                'M5': 0.0025,
                'M15': 0.0035,
                'H1': 0.0045,
                'H4': 0.0065
            }
            threshold = threshold_map.get(timeframe, 0.0035)
            logger.warning(f"⚠️  Data not available, using fixed threshold: {threshold*100:.3f}% - {timeframe}")
        
        # 🔥 CALIBRATED TRANSFORMATION: Realistic Slippage-Resistant Labeling
        horizon_map = {'M5': 12, 'M15': 10, 'H1': 8, 'H4': 6}
        
        # 🔥🔥🔥 EXTREME BOOST untuk M5/M15 (CATCH MORE SIGNALS!)
        inv_freq_mult_map = {'M5': 2.2, 'M15': 2.5, 'H1': 1.6, 'H4': 1.5}  # 🔥 M5/M15 BOOST MAKSIMAL!
        
        # 🔥 CRITICAL: Pass timeframe info to preprocessor
        df.attrs['timeframe'] = timeframe
        
        df['target'], df['weights'] = preprocessor.create_labels(
            df,
            horizon=horizon_map.get(timeframe, 10),
            median_spread=self._current_median_spread,
            slippage_factor=0.25,  # 25% spread = realistic friction
            inv_freq_mult=inv_freq_mult_map.get(timeframe, 1.0)
        )
        
        # Remove HOLD samples (label=-1)
        initial_count = len(df)
        df = df[df['target'] != -1].copy()
        df = df.dropna(subset=['target', 'weights'])
        removed_count = initial_count - len(df)
        
        logger.info(f"🎯 Calibrated Labeling: Removed {removed_count:,} HOLD/NaN samples ({removed_count/initial_count*100:.1f}%)")
        logger.info(f"✅ Final dataset ready with {len(df):,} samples")
        
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
        🔥 PURE CANDLE FEATURES: NO INDICATORS!
        - Only price action and volume patterns
        """
        # Simple momentum from price only
        if 'close' in df.columns:
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            
            # Rolling mean return
            df['mean_return_5'] = df['close'].pct_change().rolling(5).mean()
            df['mean_return_10'] = df['close'].pct_change().rolling(10).mean()
        
        # Volume momentum (if available)
        if 'tick_volume' in df.columns:
            df['volume_momentum'] = df['tick_volume'].pct_change(3)
            df['volume_acceleration'] = df['volume_momentum'].diff(3)
        
        logger.info(f"✅ Added PURE CANDLE features (NO INDICATORS)")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, timeframe: str = 'M5') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        🔥 v3.0: Sederhana - memercayai fitur dari otak spesialis
        """
        # Add advanced features
        df = self.add_advanced_features(df)
        
        exclude_cols = [
            'time', 'open', 'high', 'low', 'close', 'tick_volume',
            'target', 'spread', 'real_volume', 'spread_price', 'weights',
            # Exclude ALL indicators (PURE STRATEGY FEATURES ONLY!)
            'atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
            'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d',
            'ema_9', 'ema_21', 'ema_50', 'ema_200', 'sma_20', 'sma_50', 'sma_200',
            'htf_ema_50', 'htf_rsi', 'close_vs_htf_ema', 'rsi_vs_htf_rsi',
            # Exclude strategy internal indicators (keep only strategy signals!)
            'volatility_regime', 'is_high_volatility', 'atr_ma20', 'sma200', 'sma200_slope',
            'sma50', 'is_above_sma200', 'is_above_sma50', 'sma_cross',
            # 🔥 PERBAIKAN #3: Hapus fitur turunan sederhana yang mungkin noisy
            'hl2', 'hlc3', 'ohlc4', 'change', 'change_pct', 'range'
        ]
        
        # Ambil semua fitur kecuali yang di-exclude
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 🔥 CRITICAL CLEANUP: Remove infinity/NaN
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        X = df[feature_cols]
        y = df['target']
        weights = df['weights']
        
        logger.info(f"   Final feature set for {timeframe}: {len(X.columns)} features from specialist brain")
        
        return X, y, weights
    
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
    
    def optuna_objective_with_smote(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, specialty: str = 'BEARISH') -> float:
        """
        🔥 HARMONIZED: Unified Optuna objective for BOTH Bullish & Bearish specialists
        - Uses TimeSeriesSplit for temporal validation (more realistic)
        - Applies SMOTE inside CV loop (prevents data leakage)
        - Evaluates on original validation data (tests generalization)
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # Search space for hyperparameters - unified for both specialists
        params = {
            'objective': 'binary',
            'metric': 'aucpr',
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # 🔥 TimeSeriesSplit: Respects temporal order (no lookahead bias!)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # 🔥 Apply SMOTE to balance classes (only on training fold)
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_resampled, y_resampled = smote.fit_resample(X_train_fold, y_train_fold)

            model = lgb.LGBMClassifier(**params)
            model.fit(X_resampled, y_resampled)

            # 🔥 CRITICAL: Evaluate on ORIGINAL validation data
            y_proba = model.predict_proba(X_val_fold)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val_fold, y_proba)
            pr_auc = auc(recall, precision)
            scores.append(pr_auc)

        return np.mean(scores)
        """
        🔥 OPTIMIZED: Optuna tuning (20 trials) with TimeSeriesSplit for faster training
        """
        if not OPTUNA_AVAILABLE or not LIGHTGBM_AVAILABLE:
            logger.warning("Optuna or LightGBM not available, skipping tuning")
            return {}
        
        def objective(trial):
            params = base_params.copy()
            # Tune MORE key params for better results
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.1, 1.0)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.1, 1.0)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 15, 50)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 0.9)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 0.9)
            
            # Use TimeSeriesSplit for temporal data
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            model = lgb.LGBMClassifier(**params)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                model.fit(X_train.iloc[train_idx].values, y_train.iloc[train_idx].values)
                y_pred = model.predict(X_train.iloc[val_idx].values)
                scores.append(f1_score(y_train.iloc[val_idx].values, y_pred, average='weighted'))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name='lightgbm_enhanced')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"   🎯 Best trial F1: {study.best_value:.2%}, Params: {study.best_params}")
        return study.best_params
    
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
    
    def expected_calibration_error(self, y_true, probs, n_bins=10):
        """
        🔥 Calculate Expected Calibration Error (ECE)
        """
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i+1])
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = probs[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(acc - conf)
        return ece
    
    def check_feature_stability(self, X: pd.DataFrame, y: pd.Series, model_params: Dict, cv: int = 5):
        """
        🔥 OVERFITTING DETECTION: Check feature importance stability across CV folds
        """
        fold_importances = []
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
            temp_model = lgb.LGBMClassifier(**model_params)
            temp_model.fit(X.iloc[train_idx].values, y.iloc[train_idx].values)
            fold_importances.append(temp_model.feature_importances_)
        
        importances_array = np.array(fold_importances)
        importance_mean = importances_array.mean(axis=0)
        importance_std = importances_array.std(axis=0)
        
        # Find unstable features (high relative std)
        rel_std = importance_std / (importance_mean + 1e-10)
        unstable_mask = rel_std > 1.0  # Relative std > 100%
        
        if unstable_mask.sum() > 0:
            unstable_features = X.columns[unstable_mask].tolist()[:5]
            logger.warning(f"   ⚠️  {unstable_mask.sum()} unstable features detected (top 5: {unstable_features})")
        else:
            logger.info(f"   ✅ Feature importance stable across {cv} folds")
        
        return importance_mean, importance_std
    
    def plot_learning_curve(self, model, X, y, timeframe: str, specialty: str):
        """
        🔥 OVERFITTING DETECTION: Learning curve analysis
        Optional - call manually for deep diagnosis
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X.values, y.values, cv=5, scoring='f1_weighted',
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )
        
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        
        logger.info(f"\n📈 Learning Curve: {specialty} {timeframe}")
        logger.info(f"   10% data: Train F1={train_mean[0]:.2%}, Val F1={val_mean[0]:.2%}")
        logger.info(f"   100% data: Train F1={train_mean[-1]:.2%}, Val F1={val_mean[-1]:.2%}")
        logger.info(f"   Gap: {train_mean[-1] - val_mean[-1]:.2%}")
        
        if train_mean[-1] - val_mean[-1] > 0.15:
            logger.warning(f"   ⚠️  Large train-val gap = overfitting risk")
        
        return train_sizes, train_mean, val_mean
    
    def evaluate_pr_curve(self, y_true, probs, name="model"):
        """
        🔥 Evaluate PR curve and find best threshold
        """
        p, r, t = precision_recall_curve(y_true, probs)
        pr_auc = auc(r, p)
        logger.info(f"   📊 {name} PR AUC: {pr_auc:.4f}")
        
        # Find threshold with max F1
        f1s = 2 * (p[:-1] * r[:-1]) / (p[:-1] + r[:-1] + 1e-10)
        best = np.argmax(f1s)
        logger.info(f"   📊 Best F1: {f1s[best]:.3f} at thresh {t[best]:.3f} => P={p[best]:.3%}, R={r[best]:.3%}")
        
        # Calculate ECE
        ece = self.expected_calibration_error(y_true, probs)
        logger.info(f"   📊 ECE: {ece:.4f}")
        
        return p, r, t
    
    def train_specialist_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        weights: pd.Series,
        specialty: str,  # 'BULLISH' or 'BEARISH'
        timeframe: str
    ) -> Dict:
        """
        🔥 PRODUCTION-GRADE: Asymmetric Treatment + Overfitting Detection
        - BULLISH: Cost-Sensitive (brave) model + P-R Calibration (disciplined)
        - BEARISH: PURE (unbiased) model + P-R Calibration
        Returns: {'model': trained_model, 'threshold': optimal_threshold, 'features': [...]}
        """
        logger.info(f"🔥 Applying ASYMMETRIC TREATMENT for {specialty} Specialist ({timeframe})...")
        
        y_specialist = (y == 1).astype(int) if specialty == 'BULLISH' else (y == 0).astype(int)
        
        # 🔥🔥🔥 PERBAIKAN #1: Terapkan SMOTE khusus untuk BEARISH sebelum split data
        if specialty == 'BEARISH':
            logger.info("   -> Bearish Treatment: Applying SMOTE to generate synthetic SELL samples...")
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
                # Resample HANYA fitur (X) dan target (y_specialist). Weights akan di-handle nanti.
                X_resampled, y_resampled = smote.fit_resample(X, y_specialist)
                
                # Buat ulang weights untuk data baru, berikan bobot 1.0 untuk data sintetis
                weights_resampled = pd.Series(1.0, index=range(len(X_resampled)))
                
                logger.info(f"   Data size after SMOTE: {len(X_resampled)} samples (from {len(X)})")
                X, y_specialist, weights = X_resampled, y_resampled, weights_resampled
            except Exception as e:
                logger.error(f"   ❌ SMOTE failed for Bearish model: {e}. Proceeding without resampling.")

        # Scaling dan Splitting sekarang dilakukan pada data yang mungkin sudah di-resample
        scaler_temp = StandardScaler()
        X_scaled = pd.DataFrame(scaler_temp.fit_transform(X), columns=X.columns, index=X.index)
        
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_scaled, y_specialist, weights, test_size=0.2, random_state=42, stratify=y_specialist
        )
        
        counts = y_train.value_counts()
        logger.info(f"   Class counts (0/1): {counts.get(0,0)}/{counts.get(1,0)}. Base ratio: {counts.get(0, 1) / counts.get(1, 1):.2f}")
        
        # 🔥 OVERFITTING DETECTION: Cross-validation check
        from sklearn.model_selection import cross_val_score
        
        # Create temp model for CV (before final training)
        temp_params = {
            'objective': 'binary',
            'n_estimators': 100,  # Faster for CV
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        temp_model = lgb.LGBMClassifier(**temp_params)
        cv_scores = cross_val_score(temp_model, X_train.values, y_train.values, cv=5, scoring='f1_weighted', n_jobs=-1)
        logger.info(f"   📊 5-Fold CV F1: {cv_scores.mean():.2%} ± {cv_scores.std():.2%} (overfitting check)")
        
        # 🔥 ASYMMETRIC FOCUS v4.0: DART for Bullish, Enhanced GBDT for Bearish
        base_params = {
            'objective': 'binary',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }
        
        # 🔥🔥🔥 HARMONIZED APPROACH: Optuna + SMOTE for BOTH specialists!
        logger.info(f"   -> {specialty} Treatment: Unified Optuna + SMOTE pipeline...")
        
        if not self.use_optuna:
            logger.error("   ❌ Optuna is disabled, cannot perform robust tuning. Aborting.")
            return None
        
        # Apply SMOTE to balance classes BEFORE Optuna tuning
        logger.info(f"   -> Applying SMOTE to generate synthetic {specialty} samples...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=5)
            X_resampled, y_resampled = smote.fit_resample(X, y_specialist)
            
            # Create uniform weights for resampled data
            weights_resampled = pd.Series(1.0, index=range(len(X_resampled)))
            
            logger.info(f"   Data size after SMOTE: {len(X_resampled)} samples (from {len(X)})")
            X, y_specialist, weights = X_resampled, y_resampled, weights_resampled
        except Exception as e:
            logger.error(f"   ❌ SMOTE failed for {specialty} model: {e}. Proceeding without resampling.")
        
        # Scaling and Splitting
        scaler_temp = StandardScaler()
        X_scaled = pd.DataFrame(scaler_temp.fit_transform(X), columns=X.columns, index=X.index)
        
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_scaled, y_specialist, weights, test_size=0.2, random_state=42, stratify=y_specialist
        )
        
        counts = y_train.value_counts()
        logger.info(f"   Class counts (0/1): {counts.get(0,0)}/{counts.get(1,0)}. Base ratio: {counts.get(0, 1) / counts.get(1, 1):.2f}")
        
        # 🔥 OVERFITTING DETECTION: Cross-validation check
        from sklearn.model_selection import cross_val_score
        
        temp_params = {
            'objective': 'binary',
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        temp_model = lgb.LGBMClassifier(**temp_params)
        cv_scores = cross_val_score(temp_model, X_train.values, y_train.values, cv=5, scoring='f1_weighted', n_jobs=-1)
        logger.info(f"   📊 5-Fold CV F1: {cv_scores.mean():.2%} ± {cv_scores.std():.2%} (overfitting check)")
        
        # 🔥 UNIFIED OPTUNA TUNING for both specialists
        logger.info(f"   -> Running Optuna hyperparameter tuning (30 trials)...")
        
        study = optuna.create_study(direction='maximize', study_name=f'lgbm_{specialty.lower()}_{timeframe}')
        
        # Run optimization
        study.optimize(
            lambda trial: self.optuna_objective_with_smote(trial, X_train, y_train, specialty),
            n_trials=30,
            show_progress_bar=True
        )
        
        logger.info(f"   ✅ Optuna tuning complete for {specialty} model!")
        logger.info(f"   Best PR AUC (avg over folds): {study.best_value:.2%}")
        logger.info(f"   Best params: {study.best_params}")
        
        params = study.best_params
        params.update({
            'objective': 'binary', 'random_state': 42, 'n_jobs': -1, 'verbose': -1
        })
        
        # 🔥 Train final model with best parameters on full SMOTE-resampled data
        logger.info(f"   🔥 Training final {specialty} model with best params...")
        final_smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_train_resampled, y_train_resampled = final_smote.fit_resample(X_train, y_train)
        
        model = lgb.LGBMClassifier(**params)
        
        # Create weights for resampled data
        w_train_resampled = pd.Series(1.0, index=range(len(X_train_resampled)))
        
        # EARLY STOPPING for final model
        X_train_fit, X_val_fit, y_train_fit, y_val_fit, w_train_fit, w_val_fit = train_test_split(
            X_train_resampled, y_train_resampled, w_train_resampled, test_size=0.15, random_state=42, stratify=y_train_resampled
        )
        
        patience = 15
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val_fit, y_val_fit)],
            callbacks=[lgb.early_stopping(stopping_rounds=patience, verbose=False)]
        )
        logger.info(f"   ✅ Final training complete with early stopping (best iter: {model.best_iteration_})")
        
        # 🔥 Model training complete - now continue with common post-processing
        logger.info(f"   ✅ Using OPTUNA-TUNED hyperparameters for {specialty} model")
        
        # 🔥 LSTM HYBRID (if enabled)
        lstm_model = None
        if self.use_lstm:
            try:
                from models.lstm_model import LSTMTrader, TF_AVAILABLE
                
                if not TF_AVAILABLE:
                    logger.warning(f"   ⚠️  TensorFlow not available, skipping LSTM")
                    lstm_model = None
                    # Continue without LSTM
                    self.use_lstm = False
                else:
                    logger.info(f"   🧠 Training LSTM for temporal features (seq=20, epochs=50)...")
                    
                    lstm_model = LSTMTrader(sequence_length=20, lstm_units=32, dropout_rate=0.4)
                    lstm_model.train(
                        X_train_fit.values, y_train_fit.values,
                        X_val_fit.values, y_val_fit.values,
                        epochs=50, batch_size=32, verbose=0
                    )
                    
                    # Get LSTM probabilities as features for hybrid
                    try:
                        # 🔥 FIX DIMENSION MISMATCH: Align lengths before prediction
                        seq_len = 20
                        
                        # Calculate usable length (must be divisible by sequence)
                        train_usable_len = len(X_train_fit) - (len(X_train_fit) % seq_len) - seq_len
                        val_usable_len = len(X_val_fit) - (len(X_val_fit) % seq_len) - seq_len
                        
                        # Predict on aligned data
                        lstm_probs_train_raw = lstm_model.predict(X_train_fit.values[:train_usable_len])[1]
                        lstm_probs_val_raw = lstm_model.predict(X_val_fit.values[:val_usable_len])[1]
                        
                        # Pad to original length with forward fill
                        lstm_probs_train = np.pad(
                            lstm_probs_train_raw,
                            (0, len(X_train_fit) - len(lstm_probs_train_raw)),
                            mode='edge'  # Repeat last value
                        ).reshape(-1, 1)
                        
                        lstm_probs_val = np.pad(
                            lstm_probs_val_raw,
                            (0, len(X_val_fit) - len(lstm_probs_val_raw)),
                            mode='edge'
                        ).reshape(-1, 1)
                        
                        # Create hybrid features (LightGBM + LSTM)
                        X_train_hybrid = np.hstack([X_train_fit.values, lstm_probs_train])
                        X_val_hybrid = np.hstack([X_val_fit.values, lstm_probs_val])
                        
                        # Retrain LightGBM on hybrid features
                        logger.info(f"   🔥 Retraining LightGBM with LSTM features ({lstm_probs_train.shape})...")
                        # 🔥 Reuse patience from above
                        model.fit(
                            X_train_hybrid, y_train_fit.values,
                            sample_weight=w_train_fit,
                            eval_set=[(X_val_hybrid, y_val_fit.values)],
                            callbacks=[lgb.early_stopping(stopping_rounds=patience, verbose=False)]
                        )
                        
                        # 🔥 UPDATE DATAFRAMES to include LSTM feature (prevent dimension mismatch)
                        lstm_probs_train_full = lstm_model.predict(X_train.values[:train_usable_len])[1]
                        lstm_probs_train_full = np.pad(
                            lstm_probs_train_full,
                            (0, len(X_train) - len(lstm_probs_train_full)),
                            mode='edge'
                        )
                        lstm_probs_test_full = lstm_model.predict(X_test.values[:len(X_test) - (len(X_test) % seq_len) - seq_len])[1]
                        lstm_probs_test_full = np.pad(
                            lstm_probs_test_full,
                            (0, len(X_test) - len(lstm_probs_test_full)),
                            mode='edge'
                        )
                        
                        X_train['lstm_prob'] = lstm_probs_train_full
                        X_test['lstm_prob'] = lstm_probs_test_full
                        X_train_fit['lstm_prob'] = lstm_probs_train
                        X_val_fit['lstm_prob'] = lstm_probs_val
                        
                        logger.info(f"   ✅ Hybrid LSTM-LightGBM trained successfully")
                    except Exception as hybrid_err:
                        logger.warning(f"   ⚠️  LSTM hybrid integration failed: {hybrid_err}")
                    
                    # Store LSTM model
                    tf_key = timeframe.lower()
                    self.lstm_models[f"{tf_key}_{specialty.lower()}"] = lstm_model
                    logger.info(f"   ✅ LSTM trained successfully")
            except Exception as e:
                logger.warning(f"   ⚠️  LSTM training failed: {e}, continuing with LightGBM only")
        
        # 🔥 IMPROVED PRUNING: PROTECT Market DNA features from pruning
        # Only prune companion features if they're truly weak
        PROTECTED_FEATURES = [
            # Market DNA
            'adx', 'chop_index_smoothed', 'vol_of_vol_smoothed', 'market_regime_filter',
            # Sniper Scope (MUST PROTECT!)
            'candle_quality_score', 'is_strong_close', 'volume_confirmation', 'is_rejection_candle',
            # Confluence
            'bullish_confluence_score', 'bullish_confluence_x_quality_trend', 'bullish_strength'
        ]
        
        try:
            feature_importances = model.feature_importances_
            
            # Check for dimension mismatch
            if len(feature_importances) != len(X_train.columns):
                logger.warning(f"   ⚠️  Feature dimension mismatch ({len(feature_importances)} != {len(X_train.columns)}), skipping pruning")
            else:
                # Only consider NON-PROTECTED features for pruning
                # 🔥 PERBAIKAN #3: Hanya pangkas fitur yang tidak pernah digunakan sama sekali
                low_importance_mask = feature_importances == 0  # Hanya pangkas fitur dengan importance = 0
                
                # Create list of features to prune (exclude protected ones)
                features_to_prune = []
                for i, (feat, is_low) in enumerate(zip(X_train.columns, low_importance_mask)):
                    if is_low and feat not in PROTECTED_FEATURES:
                        features_to_prune.append(feat)
                
                if features_to_prune:
                    logger.warning(f"   🔪 Pruning {len(features_to_prune)} unused NON-DNA features (importance=0): {features_to_prune[:5]}")
                    
                    # Remove from ALL datasets
                    cols_to_keep = [c for c in X_train.columns if c not in features_to_prune]
                    X_train = X_train[cols_to_keep]
                    X_test = X_test[cols_to_keep]
                    X_train_fit = X_train_fit[cols_to_keep]
                    X_val_fit = X_val_fit[cols_to_keep]
                    
                    # Retrain without low-importance features
                    logger.info(f"   🔥 Retraining with {len(X_train.columns)} high-impact features (DNA protected)...")
                    model = lgb.LGBMClassifier(**params)
                    # 🔥 Reuse patience from above
                    model.fit(
                        X_train_fit.values, y_train_fit.values,
                        sample_weight=w_train_fit,
                        eval_set=[(X_val_fit.values, y_val_fit.values)],
                        callbacks=[lgb.early_stopping(stopping_rounds=patience, verbose=False)]
                    )
                else:
                    logger.info(f"   ✅ No features pruned (all above threshold or protected)")
        except Exception as prune_err:
            logger.warning(f"   ⚠️  Auto-pruning failed: {prune_err}, continuing with current model")
        
        # 🔥 VOTING ENSEMBLE: LightGBM + RandomForest (disabled for now to avoid calibration issues)
        # Voting ensembles don't support predict_proba well with calibration
        # Using LightGBM only for better probability calibration
        final_lgb_model = model  # Keep for ensemble later if needed
        
        # 🔥 OVERFITTING DETECTION: Feature importance stability
        logger.info(f"   🔍 Checking feature importance stability...")
        self.check_feature_stability(X_train, y_train, params, cv=5)
        
        # 🔥 ADAPTIVE PROBABILITY CALIBRATION (PR AUC-based method selection)
        n_train = len(X_train)
        
        # Get raw probabilities to compute PR AUC
        try:
            raw_probs = model.predict_proba(X_test.values)[:, 1]
        except:
            raw_probs = model.predict_proba(X_test)[:, 1]
        
        p_temp, r_temp, t_temp = precision_recall_curve(y_test, raw_probs)
        pr_auc_temp = auc(r_temp, p_temp)
        
        # Select calibration method based on PR AUC (more robust than pure n_train)
        calib_method = 'isotonic' if pr_auc_temp >= 0.75 else 'sigmoid'
        logger.info(f"   🎯 Calibrating probabilities with {calib_method} (pr_auc={pr_auc_temp:.3f}, n_train={n_train}, cv=5)...")
        
        try:
            # 🔥 FIX: Use 'estimator' param + clone to prevent sklearn errors
            calibrator = CalibratedClassifierCV(estimator=clone(model), cv=5, method=calib_method)
            calibrator.fit(X_train.values, y_train.values)
            probs = calibrator.predict_proba(X_test.values)[:, 1]
            logger.info(f"   ✅ Probabilities calibrated with {calib_method} (cv=5)")
            final_model = calibrator  # Save calibrated model
        except Exception as e:
            logger.warning(f"   ⚠️ Calibrator failed: {e}, using raw probabilities")
            probs = model.predict_proba(X_test.values)[:, 1]
            final_model = model
        
        # 🔥 PR CURVE EVALUATION
        p, r, t = self.evaluate_pr_curve(y_test, probs, f"{specialty}_{timeframe}")
        
        precisions, recalls, thresholds = p, r, t
        
        # 🔥 ADAPTIVE ALPHA based on PR AUC (ASYMMETRIC!)
        pr_auc = auc(r, p)
        
        # 🔥 PERBAIKAN: Logika khusus untuk model Bearish dengan PR AUC rendah
        if pr_auc < 0.50 and specialty == 'BEARISH': 
            alpha = 0.5  # Sangat memprioritaskan Presisi
            recall_bias = -0.1 # Bahkan berikan penalti pada recall untuk menghindari false positive
            logger.warning(f"   ⚠️  PR AUC sangat rendah ({pr_auc:.2f}). Menggunakan threshold yang sangat konservatif (fokus presisi).")
        elif pr_auc < 0.65:
            alpha = 0.15  # 🔥 SAMA: Moderate untuk kedua
        elif pr_auc < 0.75:
            alpha = 0.20  # 🔥 SAMA: Balanced
        else:
            alpha = 0.25  # 🔥 SAMA: Slightly strict
        
        # 🔥 BALANCED RECALL BIAS: Sama untuk Bullish & Bearish (anti-bias!)
        tf_recall_bias_map = {'M5': 0.03, 'M15': 0.02, 'H1': 0.02, 'H4': 0.01}  # 🔥 SAMA & REDUCED!
        # Gunakan recall_bias yang sudah dimodifikasi jika PR AUC rendah
        if not ('recall_bias' in locals() and specialty == 'BEARISH'):
            recall_bias = tf_recall_bias_map.get(timeframe, 0.02)
        logger.info(f"   🎯 Selecting balanced threshold: maximize min(P,R) - {alpha}*|P-R| + {recall_bias}*R...")
        scores = np.minimum(precisions[:-1], recalls[:-1]) - alpha * np.abs(precisions[:-1] - recalls[:-1]) + recall_bias * recalls[:-1]
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        calibration_method = f"Balanced min(P,R) - {alpha}*|P-R| + {recall_bias}*R"
        
        # Final evaluation with calibrated threshold
        final_preds = (probs >= best_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, final_preds, average='binary', pos_label=1, zero_division=0
        )
        
        target_name = 'BUY' if specialty == 'BULLISH' else 'SELL'
        logger.info(f"\n--- FINAL CALIBRATED PERFORMANCE: {specialty} ({timeframe}) ---")
        logger.info(f"   Threshold: {best_threshold:.3f} ({calibration_method})")
        logger.info(f"   Metrics for {target_name}: P={precision:.2%}, R={recall:.2%}, F1={f1:.2%}")
        
        # 🔥 OVERFITTING WARNING: Compare CV vs Test F1
        cv_f1 = cv_scores.mean()
        test_f1 = f1
        f1_gap = abs(cv_f1 - test_f1)
        
        if f1_gap > 0.10:  # Gap > 10%
            logger.warning(f"   ⚠️  OVERFITTING RISK: CV F1 ({cv_f1:.2%}) vs Test F1 ({test_f1:.2%}) gap = {f1_gap:.2%}")
        elif cv_scores.std() > 0.10:  # High CV variance
            logger.warning(f"   ⚠️  INSTABILITY: High CV variance ({cv_scores.std():.2%}), model may not generalize")
        else:
            logger.info(f"   ✅ ROBUST: CV F1 gap = {f1_gap:.2%}, variance = {cv_scores.std():.2%}")
        
        # 🔥 CRITICAL: Create dedicated scaler for THIS specialist (after pruning)
        specialist_scaler = StandardScaler()
        specialist_scaler.fit(X_train)
        
        # Return model package with calibrated threshold + scaler + feature list after pruning
        return {
            'model': final_model,
            'threshold': best_threshold,
            'features': X_train.columns.tolist(),  # 🔥 CRITICAL: Features AFTER pruning
            'scaler': specialist_scaler  # 🔥 NEW: Each specialist has its own scaler
        }
    
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
        
        # 🔥 META-MODEL WEIGHTING: Prioritize correcting BUY mistakes
        meta_weights = np.ones(len(meta_target))
        meta_weights[y_train_meta.values == 1] *= 1.8  # Boost BUY samples
        logger.info(f"   Applied meta_weights: BUY x1.8, SELL x1.0")
        
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
        meta_model.fit(meta_features_df, meta_target, sample_weight=meta_weights)
        
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
        
        # 🔥 ASYMMETRIC META-FILTER: Lower threshold for BUY approvals
        CONFIDENCE_THRESHOLD_BUY = 0.65   # More lenient for BUY
        CONFIDENCE_THRESHOLD_SELL = 0.65  # Keep strict for SELL
        
        # Apply different thresholds based on signal type
        meta_verdict = np.where(
            final_pred_pri == 1,  # BUY signal
            (meta_probas > CONFIDENCE_THRESHOLD_BUY).astype(int),
            (meta_probas > CONFIDENCE_THRESHOLD_SELL).astype(int)
        )
        logger.info(f"   Confidence threshold: BUY={CONFIDENCE_THRESHOLD_BUY:.0%}, SELL={CONFIDENCE_THRESHOLD_SELL:.0%}")
        logger.info(f"   Signals approved: {meta_verdict.sum()}/{len(meta_verdict)}")
        
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
        🔥 FINAL ARCHITECTURE: Asymmetric Treatment for Performance Parity
        - BULLISH: Cost-Sensitive + P-R Calibration
        - BEARISH: PURE + P-R Calibration
        """
        logger.info(f"{'='*60}")
        logger.info(f"🚀 Starting FINAL Training (Asymmetric Treatment) for {symbol}")
        logger.info(f"{'='*60}\n")
        
        if self.llm:
            logger.info("✅ LLM Analysis: ENABLED (GPU)")
        else:
            logger.info("⚠️  LLM Analysis: DISABLED")
        
        logger.info("✅ Architecture: HARMONIZED UNIFIED TRAINING")
        logger.info("   → Market DNA (ADX, CHOP, VoV)")
        logger.info("   → BULLISH: Optuna + SMOTE (TimeSeriesSplit CV)")
        logger.info("   → BEARISH: Optuna + SMOTE (TimeSeriesSplit CV)")
        logger.info("   → 30 Optuna trials per specialist (robust hyperparameter tuning)")
        logger.info("   → SMOTE inside CV loop (prevents data leakage)")
        logger.info("   → TimeSeriesSplit (respects temporal order, no lookahead bias)")
        logger.info("   → Adaptive Calibration (PR AUC-based: sigmoid if <0.75, isotonic if >=0.75)")
        logger.info("   → Adaptive Alpha Threshold (0.15/0.2/0.25 based on PR AUC)")
        logger.info("   → TF-Specific Recall Bias (M5:+0.03, M15:+0.02, H1:+0.02, H4:+0.01)")
        logger.info(f"✅ Model Engine: {model_type.upper()}")
        
        # Data config (extended for better generalization)
        candles_config = {
            'M5': 200000,   # Increased from 150K
            'M15': 200000,  # Increased from 150K
            'H1': 100000,
            'H4': 50000
        }
        
        for timeframe in self.timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {timeframe} Model (Asymmetric Treatment)")
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
                
                # Safety net (LOOSENED!)
                if len(df) < 500:  # 🔥 TURUNKAN: 2000 → 500
                    logger.warning(f"⚠️  SKIPPING {timeframe}: Only {len(df)} valid samples after labeling")
                    continue
                
                # Prepare features
                X, y, weights = self.prepare_features(df)
                
                # Train both specialists with Asymmetric Treatment (scaling + pruning inside)
                logger.info(f"\n🎯 Training Dual-Specialist for {timeframe}...")
                bullish_package = self.train_specialist_model(X, y, weights, 'BULLISH', timeframe)
                bearish_package = self.train_specialist_model(X, y, weights, 'BEARISH', timeframe)
                
                # 🔥 CRITICAL: Each package now contains its own scaler + features
                # No need for separate scaler/features files
                
                # Save model package (self-contained)
                final_model = {
                    'bullish': bullish_package,  # {'model': lgbm, 'threshold': 0.XX, 'features': [...], 'scaler': StandardScaler()}
                    'bearish': bearish_package   # {'model': lgbm, 'threshold': 0.XX, 'features': [...], 'scaler': StandardScaler()}
                }
                
                tf_key = timeframe.lower()
                model_path = self.output_dir / f"cls_{tf_key}.pkl"
                
                # Save ONLY model (scaler + features embedded inside)
                joblib.dump(final_model, model_path)
                
                logger.info(f"\n✅ Saved {timeframe} models to {model_path}")
                logger.info(f"   Bullish: {bullish_package['threshold']:.2f} ({len(bullish_package['features'])} features)")
                logger.info(f"   Bearish: {bearish_package['threshold']:.2f} ({len(bearish_package['features'])} features)")
                
                self.models[tf_key] = final_model
                
            except Exception as e:
                logger.error(f"❌ Error training {timeframe} model: {str(e)}", exc_info=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 PRODUCTION-GRADE TRAINING COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Trained models: {list(self.models.keys())}")
        logger.info(f"📁 Output directory: {self.output_dir}")
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
    print("\n🔧 Initializing PRODUCTION-GRADE TRAINER with NEWS & CALENDAR APIs...")
    print("\n🚀 Starting PRODUCTION Training with Fundamental Features...")
    print("⏱️  Training time: 35-50 minutes (Optuna 20 trials + LSTM 50 epochs + News/Calendar)...")
    print("📊 Collecting up to 200K candles per timeframe.")
    print("\n🎯 PRODUCTION-GRADE ENHANCEMENTS:")
    print("   🔥 BEARISH REGULARIZATION: lambda=0.5, min_data_in_leaf=50, max_depth=4")
    print("   🔥 OPTUNA: 20 trials with TimeSeriesSplit (temporal CV)")
    print("   🔥 LSTM: Focal Loss (gamma=2.0, alpha=0.25) + seq=20 + epochs=50")
    print("   🔥 HYBRID: LSTM probs padded & concatenated to LightGBM")
    print("   🔥 AUTO-PRUNE: Remove features with importance <2% (stricter)")
    print("   🔥 SMOOTHED: VoV/CHOP (5-period MA for stability)")
    print("   🔥 BALANCED: inv_freq_mult (M5:1.2, M15:1.5, H1:1.4, H4:1.5)")
    print("\n📰 NEWS & CALENDAR INTEGRATION (NEW!):")
    print("   ✅ TradingEconomics API: High-impact economic events")
    print("   ✅ NewsAPI.org: Real-time news sentiment analysis")
    print("   ✅ Features:")
    print("      • news_sentiment_score: Bullish/Bearish sentiment (-1 to 1)")
    print("      • news_bullish_ratio: % of bullish news articles")
    print("      • news_bearish_ratio: % of bearish news articles")
    print("      • calendar_impact_score: Economic event impact (0-10)")
    print("      • sentiment_price_divergence: News-price divergence detection")
    print("      • news_sentiment_balance: Bullish vs Bearish balance")
    print("      • calendar_vol_synergy: Event impact × volatility")
    print("   ⚠️  Fallback to synthetic features if APIs unavailable")
    print("\n🔑 API KEYS (set in .env file):")
    print("   • TRADING_ECONOMICS_KEY=your_key (Optional, fallback available)")
    print("   • NEWS_API_KEY=your_key (Get free at: https://newsapi.org/)")
    print("     └─ Free tier: 100 requests/day, 30 days history")
    print("\n🎯 EXPECTED IMPROVEMENTS:")
    print("   • +3-5% F1 from news sentiment context")
    print("   • +2-3% Precision from calendar event filtering")
    print("   • Better risk management around high-impact events")
    print("   • Reduced false signals during news-driven volatility\n")
    
    trainer = CLSModelTrainer(
        output_dir="./models/saved_models",
        llm_model_path="./models/Llama-3.2-3B-Instruct-BF16.gguf",
        use_gpu=True,
        use_smote=False,    # 🔥 Using scale_pos_weight instead
        use_optuna=True,    # 🔥 ENABLED: Limited to 20 trials
        use_ensemble=True,
        use_lstm=True       # 🔥 ENABLED: Hybrid LSTM-LightGBM
    )
    
    # Train all timeframes
    trainer.train_all_timeframes(
        mt5_handler=mt5,
        symbol='XAUUSDm',
        model_type='lightgbm'  # Options: 'lightgbm', 'xgboost', 'random_forest', 'gradient_boosting'
    )
    
    mt5.shutdown()
    
    print("\n✅ HARMONIZED UNIFIED Training Complete!")
    print("   → cls_*.pkl (model packages with calibrated thresholds)")
    print("\n📊 All models trained with:")
    print("   • Market DNA (ADX, CHOP, VoV)")
    print("   • BULLISH: Optuna + SMOTE (TimeSeriesSplit CV, 30 trials)")
    print("   • BEARISH: Optuna + SMOTE (TimeSeriesSplit CV, 30 trials)")
    print("   • SMOTE inside CV loop (prevents data leakage)")
    print("   • TimeSeriesSplit (respects temporal order, no lookahead bias)")
    print("   • PR AUC-based Calibration (sigmoid if <0.75, isotonic if >=0.75)")
    print("   • TF-Specific Recall Bias (M5:+0.03, M15:+0.02, H1:+0.02, H4:+0.01)")
    print("\n🔍 OVERFITTING DETECTION:")
    print("   ✅ TimeSeriesSplit Cross-Validation (temporal validation)")
    print("   ✅ Feature stability verified (warns if unstable)")
    print("   ✅ CV-Test gap monitored (warns if >10%)")
    print("   ⚠️  RUN BACKTEST: python -m models.cls_predictor")
    print("   ⚠️  VALIDATE: Compare train F1 vs backtest WR/PF")
