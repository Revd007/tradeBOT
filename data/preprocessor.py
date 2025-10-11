"""
Data Preprocessing for ML Models
Feature engineering and data transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses market data for ML models"""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Args:
            scaler_type: 'standard', 'minmax', or 'robust'
        """
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler()
        self.feature_names = []
    
    def _create_scaler(self):
        """Create appropriate scaler"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(self.scaler_type, StandardScaler())
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data
        
        Args:
            df: DataFrame with OHLC data and technical indicators
        
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Price action features
        df['body_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        df['range_pct'] = ((df['high'] - df['low']) / df['low']) * 100
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Upper/Lower wicks
        df['upper_wick_pct'] = ((df['high'] - df[['open', 'close']].max(axis=1)) / df['high']) * 100
        df['lower_wick_pct'] = ((df[['open', 'close']].min(axis=1) - df['low']) / df['low']) * 100
        
        # Volume features
        df['volume_ma'] = df['tick_volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
        df['volume_std'] = df['tick_volume'].rolling(window=20).std()
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # Rate of change
        for period in [3, 7, 14]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        # Moving average distances
        if 'ema_9' in df.columns:
            df['dist_ema9'] = ((df['close'] - df['ema_9']) / df['close']) * 100
        if 'ema_21' in df.columns:
            df['dist_ema21'] = ((df['close'] - df['ema_21']) / df['close']) * 100
        if 'ema_50' in df.columns:
            df['dist_ema50'] = ((df['close'] - df['ema_50']) / df['close']) * 100
        
        # Bollinger Band features
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['close']) * 100
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI features
        if 'rsi' in df.columns:
            df['rsi_normalized'] = df['rsi'] / 100
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_change'] = df['rsi'].diff()
        
        # MACD features
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_normalized'] = df['macd'] / df['close']
            df['macd_divergence'] = df['macd'] - df['macd_signal']
            df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
            df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                     (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # Stochastic features
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            df['stoch_k_normalized'] = df['stoch_k'] / 100
            df['stoch_divergence'] = df['stoch_k'] - df['stoch_d']
        
        # ATR features
        if 'atr' in df.columns:
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            df['atr_ratio'] = df['atr'] / df['atr'].rolling(window=14).mean()
        
        # Candle patterns (binary features)
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (abs(df['body_pct']) < 0.1).astype(int)
        df['is_hammer'] = self._detect_hammer(df)
        df['is_shooting_star'] = self._detect_shooting_star(df)
        
        # 🔥 NEW: Fitur Kontekstual untuk Kualitas Tren
        # 1. Interaksi Momentum dengan Tren Jangka Panjang
        if 'momentum_5' in df.columns and 'is_above_sma200' in df.columns:
            df['trend_momentum_interaction'] = df['momentum_5'] * (df['is_above_sma200'] * 2 - 1)
        
        # 2. Momentum yang Dinormalisasi oleh Volatilitas
        if 'momentum_10' in df.columns and 'atr' in df.columns:
            df['volatility_normalized_momentum'] = df['momentum_10'] / df['atr'].replace(0, np.nan)
        
        # 3. Skor Mean Reversion
        if 'close' in df.columns and 'ema_50' in df.columns and 'atr' in df.columns:
            df['mean_reversion_score'] = (df['close'] - df['ema_50']) / df['atr'].replace(0, np.nan)
        
        # 🔥 STRATEGI 2: FITUR ALPHA LANJUTAN
        # 2a. Alpha Decay - Tren Indikator
        if 'rsi' in df.columns:
            df['rsi_ema_5'] = df['rsi'].ewm(span=5, adjust=False).mean()
            df['rsi_trend'] = (df['rsi'] - df['rsi_ema_5'])
        
        if 'macd_divergence' in df.columns:
            df['macd_div_ema_5'] = df['macd_divergence'].ewm(span=5, adjust=False).mean()
            df['macd_div_trend'] = df['macd_divergence'] - df['macd_div_ema_5']
        
        # 2b. Order Flow Proxy
        df['order_flow_pressure'] = df['tick_volume'] * (df['close'] - df['open'])
        df['order_flow_pressure_ma_10'] = df['order_flow_pressure'].rolling(window=10).mean()
        
        # 2c. Normalisasi Fitur dengan Volatilitas
        if 'macd_divergence' in df.columns and 'atr' in df.columns:
            df['macd_div_norm_atr'] = df['macd_divergence'] / df['atr'].replace(0, np.nan)
        
        if 'dist_ema50' in df.columns and 'atr' in df.columns:
            df['dist_ema50_norm_atr'] = df['dist_ema50'] / df['atr'].replace(0, np.nan)
        
        # Trend features (EMA alignment)
        if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
            df['ema_bullish_alignment'] = ((df['ema_9'] > df['ema_21']) & 
                                           (df['ema_21'] > df['ema_50'])).astype(int)
            df['ema_bearish_alignment'] = ((df['ema_9'] < df['ema_21']) & 
                                           (df['ema_21'] < df['ema_50'])).astype(int)
        
        # Time-based features
        if 'time' in df.columns:
            df['hour'] = pd.to_datetime(df['time']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
            df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
            df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
            df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        
        # Lag features (previous candle data)
        for lag in [1, 2, 3]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['tick_volume'].shift(lag)
            if 'rsi' in df.columns:
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'high_max_{window}'] = df['high'].rolling(window=window).max()
            df[f'low_min_{window}'] = df['low'].rolling(window=window).min()
        
        # 🔥 NEW: Volatility & Market Regime Features
        if 'close' in df.columns and 'atr' in df.columns:
            # Volatility of Volatility (how stable is current volatility?)
            df['atr_std_5'] = df['atr'].rolling(window=5).std()
            
            # Distance from EMA 200 normalized by ATR (how strong is the trend?)
            if 'sma200' in df.columns:  # Use sma200 (already in dataset) instead of ema_200
                df['dist_sma200_atr'] = (df['close'] - df['sma200']) / df['atr'].replace(0, np.nan)
        
        # 🔥 NEW: Fitur Kekuatan dan Kesehatan Tren (ADX)
        if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
            # Average Directional Index (ADX) - Mengukur kekuatan tren (bukan arah)
            # Nilai di atas 25 menunjukkan tren yang kuat
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
            minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
            
            # True Range
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            
            atr_14 = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_14.replace(0, np.nan))
            minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_14.replace(0, np.nan))
            
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1))
            df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            df['is_strong_trend'] = (df['adx'] > 25).astype(int)
        
        # 🔥 STRATEGI 1: FILTER REZIM PASAR
        # 1a. Rezim Tren vs. Ranging menggunakan ADX
        if 'adx' in df.columns:
            df['regime_trending'] = (df['adx'] > 25).astype(int)
            df['regime_ranging'] = (df['adx'] < 20).astype(int)
        
        # 1b. Rezim Volatilitas
        if 'atr' in df.columns:
            df['atr_ma_10'] = df['atr'].rolling(window=10).mean()
            df['atr_ma_50'] = df['atr'].rolling(window=50).mean()
            df['regime_high_vol'] = (df['atr_ma_10'] > df['atr_ma_50'] * 1.2).astype(int)
            df['regime_low_vol'] = (df['atr_ma_10'] < df['atr_ma_50'] * 0.8).astype(int)
        
        # 1c. INTERAKSI FITUR DENGAN REZIM
        if 'rsi' in df.columns and 'regime_trending' in df.columns:
            df['rsi_x_trending'] = df['rsi'] * df['regime_trending']
            df['rsi_x_ranging'] = df['rsi'] * df['regime_ranging']
        
        if 'momentum_10' in df.columns and 'regime_high_vol' in df.columns:
            df['momentum_x_high_vol'] = df['momentum_10'] * df['regime_high_vol']
        
        # 🔥 Z-Score Normalization untuk Indikator Kunci
        z_score_window = 50
        if 'rsi' in df.columns:
            rsi_mean = df['rsi'].rolling(z_score_window).mean()
            rsi_std = df['rsi'].rolling(z_score_window).std()
            df['rsi_z_score'] = (df['rsi'] - rsi_mean) / rsi_std.replace(0, np.nan)
        
        if 'momentum_10' in df.columns:
            momentum_mean = df['momentum_10'].rolling(z_score_window).mean()
            momentum_std = df['momentum_10'].rolling(z_score_window).std()
            df['momentum_z_score'] = (df['momentum_10'] - momentum_mean) / momentum_std.replace(0, np.nan)
        
        # 🔥 Fitur Asimetris untuk Bull/Bear Specialists
        logger.info("   Adding Asymmetric Features for Bull/Bear setups...")
        
        # Fitur untuk Bullish Specialist
        # 1. Volatility Squeeze (Bollinger Bands di dalam Keltner Channel)
        if all(k in df.columns for k in ['bb_upper', 'bb_lower', 'atr']) and 'ema_20' in df.columns:
            kc_upper = df['ema_20'] + (df['atr'] * 1.5)
            kc_lower = df['ema_20'] - (df['atr'] * 1.5)
            df['bull_squeeze'] = ((df['bb_upper'] < kc_upper) & (df['bb_lower'] > kc_lower)).astype(int)
        else:
            df['bull_squeeze'] = 0
        
        # 2. Bullish Structure (Higher Lows)
        df['bull_structure_hl'] = (df['low'] > df['low'].rolling(10).min().shift(1)).astype(int)
        
        # Fitur untuk Bearish Specialist
        # 1. Expansion Volatilitas setelah Squeeze
        if 'ema_20' in df.columns:
            df['bear_expansion'] = ((df['bull_squeeze'].shift(1) == 1) & (df['bull_squeeze'] == 0) & (df['close'] < df['ema_20'])).astype(int)
        else:
            df['bear_expansion'] = 0
        
        # 2. Bearish Structure (Lower Highs)
        df['bear_structure_lh'] = (df['high'] < df['high'].rolling(10).max().shift(1)).astype(int)
        
        # 🔥🔥 FITUR ALPHA-CENTRIC KHUSUS UNTUK BULLISH SPECIALIST 🔥🔥
        logger.info("   Creating Alpha-Centric features for Bullish setups...")
        
        # 1. Fitur "Buy the Dip": Mengukur kedalaman pullback dalam tren naik
        rolling_high_20 = df['high'].rolling(20).max()
        df['bull_pullback_depth_pct'] = ((df['close'] - rolling_high_20) / rolling_high_20.replace(0, np.nan)) * 100
        
        # 2. Fitur "Selling Exhaustion": Mengukur perlambatan momentum negatif (akselerasi)
        if 'roc_3' in df.columns:
            df['bull_momentum_acceleration'] = df['roc_3'].diff()
        
        # 3. Fitur "Reversal Confirmation": Spike volume bullish setelah periode negatif
        if 'is_bullish' in df.columns and 'volume_ratio' in df.columns and 'roc_3' in df.columns:
            df['bull_reversal_confirmation'] = ((df['is_bullish'] == 1) & 
                                               (df['volume_ratio'] > 1.5) & 
                                               (df['roc_3'].shift(1) < 0)).astype(int)
        else:
            df['bull_reversal_confirmation'] = 0
        
        # 4. Fitur "Relative Volume Strength": Mengukur kekuatan volume pada candle bullish
        if 'is_bullish' in df.columns and 'volume_ratio' in df.columns:
            df['bullish_volume_spike'] = df['is_bullish'] * df['volume_ratio']
        else:
            df['bullish_volume_spike'] = 0
        
        # 🔥🔥 STRATEGI 1: FITUR KESEHATAN TREN 🔥🔥
        logger.info("   Adding Trend Health & Quality features (CHOP, Slope)...")
        
        # 1. Choppiness Index (CHOP) - 14 periods
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns and 'atr' in df.columns:
            atr_1 = df['atr'].rolling(1).mean()
            sum_atr_14 = atr_1.rolling(14).sum()
            highest_high_14 = df['high'].rolling(14).max()
            lowest_low_14 = df['low'].rolling(14).min()
            range_14 = (highest_high_14 - lowest_low_14).replace(0, np.nan)
            chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
            df['chop_index'] = chop
        else:
            df['chop_index'] = 50  # Neutral default
        
        # 2. Slope of Short-Term EMA (e.g., EMA 9)
        if 'ema_9' in df.columns:
            df['ema_9_slope'] = df['ema_9'].diff()
        else:
            df['ema_9_slope'] = 0
        
        # 3. Interaksi: Sinyal Bullish saat Pasar TIDAK Choppy
        if 'bull_structure_hl' in df.columns and 'chop_index' in df.columns:
            df['bull_hl_low_chop'] = df['bull_structure_hl'] * (df['chop_index'] < 40).astype(int)
        else:
            df['bull_hl_low_chop'] = 0
        
        # 🔥🔥 STRATEGI 2: VOLATILITY OF VOLATILITY 🔥🔥
        logger.info("   Adding Volatility of Volatility features (ATR StdDev)...")
        if 'atr' in df.columns:
            # Menghitung standar deviasi dari ATR. Nilai rendah = volatilitas stabil.
            df['atr_std_14'] = df['atr'].rolling(14).std()
            
            # Normalisasi dengan ATR itu sendiri untuk perbandingan
            atr_mean_14 = df['atr'].rolling(14).mean()
            df['vol_of_vol'] = df['atr_std_14'] / atr_mean_14.replace(0, np.nan)
        else:
            df['atr_std_14'] = 0
            df['vol_of_vol'] = 0
        
        return df
    
    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer candle pattern"""
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        is_hammer = ((lower_wick > body * 2) & 
                     (upper_wick < body * 0.5) & 
                     (df['close'] > df['open'])).astype(int)
        
        return is_hammer
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect shooting star candle pattern"""
        body = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        is_shooting_star = ((upper_wick > body * 2) & 
                            (lower_wick < body * 0.5) & 
                            (df['close'] < df['open'])).astype(int)
        
        return is_shooting_star
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'target',
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            feature_columns: List of feature columns (None = auto-select)
            test_size: Proportion of test set
            random_state: Random seed
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Select features
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume',
                           target_column, 'spread', 'real_volume']
            feature_columns = [col for col in df_clean.columns if col not in exclude_cols]
        
        self.feature_names = feature_columns
        
        # Prepare X and y
        X = df_clean[feature_columns].values
        y = df_clean[target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training data prepared: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        logger.info(f"Features: {len(feature_columns)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_labels(
        self,
        df: pd.DataFrame,
        method: str = 'triple_barrier',  # 🔥 DEFAULT: Triple Barrier Method
        horizon: int = 10,
        threshold: float = 0.2,
        binary: bool = False,
        risk_reward_ratio: float = 1.5,  # TP/SL ratio
        atr_multiplier_sl: float = 1.5   # Stop Loss distance in ATR
    ) -> pd.Series:
        """
        Create training labels
        
        Args:
            df: DataFrame with price data
            method: 'fixed_horizon', 'adaptive', or 'triple_barrier'
            horizon: Number of candles ahead to look (vertical barrier)
            threshold: Price movement threshold in % (for fixed_horizon)
            binary: If True, return only 0=SELL, 1=BUY (skip HOLD samples)
            risk_reward_ratio: TP/SL ratio (for triple_barrier)
            atr_multiplier_sl: SL distance in ATR units (for triple_barrier)
        
        Returns:
            Series with labels:
            - If binary=False: (0=SELL, 1=HOLD, 2=BUY)
            - If binary=True: (0=SELL, 1=BUY) with HOLD samples marked as -1
        """
        if method == 'fixed_horizon':
            return self._create_fixed_horizon_labels(df, horizon, threshold, binary)
        elif method == 'adaptive':
            return self._create_adaptive_labels(df, horizon, threshold, binary)
        elif method == 'triple_barrier':
            return self._create_triple_barrier_labels_v2(
                df, horizon, atr_multiplier_sl, risk_reward_ratio, binary
            )
        else:
            raise ValueError(f"Unknown labeling method: {method}")
    
    def _create_fixed_horizon_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        threshold: float,
        binary: bool = False
    ) -> pd.Series:
        """Create labels based on fixed future horizon"""
        labels = []
        
        for i in range(len(df)):
            if i + horizon >= len(df):
                labels.append(-1 if binary else 1)  # Mark for removal in binary mode
                continue
            
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + horizon]['close']
            
            price_change = ((future_price - current_price) / current_price) * 100
            
            if price_change > threshold:
                labels.append(1 if binary else 2)  # BUY (1 in binary, 2 in ternary)
            elif price_change < -threshold:
                labels.append(0)  # SELL (0 in both modes)
            else:
                labels.append(-1 if binary else 1)  # HOLD → mark for removal in binary mode
        
        return pd.Series(labels, index=df.index)
    
    def _create_adaptive_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        threshold: float
    ) -> pd.Series:
        """Create labels with adaptive threshold based on ATR"""
        labels = []
        
        for i in range(len(df)):
            if i + horizon >= len(df):
                labels.append(1)
                continue
            
            current_price = df.iloc[i]['close']
            future_high = df.iloc[i:i+horizon]['high'].max()
            future_low = df.iloc[i:i+horizon]['low'].min()
            
            # Use ATR for adaptive threshold
            atr = df.iloc[i].get('atr', current_price * 0.01)
            adaptive_threshold = (atr / current_price) * 100
            
            upside = ((future_high - current_price) / current_price) * 100
            downside = ((current_price - future_low) / current_price) * 100
            
            if upside > adaptive_threshold and upside > downside * 1.5:
                labels.append(2)  # BUY
            elif downside > adaptive_threshold and downside > upside * 1.5:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD
        
        return pd.Series(labels, index=df.index)
    
    def _create_triple_barrier_labels_v2(
        self,
        df: pd.DataFrame,
        horizon: int,
        atr_multiplier_sl: float,
        risk_reward_ratio: float,
        binary: bool
    ) -> Tuple[pd.Series, pd.Series]:
        """
        🔥 STRATEGI 1: Dynamic Sample Weighting
        
        Returns:
            (labels, weights) - labels untuk klasifikasi, weights untuk prioritas training
        """
        labels = pd.Series(np.nan, index=df.index)
        weights = pd.Series(1.0, index=df.index)  # 🔥 INISIALISASI BOBOT
        
        # Validate ATR column exists
        if 'atr' not in df.columns:
            logger.error("ATR column required for Triple Barrier! Using fallback fixed_horizon")
            return self._create_fixed_horizon_labels(df, horizon, 0.3, binary), weights
        
        logger.info(f"🎯 Triple Barrier: horizon={horizon}, SL={atr_multiplier_sl}xATR, RR={risk_reward_ratio}:1")
        
        for i in range(len(df) - horizon):
            entry_price = df['close'].iloc[i]
            atr_at_entry = df['atr'].iloc[i]
            
            # Calculate barriers
            sl_distance = atr_at_entry * atr_multiplier_sl
            tp_distance = sl_distance * risk_reward_ratio
            
            upper_barrier = entry_price + tp_distance  # Take Profit
            lower_barrier = entry_price - sl_distance  # Stop Loss
            
            # Look ahead for barrier hits
            future_slice = df.iloc[i+1 : i+1+horizon]
            
            # Find when each barrier is hit
            hit_tp_idx = future_slice[future_slice['high'] >= upper_barrier].index
            hit_sl_idx = future_slice[future_slice['low'] <= lower_barrier].index
            
            # Determine which barrier was hit first
            if len(hit_tp_idx) > 0 and len(hit_sl_idx) > 0:
                # Both hit - check which came first
                if hit_tp_idx[0] < hit_sl_idx[0]:
                    labels.iloc[i] = 1  # BUY (TP hit first)
                    # 🔥 HITUNG BOBOT: TP cepat = bobot lebih tinggi
                    time_to_tp = hit_tp_idx[0] - i
                    weights.iloc[i] = 1.0 + (1.0 / (time_to_tp + 1e-5))
                else:
                    labels.iloc[i] = 0  # SELL (SL hit first)
            elif len(hit_tp_idx) > 0:
                labels.iloc[i] = 1  # BUY (only TP hit)
                # 🔥 HITUNG BOBOT juga di sini
                time_to_tp = hit_tp_idx[0] - i
                weights.iloc[i] = 1.0 + (1.0 / (time_to_tp + 1e-5))
            elif len(hit_sl_idx) > 0:
                labels.iloc[i] = 0  # SELL (only SL hit)
            else:
                labels.iloc[i] = -1  # HOLD (time expired, no barrier hit)
        
        # Fill remaining NaN with -1 (will be removed)
        labels.fillna(-1, inplace=True)
        weights.fillna(1.0, inplace=True)
        
        # Log distribution
        label_counts = labels.value_counts().sort_index()
        logger.info(f"Triple Barrier distribution: SELL={label_counts.get(0, 0)}, "
                   f"HOLD={label_counts.get(-1, 0)}, BUY={label_counts.get(1, 0)}")
        
        return labels, weights
    
    def _create_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        threshold: float,
        binary: bool = False
    ) -> pd.Series:
        """OLD: Create labels using triple-barrier method (legacy, kept for compatibility)"""
        labels = []
        
        for i in range(len(df)):
            if i + 50 >= len(df):  # Need lookahead
                labels.append(-1 if binary else 1)
                continue
            
            entry_price = df.iloc[i]['close']
            
            # Set barriers
            upper_barrier = entry_price * (1 + threshold / 100)
            lower_barrier = entry_price * (1 - threshold / 100)
            
            # Check future prices
            future_slice = df.iloc[i+1:i+50]
            
            # Find first barrier hit
            upper_hit = future_slice[future_slice['high'] >= upper_barrier]
            lower_hit = future_slice[future_slice['low'] <= lower_barrier]
            
            if len(upper_hit) > 0 and len(lower_hit) > 0:
                if upper_hit.index[0] < lower_hit.index[0]:
                    labels.append(1 if binary else 2)  # BUY
                else:
                    labels.append(0)  # SELL
            elif len(upper_hit) > 0:
                labels.append(1 if binary else 2)  # BUY
            elif len(lower_hit) > 0:
                labels.append(0)  # SELL
            else:
                labels.append(-1 if binary else 1)  # HOLD
        
        return pd.Series(labels, index=df.index)
    
    def get_feature_importance(
        self,
        model,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            top_n: Number of top features to return
        
        Returns:
            DataFrame with features and importances
        """
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test preprocessor
    from core.mt5_handler import MT5Handler
    from strategies.base_strategy import BaseStrategy
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        # Get data
        df = mt5.get_candles("XAUUSDm", "M5", count=1000)
        
        # Add indicators
        strategy = BaseStrategy.__new__(BaseStrategy)
        BaseStrategy.__init__(strategy, "Test", "MEDIUM")
        df = strategy.add_all_indicators(df)
        
        # Create preprocessor
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        # Create features
        df_features = preprocessor.create_features(df)
        
        print(f"\nOriginal columns: {len(df.columns)}")
        print(f"After feature engineering: {len(df_features.columns)}")
        print(f"\nNew features sample:\n{df_features.columns.tolist()[:20]}")
        
        # Create labels
        df_features['target'] = preprocessor.create_labels(
            df_features, 
            method='adaptive',
            horizon=10,
            threshold=0.15
        )
        
        # Check label distribution
        print(f"\nLabel distribution:")
        print(df_features['target'].value_counts())
        
        # Prepare training data
        X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(
            df_features,
            target_column='target',
            test_size=0.2
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        mt5.shutdown()

