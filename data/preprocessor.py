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
        method: str = 'fixed_horizon',
        horizon: int = 10,
        threshold: float = 0.2
    ) -> pd.Series:
        """
        Create training labels
        
        Args:
            df: DataFrame with price data
            method: 'fixed_horizon', 'adaptive', or 'triple_barrier'
            horizon: Number of candles ahead to look
            threshold: Price movement threshold in %
        
        Returns:
            Series with labels (0=SELL, 1=HOLD, 2=BUY)
        """
        if method == 'fixed_horizon':
            return self._create_fixed_horizon_labels(df, horizon, threshold)
        elif method == 'adaptive':
            return self._create_adaptive_labels(df, horizon, threshold)
        elif method == 'triple_barrier':
            return self._create_triple_barrier_labels(df, threshold)
        else:
            raise ValueError(f"Unknown labeling method: {method}")
    
    def _create_fixed_horizon_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        threshold: float
    ) -> pd.Series:
        """Create labels based on fixed future horizon"""
        labels = []
        
        for i in range(len(df)):
            if i + horizon >= len(df):
                labels.append(1)  # HOLD for last candles
                continue
            
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + horizon]['close']
            
            price_change = ((future_price - current_price) / current_price) * 100
            
            if price_change > threshold:
                labels.append(2)  # BUY
            elif price_change < -threshold:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD
        
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
    
    def _create_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> pd.Series:
        """Create labels using triple-barrier method"""
        labels = []
        
        for i in range(len(df)):
            if i + 50 >= len(df):  # Need lookahead
                labels.append(1)
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
                    labels.append(2)  # BUY
                else:
                    labels.append(0)  # SELL
            elif len(upper_hit) > 0:
                labels.append(2)  # BUY
            elif len(lower_hit) > 0:
                labels.append(0)  # SELL
            else:
                labels.append(1)  # HOLD
        
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

