"""
CLS Model Trainer
Train classifier models for different timeframes to predict trade direction
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CLSModelTrainer:
    """Train CLS classifier models for trade direction prediction"""
    
    def __init__(self, output_dir: str = "./models/saved_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timeframes = ['M5', 'M15', 'H1', 'H4']
        self.models = {}
        self.scalers = {}
    
    def collect_training_data(
        self,
        mt5_handler,
        symbol: str,
        timeframe: str,
        candles: int = 5000
    ) -> pd.DataFrame:
        """
        Collect historical data for training
        
        Args:
            mt5_handler: MT5 connection
            symbol: Trading symbol (e.g., XAUUSDm)
            timeframe: Timeframe (M5, M15, H1, H4)
            candles: Number of candles to collect
        """
        logger.info(f"Collecting {candles} candles for {symbol} {timeframe}...")
        
        df = mt5_handler.get_candles(symbol, timeframe, count=candles)
        
        # Add technical indicators
        from strategies.base_strategy import BaseStrategy
        strategy = BaseStrategy.__new__(BaseStrategy)
        BaseStrategy.__init__(strategy, "Trainer", "MEDIUM")
        df = strategy.add_all_indicators(df)
        
        # Add features
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        df = preprocessor.create_features(df)
        
        # Create labels
        df['target'] = preprocessor.create_labels(
            df,
            method='adaptive',
            horizon=10,
            threshold=0.15
        )
        
        logger.info(f"✅ Collected {len(df)} candles with features")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        
        Returns:
            X (features), y (target)
        """
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
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Samples: {len(df_clean)}")
        logger.info(f"Label distribution:\n{y.value_counts()}")
        
        return X, y
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest'
    ) -> Tuple[object, StandardScaler]:
        """
        Train classifier model
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: 'random_forest' or 'gradient_boosting'
        
        Returns:
            (trained_model, scaler)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        logger.info(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        logger.info(f"Train accuracy: {train_score:.2%}")
        logger.info(f"Test accuracy: {test_score:.2%}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        logger.info(f"CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=['SELL', 'HOLD', 'BUY']
        ))
        
        # Confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info("\n" + str(confusion_matrix(y_test, y_pred)))
        
        # Feature importance (top 20)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 20 Important Features:")
            for idx, row in feature_importance.head(20).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return model, scaler
    
    def train_all_timeframes(
        self,
        mt5_handler,
        symbol: str = 'XAUUSDm',
        model_type: str = 'random_forest'
    ):
        """
        Train models for all timeframes
        
        Args:
            mt5_handler: MT5 connection
            symbol: Trading symbol
            model_type: Model type to train
        """
        logger.info(f"{'='*60}")
        logger.info(f"Training CLS Models for {symbol}")
        logger.info(f"{'='*60}\n")
        
        for timeframe in self.timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {timeframe} Model")
            logger.info(f"{'='*60}\n")
            
            try:
                # Collect data
                df = self.collect_training_data(
                    mt5_handler,
                    symbol,
                    timeframe,
                    candles=5000 if timeframe in ['M5', 'M15'] else 3000
                )
                
                # Prepare features
                X, y = self.prepare_features(df)
                
                # Train model
                model, scaler = self.train_model(X, y, model_type)
                
                # Save model and scaler
                tf_key = timeframe.lower()
                
                model_path = self.output_dir / f"cls_{tf_key}.pkl"
                scaler_path = self.output_dir / f"scaler_{tf_key}.pkl"
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                
                logger.info(f"✅ Saved model to {model_path}")
                logger.info(f"✅ Saved scaler to {scaler_path}")
                
                self.models[tf_key] = model
                self.scalers[tf_key] = scaler
                
            except Exception as e:
                logger.error(f"❌ Error training {timeframe} model: {str(e)}", exc_info=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Trained models: {list(self.models.keys())}")
        logger.info(f"Output directory: {self.output_dir}")
    
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
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    login = int(os.getenv('MT5_LOGIN_DEMO', '12345678'))
    password = os.getenv('MT5_PASSWORD_DEMO', 'password')
    server = os.getenv('MT5_SERVER_DEMO', 'MetaQuotes-Demo')
    
    mt5 = MT5Handler(login, password, server)
    
    if not mt5.initialize():
        print("❌ Failed to connect to MT5")
        exit(1)
    
    # Create trainer
    trainer = CLSModelTrainer(output_dir="./models/saved_models")
    
    # Train all timeframes
    print("\n🚀 Starting training process...")
    print("⏱️  This may take 10-30 minutes depending on your system...\n")
    
    trainer.train_all_timeframes(
        mt5_handler=mt5,
        symbol='XAUUSDm',
        model_type='random_forest'  # or 'gradient_boosting'
    )
    
    mt5.shutdown()
    
    print("\n✅ Training complete! Models saved to ./models/saved_models/")
    print("\nYou can now use these models with CLSPredictor for live trading.")

