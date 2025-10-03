import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CLSPredictor:
    """Trained classifier model for trade direction prediction"""
    
    def __init__(self, model_dir: str = "./models/saved_models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models for different timeframes"""
        timeframes = ['m5', 'm15', 'h1', 'h4']
        
        for tf in timeframes:
            model_path = self.model_dir / f"cls_{tf}.pkl"
            scaler_path = self.model_dir / f"scaler_{tf}.pkl"
            
            try:
                if model_path.exists():
                    self.models[tf] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {tf} model")
                
                if scaler_path.exists():
                    self.scalers[tf] = joblib.load(scaler_path)
            
            except Exception as e:
                logger.error(f"Error loading {tf} model: {str(e)}")
        
        if not self.models:
            logger.warning("⚠️ No CLS models found. Using default predictions.")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from candle data
        
        Features:
        - Price action (OHLC ratios)
        - Technical indicators (RSI, MACD, etc.)
        - Candle patterns
        - Volume
        """
        features = []
        
        last_candle = df.iloc[-1]
        
        # Price action features
        features.extend([
            (last_candle['close'] - last_candle['open']) / last_candle['open'],  # Body %
            (last_candle['high'] - last_candle['low']) / last_candle['low'],     # Range %
            (last_candle['close'] - last_candle['low']) / (last_candle['high'] - last_candle['low']),  # Close position
        ])
        
        # Technical indicators
        features.extend([
            last_candle['rsi'] / 100,  # Normalize RSI
            last_candle['macd'] / last_candle['close'],  # MACD ratio
            (last_candle['close'] - last_candle['ema_21']) / last_candle['close'],  # Distance from EMA
            (last_candle['bb_upper'] - last_candle['bb_lower']) / last_candle['close'],  # BB width
            (last_candle['close'] - last_candle['bb_middle']) / (last_candle['bb_upper'] - last_candle['bb_lower']),  # BB position
        ])
        
        # Momentum features
        features.extend([
            last_candle['change_pct'],
            last_candle['stoch_k'] / 100,
            (last_candle['stoch_k'] - last_candle['stoch_d']) / 100,
        ])
        
        # Trend features (EMA alignment)
        features.extend([
            1 if last_candle['ema_9'] > last_candle['ema_21'] else 0,
            1 if last_candle['ema_21'] > last_candle['ema_50'] else 0,
            (last_candle['ema_9'] - last_candle['ema_50']) / last_candle['ema_50'],
        ])
        
        # Volume
        avg_volume = df['tick_volume'].iloc[-20:].mean()
        features.append(last_candle['tick_volume'] / avg_volume if avg_volume > 0 else 1.0)
        
        # Recent price action (last 5 candles)
        recent_changes = df['change_pct'].iloc[-5:].tolist()
        features.extend(recent_changes)
        
        return np.array(features).reshape(1, -1)
    
    def predict(
        self, 
        df: pd.DataFrame, 
        timeframe: str = 'm5'
    ) -> Tuple[str, float]:
        """
        Predict trade direction
        
        Returns:
            (direction: 'BUY' or 'SELL', confidence: 0.0-1.0)
        """
        tf_key = timeframe.lower()
        
        if tf_key not in self.models:
            # Default prediction if no model
            logger.warning(f"No model for {timeframe}, using default")
            return 'HOLD', 0.5
        
        try:
            # Prepare features
            features = self.prepare_features(df)
            
            # Scale features
            if tf_key in self.scalers:
                features = self.scalers[tf_key].transform(features)
            
            # Predict
            model = self.models[tf_key]
            prediction = model.predict(features)[0]
            
            # Get probability (confidence)
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(features)[0]
                confidence = max(probas)
            else:
                confidence = 0.7  # Default confidence
            
            # Map prediction to action
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            action = action_map.get(prediction, 'HOLD')
            
            return action, confidence
        
        except Exception as e:
            logger.error(f"Error in CLS prediction: {str(e)}")
            return 'HOLD', 0.0
    
    def multi_timeframe_consensus(
        self, 
        symbol: str, 
        mt5_handler,
        timeframes: list = ['m5', 'm15', 'h1']
    ) -> Dict:
        """
        Get predictions from multiple timeframes and calculate consensus
        """
        predictions = {}
        
        for tf in timeframes:
            try:
                df = mt5_handler.get_candles(symbol, tf.upper(), count=200)
                
                # Add indicators
                from strategies.base_strategy import BaseStrategy
                strategy = BaseStrategy.__new__(BaseStrategy)
                BaseStrategy.__init__(strategy, "Temp", "MEDIUM")
                df = strategy.add_all_indicators(df)
                
                action, confidence = self.predict(df, tf)
                predictions[tf] = {
                    'action': action,
                    'confidence': confidence
                }
            
            except Exception as e:
                logger.error(f"Error in {tf} prediction: {str(e)}")
                predictions[tf] = None
        
        # Calculate consensus
        buy_votes = sum(1 for p in predictions.values() if p and p['action'] == 'BUY')
        sell_votes = sum(1 for p in predictions.values() if p and p['action'] == 'SELL')
        hold_votes = sum(1 for p in predictions.values() if p and p['action'] == 'HOLD')
        
        total_votes = buy_votes + sell_votes + hold_votes
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            consensus = 'BUY'
            consensus_confidence = buy_votes / total_votes
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            consensus = 'SELL'
            consensus_confidence = sell_votes / total_votes
        else:
            consensus = 'HOLD'
            consensus_confidence = hold_votes / total_votes if total_votes > 0 else 0.5
        
        # Calculate weighted confidence (higher timeframes have more weight)
        weights = {'m5': 1.0, 'm15': 1.5, 'h1': 2.0, 'h4': 2.5}
        weighted_confidence = 0
        total_weight = 0
        
        for tf, pred in predictions.items():
            if pred and pred['action'] == consensus:
                weight = weights.get(tf, 1.0)
                weighted_confidence += pred['confidence'] * weight
                total_weight += weight
        
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        return {
            'consensus': consensus,
            'confidence': final_confidence,
            'predictions': predictions,
            'votes': {
                'buy': buy_votes,
                'sell': sell_votes,
                'hold': hold_votes
            }
        }


# models/trend_fusion.py
"""
Trend Fusion - Meta-analysis combining multiple signals
"""

class TrendFusion:
    """Combines CLS, technical analysis, and news sentiment"""
    
    def __init__(self, cls_predictor: CLSPredictor):
        self.cls = cls_predictor
    
    def analyze(
        self,
        symbol: str,
        mt5_handler,
        strategy_signals: Dict,
        news_context: Optional[Dict] = None
    ) -> Dict:
        """
        Fuse all signals into final decision
        
        Args:
            symbol: Trading symbol
            mt5_handler: MT5 connection
            strategy_signals: Signals from strategy manager
            news_context: News and calendar context
        
        Returns:
            {
                'final_action': 'BUY', 'SELL', or 'HOLD',
                'confidence': 0.0-1.0,
                'meta_score': float,
                'breakdown': dict,
                'reason': str
            }
        """
        scores = {
            'cls': 0.0,
            'technical': 0.0,
            'strategy': 0.0,
            'news': 0.0
        }
        
        breakdown = {}
        
        # 1. CLS Model Score (30% weight)
        cls_result = self.cls.multi_timeframe_consensus(symbol, mt5_handler)
        breakdown['cls'] = cls_result
        
        if cls_result['consensus'] == 'BUY':
            scores['cls'] = cls_result['confidence'] * 0.3
        elif cls_result['consensus'] == 'SELL':
            scores['cls'] = -cls_result['confidence'] * 0.3
        
        # 2. Technical Analysis Score (25% weight)
        df = mt5_handler.get_candles(symbol, 'M15', count=200)
        from strategies.base_strategy import BaseStrategy
        base_strat = BaseStrategy.__new__(BaseStrategy)
        BaseStrategy.__init__(base_strat, "Tech", "MEDIUM")
        df = base_strat.add_all_indicators(df)
        
        trend = base_strat.detect_trend(df)
        breakdown['trend'] = trend
        
        if trend == 'BULLISH':
            scores['technical'] = 0.25
        elif trend == 'BEARISH':
            scores['technical'] = -0.25
        
        # 3. Strategy Signals Score (35% weight)
        if strategy_signals.get('best_signal'):
            best_signal = strategy_signals['best_signal']
            breakdown['best_strategy'] = best_signal
            
            signal_score = best_signal['confidence'] * 0.35
            if best_signal['action'] == 'BUY':
                scores['strategy'] = signal_score
            else:
                scores['strategy'] = -signal_score
        
        # 4. News Sentiment Score (10% weight)
        if news_context and news_context.get('news_sentiment'):
            sentiment = news_context['news_sentiment']['sentiment']
            breakdown['news_sentiment'] = sentiment
            
            if sentiment == 'BULLISH':
                scores['news'] = 0.10
            elif sentiment == 'BEARISH':
                scores['news'] = -0.10
        
        # Calculate meta score (-1 to +1)
        meta_score = sum(scores.values())
        breakdown['scores'] = scores
        breakdown['meta_score'] = meta_score
        
        # Determine final action
        if meta_score > 0.40:
            final_action = 'BUY'
            confidence = min(abs(meta_score), 1.0)
            reason = "Strong bullish confluence across multiple signals"
        elif meta_score < -0.40:
            final_action = 'SELL'
            confidence = min(abs(meta_score), 1.0)
            reason = "Strong bearish confluence across multiple signals"
        else:
            final_action = 'HOLD'
            confidence = 0.5
            reason = "Mixed or weak signals, no clear direction"
        
        # Adjust confidence based on news impact
        if news_context and news_context.get('impact_score', 0) > 5:
            confidence *= 0.7  # Reduce confidence during high impact news
            reason += f" (High news impact: {news_context['impact_score']:.1f}/10)"
        
        return {
            'final_action': final_action,
            'confidence': confidence,
            'meta_score': meta_score,
            'breakdown': breakdown,
            'reason': reason,
            'components': {
                'cls_weight': scores['cls'],
                'technical_weight': scores['technical'],
                'strategy_weight': scores['strategy'],
                'news_weight': scores['news']
            }
        }
    
    def should_enter_trade(self, fusion_result: Dict, min_confidence: float = 0.65) -> bool:
        """Determine if trade should be entered based on fusion analysis"""
        return (fusion_result['final_action'] != 'HOLD' and 
                fusion_result['confidence'] >= min_confidence)


if __name__ == "__main__":
    # Test CLS predictor
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        cls = CLSPredictor()
        
        # Single prediction
        df = mt5.get_candles("XAUUSD", "M5", count=200)
        
        from strategies.base_strategy import BaseStrategy
        strat = BaseStrategy.__new__(BaseStrategy)
        BaseStrategy.__init__(strat, "Test", "MEDIUM")
        df = strat.add_all_indicators(df)
        
        action, confidence = cls.predict(df, 'm5')
        print(f"Single prediction: {action} (confidence: {confidence:.2%})")
        
        # Multi-timeframe consensus
        result = cls.multi_timeframe_consensus("XAUUSD", mt5)
        print(f"\nMTF Consensus: {result['consensus']} (confidence: {result['confidence']:.2%})")
        print(f"Votes: BUY={result['votes']['buy']}, SELL={result['votes']['sell']}, HOLD={result['votes']['hold']}")
        
        # Trend Fusion
        fusion = TrendFusion(cls)
        
        from strategies.strategy_manager import StrategyManager
        strategy_mgr = StrategyManager(mt5)
        strategy_signals = strategy_mgr.analyze_all("XAUUSD")
        
        fusion_result = fusion.analyze("XAUUSD", mt5, strategy_signals)
        
        print(f"\n--- TREND FUSION ---")
        print(f"Final Action: {fusion_result['final_action']}")
        print(f"Confidence: {fusion_result['confidence']:.2%}")
        print(f"Meta Score: {fusion_result['meta_score']:.2f}")
        print(f"Reason: {fusion_result['reason']}")
        print(f"\nComponent Scores:")
        for component, score in fusion_result['components'].items():
            print(f"  {component}: {score:+.3f}")
        
        mt5.shutdown()