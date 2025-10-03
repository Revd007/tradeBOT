"""
Trend Fusion - Meta-analysis combining multiple signals
Combines CLS predictions, technical analysis, strategy signals, and news sentiment
"""

import logging
from typing import Dict, Optional
from models.cls_predictor import CLSPredictor

logger = logging.getLogger(__name__)


class TrendFusion:
    """Combines CLS, technical analysis, strategy signals, and news sentiment"""
    
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
        try:
            cls_result = self.cls.multi_timeframe_consensus(symbol, mt5_handler)
            breakdown['cls'] = cls_result
            
            if cls_result['consensus'] == 'BUY':
                scores['cls'] = cls_result['confidence'] * 0.3
            elif cls_result['consensus'] == 'SELL':
                scores['cls'] = -cls_result['confidence'] * 0.3
        except Exception as e:
            logger.error(f"Error in CLS analysis: {str(e)}")
            breakdown['cls'] = None
        
        # 2. Technical Analysis Score (25% weight)
        try:
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
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            breakdown['trend'] = 'NEUTRAL'
        
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
    
    def get_signal_strength(self, fusion_result: Dict) -> str:
        """
        Classify signal strength
        
        Returns:
            'VERY_STRONG', 'STRONG', 'MODERATE', 'WEAK'
        """
        confidence = fusion_result['confidence']
        
        if confidence >= 0.85:
            return 'VERY_STRONG'
        elif confidence >= 0.70:
            return 'STRONG'
        elif confidence >= 0.55:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def get_risk_adjustment(self, fusion_result: Dict) -> float:
        """
        Get risk adjustment multiplier based on signal quality
        
        Returns:
            Multiplier (0.5 to 1.5)
        """
        strength = self.get_signal_strength(fusion_result)
        
        adjustments = {
            'VERY_STRONG': 1.5,
            'STRONG': 1.2,
            'MODERATE': 1.0,
            'WEAK': 0.7
        }
        
        return adjustments.get(strength, 1.0)
    
    def format_analysis_report(self, fusion_result: Dict) -> str:
        """Generate human-readable analysis report"""
        action = fusion_result['final_action']
        confidence = fusion_result['confidence']
        meta_score = fusion_result['meta_score']
        strength = self.get_signal_strength(fusion_result)
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║              TREND FUSION ANALYSIS REPORT                 ║
╚═══════════════════════════════════════════════════════════╝

🎯 FINAL DECISION: {action}
📊 Confidence: {confidence:.1%}
⚡ Signal Strength: {strength}
📈 Meta Score: {meta_score:+.3f}

COMPONENT BREAKDOWN:
"""
        
        components = fusion_result['components']
        for component, weight in components.items():
            bar_length = int(abs(weight) * 50)
            bar = '█' * bar_length
            sign = '+' if weight >= 0 else ''
            report += f"  {component:20s}: {sign}{weight:+.3f} {bar}\n"
        
        report += f"\n💡 REASON: {fusion_result['reason']}\n"
        
        # Add breakdown details
        breakdown = fusion_result.get('breakdown', {})
        
        if breakdown.get('cls'):
            cls_data = breakdown['cls']
            report += f"\n📊 CLS Consensus: {cls_data['consensus']} ({cls_data['confidence']:.1%})"
            report += f"\n   Votes - BUY: {cls_data['votes']['buy']}, SELL: {cls_data['votes']['sell']}, HOLD: {cls_data['votes']['hold']}"
        
        if breakdown.get('trend'):
            report += f"\n📈 Technical Trend: {breakdown['trend']}"
        
        if breakdown.get('best_strategy'):
            strat = breakdown['best_strategy']
            report += f"\n🎲 Best Strategy: {strat['risk_level']} - {strat['action']} ({strat['confidence']:.1%})"
        
        report += "\n" + "═" * 60
        
        return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    from models.cls_predictor import CLSPredictor
    from strategies.strategy_manager import StrategyManager
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        cls = CLSPredictor()
        fusion = TrendFusion(cls)
        
        # Get strategy signals
        strategy_mgr = StrategyManager(mt5)
        strategy_signals = strategy_mgr.analyze_all("XAUUSD")
        
        # Run fusion analysis
        fusion_result = fusion.analyze("XAUUSD", mt5, strategy_signals)
        
        # Print report
        print(fusion.format_analysis_report(fusion_result))
        
        # Check if should trade
        if fusion.should_enter_trade(fusion_result):
            print(f"\n✅ TRADE RECOMMENDED")
            print(f"Risk Adjustment: {fusion.get_risk_adjustment(fusion_result):.2f}x")
        else:
            print(f"\n❌ NO TRADE")
        
        mt5.shutdown()

