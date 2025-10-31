"""
Strategy Manager - Coordinates multiple trading strategies
Analyzes all strategies and provides consensus signals
"""

import logging
from typing import Dict, List, Optional
from strategies.counter_trend import CounterTrendStrategy
from strategies.breakout import BreakoutStrategy
from strategies.fibonacci_atr import FibonacciATRStrategy
from strategies.mean_reversion import MeanReversionStrategy  # 🔥 NEW: Mean reversion strategy

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages and coordinates multiple trading strategies"""
    
    def __init__(self, mt5_handler, enabled_strategies: Optional[List[str]] = None):
        """
        Initialize strategy manager
        
        Args:
            mt5_handler: MT5 connection handler
            enabled_strategies: List of strategy names to enable (None = all)
        """
        self.mt5 = mt5_handler
        
        # Initialize all strategies
        self.strategies = {
            'counter_trend': CounterTrendStrategy(),
            'breakout': BreakoutStrategy(),
            'fibonacci_atr': FibonacciATRStrategy(),
            'mean_reversion': MeanReversionStrategy()  # 🔥 NEW: Mean reversion strategy
        }
        
        # Filter enabled strategies
        if enabled_strategies:
            self.strategies = {
                k: v for k, v in self.strategies.items() 
                if k in enabled_strategies
            }
        
        logger.info(f"✅ Initialized strategies: {list(self.strategies.keys())}")
    
    def analyze_all(self, symbol: str, timeframe: str = 'M5') -> Dict:
        """
        🔥 IMPROVED: Run all enabled strategies with:
        - Market Regime Filter (ADX-based)
        - Weighted Consensus (risk-adjusted)
        - Confluence Score (signal overlap bonus)
        
        Args:
            symbol: Trading symbol (e.g., XAUUSDm)
            timeframe: Timeframe to analyze
        
        Returns:
            {
                'best_signal': signal dict or None,
                'all_signals': {strategy_name: signal},
                'market_regime': Dict with regime info,
                'consensus': 'BUY', 'SELL', or 'NEUTRAL',
                'buy_count': int,
                'sell_count': int,
                'weighted_score': float,
                'confluence_score': float,
                'agreement_score': float (0-1)
            }
        """
        try:
            df = self.mt5.get_candles(symbol, timeframe, count=200)
            symbol_info = self.mt5.get_symbol_info(symbol)
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return {
                'best_signal': None,
                'all_signals': {},
                'market_regime': None,
                'consensus': 'NEUTRAL',
                'buy_count': 0,
                'sell_count': 0,
                'weighted_score': 0.0,
                'confluence_score': 0.0,
                'agreement_score': 0.0
            }
        
        # 🔥 STEP 1: Detect Market Regime FIRST (most important!)
        df_with_indicators = df.copy()
        temp_strategy = self.strategies[list(self.strategies.keys())[0]]  # Use any strategy
        df_with_indicators = temp_strategy.add_all_indicators(df_with_indicators)
        
        market_regime = temp_strategy.detect_market_regime(df_with_indicators)
        
        logger.info(f"📊 Market Regime: {market_regime['regime']} (ADX: {market_regime['adx']:.1f})")
        logger.info(f"   Suitable strategies: {', '.join(market_regime['suitable_strategies']) if market_regime['suitable_strategies'] else 'None (avoid trading)'}")
        
        # 🔥 STEP 2: Filter strategies based on market regime
        suitable_strategies = market_regime['suitable_strategies']
        
        if market_regime['regime'] == 'VOLATILE':
            logger.warning("⚠️  VOLATILE market detected - skipping all strategies!")
            return {
                'best_signal': None,
                'all_signals': {},
                'market_regime': market_regime,
                'consensus': 'NEUTRAL',
                'buy_count': 0,
                'sell_count': 0,
                'weighted_score': 0.0,
                'confluence_score': 0.0,
                'agreement_score': 0.0
            }
        
        signals = {}
        
        # 🔥 STEP 3: Run only SUITABLE strategies
        for name, strategy in self.strategies.items():
            try:
                # 🔥 NEW: Check if strategy is suitable for current regime
                strategy_name_formatted = strategy.name  # Use strategy's actual name
                
                if suitable_strategies and strategy_name_formatted not in suitable_strategies:
                    logger.info(f"⚪ {name}: Skipped (not suitable for {market_regime['regime']} market)")
                    signals[name] = None
                    continue
                
                df_copy = df.copy()
                df_copy = strategy.add_all_indicators(df_copy)
                signal = strategy.analyze(df_copy, symbol_info)
                signals[name] = signal
                
                if signal:
                    logger.info(
                        f"✅ {name}: {signal['action']} "
                        f"(confidence: {signal['confidence']:.2%}, "
                        f"risk: {signal['risk_level']})"
                    )
                else:
                    logger.info(f"⚪ {name}: No signal")
            
            except Exception as e:
                logger.error(f"❌ Error in {name} strategy: {str(e)}", exc_info=True)
                signals[name] = None
        
        # 🔥 STEP 4: Calculate WEIGHTED consensus
        # Risk weights: LOW=1.5, MEDIUM=1.2, HIGH=1.0, EXTREME=0.8
        risk_weights = {
            'LOW': 1.5,
            'MEDIUM': 1.2,
            'HIGH': 1.0,
            'EXTREME': 0.8
        }
        
        buy_signals = []
        sell_signals = []
        buy_weighted_score = 0.0
        sell_weighted_score = 0.0
        
        for name, signal in signals.items():
            if signal:
                weight = risk_weights.get(signal['risk_level'], 1.0)
                weighted_confidence = signal['confidence'] * weight
                
                if signal['action'] == 'BUY':
                    buy_signals.append(signal)
                    buy_weighted_score += weighted_confidence
                elif signal['action'] == 'SELL':
                    sell_signals.append(signal)
                    sell_weighted_score += weighted_confidence
        
        total_signals = len([s for s in signals.values() if s])
        
        # 🔥 STEP 5: Calculate Confluence Score
        # Bonus if multiple strategies agree on same direction
        confluence_score = 0.0
        
        if len(buy_signals) >= 2:
            # Multiple BUY signals = strong confluence
            confluence_score = min(len(buy_signals) * 0.15, 0.50)  # Cap at 50%
            logger.info(f"   🎯 BUY Confluence: {len(buy_signals)} strategies agree (+{confluence_score:.1%} bonus)")
        
        if len(sell_signals) >= 2:
            # Multiple SELL signals = strong confluence
            confluence_score = min(len(sell_signals) * 0.15, 0.50)  # Cap at 50%
            logger.info(f"   🎯 SELL Confluence: {len(sell_signals)} strategies agree (+{confluence_score:.1%} bonus)")
        
        # Apply confluence bonus to weighted scores
        if buy_signals:
            buy_weighted_score *= (1 + confluence_score)
        if sell_signals:
            sell_weighted_score *= (1 + confluence_score)
        
        # 🔥 STEP 6: Determine consensus based on weighted scores
        if buy_weighted_score > sell_weighted_score and buy_signals:
            consensus = 'BUY'
            agreement_score = buy_weighted_score / max(buy_weighted_score + sell_weighted_score, 0.01)
            weighted_score = buy_weighted_score
        elif sell_weighted_score > buy_weighted_score and sell_signals:
            consensus = 'SELL'
            agreement_score = sell_weighted_score / max(buy_weighted_score + sell_weighted_score, 0.01)
            weighted_score = sell_weighted_score
        else:
            consensus = 'NEUTRAL'
            agreement_score = 0.5
            weighted_score = 0.0
        
        # 🔥 STEP 7: Handle conflicts (strong disagreement)
        if len(buy_signals) >= 2 and len(sell_signals) >= 2:
            logger.warning(f"⚠️  CONFLICT: {len(buy_signals)} BUY vs {len(sell_signals)} SELL signals")
            logger.warning(f"   Scores: BUY={buy_weighted_score:.2f}, SELL={sell_weighted_score:.2f}")
            
            # If scores are very close, stay neutral
            if abs(buy_weighted_score - sell_weighted_score) < 0.2:
                consensus = 'NEUTRAL'
                agreement_score = 0.0
                weighted_score = 0.0
                logger.warning(f"   🚫 NEUTRAL consensus due to conflicting signals")
        
        # 🔥 STEP 8: Select best signal (highest weighted confidence from consensus direction)
        best_signal = None
        if consensus == 'BUY':
            # Select BUY signal with highest weighted confidence
            best_signal = max(
                buy_signals, 
                key=lambda s: s['confidence'] * risk_weights.get(s['risk_level'], 1.0)
            ) if buy_signals else None
        elif consensus == 'SELL':
            # Select SELL signal with highest weighted confidence
            best_signal = max(
                sell_signals, 
                key=lambda s: s['confidence'] * risk_weights.get(s['risk_level'], 1.0)
            ) if sell_signals else None
        
        logger.info(f"📊 Final Consensus: {consensus} (weighted score: {weighted_score:.2f}, confluence: {confluence_score:.1%})")
        
        return {
            'best_signal': best_signal,
            'all_signals': signals,
            'market_regime': market_regime,
            'consensus': consensus,
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'weighted_score': weighted_score,
            'confluence_score': confluence_score,
            'agreement_score': agreement_score,
            'total_signals': total_signals
        }
    
    def get_strategy_performance(self, strategy_name: str) -> Dict:
        """
        Get performance metrics for a specific strategy
        
        Note: This connects to performance tracker
        For now returns placeholder data
        """
        # TODO: Integrate with performance tracker
        return {
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'total_trades': 0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0
        }
    
    def get_all_strategies_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all strategies"""
        performance = {}
        for strategy_name in self.strategies.keys():
            performance[strategy_name] = self.get_strategy_performance(strategy_name)
        return performance
    
    def enable_strategy(self, strategy_name: str):
        """Enable a specific strategy"""
        if strategy_name == 'counter_trend':
            self.strategies[strategy_name] = CounterTrendStrategy()
        elif strategy_name == 'breakout':
            self.strategies[strategy_name] = BreakoutStrategy()
        elif strategy_name == 'fibonacci_atr':
            self.strategies[strategy_name] = FibonacciATRStrategy()
        elif strategy_name == 'mean_reversion':
            self.strategies[strategy_name] = MeanReversionStrategy()
        else:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return
        
        logger.info(f"✅ Enabled strategy: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a specific strategy"""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"❌ Disabled strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
    
    def format_analysis_report(self, analysis_result: Dict) -> str:
        """🔥 IMPROVED: Generate formatted analysis report with market regime info"""
        consensus = analysis_result['consensus']
        buy_count = analysis_result['buy_count']
        sell_count = analysis_result['sell_count']
        agreement = analysis_result['agreement_score']
        best_signal = analysis_result['best_signal']
        market_regime = analysis_result.get('market_regime')
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║           STRATEGY MANAGER ANALYSIS REPORT                ║
╚═══════════════════════════════════════════════════════════╝
"""
        
        # 🔥 NEW: Market Regime Section
        if market_regime:
            report += f"""
🌐 MARKET REGIME: {market_regime['regime']}
   ADX: {market_regime['adx']:.1f}
   Trend Direction: {market_regime['trend_direction']}
   Suitable Strategies: {', '.join(market_regime['suitable_strategies']) if market_regime['suitable_strategies'] else 'None (avoid trading)'}
   Confidence: {market_regime['confidence']:.1%}

"""
        
        # Consensus Section
        report += f"""📊 CONSENSUS: {consensus}
🎯 Agreement Score: {agreement:.1%}
"""
        
        # 🔥 NEW: Weighted score and confluence
        if 'weighted_score' in analysis_result:
            report += f"""⚖️  Weighted Score: {analysis_result['weighted_score']:.2f}
🎯 Confluence Bonus: {analysis_result.get('confluence_score', 0):.1%}
"""
        
        report += f"""📈 Buy Signals: {buy_count}
📉 Sell Signals: {sell_count}
💡 Total Signals: {analysis_result['total_signals']}

"""
        
        if best_signal:
            report += f"""BEST SIGNAL:
  Action: {best_signal['action']}
  Entry: {best_signal['entry_price']}
  Stop Loss: {best_signal['stop_loss']}
  Take Profit: {best_signal['take_profit']}
  Confidence: {best_signal['confidence']:.1%}
  Risk Level: {best_signal['risk_level']}
  Reason: {best_signal['reason']}

"""
        
        report += "INDIVIDUAL STRATEGY RESULTS:\n"
        all_signals = analysis_result['all_signals']
        
        for strategy_name, signal in all_signals.items():
            if signal:
                report += f"  ✅ {strategy_name.upper()}: {signal['action']} ({signal['confidence']:.1%})\n"
            else:
                report += f"  ⚪ {strategy_name.upper()}: No Signal\n"
        
        report += "\n" + "═" * 60
        
        return report


if __name__ == "__main__":
    # Test strategy manager
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    
    if mt5.initialize():
        # Initialize manager with all strategies
        manager = StrategyManager(mt5)
        
        # Analyze XAUUSDm on M5 timeframe
        print("\n" + "="*70)
        print("ANALYZING XAUUSDm M5")
        print("="*70)
        
        result = manager.analyze_all("XAUUSDm", "M5")
        
        # Print formatted report
        print(manager.format_analysis_report(result))
        
        # Test with only specific strategies
        print("\n" + "="*70)
        print("ANALYZING WITH ONLY BREAKOUT & FIBONACCI STRATEGIES")
        print("="*70)
        
        manager2 = StrategyManager(mt5, enabled_strategies=['breakout', 'fibonacci_atr'])
        result2 = manager2.analyze_all("XAUUSDm", "H1")
        print(manager2.format_analysis_report(result2))
        
        # Test strategy enable/disable
        print("\n" + "="*70)
        print("TESTING ENABLE/DISABLE")
        print("="*70)
        
        manager.disable_strategy('counter_trend')
        print(f"Active strategies: {list(manager.strategies.keys())}")
        
        manager.enable_strategy('counter_trend')
        print(f"Active strategies: {list(manager.strategies.keys())}")
        
        mt5.shutdown()

