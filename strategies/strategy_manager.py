"""
Strategy Manager - Coordinates multiple trading strategies
Analyzes all strategies and provides consensus signals
"""

import logging
from typing import Dict, List, Optional
from strategies.counter_trend import CounterTrendStrategy
from strategies.breakout import BreakoutStrategy
from strategies.fibonacci_atr import FibonacciATRStrategy

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
            'fibonacci_atr': FibonacciATRStrategy()
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
        Run all enabled strategies and return aggregated results
        
        Args:
            symbol: Trading symbol (e.g., XAUUSD)
            timeframe: Timeframe to analyze
        
        Returns:
            {
                'best_signal': signal dict or None,
                'all_signals': {strategy_name: signal},
                'consensus': 'BUY', 'SELL', or 'NEUTRAL',
                'buy_count': int,
                'sell_count': int,
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
                'consensus': 'NEUTRAL',
                'buy_count': 0,
                'sell_count': 0,
                'agreement_score': 0.0
            }
        
        signals = {}
        
        # Run each strategy
        for name, strategy in self.strategies.items():
            try:
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
                logger.error(f"❌ Error in {name} strategy: {str(e)}")
                signals[name] = None
        
        # Calculate consensus
        buy_signals = [s for s in signals.values() if s and s['action'] == 'BUY']
        sell_signals = [s for s in signals.values() if s and s['action'] == 'SELL']
        total_signals = len([s for s in signals.values() if s])
        
        # Determine consensus
        if len(buy_signals) > len(sell_signals) and buy_signals:
            consensus = 'BUY'
            agreement_score = len(buy_signals) / max(total_signals, 1)
        elif len(sell_signals) > len(buy_signals) and sell_signals:
            consensus = 'SELL'
            agreement_score = len(sell_signals) / max(total_signals, 1)
        else:
            consensus = 'NEUTRAL'
            agreement_score = 0.5
        
        # Select best signal (highest confidence from consensus direction)
        best_signal = None
        if consensus == 'BUY':
            best_signal = max(buy_signals, key=lambda s: s['confidence']) if buy_signals else None
        elif consensus == 'SELL':
            best_signal = max(sell_signals, key=lambda s: s['confidence']) if sell_signals else None
        else:
            # If no consensus, take highest confidence overall
            all_valid_signals = [s for s in signals.values() if s]
            if all_valid_signals:
                best_signal = max(all_valid_signals, key=lambda s: s['confidence'])
        
        return {
            'best_signal': best_signal,
            'all_signals': signals,
            'consensus': consensus,
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
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
        """Generate formatted analysis report"""
        consensus = analysis_result['consensus']
        buy_count = analysis_result['buy_count']
        sell_count = analysis_result['sell_count']
        agreement = analysis_result['agreement_score']
        best_signal = analysis_result['best_signal']
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║           STRATEGY MANAGER ANALYSIS REPORT                ║
╚═══════════════════════════════════════════════════════════╝

📊 CONSENSUS: {consensus}
🎯 Agreement Score: {agreement:.1%}
📈 Buy Signals: {buy_count}
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
        
        # Analyze XAUUSD on M5 timeframe
        print("\n" + "="*70)
        print("ANALYZING XAUUSD M5")
        print("="*70)
        
        result = manager.analyze_all("XAUUSD", "M5")
        
        # Print formatted report
        print(manager.format_analysis_report(result))
        
        # Test with only specific strategies
        print("\n" + "="*70)
        print("ANALYZING WITH ONLY BREAKOUT & FIBONACCI STRATEGIES")
        print("="*70)
        
        manager2 = StrategyManager(mt5, enabled_strategies=['breakout', 'fibonacci_atr'])
        result2 = manager2.analyze_all("XAUUSD", "H1")
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

