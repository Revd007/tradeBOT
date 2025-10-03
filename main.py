import sys
import os
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all components
from core.mt5_handler import MT5Handler, ConnectionManager
from core.order_executor import OrderExecutor, TrailingStopManager
from core.risk_manager import RiskManager
from strategies.strategy_manager import StrategyManager
from models.cls_predictor import CLSPredictor
from models.trend_fusion import TrendFusion
from data.news_scraper import NewsAPI
from data.calendar_scraper import EconomicCalendar, NewsFilter
from cli.menu import CLIMenu
from monitoring.telegram_bot import TelegramNotifier
from monitoring.performance_tracker import PerformanceTracker

# Setup logging
from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


class TradingBotEngine:
    """Main trading bot engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.auto_trade_enabled = False
        self.default_lot_size = config.get('default_lot_size', 0.01)
        self.max_slippage = config.get('max_slippage', 2.0)
        
        # Initialize components
        logger.info("🚀 Initializing Trading Bot...")
        
        # 1. MT5 Connection
        self.mt5 = self._init_mt5()
        self.connection_manager = ConnectionManager(self.mt5)
        
        # 2. Order execution
        self.executor = OrderExecutor(self.mt5, self.max_slippage)
        self.trailing_stop = TrailingStopManager(self.executor)
        
        # 3. Risk management
        self.risk_manager = RiskManager(
            self.mt5,
            max_risk_per_trade=config.get('max_risk_per_trade', 1.0),
            max_daily_loss=config.get('max_daily_loss', 5.0),
            max_trades_per_day=config.get('max_trades_per_day', 10),
            max_open_positions=config.get('max_open_positions', 3)
        )
        
        # 4. Strategy manager
        enabled_strategies = config.get('enabled_strategies', ['counter_trend', 'breakout', 'fibonacci_atr'])
        self.strategy_manager = StrategyManager(self.mt5, enabled_strategies)
        
        # 5. AI/ML Models
        self.cls_predictor = CLSPredictor(config.get('model_dir', './models/saved_models'))
        self.trend_fusion = TrendFusion(self.cls_predictor)
        
        # 6. News and calendar
        news_api_key = os.getenv('NEWS_API_KEY')
        calendar_api_key = os.getenv('TRADING_ECONOMICS_KEY')
        self.news_filter = NewsFilter(news_api_key, calendar_api_key)
        
        # 7. Monitoring
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
        else:
            self.telegram = None
            logger.warning("⚠️  Telegram notifications disabled (no credentials)")
        
        # 8. Performance tracking
        self.performance = PerformanceTracker(self.mt5, save_to_firebase=config.get('save_to_firebase', False))
        
        logger.info("✅ Trading Bot initialized successfully!")
    
    def _init_mt5(self) -> MT5Handler:
        """Initialize MT5 connection"""
        account_type = os.getenv('ACCOUNT_TYPE', 'DEMO')
        
        if account_type == 'EXNESS':
            login = int(os.getenv('MT5_LOGIN_EXNESS'))
            password = os.getenv('MT5_PASSWORD_EXNESS')
            server = os.getenv('MT5_SERVER_EXNESS')
        else:
            login = int(os.getenv('MT5_LOGIN_DEMO'))
            password = os.getenv('MT5_PASSWORD_DEMO')
            server = os.getenv('MT5_SERVER_DEMO')
        
        mt5 = MT5Handler(login, password, server)
        
        if not mt5.initialize():
            logger.error("❌ Failed to connect to MT5")
            raise Exception("MT5 connection failed")
        
        return mt5
    
    def analyze_market(self, symbol: str, timeframe: str = 'M5') -> Dict:
        """
        Comprehensive market analysis
        
        Returns complete analysis including:
        - Strategy signals
        - CLS predictions
        - Trend fusion
        - News context
        - Risk assessment
        """
        logger.info(f"🔍 Analyzing {symbol} on {timeframe}...")
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now()
        }
        
        try:
            # 1. Check news/calendar filter
            news_context = self.news_filter.get_trading_context(symbol)
            results['news_context'] = news_context
            
            if not news_context['can_trade']:
                logger.warning(f"⚠️  Trading paused: {news_context['reason']}")
                results['can_trade'] = False
                return results
            
            # 2. Strategy analysis
            strategy_result = self.strategy_manager.analyze_all(symbol, timeframe)
            results['all_strategies'] = strategy_result['all_signals']
            results['best_signal'] = strategy_result['best_signal']
            results['strategy_consensus'] = strategy_result['consensus']
            
            # 3. CLS multi-timeframe prediction
            cls_result = self.cls_predictor.multi_timeframe_consensus(symbol, self.mt5)
            results['mtf_consensus'] = cls_result
            
            # 4. Trend Fusion (meta-analysis)
            fusion_result = self.trend_fusion.analyze(
                symbol, self.mt5, strategy_result, news_context
            )
            results['fusion'] = fusion_result
            
            # 5. Risk assessment
            can_trade, reason = self.risk_manager.can_trade(symbol)
            results['can_trade'] = can_trade
            results['risk_reason'] = reason
            
            if can_trade:
                risk_metrics = self.risk_manager.get_risk_metrics()
                results['risk_metrics'] = risk_metrics
            
            # 6. Determine if should enter trade
            should_enter = (
                can_trade and
                news_context['can_trade'] and
                self.trend_fusion.should_enter_trade(fusion_result, min_confidence=0.65)
            )
            
            results['should_enter_trade'] = should_enter
            
            if should_enter:
                # Calculate position details
                signal = fusion_result
                
                if fusion_result['final_action'] in ['BUY', 'SELL']:
                    # Get best signal for SL/TP
                    best_strat_signal = strategy_result['best_signal']
                    
                    # Calculate position size
                    df = self.mt5.get_candles(symbol, timeframe, count=50)
                    from strategies.base_strategy import TechnicalIndicators
                    atr = TechnicalIndicators.calculate_atr(df).iloc[-1]
                    
                    stop_loss_pips = abs(best_strat_signal['entry_price'] - best_strat_signal['stop_loss']) / 0.1
                    
                    lot_size = self.risk_manager.calculate_position_size(
                        symbol,
                        stop_loss_pips,
                        confidence=fusion_result['confidence'],
                        news_impact=news_context['impact_score'],
                        trade_mode=self.config.get('trade_mode', 'NORMAL')
                    )
                    
                    results['recommended_trade'] = {
                        'action': fusion_result['final_action'],
                        'symbol': symbol,
                        'lot_size': lot_size,
                        'entry_price': best_strat_signal['entry_price'],
                        'stop_loss': best_strat_signal['stop_loss'],
                        'take_profit': best_strat_signal['take_profit'],
                        'confidence': fusion_result['confidence'],
                        'reason': fusion_result['reason']
                    }
            
            logger.info(f"✅ Analysis complete: {fusion_result['final_action']} (confidence: {fusion_result['confidence']:.2%})")
            
        except Exception as e:
            logger.error(f"❌ Error during analysis: {str(e)}", exc_info=True)
            results['error'] = str(e)
        
        return results
    
    def execute_trade(self, trade_params: Dict) -> bool:
        """Execute a trade based on analysis"""
        try:
            logger.info(f"🎯 Executing {trade_params['action']} trade on {trade_params['symbol']}")
            
            # Place order
            success, result = self.executor.place_market_order(
                symbol=trade_params['symbol'],
                order_type=trade_params['action'],
                lot_size=trade_params['lot_size'],
                stop_loss=trade_params['stop_loss'],
                take_profit=trade_params['take_profit'],
                comment=f"AI-Bot: {trade_params['reason'][:50]}"
            )
            
            if success:
                logger.info(f"✅ Trade executed successfully! Ticket: {result['order']}")
                
                # Update risk manager
                self.risk_manager.update_trade_stats({
                    'symbol': trade_params['symbol'],
                    'type': trade_params['action'],
                    'lot_size': trade_params['lot_size'],
                    'entry_price': result['price'],
                    'timestamp': datetime.now()
                })
                
                # Send Telegram notification
                if self.telegram:
                    self.telegram.send_trade_alert({
                        'symbol': trade_params['symbol'],
                        'type': trade_params['action'],
                        'entry': result['price'],
                        'sl': trade_params['stop_loss'],
                        'tp': trade_params['take_profit'],
                        'confidence': trade_params['confidence'],
                        'reason': trade_params['reason']
                    })
                
                return True
            
            else:
                logger.error(f"❌ Trade failed: {result.get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Exception during trade execution: {str(e)}", exc_info=True)
            return False
    
    def auto_trade_loop(self, symbol: str, timeframe: str = 'M5', interval: int = 300):
        """
        Automated trading loop
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            interval: Check interval in seconds (default 5 minutes)
        """
        logger.info(f"🤖 Starting auto-trade loop for {symbol} ({timeframe})")
        logger.info(f"   Check interval: {interval}s")
        
        import time
        
        while self.auto_trade_enabled:
            try:
                # Health check
                if not self.connection_manager.health_check():
                    logger.error("❌ MT5 connection lost")
                    time.sleep(30)
                    continue
                
                # Check if should stop trading
                should_stop, reason = self.risk_manager.should_stop_trading()
                if should_stop:
                    logger.warning(f"⚠️  Trading stopped: {reason}")
                    self.auto_trade_enabled = False
                    break
                
                # Analyze market
                analysis = self.analyze_market(symbol, timeframe)
                
                # Execute trade if recommended
                if analysis.get('should_enter_trade') and analysis.get('recommended_trade'):
                    trade = analysis['recommended_trade']
                    
                    logger.info(f"📊 Trade Signal: {trade['action']} @ {trade['entry_price']}")
                    logger.info(f"   Confidence: {trade['confidence']:.2%}")
                    logger.info(f"   Reason: {trade['reason']}")
                    
                    # Execute
                    self.execute_trade(trade)
                
                # Update trailing stops
                self.trailing_stop.update_trailing_stops()
                
                # Sleep until next check
                logger.info(f"💤 Sleeping for {interval}s...")
                time.sleep(interval)
            
            except KeyboardInterrupt:
                logger.info("⚠️  Auto-trade interrupted by user")
                self.auto_trade_enabled = False
                break
            
            except Exception as e:
                logger.error(f"❌ Error in auto-trade loop: {str(e)}", exc_info=True)
                time.sleep(60)  # Wait 1 minute before retry
        
        logger.info("🛑 Auto-trade loop stopped")
    
    def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("🛑 Shutting down trading bot...")
        
        # Disable auto-trading
        self.auto_trade_enabled = False
        
        # Close all positions (optional - comment out if you want to keep positions open)
        # self.executor.close_all_positions()
        
        # Close MT5 connection
        self.mt5.shutdown()
        
        logger.info("✅ Bot shutdown complete")


def load_config() -> Dict:
    """Load configuration from environment and defaults"""
    return {
        'default_lot_size': float(os.getenv('DEFAULT_LOT_SIZE', '0.01')),
        'max_slippage': float(os.getenv('MAX_SLIPPAGE', '2.0')),
        'max_risk_per_trade': float(os.getenv('RISK_PER_TRADE', '1.0')),
        'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '5.0')),
        'max_trades_per_day': int(os.getenv('MAX_TRADES_PER_DAY', '10')),
        'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', '3')),
        'enabled_strategies': ['counter_trend', 'breakout', 'fibonacci_atr'],
        'model_dir': './models/saved_models',
        'trade_mode': 'NORMAL',
        'save_to_firebase': os.getenv('FIREBASE_ENABLED', 'false').lower() == 'true'
    }


def main():
    """Main entry point"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           🤖 AUTO-TRADING BOT v2.0                      ║
    ║           Advanced AI-Powered Forex Trading System       ║
    ║                                                           ║
    ║  Features:                                                ║
    ║  ✅ Multi-Strategy Analysis (Counter-Trend, Breakout)    ║
    ║  ✅ CLS Machine Learning Predictions                     ║
    ║  ✅ Trend Fusion Meta-Analysis                           ║
    ║  ✅ News & Economic Calendar Integration                 ║
    ║  ✅ Advanced Risk Management                             ║
    ║  ✅ Auto-Reconnection & Error Handling                   ║
    ║  ✅ Telegram Notifications                               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize bot
        bot = TradingBotEngine(config)
        
        # Start CLI
        cli = CLIMenu(bot)
        cli.run()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}", exc_info=True)
        print(f"\n❌ Fatal Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()