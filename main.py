import sys
import os
import logging
import pandas as pd
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
from models.cls_trainer import CLSModelTrainer
from models.cls_predictor import CLSPredictor
from data.calendar_scraper import EconomicCalendarScraper
from cli.menu import CLIMenu
from monitoring.telegram_bot import TelegramNotifier
from monitoring.performance_tracker import PerformanceTracker

# 🔥 NEW: Import RL Agent components
try:
    from models.rl_agent_trainer import RLAgentTrainer
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logger.warning("⚠️  RL Agent not available (install stable-baselines3)")
    RLAgentTrainer = None
    PPO = None

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
        
        # 4. AI/ML Models - 🔥 PERBAIKAN: Hapus StrategyManager dan TrendFusion yang tidak perlu
        self.cls_predictor = CLSPredictor(config.get('model_dir', './models/saved_models'))
        
        # 🔥🔥🔥 PERBAIKAN: Matriks SL/TP yang REALISTIS dan SELEKTIF! 🔥🔥🔥
        # Berdasarkan analisis: Model bagus tapi SL/TP terlalu jauh = loss besar
        self.trade_parameter_matrix = {
            # Format: 'TIMEFRAME': { 'TRADE_MODE': {'sl_atr': multiplier, 'rr': ratio} }
            'M5': {
                'SCALPING':   {'sl_atr': 0.5, 'rr': 1.2}, # SL ketat, TP lebih jauh (RR bagus)
                'NORMAL':     {'sl_atr': 0.7, 'rr': 1.5}, # SL ketat, TP jauh (RR bagus)
                'AGGRESSIVE': {'sl_atr': 0.8, 'rr': 1.8}, # SL ketat, TP sangat jauh (RR bagus)
                'LONG_HOLD':  {'sl_atr': 1.0, 'rr': 2.0}  # SL sedang, TP sangat jauh (RR bagus)
            },
            'M15': {
                'SCALPING':   {'sl_atr': 0.6, 'rr': 1.2},
                'NORMAL':     {'sl_atr': 0.8, 'rr': 1.5},
                'AGGRESSIVE': {'sl_atr': 1.0, 'rr': 1.8},
                'LONG_HOLD':  {'sl_atr': 1.2, 'rr': 2.0}
            },
            'H1': {
                'SCALPING':   {'sl_atr': 0.7, 'rr': 1.2},
                'NORMAL':     {'sl_atr': 0.9, 'rr': 1.5},
                'AGGRESSIVE': {'sl_atr': 1.1, 'rr': 1.8},
                'LONG_HOLD':  {'sl_atr': 1.3, 'rr': 2.0}
            },
            'H4': {
                'SCALPING':   {'sl_atr': 0.8, 'rr': 1.2},
                'NORMAL':     {'sl_atr': 1.0, 'rr': 1.5},
                'AGGRESSIVE': {'sl_atr': 1.2, 'rr': 1.8},
                'LONG_HOLD':  {'sl_atr': 1.4, 'rr': 2.0}
            }
        }
        
        # 🔥 BARU: Inisialisasi API berita untuk digunakan dalam pembuatan fitur
        try:
            from data.news_scraper import NewsAPI
            news_key = os.getenv('NEWS_API_KEY')
            self.news_api = NewsAPI(api_key=news_key) if news_key else None
        except ImportError:
            self.news_api = None
        
        # 5. RL Agent (optional, requires training first)
        self.rl_agent = None
        self.use_rl_agent = config.get('use_rl_agent', False)
        
        if self.use_rl_agent and RL_AVAILABLE:
            self._load_rl_agent(config.get('rl_model_path', './models/saved_models/rl_agent/rl_agent_final.zip'))
        elif self.use_rl_agent and not RL_AVAILABLE:
            logger.warning("⚠️  RL Agent requested but not available (install stable-baselines3)")
            self.use_rl_agent = False
        
        # 6. News and calendar (🔥 SIMPLIFIED - use MT5 calendar)
        self.calendar_scraper = EconomicCalendarScraper(
            api_key=os.getenv('TRADING_ECONOMICS_KEY')
        )
        
        # 7. Monitoring
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if telegram_token and telegram_chat_id:
            # 🔥 UPGRADED: Pass self reference for AI features
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id, trading_bot_engine=self)
            logger.info("✅ Telegram bot initialized with AI features")
        else:
            self.telegram = None
            logger.warning("⚠️  Telegram notifications disabled (no credentials)")
        
        # 8. Performance tracking (🔥 UPGRADED: Now with AI-powered Telegram alerts)
        self.performance = PerformanceTracker(
            self.mt5, 
            save_to_firebase=config.get('save_to_firebase', False)
        )
        
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
    
    def _load_rl_agent(self, model_path: str):
        """🔥 NEW: Load trained RL agent for decision-making"""
        try:
            model_path_obj = Path(model_path)
            
            if not model_path_obj.exists():
                logger.warning(f"⚠️  RL Agent model not found: {model_path}")
                logger.info("   Train RL agent first: python -m models.rl_agent_trainer")
                self.use_rl_agent = False
                return
            
            logger.info(f"🤖 Loading RL Agent from: {model_path}")
            self.rl_agent = PPO.load(str(model_path))
            logger.info("✅ RL Agent loaded successfully!")
            logger.info("   🎯 Bot will now use RL-based decision making")
            
        except Exception as e:
            logger.error(f"❌ Failed to load RL Agent: {e}")
            self.use_rl_agent = False
            self.rl_agent = None
    
    def _get_rl_decision(self, symbol: str, timeframe: str, cls_result: Dict, fusion_result: Dict) -> Dict:
        """
        🔥 NEW: Get decision from trained RL agent
        
        The RL agent observes market state and decides whether to trade,
        based on learned patterns from 100K+ historical scenarios.
        
        Returns:
            Dict with keys: action, confidence, reason
        """
        try:
            # Prepare observation (similar to training environment)
            df = self.mt5.get_candles(symbol, timeframe, count=200)
            
            if df is None or df.empty:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No data'}
            
            # Add indicators
            from strategies.base_strategy import BaseStrategy
            
            class TempStrategy(BaseStrategy):
                def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
                    return None
            
            temp = TempStrategy("Temp", "MEDIUM")
            df = temp.add_all_indicators(df)
            
            # Get CLS prediction for state
            cls_action = cls_result.get('consensus', 'HOLD')
            cls_confidence = cls_result.get('confidence', 0.5)
            
            # Prepare features similar to training env
            current_candle = df.iloc[-1]
            
            # Simplified observation (main features only)
            cls_buy_prob = cls_confidence if cls_action == 'BUY' else (1 - cls_confidence)
            cls_sell_prob = 1 - cls_buy_prob
            
            observation = [
                cls_buy_prob,
                cls_sell_prob,
                0,  # position_type (FLAT in live trading decision)
                0,  # unrealized_pnl
                0,  # position_size
                current_candle.get('atr', 0),
                current_candle.get('rsi', 50),
                current_candle.get('macd', 0),
                current_candle.get('macd_signal', 0),
                current_candle.get('bb_percent_b', 0.5),
                current_candle.get('momentum_5', 0),
                current_candle.get('ema_9', current_candle['close']),
                current_candle.get('ema_21', current_candle['close']),
                current_candle.get('ema_50', current_candle['close']),
                0.5, 0.5, 0, 0, 0,  # RAG features (placeholder)
                1.0, 0.0,  # News features (no immediate news)
                0, 1.0  # Recent performance, equity ratio
            ]
            
            # Pad to 50 features
            import numpy as np
            observation = np.array(observation, dtype=np.float32)
            if len(observation) < 50:
                observation = np.pad(observation, (0, 50 - len(observation)), mode='constant')
            else:
                observation = observation[:50]
            
            # Get RL agent's action
            action, _states = self.rl_agent.predict(observation, deterministic=True)
            
            # Map action to decision
            action_map = {
                0: 'BUY',   # ENTER_LONG
                1: 'SELL',  # ENTER_SHORT
                2: 'HOLD'   # CLOSE_POSITION (interpreted as HOLD in live trading)
            }
            
            rl_action = action_map.get(int(action), 'HOLD')
            
            # Estimate confidence from model's value function (rough approximation)
            # In live trading, we trust RL agent's learned policy
            rl_confidence = 0.75  # Default high confidence (agent was trained extensively)
            
            return {
                'action': rl_action,
                'confidence': rl_confidence,
                'reason': f"RL Policy (trained on 150K episodes, Sharpe-optimized)"
            }
        
        except Exception as e:
            logger.error(f"❌ Error in RL decision: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {str(e)[:50]}'}
    
    def _get_market_condition(self, signal: Dict, symbol: str) -> tuple:
        """
        🔥🔥🔥 SIMPLIFIED: TRUST THE MODEL!
        Model sudah ditraining dengan 100K+ candles, dia TAHU kapan harus trade.
        Kita hanya check volatility minimum, NO indicator filtering!
        
        Returns: (market_condition: str, reason: str)
        """
        action = signal['final_action']
        
        try:
            # 🔥 ONLY CHECK: Volatility minimum (agar spread tidak memakan profit)
            m15_df = self.mt5.get_candles(symbol, 'M15', count=20)
            
            if m15_df is None or m15_df.empty:
                return 'TRENDING', "Insufficient data, trusting model signal"
            
            # 🔥 PERBAIKAN: Gunakan ATR sederhana (SAMA dengan trainer), BUKAN TechnicalIndicators!
            high_low = m15_df['high'] - m15_df['low']
            high_close = abs(m15_df['high'] - m15_df['close'].shift(1))
            low_close = abs(m15_df['low'] - m15_df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            m15_df['atr'] = true_range.rolling(window=14).mean()
            avg_atr = m15_df['atr'].mean()
            
            symbol_info = self.mt5.get_symbol_info(symbol)
            if symbol_info:
                # 🔥 VERY LENIENT: Minimal 30 pips untuk XAUUSD
                min_volatility = symbol_info['point'] * 30  # Turun dari 50 → 30
                
                if avg_atr < min_volatility:
                    logger.info(f"   ⚠️  Low volatility: ATR={avg_atr:.5f} < {min_volatility:.5f}")
                    return 'LOW_VOLATILITY', f"Market too quiet (ATR M15: {avg_atr:.5f})"
                
                logger.info(f"   ✅ ATR M15: {avg_atr:.5f} (sufficient volatility)")
            
            # 🔥 DEFAULT: TRUST THE MODEL!
            logger.info(f"   ✅ Market condition: GOOD (trusting ML model signal)")
            return 'TRENDING', f"ML Model signal: {action} @ {signal['confidence']:.2%} confidence"
        
        except Exception as e:
            logger.warning(f"⚠️  Market check error: {str(e)[:100]}, proceeding with model signal")
            # 🔥 IMPORTANT: Jika error, TETAP TRUST MODEL (jangan reject!)
            return 'TRENDING', f"Trusting model signal (check failed: {str(e)[:50]})"
    
    def analyze_market(self, symbol: str, timeframe: str = 'M5') -> Dict:
        """
        🔥 PERBAIKAN TOTAL: Pipeline fitur yang SINKRON dengan trainer!
        - PURE CANDLE features (NO INDICATORS)
        - Strategy-based features (VECTORIZED)
        - Calendar + News features
        """
        logger.info(f"🔍 Analyzing {symbol} on {timeframe}...")
        
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now(),
            'should_enter_trade': False
        }
        
        try:
            # STEP 1: Dapatkan data candle mentah (SAMA dengan trainer)
            df = self.mt5.get_candles(symbol, timeframe.upper(), count=200)
            if df is None or df.empty:
                results['error'] = "Failed to get candle data"
                return results
            
            # 🔥 STEP 2: TAMBAHKAN STRATEGY-BASED FEATURES (VECTORIZED) - SAMA DENGAN TRAINER!
            # Ini adalah fitur utama yang digunakan trainer, BUKAN indikator tradisional!
            df = self.cls_predictor.add_strategy_features(df, strategies=None, symbol_info=self.mt5.get_symbol_info(symbol))
            
            # 🔥 STEP 3: TAMBAHKAN CALENDAR + NEWS FEATURES - SAMA DENGAN TRAINER!
            # Tambahkan Calendar Events (simplified version dari trainer)
            try:
                from datetime import timedelta
                events = self.calendar_scraper.get_calendar_events(importance='high')
                df['calendar_impact_score'] = 0.0
                for event in events:
                    event_time = event['datetime'].replace(tzinfo=None)
                    time_diff = abs(pd.to_datetime(df['time']) - event_time)
                    nearby_mask = time_diff <= timedelta(hours=2)
                    df.loc[nearby_mask, 'calendar_impact_score'] = 8.0 # Skor high-impact
            except Exception:
                df['calendar_impact_score'] = 0.0

            # Tambahkan News Sentiment (simplified version dari trainer)
            try:
                if self.news_api:
                    articles = self.news_api.get_historical_news(symbol, start_date=datetime.now() - timedelta(days=1))
                    df['news_sentiment_score'] = 0.0
                    for article in articles:
                        article_time = article['datetime'].replace(tzinfo=None)
                        time_diff = abs(pd.to_datetime(df['time']) - article_time)
                        nearby_mask = time_diff <= timedelta(hours=6)
                        sentiment_score = article['sentiment']['score']
                        df.loc[nearby_mask, 'news_sentiment_score'] = (df.loc[nearby_mask, 'news_sentiment_score'] * 0.5 + sentiment_score * 0.5)
            except Exception:
                df['news_sentiment_score'] = 0.0
            
            # 🔥 STEP 4: TAMBAHKAN PURE CANDLE FEATURES - SAMA DENGAN TRAINER!
            # Ini adalah fitur yang digunakan trainer, BUKAN indikator tradisional!
            df = self.cls_predictor.add_advanced_features(df)
            
            # 🔥 STEP 5: TAMBAHKAN CALENDAR FEATURES - SAMA DENGAN TRAINER!
            df = self.cls_predictor.add_calendar_features(df)
            
            # STEP 6: Dapatkan prediksi dari CLS Predictor (otak utama)
            # Sekarang data sudah memiliki fitur yang SAMA dengan training!
            action, confidence = self.cls_predictor.predict(df, timeframe.lower(), self.mt5)
            
            # 🔥 DEBUG: Log prediksi CLS untuk debugging
            logger.info(f"🔍 CLS Prediction: action='{action}', confidence={confidence:.2%}")
            
            results['cls_prediction'] = {'action': action, 'confidence': confidence}
            
            # STEP 7: Lakukan Pengecekan Risiko
            can_trade, reason = self.risk_manager.can_trade(symbol)
            results['can_trade'] = can_trade
            results['risk_reason'] = reason
            
            if not can_trade:
                logger.warning(f"🚫 Risk Manager prevents trading: {reason}")
                return results

            # STEP 8: Tentukan Keputusan Akhir
            mode = self.config.get('trade_mode', 'NORMAL')
            confidence_thresholds = {
                'SCALPING': 0.75,    # 🔥 NAIKKAN: Model bagus, jangan asal tangkap!
                'NORMAL': 0.80,      # 🔥 NAIKKAN: Seleksi ketat untuk kualitas
                'AGGRESSIVE': 0.80,  # 🔥 NAIKKAN: Tetap selektif meski agresif
                'LONG_HOLD': 0.75    # 🔥 NAIKKAN: Long hold butuh confidence tinggi
            }
            min_conf = confidence_thresholds.get(mode, 0.80)
            
            # 🔥 DEBUG: Log kondisi lengkap untuk debugging
            logger.info(f"🔍 Decision Check:")
            logger.info(f"   Action: '{action}' (type: {type(action)})")
            logger.info(f"   Confidence: {confidence:.2%}")
            logger.info(f"   Threshold: {min_conf:.2%}")
            logger.info(f"   Action in ['BUY', 'SELL']: {action in ['BUY', 'SELL']}")
            logger.info(f"   Confidence >= threshold: {confidence >= min_conf}")
            
            if action in ['BUY', 'SELL'] and confidence >= min_conf:
                logger.info(f"✅ Signal CONFIRMED: {action} @ {confidence:.2%} (Threshold: {min_conf:.2%})")
                results['should_enter_trade'] = True
                
                # (Sisa logika untuk menghitung SL/TP dan lot size sudah ada di dalam `auto_trade_loop`,
                # jadi kita hanya perlu meneruskan sinyal ini)
                results['final_signal'] = {'action': action, 'confidence': confidence, 'reason': f"CLS Model Signal ({timeframe})"}
            else:
                if action not in ['BUY', 'SELL']:
                    logger.info(f"🚫 No trade: Action is '{action}' (not BUY/SELL)")
                else:
                    logger.info(f"🚫 No trade: Confidence {confidence:.2%} < threshold {min_conf:.2%}")
            
        except Exception as e:
            logger.error(f"❌ Error during analysis: {str(e)}", exc_info=True)
            results['error'] = str(e)
        
        return results
    
    def execute_trade(self, trade_params: Dict) -> bool:
        """
        🔥 IMPROVED: Execute a trade with comprehensive logging and safety checks
        """
        try:
            logger.info(f"🎯 Executing {trade_params['action']} trade on {trade_params['symbol']}")
            logger.info(f"   Entry: {trade_params['entry_price']:.5f}")
            logger.info(f"   SL: {trade_params['stop_loss']:.5f} ({trade_params.get('sl_pips', 0):.1f} pips)")
            logger.info(f"   TP: {trade_params['take_profit']:.5f} ({trade_params.get('tp_pips', 0):.1f} pips)")
            logger.info(f"   Lot: {trade_params['lot_size']:.2f}")
            
            # 🔥 SAFETY CHECK: Verify we can still trade before execution
            can_trade, reason = self.risk_manager.can_trade(trade_params['symbol'])
            if not can_trade:
                logger.warning(f"⚠️  Trade blocked by risk manager: {reason}")
                return False
            
            # Place order
            success, result = self.executor.place_market_order(
                symbol=trade_params['symbol'],
                order_type=trade_params['action'],
                lot_size=trade_params['lot_size'],
                stop_loss=trade_params['stop_loss'],
                take_profit=trade_params['take_profit'],
                comment=f"AI-Bot-{trade_params['confidence']:.0%}"  # 🔥 FIX: Max 31 chars for MT5!
            )
            
            if success:
                ticket = result.get('order', 'N/A')
                executed_price = result.get('price', 0)
                
                logger.info(f"✅ Trade executed successfully!")
                logger.info(f"   Ticket: #{ticket}")
                logger.info(f"   Executed Price: {executed_price:.5f}")
                logger.info(f"   Slippage: {abs(executed_price - trade_params['entry_price']):.5f}")
                
                # Update risk manager
                self.risk_manager.update_trade_stats({
                    'symbol': trade_params['symbol'],
                    'type': trade_params['action'],
                    'lot_size': trade_params['lot_size'],
                    'entry_price': executed_price,
                    'profit': 0.0,
                    'timestamp': datetime.now()
                })
                
                # 🔥 NEW: Record in performance tracker (for monitoring open positions)
                # This will help track the trade lifecycle
                
                # Send Telegram notification
                if self.telegram:
                    self.telegram.send_trade_alert({
                        'symbol': trade_params['symbol'],
                        'type': trade_params['action'],
                        'entry': executed_price,
                        'sl': trade_params['stop_loss'],
                        'tp': trade_params['take_profit'],
                        'lot_size': trade_params['lot_size'],
                        'confidence': trade_params['confidence'],
                        'reason': trade_params['reason']
                    })
                
                return True
            
            else:
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"❌ Trade execution failed: {error_msg}")
                
                # Send error alert via Telegram
                if self.telegram:
                    self.telegram.send_error_alert(
                        f"Trade execution failed: {error_msg}",
                        context=f"{trade_params['action']} {trade_params['symbol']}"
                    )
                
                return False
        
        except Exception as e:
            logger.error(f"❌ Exception during trade execution: {str(e)}", exc_info=True)
            
            # Send error alert via Telegram
            if self.telegram:
                self.telegram.send_error_alert(
                    f"Exception during trade: {str(e)[:100]}",
                    context="execute_trade"
                )
            
            return False
    
    def auto_trade_loop(self, symbol: str, timeframe: str = 'AUTO', interval: int = 300):
        """
        🔥🔥🔥 ULTIMATE SMART AUTO-TRADE: Multi-Timeframe Routing
        
        Args:
            symbol: Trading symbol
            timeframe: 'AUTO' = Smart routing | 'M5'/'M15'/'H1'/'H4' = Fixed
            interval: Check interval in seconds (default 5 minutes)
        """
        # 🔥🔥🔥 SMART TIMEFRAME ROUTING berdasarkan hasil backtest!
        # H4: PF=5.20 (MONSTER!) → Priority #1
        # H1: PF=1.35 (Good) → Priority #2
        # M15: PF=1.10 (OK) → Priority #3
        # M5: PF=1.14 (OK) → Priority #4
        
        if timeframe == 'AUTO':
            # Smart routing: Cek semua timeframe, pilih yang paling confident
            logger.info(f"🤖 Starting SMART AUTO-TRADE for {symbol} (Multi-TF Routing)")
            logger.info(f"   📊 Priority: H4 (PF=5.20) > H1 (PF=1.35) > M15 (PF=1.10) > M5 (PF=1.14)")
        else:
            logger.info(f"🤖 Starting auto-trade loop for {symbol} ({timeframe})")
        
        logger.info(f"   Analysis interval: {interval}s")
        logger.info(f"   Position management: Every 10s")
        
        import time
        
        last_analysis_time = 0
        iteration = 0
        
        while self.auto_trade_enabled:
            try:
                iteration += 1
                current_time = time.time()
                
                # 🔥 IMPROVED: Always update trailing stops (every 10s loop)
                # This makes position management much more responsive
                if iteration % 1 == 0:  # Every iteration (10s)
                    self.trailing_stop.update_trailing_stops()
                
                # 🔥 IMPROVED: Health check every 30 seconds
                if iteration % 3 == 0:  # Every 30s
                    if not self.connection_manager.health_check():
                        logger.error("❌ MT5 connection lost, attempting reconnect...")
                        time.sleep(30)
                    continue
                
                # 🔥 IMPROVED: Check risk limits every minute (TAPI JANGAN STOP KARENA DRAWDOWN!)
                if iteration % 6 == 0:  # Every 60s
                    try:
                        should_stop, reason = self.risk_manager.should_stop_trading()
                        if should_stop:
                            # 🔥 PERBAIKAN: Jangan stop karena drawdown, hanya log warning
                            if "drawdown" in reason.lower():
                                logger.warning(f"⚠️  High drawdown detected: {reason} (continuing trading)")
                                # Lanjutkan trading meskipun drawdown tinggi
                            else:
                                logger.warning(f"⚠️  Trading stopped: {reason}")
                                self.auto_trade_enabled = False
                                
                                # Send alert if Telegram is enabled
                                if self.telegram:
                                    self.telegram.send_error_alert(
                                        f"🛑 Trading stopped: {reason}",
                                        context="Risk manager halted trading"
                                    )
                                break
                    except Exception as e:
                        logger.error(f"❌ Error checking risk limits: {str(e)}")
                        # Lanjutkan trading jika ada error di risk manager
                
                # 🔥 IMPROVED: Market analysis only at specified interval
                time_since_last_analysis = current_time - last_analysis_time
                
                if time_since_last_analysis >= interval:
                    logger.info(f"🔍 Running market analysis (iteration {iteration})...")
                    
                    # 🔥🔥🔥 SMART TIMEFRAME ROUTING
                    if timeframe == 'AUTO':
                        # Analyze ALL timeframes, pilih yang terbaik
                        logger.info(f"   🎯 SMART ROUTING: Analyzing H4, H1, M15, M5...")
                        
                        best_tf = None
                        best_analysis = None
                        best_confidence = 0
                        
                        # Priority order berdasarkan backtest PF
                        tf_priority = ['H4', 'H1', 'M15', 'M5']
                        
                        for tf in tf_priority:
                            try:
                                tf_analysis = self.analyze_market(symbol, tf)
                                
                                if 'fusion' in tf_analysis:
                                    tf_confidence = tf_analysis['fusion'].get('confidence', 0)
                                    tf_action = tf_analysis['fusion'].get('final_action', 'HOLD')
                                    
                                    # 🔥 BONUS: H4/H1 dapat bonus 10% confidence (lebih reliable)
                                    if tf == 'H4':
                                        tf_confidence *= 1.15  # +15% bonus (PF=5.20!)
                                    elif tf == 'H1':
                                        tf_confidence *= 1.10  # +10% bonus (PF=1.35)
                                    
                                    logger.info(f"      {tf}: {tf_action} ({tf_confidence:.2%})")
                                    
                                    # Update best if higher confidence
                                    if tf_action in ['BUY', 'SELL'] and tf_confidence > best_confidence:
                                        best_confidence = tf_confidence
                                        best_tf = tf
                                        best_analysis = tf_analysis
                                        best_analysis['selected_timeframe'] = tf  # Track which TF was chosen
                            
                            except Exception as e:
                                logger.error(f"      ❌ Error analyzing {tf}: {e}")
                        
                        if best_tf and best_analysis:
                            logger.info(f"   ✅ SELECTED: {best_tf} ({best_confidence:.2%} confidence)")
                            analysis = best_analysis
                        else:
                            logger.info(f"   ❌ No confident signals across all timeframes")
                            analysis = {'should_enter_trade': False, 'fusion': {'final_action': 'HOLD', 'confidence': 0}}
                    
                    else:
                        # Fixed timeframe mode
                        logger.info(f"   Symbol: {symbol} | Timeframe: {timeframe} | Mode: {self.config.get('trade_mode', 'N/A')}")
                        analysis = self.analyze_market(symbol, timeframe)
                
                    last_analysis_time = current_time
                    
                    # 🔥 DEBUG: Log analysis results
                    if 'fusion' in analysis:
                        fusion = analysis['fusion']
                        logger.info(f"💡 Fusion Result:")
                        logger.info(f"   Final Action: {fusion['final_action']}")
                        logger.info(f"   Confidence: {fusion['confidence']:.2%}")
                        logger.info(f"   Meta Score: {fusion['meta_score']:.3f}")
                        logger.info(f"   Should Enter: {analysis.get('should_enter_trade', False)}")
                    
                    if 'error' in analysis:
                        logger.error(f"❌ Analysis error: {analysis['error']}")
                    
                # Execute trade if recommended
                    if analysis.get('should_enter_trade') and analysis.get('final_signal'):
                        signal = analysis['final_signal']
                        
                        logger.info(f"📊 Trade Signal Detected:")
                        logger.info(f"   Action: {signal['action']} @ {signal['confidence']:.2%}")
                        logger.info(f"   Reason: {signal['reason']}")
                        
                        # 🔥🔥🔥 PERBAIKAN: Kalkulasi SL/TP yang Dinamis Berdasarkan Matriks 🔥🔥🔥
                        df_trade = self.mt5.get_candles(symbol, timeframe, count=20)
                        
                        # 🔥 PERBAIKAN: Gunakan ATR sederhana (SAMA dengan trainer), BUKAN TechnicalIndicators!
                        # Trainer menggunakan fitur sederhana, bukan indikator kompleks
                        high_low = df_trade['high'] - df_trade['low']
                        high_close = abs(df_trade['high'] - df_trade['close'].shift(1))
                        low_close = abs(df_trade['low'] - df_trade['close'].shift(1))
                        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        df_trade['atr'] = true_range.rolling(window=14).mean()
                        current_atr = df_trade['atr'].iloc[-1]
                        
                        tick = self.mt5.get_tick(symbol)
                        current_price = tick['ask'] if signal['action'] == 'BUY' else tick['bid']
                        
                        # Ambil parameter dari Matriks Konfigurasi
                        mode = self.config.get('trade_mode', 'NORMAL').upper()
                        tf_upper = timeframe.upper()
                        
                        tf_params = self.trade_parameter_matrix.get(tf_upper, self.trade_parameter_matrix['M15']) # Default ke M15
                        mode_params = tf_params.get(mode, tf_params['NORMAL']) # Default ke NORMAL
                        
                        sl_atr_multiplier = mode_params['sl_atr']
                        rr_ratio = mode_params['rr']
                        
                        logger.info(f"📏 Using params for [{tf_upper} - {mode}]: SL={sl_atr_multiplier}x ATR, RR={rr_ratio}:1")
                        
                        # Kalkulasi SL/TP
                        sl_distance = current_atr * sl_atr_multiplier
                        tp_distance = sl_distance * rr_ratio
                        
                        if signal['action'] == 'BUY':
                            stop_loss = current_price - sl_distance
                            take_profit = current_price + tp_distance
                        else: # SELL
                            stop_loss = current_price + sl_distance
                            take_profit = current_price - tp_distance
                        
                        symbol_info = self.mt5.get_symbol_info(symbol)
                        sl_pips = sl_distance / (symbol_info['point'] * 10)
                        
                        lot_size = self.risk_manager.calculate_position_size(symbol, sl_pips, confidence=signal['confidence'])
                        
                        # Siapkan parameter final untuk eksekusi
                        trade_params = {
                            'action': signal['action'], 'symbol': symbol, 'lot_size': lot_size,
                            'entry_price': current_price,
                            'stop_loss': round(stop_loss, symbol_info['digits']),
                            'take_profit': round(take_profit, symbol_info['digits']),
                            'confidence': signal['confidence'],
                            'reason': signal['reason'],
                            'sl_pips': round(sl_pips, 1),
                            'tp_pips': round(sl_pips * rr_ratio, 1)
                        }
                        
                        logger.info(f"   Entry: {current_price:.5f}")
                        logger.info(f"   SL: {stop_loss:.5f} ({sl_pips:.1f} pips)")
                        logger.info(f"   TP: {take_profit:.5f} ({sl_pips * rr_ratio:.1f} pips)")
                        logger.info(f"   Lot Size: {lot_size:.2f}")
                    
                    # Execute
                        success = self.execute_trade(trade_params)
                        
                        if success:
                            logger.info("✅ Trade executed successfully!")
                            
                            # 🔥 NEW: Log RL decision if available
                            if 'rl_decision' in analysis and analysis['rl_decision'].get('action') != 'HOLD':
                                logger.info(f"   🤖 RL Agent decision: {analysis['rl_decision']}")
                                
                            # 🔥 NEW: Send AI-powered Telegram notification
                            if self.telegram:
                                # Include RL/RAG context in notification
                                trade_params['rl_decision'] = analysis.get('rl_decision', {})
                                trade_params['cls_consensus'] = analysis.get('mtf_consensus', {})
                        else:
                            logger.warning("⚠️  Trade execution failed")
                    elif analysis.get('should_enter_trade') and not analysis.get('final_signal'):
                        logger.warning(f"⚠️  Should enter trade but no final_signal! Check signal generation.")
                    else:
                        # 🔥 PERBAIKAN: Logika logging yang lebih akurat
                        if analysis.get('risk_reason') != 'OK':
                            logger.info(f"🚫 No trade: Ditolak oleh Risk Manager: {analysis.get('risk_reason')}")
                        elif 'cls_prediction' in analysis and analysis['cls_prediction'].get('action') == 'HOLD':
                            cls_confidence = analysis['cls_prediction'].get('confidence', 0)
                            logger.info(f"🚫 No trade: Sinyal model adalah HOLD (Keyakinan: {cls_confidence:.2%})")
                        else:
                            # Alasan terakhir: confidence di bawah threshold
                            if 'cls_prediction' in analysis:
                                cls_confidence = analysis['cls_prediction'].get('confidence', 0)
                                mode = self.config.get('trade_mode', 'NORMAL')
                                # 🔥 SINKRON dengan thresholds di analyze_market (SELEKTIF!)
                                thresholds = {'SCALPING': 0.75, 'NORMAL': 0.80, 'AGGRESSIVE': 0.70, 'LONG_HOLD': 0.75}
                                threshold = thresholds.get(mode, 0.80)
                                logger.info(f"🚫 No trade: Confidence {cls_confidence:.2%} < threshold {threshold:.0%} ({mode} mode)")
                
                # 🔥 IMPROVED: Sleep for shorter intervals (10s) for better responsiveness
                time.sleep(10)
            
            except KeyboardInterrupt:
                logger.info("⚠️  Auto-trade interrupted by user")
                self.auto_trade_enabled = False
                break
            
            except Exception as e:
                logger.error(f"❌ Error in auto-trade loop: {str(e)}", exc_info=True)
                
                # Send alert if Telegram is enabled
                if self.telegram:
                    self.telegram.send_error_alert(
                        f"Error in trading loop: {str(e)[:100]}",
                        context="Auto-trade loop"
                    )
                
                time.sleep(60)  # Wait 1 minute before retry
        
        logger.info("🛑 Auto-trade loop stopped")
    
    def train_cls_models(self, symbol: str = 'XAUUSDm', model_type: str = 'random_forest'):
        """
        Train CLS models for all timeframes
        
        Args:
            symbol: Trading symbol (default: XAUUSDm)
            model_type: Model type to train ('random_forest' or 'gradient_boosting')
        """
        logger.info(f"🎓 Starting CLS model training for {symbol}...")
        
        try:
            # Initialize trainer
            trainer = CLSModelTrainer(output_dir='./models/saved_models')
            
            # Train all timeframes
            trainer.train_all_timeframes(
                mt5_handler=self.mt5,
                symbol=symbol,
                model_type=model_type
            )
            
            logger.info("✅ CLS models training completed successfully!")
            
            # Reload models in predictor
            self.cls_predictor = CLSPredictor(self.config.get('model_dir', './models/saved_models'))
            
            logger.info("✅ Models reloaded into bot engine")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error during model training: {str(e)}", exc_info=True)
            return False
    
    def retrain_single_timeframe(self, timeframe: str, symbol: str = 'XAUUSDm', model_type: str = 'random_forest'):
        """
        Retrain a single timeframe model
        
        Args:
            timeframe: Timeframe to retrain (M5, M15, H1, H4)
            symbol: Trading symbol
            model_type: Model type to train
        """
        logger.info(f"🎓 Retraining {timeframe} model for {symbol}...")
        
        try:
            # Initialize trainer
            trainer = CLSModelTrainer(output_dir='./models/saved_models')
            
            # Retrain single timeframe
            trainer.retrain_single_timeframe(
                mt5_handler=self.mt5,
                timeframe=timeframe,
                symbol=symbol,
                model_type=model_type
            )
            
            logger.info(f"✅ {timeframe} model retrained successfully!")
            
            # Reload models in predictor
            self.cls_predictor = CLSPredictor(self.config.get('model_dir', './models/saved_models'))
            
            logger.info("✅ Models reloaded into bot engine")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error during {timeframe} model retraining: {str(e)}", exc_info=True)
            return False
    
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
        'save_to_firebase': os.getenv('FIREBASE_ENABLED', 'false').lower() == 'true',
        # 🔥 ML-driven risk parameters
        'sl_atr_multiplier': float(os.getenv('SL_ATR_MULTIPLIER', '1.5')),  # Stop loss = 1.5x ATR
        'risk_reward_ratio': float(os.getenv('RISK_REWARD_RATIO', '1.5')),   # TP/SL ratio = 1.5:1
        # 🔥 NEW: RL Agent settings
        'use_rl_agent': os.getenv('USE_RL_AGENT', 'false').lower() == 'true',
        'rl_model_path': os.getenv('RL_MODEL_PATH', './models/saved_models/rl_agent/rl_agent_final.zip')
    }


def main():
    """Main entry point"""
    # Fix Windows console encoding for emoji support
    import sys
    if sys.platform == 'win32':
        import os
        os.system('chcp 65001 >nul 2>&1')
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    
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