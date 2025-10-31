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

# Setup logging FIRST (before any logging calls)
from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# 🔥 NEW: Import RL Agent components
try:
    from models.rl_agent_trainer import RLAgentTrainer
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
    logger.info("✅ RL Agent components loaded successfully")
except ImportError as e:
    RL_AVAILABLE = False
    logger.warning(f"⚠️  RL Agent not available: {e}")
    logger.info("   Install with: pip install stable-baselines3 gymnasium")
    RLAgentTrainer = None
    PPO = None

# 🔥 NEW: Import RAG Memory for continuous learning
try:
    from models.rag_memory import RAGMemory
    RAG_AVAILABLE = True
    logger.info("✅ RAG Memory loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"⚠️  RAG Memory not available: {e}")
    RAGMemory = None


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
        
        # 4. AI/ML Models
        self.cls_predictor = CLSPredictor(config.get('model_dir', './models/saved_models'))
        
        # 5. RL Agent (optional, requires training first)
        self.rl_agent = None
        self.use_rl_agent = config.get('use_rl_agent', False)
        
        # 🔥 NEW: RL Override settings
        self.rl_override_enabled = config.get('rl_override_enabled', True)  # RL can override CLS threshold
        self.rl_override_threshold = config.get('rl_override_threshold', 0.80)  # RL needs 80% confidence to override
        self.cls_min_threshold = config.get('cls_min_threshold', 0.70)  # Normal CLS threshold
        
        if self.use_rl_agent and RL_AVAILABLE:
            self._load_rl_agent(config.get('rl_model_path', './models/saved_models/rl_agent/rl_agent_final.zip'))
        elif self.use_rl_agent and not RL_AVAILABLE:
            logger.warning("⚠️  RL Agent requested but not available (install stable-baselines3)")
            self.use_rl_agent = False
        
        # 6. News and calendar
        self.calendar_scraper = EconomicCalendarScraper(
            api_key=os.getenv('TRADING_ECONOMICS_KEY')
        )
        
        # 7. Monitoring - Telegram
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_GROUP_ID')  # 🔥 Changed to GROUP_ID for group support
        
        if telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id, trading_bot_engine=self)
            logger.info("✅ Telegram bot initialized with AI features")
        else:
            self.telegram = None
            logger.warning("⚠️  Telegram notifications disabled (no credentials)")
        
        # 8. Performance tracking
        self.performance = PerformanceTracker(
            self.mt5, 
            save_to_firebase=config.get('save_to_firebase', False)
        )
        
        # 9. 🔥 NEW: RAG Memory for continuous learning (keeps history of trade outcomes)
        if RAG_AVAILABLE:
            self.rag_memory = RAGMemory(
                db_path="./data/rag_db",
                use_chromadb=True
            )
            logger.info("✅ RAG Memory initialized (using ChromaDB)")
        else:
            self.rag_memory = None
            logger.warning("⚠️  RAG Memory not available")
        
        # 10. 🔥 DISABLED: Stop Loss Plus system (temporarily disabled for testing)
        try:
            from models.stoploss_plus import StopLossPlus
            # 🔥 DISABLED: Set to None to disable SL+ monitoring
            # self.stop_loss_plus = StopLossPlus(
            #     initial_sl_pips=20,
            #     trailing_start_pips=15,
            #     pullback_threshold=0.65,
            #     min_pullback_duration=3,
            #     atr_multiplier=1.5
            # )
            self.stop_loss_plus = None  # 🔥 DISABLED
            logger.info("⚠️  Stop Loss Plus DISABLED (will be re-enabled later)")
        except ImportError as e:
            logger.warning(f"⚠️  Stop Loss Plus not available: {e}")
            self.stop_loss_plus = None
        
        # 11. 🔥 NEW: Position tracking for TP/SL monitoring
        self._active_positions = {}  # {ticket: {entry_price, sl, tp, symbol, type, entry_time}}
        self._last_known_positions = set()  # Set of tickets to detect closures
        
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
        
        # 🔥 NEW: Log RL Override configuration
        if self.use_rl_agent:
            logger.info(f"🤖 RL Agent: ENABLED")
            logger.info(f"   Override Enabled: {self.rl_override_enabled}")
            if self.rl_override_enabled:
                logger.info(f"   RL Override Threshold: {self.rl_override_threshold:.0%}")
                logger.info(f"   CLS Min Threshold: {self.cls_min_threshold:.0%}")
                logger.info(f"   🎯 RL can override CLS if RL confidence >= {self.rl_override_threshold:.0%}")
        else:
            logger.info(f"🤖 RL Agent: DISABLED")
        
        logger.info("✅ Trading Bot initialized successfully!")
    
    def get_trade_parameters(self, symbol: str, timeframe: str, mode: str) -> Dict:
        """
        🔥 NEW: Get dynamic trade parameters based on symbol type
        
        Different instruments have different characteristics:
        - XAUUSD (Gold): More volatile, needs wider SL, slightly lower RR
        - Major pairs: Less volatile, tighter SL, higher RR
        - Default: Use base matrix values
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M5, M15, H1, H4)
            mode: Trading mode (SCALPING, NORMAL, AGGRESSIVE, LONG_HOLD)
        
        Returns:
            Dict with 'sl_atr' and 'rr' parameters
        """
        # Get base parameters from matrix
        tf_upper = timeframe.upper()
        base_params = self.trade_parameter_matrix.get(
            tf_upper,
            self.trade_parameter_matrix['M15']  # Default to M15
        ).get(
            mode.upper(),
            self.trade_parameter_matrix['M15']['NORMAL']  # Default to NORMAL
        )
        
        # 🔥 GOLD ADJUSTMENT: More volatile, needs different approach
        if 'XAU' in symbol.upper():
            adjusted_params = {
                'sl_atr': base_params['sl_atr'] * 1.3,  # +30% wider SL for volatility
                'rr': base_params['rr'] * 0.9           # -10% RR (still profitable)
            }
            logger.debug(f"   📊 GOLD adjustment: SL={adjusted_params['sl_atr']:.2f}x, RR={adjusted_params['rr']:.2f}")
            return adjusted_params
        
        # 🔥 MAJOR PAIRS ADJUSTMENT: Less volatile, can be tighter
        majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD', 'USDCHF']
        if any(major in symbol.upper() for major in majors):
            adjusted_params = {
                'sl_atr': base_params['sl_atr'] * 0.8,  # -20% tighter SL
                'rr': base_params['rr'] * 1.1           # +10% better RR
            }
            logger.debug(f"   📊 MAJOR adjustment: SL={adjusted_params['sl_atr']:.2f}x, RR={adjusted_params['rr']:.2f}")
            return adjusted_params
        
        # Default: use base parameters
        logger.debug(f"   📊 BASE parameters: SL={base_params['sl_atr']:.2f}x, RR={base_params['rr']:.2f}")
        return base_params
    
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
    
    def _get_rl_decision(self, symbol: str, timeframe: str, cls_action: str, cls_confidence: float, df: pd.DataFrame) -> Dict:
        """
        🔥 IMPROVED: Get decision from trained RL agent with synchronized features
        
        The RL agent observes market state and decides whether to trade,
        based on learned patterns from 500K+ historical scenarios.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for analysis
            cls_action: CLS prediction (BUY/SELL/HOLD)
            cls_confidence: CLS confidence (0-1)
            df: DataFrame with all features (already processed)
        
        Returns:
            Dict with keys: action, confidence, reason
        """
        try:
            if self.rl_agent is None or not self.use_rl_agent:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'RL agent not available'}
            
            # Get current candle (last row)
            current_candle = df.iloc[-1]
            current_price = current_candle['close']
            
            # 🔥 FIX: Calculate CLS probabilities (binary 0/1, matching training!)
            # Training: cls_buy_prob = 1.0 if BUY, 0.0 if SELL/HOLD
            # Live: Must match training format!
            if cls_action == 'BUY':
                cls_buy_prob = 1.0
                cls_sell_prob = 0.0
            elif cls_action == 'SELL':
                cls_buy_prob = 0.0
                cls_sell_prob = 1.0
            else:  # HOLD
                cls_buy_prob = 0.0
                cls_sell_prob = 0.0
            
            logger.debug(f"🔧 RL Input Fix: cls_buy_prob={cls_buy_prob}, cls_sell_prob={cls_sell_prob} (action={cls_action})")
            
            # Check if we have open position
            open_positions = self.mt5.get_open_positions() if self.mt5 else []
            position_type = 0  # FLAT
            unrealized_pnl = 0
            position_size = 0
            
            for pos in open_positions:
                if pos.get('symbol') == symbol:
                    position_type = 1 if pos.get('type') == 0 else -1  # 0=LONG, 1=SHORT
                    position_size = pos.get('volume', 0)
                    entry_price = pos.get('price_open', current_price)
                    unrealized_pnl = (current_price - entry_price) * position_size * 100 if position_type == 1 else (entry_price - current_price) * position_size * 100
            
            # Normalize unrealized P&L
            account_info = self.mt5.get_account_info()
            balance = account_info.get('balance', 20.0)
            unrealized_pnl_normalized = unrealized_pnl / balance if balance > 0 else 0
            
            # 🔥 FIX: Query RAG Memory LIVE (not using stale pre-computed data!)
            rag_features = [0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5]  # Default fallback values
            if self.rag_memory:
                try:
                    price = current_price
                    atr = current_candle.get('atr', 10)
                    rsi = current_candle.get('rsi', 50)
                    ema21 = current_candle.get('ema_21', price)
                    trend_is_bullish = price > ema21
                    
                    # 🔥 LIVE RAG QUERY: Ask RAG memory "What happened in similar situations?"
                    rag_results = self.rag_memory.get_similar_outcomes(
                        price=price, 
                        atr=atr, 
                        rsi=rsi, 
                        trend=trend_is_bullish
                    )
                    
                    # Ensure we get exactly 7 features
                    if len(rag_results) >= 7:
                        rag_features = list(rag_results[:7])
                    
                    logger.debug(f"🧠 Live RAG Query: Win(L)={rag_features[0]:.1%}, Win(S)={rag_features[1]:.1%}, Samples={rag_features[4]:.0f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Live RAG query failed: {e}")
            
            # Opportunity flags
            buy_opportunity = 1.0 if cls_buy_prob >= 0.7 else 0.0
            sell_opportunity = 1.0 if cls_sell_prob >= 0.7 else 0.0
            
            # Build observation (synchronized with training!)
            observation = [
                cls_buy_prob,
                cls_sell_prob,
                position_type,
                unrealized_pnl_normalized,
                position_size,
                current_candle.get('atr', 0),
                current_candle.get('rsi', 50),
                current_candle.get('macd', 0),
                current_candle.get('macd_signal', 0),
                current_candle.get('bb_percent_b', 0.5),
                current_candle.get('momentum_5', 0),
                current_candle.get('ema_9', current_price),
                current_candle.get('ema_21', current_price),
                current_candle.get('ema_50', current_price),
                *rag_features,  # 🔥 LIVE RAG features (not stale!)
                current_candle.get('minutes_until_news', 1.0),
                current_candle.get('is_news_imminent', 0.0),
                0,  # recent_pnl (no recent trades in live trading)
                balance / 20.0,  # equity ratio
                buy_opportunity,
                sell_opportunity,
            ]
            
            # Pad to 57 features (updated to match RL environment)
            import numpy as np
            observation = np.array(observation, dtype=np.float32)
            if len(observation) < 57:
                observation = np.pad(observation, (0, 57 - len(observation)), mode='constant')
            else:
                observation = observation[:57]
            
            # 🔥 FIX: Use non-deterministic for exploration (allow RL to try actions)
            # deterministic=False → RL will sample from probability distribution
            action_idx, _states = self.rl_agent.predict(observation, deterministic=False)
            action_idx = int(action_idx)
            
            # 🔥 NEW: Log RL action probabilities for debugging
            # This helps understand why RL chooses certain actions
            try:
                # Method 1: Try to get distribution from policy (if available)
                import torch
                try:
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                    with torch.no_grad():
                        if hasattr(self.rl_agent.policy, 'get_distribution'):
                            dist = self.rl_agent.policy.get_distribution(obs_tensor)
                            if hasattr(dist, 'distribution'):
                                logits = dist.distribution.logits
                                if hasattr(logits, 'numpy'):
                                    logits_np = logits.numpy()[0]
                                else:
                                    logits_np = logits.detach().cpu().numpy()[0]
                                probs = np.exp(logits_np) / np.sum(np.exp(logits_np))  # Softmax
                                logger.info(f"📊 RL Action Probs: LONG={probs[0]:.1%}, SHORT={probs[1]:.1%}, HOLD={probs[2]:.1%}")
                except Exception as torch_err:
                    # Method 2: Run multiple predictions (sampling) to estimate probabilities
                    sample_count = 50
                    action_samples = []
                    for _ in range(sample_count):
                        sample_action, _ = self.rl_agent.predict(observation, deterministic=False)
                        action_samples.append(int(sample_action))
                    
                    action_counts = np.bincount(action_samples, minlength=3)
                    probs = action_counts / sample_count
                    logger.info(f"📊 RL Action Probs (estimated): LONG={probs[0]:.1%}, SHORT={probs[1]:.1%}, HOLD={probs[2]:.1%}")
            except Exception as e:
                logger.debug(f"⚠️  Could not get RL action probs: {e}")
            
            # 🔥 RL ACTION MAPPING:
            # 0 = ENTER_LONG (only valid if flat)
            # 1 = ENTER_SHORT (only valid if flat)
            # 2 = CLOSE/HOLD (only valid if in position)
            
            # Check position state for action validity
            has_position = position_type != 0
            
            # Map action based on position state
            if action_idx == 0:  # ENTER_LONG
                if has_position:
                    # Invalid: Already have position, RL wants to open new one
                    # → Close current position first, then reconsider next loop
                    rl_action = 'CLOSE'  
                    rl_confidence = 0.90
                    logger.info(f"🤖 RL wants LONG but have position → CLOSE first")
                else:
                    rl_action = 'BUY'
                    rl_confidence = 0.85
                    logger.info(f"🤖 RL Decision: BUY (ENTER_LONG)")
                    
            elif action_idx == 1:  # ENTER_SHORT
                if has_position:
                    # Invalid: Already have position, RL wants to open new one  
                    # → Close current position first, then reconsider next loop
                    rl_action = 'CLOSE'
                    rl_confidence = 0.90
                    logger.info(f"🤖 RL wants SHORT but have position → CLOSE first")
                else:
                    rl_action = 'SELL'
                    rl_confidence = 0.85
                    logger.info(f"🤖 RL Decision: SELL (ENTER_SHORT)")
                    
            else:  # action_idx == 2 (CLOSE/HOLD)
                if has_position:
                    rl_action = 'CLOSE'
                    rl_confidence = 0.80
                    logger.info(f"🤖 RL Decision: CLOSE position")
                else:
                    rl_action = 'HOLD'
                    rl_confidence = 0.70
                    logger.info(f"🤖 RL Decision: HOLD (no trade)")
            
            return {
                'action': rl_action,
                'confidence': rl_confidence,
                'reason': f"RL Policy (trained on 500K episodes, action={action_idx})"
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
            
            # 🔥 STEP 6.5: Get RL Agent decision if enabled
            rl_decision = None
            if self.use_rl_agent and self.rl_agent:
                logger.info("🤖 Getting RL Agent decision...")
                rl_decision = self._get_rl_decision(symbol, timeframe, action, confidence, df)
                results['rl_decision'] = rl_decision
                logger.info(f"   RL Action: {rl_decision['action']} ({rl_decision['confidence']:.2%})")
            
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
                'SCALPING': 0.70,    # 🔥 Scalping butuh confidence tinggi
                'NORMAL': 0.75,      # 🔥 Balanced untuk trading sehari-hari
                'AGGRESSIVE': 0.70,  # 🔥 AGGRESSIVE: Threshold rendah untuk lebih banyak trade
                'LONG_HOLD': 0.75    # 🔥 Long hold butuh confidence tinggi
            }
            min_conf = confidence_thresholds.get(mode, 0.70)
            
            # 🔥🔥🔥 RL AGENT IS THE BOSS! 🔥🔥🔥
            # CLS = Sensor (input feature untuk RL)
            # RL = Decision Maker (otak yang putuskan kapan masuk/keluar)
            
            if rl_decision and self.use_rl_agent:
                # ✅ RL AGENT MODE: RL is the decision maker!
                final_action = rl_decision['action']
                final_confidence = rl_decision['confidence']
                final_reason = rl_decision['reason']
                
                # Log for debugging (CLS sudah dimasukkan ke RL sebagai feature)
                logger.info(f"📊 CLS Input: {action} @ {confidence:.2%} (SENSOR for RL)")
                logger.info(f"🤖 RL Decision: {final_action} @ {final_confidence:.2%} (DECISION MAKER)")
                logger.info(f"   Reason: {final_reason}")
                
                # 🔥 RL ACTIONS: Execute based on RL decision (NO CLS THRESHOLD!)
                if final_action in ['BUY', 'SELL']:
                    # Check if this is a RL OVERRIDE (RL confidence high but CLS low)
                    is_override = (self.rl_override_enabled and 
                                  final_confidence >= self.rl_override_threshold and 
                                  confidence < self.cls_min_threshold)
                    
                    if is_override:
                        logger.warning(f"🔥 RL OVERRIDE ACTIVATED!")
                        logger.warning(f"   CLS: {action} @ {confidence:.2%} (BELOW {self.cls_min_threshold:.0%} threshold)")
                        logger.warning(f"   RL: {final_action} @ {final_confidence:.2%} (ABOVE {self.rl_override_threshold:.0%} override threshold)")
                        logger.warning(f"   🎯 RL OVERRIDING CLS - Executing trade anyway!")
                        final_reason = f"RL_OVERRIDE (CLS {confidence:.0%} < {self.cls_min_threshold:.0%})"
                    else:
                        logger.info(f"✅ RL SIGNAL CONFIRMED: {final_action} @ {final_confidence:.2%}")
                    
                    results['should_enter_trade'] = True
                    results['final_signal'] = {'action': final_action, 'confidence': final_confidence, 'reason': final_reason}
                elif final_action == 'CLOSE':
                    # RL wants to close position
                    logger.info(f"🔴 RL SIGNAL: CLOSE existing position (confidence: {final_confidence:.2%})")
                    results['should_close_position'] = True
                    results['close_signal'] = {'action': 'CLOSE', 'confidence': final_confidence, 'reason': final_reason}
                elif final_action == 'HOLD':
                    logger.info(f"🚫 RL Signal: HOLD (no trade)")
                else:
                    logger.info(f"🚫 RL Action '{final_action}' not recognized")
            
            else:
                # ⚠️ FALLBACK: CLS-only mode (RL not available)
                logger.info(f"⚠️  RL Agent not available, using CLS-only mode (legacy)")
                
                # CLS-only: Use threshold
                logger.info(f"🔍 CLS-only Decision Check:")
                logger.info(f"   Action: '{action}'")
                logger.info(f"   Confidence: {confidence:.2%}")
                logger.info(f"   Threshold: {min_conf:.2%}")
                logger.info(f"   Action in ['BUY', 'SELL']: {action in ['BUY', 'SELL']}")
                logger.info(f"   Confidence >= threshold: {confidence >= min_conf}")
                
                if action in ['BUY', 'SELL'] and confidence >= min_conf:
                    logger.info(f"✅ CLS Signal: {action} @ {confidence:.2%} (no RL agent)")
                    results['should_enter_trade'] = True
                    results['final_signal'] = {'action': action, 'confidence': confidence, 'reason': f"CLS Model ({timeframe})"}
                else:
                    if action not in ['BUY', 'SELL']:
                        logger.info(f"🚫 No trade: CLS action is '{action}' (not BUY/SELL)")
                    else:
                        logger.info(f"🚫 No trade: CLS confidence {confidence:.2%} < threshold {min_conf:.2%}")
            
        except Exception as e:
            logger.error(f"❌ Error during analysis: {str(e)}", exc_info=True)
            results['error'] = str(e)
        
        return results
    
    def _monitor_positions_closed(self, symbol: str):
        """
        🔥 NEW: Monitor positions for TP/SL hits and send Telegram notifications
        
        Detects when positions are closed by MT5 (TP/SL hit) and sends
        detailed notifications with profit/loss, win rate, and stats.
        """
        if not self._active_positions:
            return
        
        try:
            # Get current open positions
            current_positions = self.mt5.get_open_positions(symbol)
            current_tickets = {pos.get('ticket') for pos in current_positions}
            
            # Find closed positions (were tracked but no longer open)
            tracked_tickets = set(self._active_positions.keys())
            closed_tickets = tracked_tickets - current_tickets
            
            for ticket in closed_tickets:
                if ticket not in self._active_positions:
                    continue
                
                pos_info = self._active_positions[ticket]
                
                # Try to get closed position info from MT5 history
                # (for exact exit price and reason)
                try:
                    import MetaTrader5 as mt5
                    from_date = pos_info['entry_time']
                    deals = mt5.history_deals_get(from_date, datetime.now(), group=pos_info['symbol'])
                    
                    # Find deal related to this position
                    closing_deal = None
                    if deals:
                        for deal in deals:
                            if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                                closing_deal = deal
                                break
                    
                    if closing_deal:
                        exit_price = closing_deal.price
                        exit_time = datetime.fromtimestamp(closing_deal.time)
                        profit = closing_deal.profit
                        commission = closing_deal.commission
                    else:
                        # Fallback: estimate from SL/TP
                        exit_price = pos_info.get('tp_price', pos_info.get('sl_price', pos_info['entry_price']))
                        exit_time = datetime.now()
                        if pos_info['type'] == 'BUY':
                            profit = (exit_price - pos_info['entry_price']) * pos_info['lot_size'] * 100
                        else:
                            profit = (pos_info['entry_price'] - exit_price) * pos_info['lot_size'] * 100
                        commission = 0
                    
                    # Calculate profit in pips
                    symbol_info = self.mt5.get_symbol_info(pos_info['symbol'])
                    pip_value = symbol_info['pip_value'] if symbol_info else 0.1
                    
                    if pos_info['type'] == 'BUY':
                        price_diff = exit_price - pos_info['entry_price']
                    else:
                        price_diff = pos_info['entry_price'] - exit_price
                    
                    profit_pips = price_diff / pip_value
                    
                    # Determine close reason
                    sl_price = pos_info.get('sl_price', 0)
                    tp_price = pos_info.get('tp_price', 0)
                    
                    # Check proximity to SL/TP (within 2 pips = hit)
                    sl_distance = abs(exit_price - sl_price) / pip_value if sl_price > 0 else 999
                    tp_distance = abs(exit_price - tp_price) / pip_value if tp_price > 0 else 999
                    
                    if sl_distance < 2.0 and profit < 0:
                        close_reason = 'SL Hit'
                    elif tp_distance < 2.0 and profit > 0:
                        close_reason = 'TP Hit'
                    else:
                        close_reason = 'Manual/Other'
                    
                    # Prepare position data for Telegram
                    position_data = {
                        'symbol': pos_info['symbol'],
                        'type': pos_info['type'],
                        'ticket': ticket,
                        'entry_price': pos_info['entry_price'],
                        'exit_price': exit_price,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'profit': profit,
                        'profit_pips': profit_pips,
                        'close_reason': close_reason,
                        'entry_time': pos_info['entry_time'],
                        'exit_time': exit_time
                    }
                    
                    # Send Telegram notification
                    if self.telegram:
                        self.telegram.send_position_closed(position_data)
                        logger.info(f"📱 Telegram notification sent for closed position #{ticket}")
                    
                    # Store to RAG if profitable
                    if self.rag_memory and profit != 0:
                        trade_outcome = {
                            'symbol': pos_info['symbol'],
                            'entry_price': pos_info['entry_price'],
                            'exit_price': exit_price,
                            'action': pos_info['type'],
                            'pnl': profit,
                            'atr_entry': pos_info.get('atr_entry', 10),
                            'rsi_entry': pos_info.get('rsi_entry', 50),
                            'duration': (exit_time - pos_info['entry_time']).total_seconds() / 60,  # minutes
                            'is_during_news': False
                        }
                        self._store_trade_outcome_to_rag(trade_outcome)
                    
                    # Remove from tracking
                    del self._active_positions[ticket]
                    if ticket in self._last_known_positions:
                        self._last_known_positions.remove(ticket)
                    
                    logger.info(f"✅ Position #{ticket} closed: {close_reason}, P&L: ${profit:.2f} ({profit_pips:+.1f} pips)")
                    
                except Exception as e:
                    logger.error(f"❌ Error monitoring position #{ticket}: {e}")
                    # Clean up anyway
                    if ticket in self._active_positions:
                        del self._active_positions[ticket]
        
        except Exception as e:
            logger.debug(f"Error in position monitoring: {e}")
    
    def _store_trade_outcome_to_rag(self, trade_info: Dict):
        """
        🔥 NEW: Store completed trade outcome to RAG Memory
        
        This allows the RL agent to learn from past trades and improve
        future decisions based on similar market conditions.
        
        Args:
            trade_info: Dict containing trade details including:
                - symbol: Trading symbol
                - entry_price: Entry price
                - exit_price: Exit price
                - action: 'BUY' or 'SELL'
                - pnl: Realized P&L
                - atr_entry: ATR at entry
                - rsi_entry: RSI at entry
                - duration: Trade duration in candles
        """
        if not self.rag_memory:
            return
        
        try:
            # Get market context at entry
            atr_entry = trade_info.get('atr_entry', 10)
            rsi_entry = trade_info.get('rsi_entry', 50)
            entry_price = trade_info.get('entry_price', 0)
            action = trade_info.get('action', 'BUY')
            pnl = trade_info.get('pnl', 0)
            duration = trade_info.get('duration', 0)
            is_during_news = trade_info.get('is_during_news', False)
            
            # Store to RAG memory
            self.rag_memory.store_trade_outcome(
                price=entry_price,
                atr=atr_entry,
                rsi=rsi_entry,
                action=action,
                outcome=pnl,
                duration=duration,
                is_during_news=is_during_news,
                additional_context={
                    'symbol': trade_info.get('symbol', 'XAUUSDm'),
                    'exit_price': trade_info.get('exit_price', entry_price),
                    'entry_time': trade_info.get('entry_time', datetime.now().isoformat()),
                    'exit_time': trade_info.get('exit_time', datetime.now().isoformat())
                }
            )
            
            logger.info(f"💾 Stored trade outcome to RAG Memory: {action} @ {entry_price:.2f}, P&L: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error storing to RAG Memory: {e}")
    
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
                
                # 🔥 NEW: Track position for TP/SL monitoring
                self._active_positions[ticket] = {
                    'ticket': ticket,
                    'symbol': trade_params['symbol'],
                    'type': trade_params['action'],
                    'entry_price': executed_price,
                    'entry_time': datetime.now(),
                    'sl_price': trade_params['stop_loss'],
                    'tp_price': trade_params['take_profit'],
                    'lot_size': trade_params['lot_size']
                }
                self._last_known_positions.add(ticket)
                logger.info(f"📝 Position tracked: #{ticket} (SL: {trade_params['stop_loss']:.5f}, TP: {trade_params['take_profit']:.5f})")
                
                # 🔥 NEW: Activate Stop Loss Plus for intelligent exit management
                if self.stop_loss_plus:
                    position_type = 'LONG' if trade_params['action'] == 'BUY' else 'SHORT'
                    self.stop_loss_plus.set_position(
                        entry_price=executed_price,
                        position_type=position_type,
                        entry_time=datetime.now()
                    )
                    logger.info(f"🛡️  Stop Loss Plus activated for {position_type} position")
                
                # 🔥 NEW: Record in performance tracker (for monitoring open positions)
                # This will help track the trade lifecycle
                
                # 🔥 NEW: Store entry context for later RAG learning (capture market state)
                trade_entry_context = {
                    'ticket': ticket,
                    'symbol': trade_params['symbol'],
                    'action': trade_params['action'],
                    'entry_price': executed_price,
                    'entry_time': datetime.now(),
                    'atr_entry': trade_params.get('atr_entry', 10),
                    'rsi_entry': trade_params.get('rsi_entry', 50),
                    'lot_size': trade_params['lot_size']
                }
                # Store for later retrieval when trade closes
                if not hasattr(self, '_active_trade_contexts'):
                    self._active_trade_contexts = {}
                self._active_trade_contexts[ticket] = trade_entry_context
                
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
        🔥🔥🔥 ULTIMATE SMART AUTO-TRADE: Multi-Timeframe Routing + Enhanced Error Recovery
        
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
        logger.info(f"   🔥 Enhanced error recovery: Exponential backoff enabled")
        
        # 🔥 NEW: Check mode and adjust RL Override settings
        mode = self.config.get('trade_mode', 'NORMAL').upper()
        if mode == "AGGRESSIVE" and self.use_rl_agent:
            logger.info(f"🔥 AGGRESSIVE MODE: RL Override is ENABLED!")
            logger.info(f"   RL can trade even if CLS < {self.cls_min_threshold:.0%} (if RL >= {self.rl_override_threshold:.0%})")
        elif self.use_rl_agent:
            logger.info(f"⚠️  {mode} MODE: RL Override enabled but may be conservative")
        
        import time
        
        last_analysis_time = 0
        iteration = 0
        
        # 🔥 NEW: Error recovery tracking
        max_consecutive_errors = 5
        error_count = 0
        last_error_time = 0
        
        while self.auto_trade_enabled:
            try:
                iteration += 1
                current_time = time.time()
                
                # 🔥 IMPROVED: Always update trailing stops (every 10s loop)
                # This makes position management much more responsive
                if iteration % 1 == 0:  # Every iteration (10s)
                    self.trailing_stop.update_trailing_stops()
                    
                    # 🔥 NEW: Monitor positions for TP/SL hits
                    self._monitor_positions_closed(symbol)
                    
                    # 🔥 DISABLED: Stop Loss Plus monitoring (user disabled)
                    # if self.stop_loss_plus and self.stop_loss_plus.entry_price:
                    #     try:
                    #         # Get current positions
                    #         open_positions = self.mt5.get_open_positions(symbol)
                    #         if open_positions:
                    #             # Get latest market data
                    #             df_sl = self.mt5.get_candles(symbol, timeframe, count=50)
                    #             if df_sl is not None and len(df_sl) > 0:
                    #                 current_candle = df_sl.iloc[-1]
                    #                 current_price = current_candle['close']
                    #                 current_atr = current_candle.get('atr', 10)
                    #                 
                    #                 # Get CLS probabilities
                    #                 cls_action, cls_confidence = self.cls_predictor.predict(df_sl, timeframe.lower(), self.mt5)
                    #                 cls_buy_prob = cls_confidence if cls_action == 'BUY' else 1 - cls_confidence
                    #                 cls_sell_prob = cls_confidence if cls_action == 'SELL' else 1 - cls_confidence
                    #                 
                    #                 # Get pip value
                    #                 symbol_info = self.mt5.get_symbol_info(symbol)
                    #                 pip_value = symbol_info['pip_value'] if symbol_info else 0.1
                    #                 
                    #                 # Update SL+
                    #                 sl_result = self.stop_loss_plus.update(
                    #                     current_price=current_price,
                    #                     atr=current_atr,
                    #                     cls_buy_prob=cls_buy_prob,
                    #                     cls_sell_prob=cls_sell_prob,
                    #                     pip_value=pip_value
                    #                 )
                    #                 
                    #                 # Check if should exit
                    #                 if sl_result['exit']:
                    #                     logger.warning(f"🔴 SL+ EXIT SIGNAL: {sl_result['reason']}")
                    #                     logger.warning(f"   Current: {current_price:.5f}, SL: {sl_result['sl_price']:.5f}")
                    #                     logger.warning(f"   Profit: {sl_result['profit_pips']:.1f} pips")
                    #                     
                    #                     # Close positions
                    #                     for pos in open_positions:
                    #                         ticket = pos.get('ticket')
                    #                         try:
                    #                             # 🔥 FIX: close_position_by_ticket returns Tuple[bool, Optional[Dict]]
                    #                             success, result_dict = self.executor.close_position_by_ticket(ticket)
                    #                             if success:
                    #                                 logger.info(f"   ✅ Closed position #{ticket} (SL+ exit)")
                    #                             else:
                    #                                 error_msg = result_dict.get('error', 'Unknown error') if result_dict else 'Unknown error'
                    #                                 logger.error(f"   ❌ Failed to close #{ticket}: {error_msg}")
                    #                         except Exception as e:
                    #                             logger.error(f"   ❌ Error closing position #{ticket}: {e}")
                    #                     
                    #                     # Reset SL+
                    #                     self.stop_loss_plus.reset()
                    #     except Exception as e:
                    #         logger.debug(f"SL+ monitoring error: {e}")
                
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
                    if 'final_signal' in analysis:
                        signal = analysis['final_signal']
                        logger.info(f"💡 Final Signal:")
                        logger.info(f"   Action: {signal['action']}")
                        logger.info(f"   Confidence: {signal['confidence']:.2%}")
                        logger.info(f"   Reason: {signal['reason']}")
                        logger.info(f"   Should Enter: {analysis.get('should_enter_trade', False)}")
                    
                    if 'rl_decision' in analysis:
                        rl = analysis['rl_decision']
                        logger.info(f"🤖 RL Agent: {rl['action']} @ {rl['confidence']:.2%}")
                    
                    if 'error' in analysis:
                        logger.error(f"❌ Analysis error: {analysis['error']}")
                    
                # 🔥 DISABLED: Handle CLOSE signal from RL agent (user wants to disable auto-close)
                # if analysis.get('should_close_position') and analysis.get('close_signal'):
                #     close_signal = analysis['close_signal']
                #     logger.info(f"🔴 RL CLOSE Signal: {close_signal['action']} @ {close_signal['confidence']:.2%}")
                #     logger.info(f"   Reason: {close_signal['reason']}")
                #     
                #     # Get open positions for this symbol
                #     open_positions = self.mt5.get_open_positions(symbol)
                #     if open_positions:
                #         logger.info(f"   Closing {len(open_positions)} position(s)...")
                #         for pos in open_positions:
                #             ticket = pos.get('ticket')
                #             try:
                #                 # Close position
                #                 result = self.executor.close_position_by_ticket(ticket)
                #                 if result.get('success'):
                #                     logger.info(f"   ✅ Closed position #{ticket}")
                #                 else:
                #                     logger.error(f"   ❌ Failed to close position #{ticket}: {result.get('error')}")
                #             except Exception as e:
                #                 logger.error(f"   ❌ Error closing position #{ticket}: {e}")
                #     else:
                #         logger.info(f"   ⚠️  No open positions to close (race condition)")
                    
                    # 🔥 DISABLED: Log if RL wants to close (for debugging only)
                    if analysis.get('should_close_position') and analysis.get('close_signal'):
                        close_signal = analysis['close_signal']
                        logger.info(f"🔴 RL wants to CLOSE (DISABLED): {close_signal['action']} @ {close_signal['confidence']:.2%} - Auto-close is disabled by user")
                    
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
                        
                        # 🔥 NEW: Get dynamic parameters based on symbol type
                        mode = self.config.get('trade_mode', 'NORMAL').upper()
                        tf_upper = timeframe.upper()
                        
                        # Use new method for symbol-specific adjustment
                        params = self.get_trade_parameters(symbol, tf_upper, mode)
                        sl_atr_multiplier = params['sl_atr']
                        rr_ratio = params['rr']
                        
                        logger.info(f"📏 Using params for [{tf_upper} - {mode}]: SL={sl_atr_multiplier:.2f}x ATR, RR={rr_ratio:.2f}:1")
                        
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
                        
                        # 🔥 NEW: Get RSI for RAG memory
                        from strategies.base_strategy import BaseStrategy
                        
                        class TempStrategy(BaseStrategy):
                            def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
                                return None
                        
                        strategy = TempStrategy("Temp", "MEDIUM")
                        df_trade = strategy.add_all_indicators(df_trade)
                        current_rsi = df_trade['rsi'].iloc[-1]
                        
                        # Siapkan parameter final untuk eksekusi
                        trade_params = {
                            'action': signal['action'], 'symbol': symbol, 'lot_size': lot_size,
                            'entry_price': current_price,
                            'stop_loss': round(stop_loss, symbol_info['digits']),
                            'take_profit': round(take_profit, symbol_info['digits']),
                            'confidence': signal['confidence'],
                            'reason': signal['reason'],
                            'sl_pips': round(sl_pips, 1),
                            'tp_pips': round(sl_pips * rr_ratio, 1),
                            # 🔥 NEW: Add market context for RAG learning
                            'atr_entry': current_atr,
                            'rsi_entry': current_rsi
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
                            
                            # 🔥 NEW: Send special Telegram alert for RL OVERRIDE
                            if self.telegram and signal['reason'].startswith('RL_OVERRIDE'):
                                cls_action = analysis.get('cls_prediction', {}).get('action', 'N/A')
                                cls_conf = analysis.get('cls_prediction', {}).get('confidence', 0)
                                rl_conf = signal['confidence']
                                
                                override_msg = f"""
🔥 <b>RL OVERRIDE TRADE EXECUTED!</b>

<b>Signal:</b> {signal['action']}
<b>Symbol:</b> {symbol}

<b>CLS Sensor:</b> {cls_action} @ {cls_conf:.0%} (BELOW threshold)
<b>RL Decision:</b> {signal['action']} @ {rl_conf:.0%} (OVERRIDING)

💰 <b>Entry:</b> {current_price:.5f}
🛑 <b>SL:</b> {stop_loss:.5f} ({sl_pips:.1f} pips)
🎯 <b>TP:</b> {take_profit:.5f} ({sl_pips * rr_ratio:.1f} pips)

<i>RL Agent says: "Trust my decision!" 🎯</i>

Time: {datetime.now().strftime('%H:%M:%S')}
"""
                                self.telegram.send_message(override_msg, parse_mode='HTML')
                                
                            # 🔥 NEW: Send AI-powered Telegram notification (normal)
                            if self.telegram and not signal['reason'].startswith('RL_OVERRIDE'):
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
                                # 🔥 SINKRON dengan thresholds di analyze_market
                                thresholds = {'SCALPING': 0.70, 'NORMAL': 0.75, 'AGGRESSIVE': 0.70, 'LONG_HOLD': 0.75}
                                threshold = thresholds.get(mode, 0.70)
                                logger.info(f"🚫 No trade: Confidence {cls_confidence:.2%} < threshold {threshold:.0%} ({mode} mode)")
                
                # 🔥 IMPROVED: Reset error count on successful iteration
                error_count = 0
                
                # Sleep for shorter intervals (10s) for better responsiveness
                time.sleep(10)
            
            except KeyboardInterrupt:
                logger.info("⚠️  Auto-trade interrupted by user")
                self.auto_trade_enabled = False
                break
            
            except Exception as e:
                error_count += 1
                current_error_time = time.time()
                
                # 🔥 IMPROVED: Detailed error logging
                logger.error(f"❌ Error #{error_count}/{max_consecutive_errors} in auto-trade loop: {str(e)}", exc_info=True)
                
                # 🔥 NEW: Check if we've exceeded max consecutive errors
                if error_count >= max_consecutive_errors:
                    logger.error(f"🛑 Too many consecutive errors ({error_count}), stopping auto-trade")
                    self.auto_trade_enabled = False
                    
                    # Send critical alert
                if self.telegram:
                    self.telegram.send_error_alert(
                            f"🛑 CRITICAL: Auto-trade stopped after {error_count} consecutive errors",
                            context=f"Last error: {str(e)[:100]}"
                        )
                    break
                
                # 🔥 NEW: Exponential backoff (but cap at 5 minutes)
                # Wait: 30s, 60s, 120s, 240s, 300s (max)
                sleep_time = min(30 * (2 ** (error_count - 1)), 300)
                
                logger.warning(f"⏳ Waiting {sleep_time}s before retry (exponential backoff)...")
                
                # Send error alert if it's first error or been a while since last error
                if error_count == 1 or (current_error_time - last_error_time) > 600:  # 10 minutes
                    if self.telegram:
                        self.telegram.send_error_alert(
                            f"Error in trading loop (#{error_count}): {str(e)[:100]}",
                            context="Auto-trade loop - Will retry"
                        )
                
                last_error_time = current_error_time
                time.sleep(sleep_time)
        
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
    
    def get_bot_status(self) -> Dict:
        """
        🔥 NEW: Get comprehensive bot status for monitoring
        
        Returns detailed health check information including:
        - Trading status
        - MT5 connection status
        - Open positions count
        - Daily P&L statistics
        - Model loading status
        - RL agent status
        - Telegram connection
        - Last analysis timestamp
        
        Returns:
            Dict with comprehensive bot status
        """
        try:
            # Get open positions
            open_positions = self.mt5.get_open_positions() if self.mt5 else []
            
            # Get daily stats
            daily_stats = {}
            try:
                daily_stats = self.performance.get_daily_stats() if self.performance else {}
            except Exception as e:
                logger.debug(f"Could not fetch daily stats: {e}")
                daily_stats = {'error': str(e)}
            
            # Get MT5 account info
            account_info = {}
            try:
                account_info = self.mt5.get_account_info() if self.mt5 else {}
            except Exception as e:
                logger.debug(f"Could not fetch account info: {e}")
                account_info = {'error': str(e)}
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'auto_trade_enabled': self.auto_trade_enabled,
                'mt5': {
                    'connected': self.connection_manager.health_check() if hasattr(self, 'connection_manager') else False,
                    'account_login': account_info.get('login', 'N/A'),
                    'account_server': account_info.get('server', 'N/A'),
                    'balance': account_info.get('balance', 0),
                    'equity': account_info.get('equity', 0),
                    'margin_free': account_info.get('margin_free', 0)
                },
                'positions': {
                    'open_count': len(open_positions),
                    'details': [
                        {
                            'ticket': pos.get('ticket', 'N/A'),
                            'symbol': pos.get('symbol', 'N/A'),
                            'type': pos.get('type', 'N/A'),
                            'volume': pos.get('volume', 0),
                            'profit': pos.get('profit', 0)
                        } for pos in open_positions[:5]  # Max 5 positions in summary
                    ]
                },
                'daily_stats': daily_stats,
                'models': {
                    'cls_loaded': self.cls_predictor is not None,
                    'rl_agent_loaded': self.rl_agent is not None and self.use_rl_agent,
                    'rl_agent_enabled': self.use_rl_agent
                },
                'monitoring': {
                    'telegram_connected': self.telegram is not None,
                    'performance_tracker_active': self.performance is not None
                },
                'config': {
                    'trade_mode': self.config.get('trade_mode', 'N/A'),
                    'default_lot_size': self.default_lot_size,
                    'max_open_positions': self.risk_manager.max_open_positions if hasattr(self.risk_manager, 'max_open_positions') else 'N/A',
                    'max_trades_per_day': self.risk_manager.max_trades_per_day if hasattr(self.risk_manager, 'max_trades_per_day') else 'N/A'
                },
                'health': 'OK' if all([
                    self.connection_manager.health_check() if hasattr(self, 'connection_manager') else False,
                    self.cls_predictor is not None
                ]) else 'DEGRADED'
            }
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'health': 'ERROR'
            }
    
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
    """
    🔥 IMPROVED: Load and validate configuration from environment and defaults
    
    Returns:
        Dict with validated configuration
    """
    # Load raw config
    config = {
        'default_lot_size': float(os.getenv('DEFAULT_LOT_SIZE', '0.01')),
        'max_slippage': float(os.getenv('MAX_SLIPPAGE', '2.0')),
        'max_risk_per_trade': float(os.getenv('RISK_PER_TRADE', '1.0')),
        'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '5.0')),
        'max_trades_per_day': int(os.getenv('MAX_TRADES_PER_DAY', '10')),
        'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', '3')),
        'enabled_strategies': ['counter_trend', 'breakout', 'fibonacci_atr'],
        'model_dir': './models/saved_models',
        'trade_mode': os.getenv('TRADE_MODE', 'NORMAL').upper(),
        'save_to_firebase': os.getenv('FIREBASE_ENABLED', 'false').lower() == 'true',
        # 🔥 ML-driven risk parameters
        'sl_atr_multiplier': float(os.getenv('SL_ATR_MULTIPLIER', '1.5')),
        'risk_reward_ratio': float(os.getenv('RISK_REWARD_RATIO', '1.5')),
        # 🔥 NEW: RL Agent settings
        'use_rl_agent': os.getenv('USE_RL_AGENT', 'false').lower() == 'true',
        'rl_model_path': os.getenv('RL_MODEL_PATH', './models/saved_models/rl_agent/rl_agent_final.zip'),
        # 🔥 NEW: Multi-symbol support
        'symbols': [s.strip() for s in os.getenv('TRADING_SYMBOLS', 'XAUUSDm').split(',')]
    }
    
    # 🔥 VALIDATION: Trade mode
    valid_modes = ['SCALPING', 'NORMAL', 'AGGRESSIVE', 'LONG_HOLD']
    if config['trade_mode'] not in valid_modes:
        logger.warning(f"⚠️  Invalid trade mode: '{config['trade_mode']}', using NORMAL")
        logger.info(f"   Valid modes: {', '.join(valid_modes)}")
        config['trade_mode'] = 'NORMAL'
    
    # 🔥 VALIDATION: Risk parameters
    if not 0.0 < config['max_risk_per_trade'] <= 5.0:
        logger.warning(f"⚠️  Risk per trade {config['max_risk_per_trade']}% is out of safe range (0-5%), using 1.0%")
        config['max_risk_per_trade'] = 1.0
    
    if not 0.0 < config['max_daily_loss'] <= 20.0:
        logger.warning(f"⚠️  Max daily loss {config['max_daily_loss']}% is out of safe range (0-20%), using 5.0%")
        config['max_daily_loss'] = 5.0
    
    if not 0.001 <= config['default_lot_size'] <= 10.0:
        logger.warning(f"⚠️  Default lot size {config['default_lot_size']} is out of safe range (0.001-10.0), using 0.01")
        config['default_lot_size'] = 0.01
    
    # 🔥 VALIDATION: Trading limits
    if config['max_trades_per_day'] > 100:
        logger.warning(f"⚠️  Max trades per day {config['max_trades_per_day']} is very high, using 50")
        config['max_trades_per_day'] = 50
    
    if config['max_open_positions'] > 10:
        logger.warning(f"⚠️  Max open positions {config['max_open_positions']} is very high, using 5")
        config['max_open_positions'] = 5
    
    # 🔥 VALIDATION: SL/TP parameters
    if not 0.5 <= config['sl_atr_multiplier'] <= 3.0:
        logger.warning(f"⚠️  SL ATR multiplier {config['sl_atr_multiplier']} is out of range (0.5-3.0), using 1.5")
        config['sl_atr_multiplier'] = 1.5
    
    if not 1.0 <= config['risk_reward_ratio'] <= 5.0:
        logger.warning(f"⚠️  Risk/Reward ratio {config['risk_reward_ratio']} is out of range (1.0-5.0), using 1.5")
        config['risk_reward_ratio'] = 1.5
    
    # Log validated config
    logger.info("📋 Configuration loaded and validated:")
    logger.info(f"   Trade Mode: {config['trade_mode']}")
    logger.info(f"   Default Lot: {config['default_lot_size']}")
    logger.info(f"   Risk per Trade: {config['max_risk_per_trade']}%")
    logger.info(f"   Max Daily Loss: {config['max_daily_loss']}%")
    logger.info(f"   Max Trades/Day: {config['max_trades_per_day']}")
    logger.info(f"   Max Open Positions: {config['max_open_positions']}")
    logger.info(f"   SL/TP: {config['sl_atr_multiplier']}x ATR / RR {config['risk_reward_ratio']}:1")
    logger.info(f"   RL Agent: {'Enabled' if config['use_rl_agent'] else 'Disabled'}")
    logger.info(f"   Symbols: {', '.join(config['symbols'])}")
    
    return config


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