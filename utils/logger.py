import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "./logs", level: str = "INFO"):
    """Setup logging configuration"""
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Log file name with date
    log_file = log_path / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger.info("=" * 80)
    logger.info("Logging initialized")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


# monitoring/telegram_bot.py
"""
Telegram Bot for Notifications
"""

import logging
from typing import Dict, Optional

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logging.warning("python-telegram-bot not installed, Telegram features disabled")

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send trading notifications via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot package required for Telegram notifications")
        
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        
        # Test connection
        try:
            self.bot.get_me()
            logger.info("✅ Telegram bot connected")
        except TelegramError as e:
            logger.error(f"❌ Telegram connection failed: {e}")
    
    def send_message(self, message: str):
        """Send text message"""
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def send_trade_alert(self, trade_info: Dict):
        """Send trade execution alert"""
        emoji = "🟢" if trade_info['type'] == 'BUY' else "🔴"
        
        message = f"""
{emoji} <b>Trade Executed</b>

<b>Symbol:</b> {trade_info['symbol']}
<b>Type:</b> {trade_info['type']}
<b>Entry:</b> {trade_info['entry']:.5f}
<b>Stop Loss:</b> {trade_info['sl']:.5f}
<b>Take Profit:</b> {trade_info['tp']:.5f}

<b>AI Confidence:</b> {trade_info['confidence']:.1%}
<b>Reason:</b> {trade_info['reason'][:100]}
        """
        
        self.send_message(message)
    
    def send_position_closed(self, position_info: Dict):
        """Send position closed notification"""
        profit = position_info['profit']
        emoji = "✅" if profit > 0 else "❌"
        
        message = f"""
{emoji} <b>Position Closed</b>

<b>Symbol:</b> {position_info['symbol']}
<b>Type:</b> {position_info['type']}
<b>P&L:</b> ${profit:.2f} ({position_info.get('pips', 0):.1f} pips)
<b>Duration:</b> {position_info.get('duration', 'N/A')}
        """
        
        self.send_message(message)
    
    def send_daily_summary(self, summary: Dict):
        """Send end-of-day summary"""
        total_pnl = summary['total_pnl']
        emoji = "📈" if total_pnl > 0 else "📉"
        
        message = f"""
{emoji} <b>Daily Summary</b>

<b>Total Trades:</b> {summary['total_trades']}
<b>Wins:</b> {summary['wins']} ({summary['win_rate']:.1%})
<b>Losses:</b> {summary['losses']}
<b>Total P&L:</b> ${total_pnl:+.2f}

<b>Best Trade:</b> ${summary['best_trade']:.2f}
<b>Worst Trade:</b> ${summary['worst_trade']:.2f}
<b>Avg Profit:</b> ${summary['avg_profit']:.2f}
        """
        
        self.send_message(message)
    
    def send_error_alert(self, error_msg: str):
        """Send error notification"""
        message = f"""
⚠️ <b>Error Alert</b>

{error_msg}
        """
        
        self.send_message(message)


# monitoring/performance_tracker.py
"""
Performance Tracking and Analytics
"""

import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self, mt5_handler, save_to_firebase: bool = False):
        self.mt5 = mt5_handler
        self.save_to_firebase = save_to_firebase
        self.trades_history = []
        self.data_dir = Path("./data/performance")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing history
        self._load_history()
    
    def _load_history(self):
        """Load trade history from file"""
        history_file = self.data_dir / "trade_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.trades_history = json.load(f)
                logger.info(f"Loaded {len(self.trades_history)} historical trades")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
    
    def _save_history(self):
        """Save trade history to file"""
        history_file = self.data_dir / "trade_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.trades_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def log_trade(self, trade: Dict):
        """Log a completed trade"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': trade.get('symbol'),
            'type': trade.get('type'),
            'entry_price': trade.get('entry_price'),
            'exit_price': trade.get('exit_price'),
            'lot_size': trade.get('lot_size'),
            'profit': trade.get('profit', 0),
            'pips': trade.get('pips', 0),
            'duration_minutes': trade.get('duration_minutes', 0),
            'strategy': trade.get('strategy'),
            'confidence': trade.get('confidence', 0),
            'risk_reward': trade.get('risk_reward', 0)
        }
        
        self.trades_history.append(trade_record)
        self._save_history()
        
        logger.info(f"Trade logged: {trade['symbol']} {trade['type']} → ${trade['profit']:.2f}")
    
    def get_daily_stats(self, date: datetime = None) -> Dict:
        """Get statistics for a specific day"""
        if date is None:
            date = datetime.now()
        
        date_str = date.date().isoformat()
        
        daily_trades = [
            t for t in self.trades_history 
            if t['timestamp'][:10] == date_str
        ]
        
        if not daily_trades:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        wins = [t for t in daily_trades if t['profit'] > 0]
        losses = [t for t in daily_trades if t['profit'] < 0]
        
        total_pnl = sum(t['profit'] for t in daily_trades)
        
        return {
            'total_trades': len(daily_trades),
            'wins': len(wins),
            'losses': len(losses),
            'total_pnl': total_pnl,
            'win_rate': len(wins) / len(daily_trades) if daily_trades else 0,
            'avg_profit': total_pnl / len(daily_trades) if daily_trades else 0,
            'best_trade': max((t['profit'] for t in daily_trades), default=0),
            'worst_trade': min((t['profit'] for t in daily_trades), default=0)
        }
    
    def get_overall_stats(self, days: int = 30) -> Dict:
        """Get overall statistics for last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        recent_trades = [
            t for t in self.trades_history 
            if t['timestamp'] >= cutoff_str
        ]
        
        if not recent_trades:
            return {'error': 'No trades in period'}
        
        wins = [t for t in recent_trades if t['profit'] > 0]
        losses = [t for t in recent_trades if t['profit'] < 0]
        
        total_pnl = sum(t['profit'] for t in recent_trades)
        total_wins = sum(t['profit'] for t in wins)
        total_losses = abs(sum(t['profit'] for t in losses))
        
        # Calculate metrics
        win_rate = len(wins) / len(recent_trades) if recent_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        
        # Calculate drawdown
        cumulative_pnl = []
        running_total = 0
        for trade in sorted(recent_trades, key=lambda x: x['timestamp']):
            running_total += trade['profit']
            cumulative_pnl.append(running_total)
        
        max_drawdown = 0
        peak = cumulative_pnl[0] if cumulative_pnl else 0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'period_days': days,
            'total_trades': len(recent_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': max((t['profit'] for t in recent_trades), default=0),
            'worst_trade': min((t['profit'] for t in recent_trades), default=0),
            'max_drawdown': max_drawdown,
            'avg_duration': sum(t['duration_minutes'] for t in recent_trades) / len(recent_trades)
        }
    
    def get_strategy_performance(self) -> Dict:
        """Get performance breakdown by strategy"""
        strategy_stats = {}
        
        for trade in self.trades_history:
            strategy = trade.get('strategy', 'unknown')
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0
                }
            
            stats = strategy_stats[strategy]
            stats['trades'] += 1
            stats['total_pnl'] += trade['profit']
            
            if trade['profit'] > 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
        
        # Calculate win rates
        for strategy, stats in strategy_stats.items():
            stats['win_rate'] = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            stats['avg_profit'] = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
        
        return strategy_stats
    
    def generate_report(self, period_days: int = 7) -> str:
        """Generate text performance report"""
        overall = self.get_overall_stats(period_days)
        today = self.get_daily_stats()
        strategies = self.get_strategy_performance()
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║              PERFORMANCE REPORT ({period_days} days)                    ║
╚═══════════════════════════════════════════════════════════╝

📊 OVERALL STATISTICS
────────────────────────────────────────────────────────────
Total Trades:      {overall['total_trades']}
Wins:              {overall['wins']} ({overall['win_rate']:.1%})
Losses:            {overall['losses']}
Total P&L:         ${overall['total_pnl']:+.2f}
Profit Factor:     {overall['profit_factor']:.2f}
Max Drawdown:      ${overall['max_drawdown']:.2f}

Average Win:       ${overall['avg_win']:.2f}
Average Loss:      ${overall['avg_loss']:.2f}
Best Trade:        ${overall['best_trade']:.2f}
Worst Trade:       ${overall['worst_trade']:.2f}

📅 TODAY'S PERFORMANCE
────────────────────────────────────────────────────────────
Trades:            {today['total_trades']}
Wins/Losses:       {today['wins']}/{today['losses']}
Total P&L:         ${today['total_pnl']:+.2f}
Win Rate:          {today['win_rate']:.1%}

🎯 STRATEGY BREAKDOWN
────────────────────────────────────────────────────────────
"""
        
        for strategy, stats in strategies.items():
            report += f"""
{strategy.upper()}:
  Trades: {stats['trades']} | Win Rate: {stats['win_rate']:.1%}
  Total P&L: ${stats['total_pnl']:+.2f} | Avg: ${stats['avg_profit']:+.2f}
"""
        
        report += "\n" + "═" * 60 + "\n"
        
        return report
    
    def export_to_csv(self, filename: str = None):
        """Export trade history to CSV"""
        if not self.trades_history:
            logger.warning("No trades to export")
            return
        
        if filename is None:
            filename = self.data_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
        
        df = pd.DataFrame(self.trades_history)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(df)} trades to {filename}")


# utils/config_loader.py
"""
Configuration management
"""

import yaml
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration"""
    
    @staticmethod
    def load_yaml(config_file: str = "config.yaml") -> Dict:
        """Load configuration from YAML file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_file}")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def validate_config(config: Dict) -> bool:
        """Validate configuration structure"""
        required_keys = ['trading', 'risk_management', 'strategies']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config section: {key}")
                return False
        
        return True


# utils/decorators.py
"""
Utility decorators
"""

import time
from functools import wraps


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator


def timing(func):
    """Measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


if __name__ == "__main__":
    # Test performance tracker
    setup_logging()
    
    from core.mt5_handler import MT5Handler
    
    mt5 = MT5Handler(12345678, "password", "MetaQuotes-Demo")
    mt5.initialize()
    
    tracker = PerformanceTracker(mt5)
    
    # Log sample trade
    tracker.log_trade({
        'symbol': 'XAUUSDm',
        'type': 'BUY',
        'entry_price': 3850.00,
        'exit_price': 3865.00,
        'lot_size': 0.01,
        'profit': 15.00,
        'pips': 15.0,
        'duration_minutes': 30,
        'strategy': 'breakout',
        'confidence': 0.75,
        'risk_reward': 2.0
    })
    
    # Get stats
    daily_stats = tracker.get_daily_stats()
    print("\nDaily Stats:")
    print(daily_stats)
    
    # Generate report
    report = tracker.generate_report(period_days=7)
    print(report)
    
    mt5.shutdown()