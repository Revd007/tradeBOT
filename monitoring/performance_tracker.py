"""
Performance Tracker
Track and analyze trading performance metrics

🔥 UPGRADED: Now uses SQLite for fast, scalable performance analysis
"""

import pandas as pd
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from monitoring.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track and analyze trading performance with SQLite database"""
    
    def __init__(
        self,
        mt5_handler,
        save_to_firebase: bool = False,
        db_path: str = "./data/trading_performance.db"
    ):
        """
        Args:
            mt5_handler: MT5 connection
            save_to_firebase: Save to Firebase (optional)
            db_path: SQLite database path
        """
        self.mt5 = mt5_handler
        self.save_to_firebase = save_to_firebase
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 🔥 NEW: Connect to SQLite database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_database()
        
        logger.info(f"✅ PerformanceTracker connected to SQLite: {self.db_path}")
        
        if save_to_firebase:
            self._init_firebase()
    
    def _init_database(self):
        """🔥 NEW: Initialize SQLite database schema"""
        cursor = self.conn.cursor()
        
        # Create trade_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER,
                symbol TEXT NOT NULL,
                type TEXT NOT NULL,
                lot_size REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                profit REAL NOT NULL,
                profit_pips REAL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                strategy TEXT,
                confidence REAL,
                reason TEXT,
                recorded_at TEXT NOT NULL,
                UNIQUE(ticket, open_time)
            )
        ''')
        
        # Create indexes for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_close_time ON trade_history(close_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy ON trade_history(strategy)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON trade_history(symbol)')
        
        self.conn.commit()
        logger.info("✅ Database schema initialized")
    
    def _init_firebase(self):
        """Initialize Firebase (optional)"""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            cred_path = "./firebase-key.json"
            if Path(cred_path).exists():
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                logger.info("✅ Firebase initialized")
            else:
                logger.warning("Firebase credentials not found")
                self.save_to_firebase = False
        except Exception as e:
            logger.warning(f"Firebase init failed: {str(e)}")
            self.save_to_firebase = False
    
    def record_trade(self, trade_data: Dict, telegram_notifier=None):
        """
        🔥 UPGRADED: Record trade to SQLite database + AI-powered notifications
        
        Args:
            trade_data: Trade information dict
            telegram_notifier: TelegramNotifier instance for AI insights
        """
        trade_record = {
            **trade_data,
            'recorded_at': datetime.now().isoformat()
        }
        
        # 🔥 NEW: Use pandas to_sql for easy SQLite insertion
        df = pd.DataFrame([trade_record])
        try:
            df.to_sql('trade_history', self.conn, if_exists='append', index=False)
            logger.info(f"✅ Trade recorded to SQLite: {trade_data['symbol']} {trade_data['type']} "
                       f"Profit: ${trade_data['profit']:.2f}")
            
            # 🔥 NEW: Send AI-powered Telegram analysis
            if telegram_notifier:
                profit = trade_data.get('profit', 0)
                
                if profit > 0:
                    # Profit analysis with RL/RAG insights
                    telegram_notifier.send_profit_analysis(trade_data)
                    logger.info("📱 Sent profit analysis via Telegram")
                elif profit < 0:
                    # Loss analysis with learning insights
                    telegram_notifier.send_loss_analysis(trade_data)
                    logger.info("📱 Sent loss analysis via Telegram")
            
        except sqlite3.IntegrityError:
            logger.warning(f"⚠️  Duplicate trade (ticket={trade_data.get('ticket')}), skipped")
        except Exception as e:
            logger.error(f"❌ Error recording trade to SQLite: {e}")
        
        # Optional: Also save to Firebase
        if self.save_to_firebase:
            try:
                self.db.collection('trades').add(trade_record)
            except Exception as e:
                logger.error(f"Error saving to Firebase: {str(e)}")
    
    def get_metrics(self, period_days: Optional[int] = None) -> Dict:
        """
        🔥 UPGRADED: Calculate metrics directly from SQLite database
        
        Args:
            period_days: Period in days (None = all time)
        
        Returns:
            Performance metrics dictionary
        """
        # 🔥 NEW: Query SQLite database instead of loading JSON
        query = "SELECT * FROM trade_history"
        if period_days:
            cutoff_date = (datetime.now() - timedelta(days=period_days)).isoformat()
            query += f" WHERE close_time >= '{cutoff_date}'"
        
        try:
            df = pd.read_sql(query, self.conn)
        except (sqlite3.OperationalError, pd.io.sql.DatabaseError):
            # Table doesn't exist or is empty
            return self._empty_metrics()
        
        if df.empty:
            return self._empty_metrics()
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        losing_trades = len(df[df['profit'] < 0])
        breakeven_trades = len(df[df['profit'] == 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = df['profit'].sum()
        total_wins = df[df['profit'] > 0]['profit'].sum()
        total_losses = abs(df[df['profit'] < 0]['profit'].sum())
        
        avg_win = df[df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['profit'] < 0]['profit'].mean()) if losing_trades > 0 else 0
        
        # Profit factor
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Risk-reward ratio
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Consecutive wins/losses
        df['is_win'] = df['profit'] > 0
        df['streak'] = (df['is_win'] != df['is_win'].shift()).cumsum()
        streaks = df.groupby('streak')['is_win'].agg(['sum', 'count'])
        
        max_consecutive_wins = streaks[streaks.index.isin(df[df['is_win']]['streak'])]['count'].max() if winning_trades > 0 else 0
        max_consecutive_losses = streaks[streaks.index.isin(df[~df['is_win']]['streak'])]['count'].max() if losing_trades > 0 else 0
        
        # Drawdown
        df_sorted = df.sort_values('close_time')
        df_sorted['cumulative_profit'] = df_sorted['profit'].cumsum()
        df_sorted['peak'] = df_sorted['cumulative_profit'].cummax()
        df_sorted['drawdown'] = df_sorted['peak'] - df_sorted['cumulative_profit']
        max_drawdown = df_sorted['drawdown'].max()
        
        # Best/worst trade
        best_trade = df['profit'].max()
        worst_trade = df['profit'].min()
        
        # Average trade duration
        df['duration'] = pd.to_datetime(df['close_time']) - pd.to_datetime(df['open_time'])
        avg_duration = df['duration'].mean()
        
        # Per strategy performance
        strategy_stats = {}
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            strategy_stats[strategy] = {
                'trades': len(strategy_df),
                'win_rate': len(strategy_df[strategy_df['profit'] > 0]) / len(strategy_df),
                'total_profit': strategy_df['profit'].sum(),
                'avg_profit': strategy_df['profit'].mean()
            }
        
        return {
            'period_days': period_days or 'all_time',
            'total_trades': int(total_trades),
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'breakeven_trades': int(breakeven_trades),
            'win_rate': float(win_rate),
            'total_profit': float(total_profit),
            'total_wins': float(total_wins),
            'total_losses': float(total_losses),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'avg_risk_reward': float(avg_rr),
            'max_consecutive_wins': int(max_consecutive_wins),
            'max_consecutive_losses': int(max_consecutive_losses),
            'max_drawdown': float(max_drawdown),
            'best_trade': float(best_trade),
            'worst_trade': float(worst_trade),
            'avg_trade_duration': str(avg_duration),
            'strategy_performance': strategy_stats
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'profit_factor': 0.0
        }
    
    def generate_report(self, period_days: int = 7) -> str:
        """Generate human-readable performance report"""
        metrics = self.get_metrics(period_days)
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║          PERFORMANCE REPORT ({period_days} DAYS)
╚═══════════════════════════════════════════════════════════╝

OVERVIEW:
  Total Trades: {metrics['total_trades']}
  Winning: {metrics['winning_trades']} | Losing: {metrics['losing_trades']}
  Win Rate: {metrics['win_rate']:.1%}

PROFIT & LOSS:
  Total P&L: ${metrics['total_profit']:+.2f}
  Total Wins: ${metrics['total_wins']:.2f}
  Total Losses: ${metrics['total_losses']:.2f}
  
  Average Win: ${metrics['avg_win']:.2f}
  Average Loss: ${metrics['avg_loss']:.2f}
  
  Profit Factor: {metrics['profit_factor']:.2f}
  Avg Risk/Reward: 1:{metrics['avg_risk_reward']:.2f}

STREAKS:
  Max Consecutive Wins: {metrics['max_consecutive_wins']}
  Max Consecutive Losses: {metrics['max_consecutive_losses']}

RISK:
  Max Drawdown: ${metrics['max_drawdown']:.2f}
  Best Trade: ${metrics['best_trade']:+.2f}
  Worst Trade: ${metrics['worst_trade']:+.2f}

STRATEGY PERFORMANCE:
"""
        
        for strategy, stats in metrics.get('strategy_performance', {}).items():
            report += f"""
  {strategy.upper()}:
    Trades: {stats['trades']}
    Win Rate: {stats['win_rate']:.1%}
    Total P&L: ${stats['total_profit']:+.2f}
"""
        
        report += "\n" + "═" * 60
        
        return report
    
    def export_to_csv(self, filename: str = "./data/trades_export.csv"):
        """Export trade history to CSV"""
        if not self.trade_history:
            logger.warning("No trade history to export")
            return
        
        df = pd.DataFrame(self.trade_history)
        df.to_csv(filename, index=False)
        logger.info(f"✅ Exported {len(df)} trades to {filename}")


if __name__ == "__main__":
    # Test performance tracker
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mt5 = MT5Handler(
        int(os.getenv('MT5_LOGIN_DEMO', 12345678)),
        os.getenv('MT5_PASSWORD_DEMO', 'password'),
        os.getenv('MT5_SERVER_DEMO', 'MetaQuotes-Demo')
    )
    
    if mt5.initialize():
        tracker = PerformanceTracker(mt5)
        
        # Add sample trades
        sample_trades = [
            {
                'ticket': 12345,
                'symbol': 'XAUUSDm',
                'type': 'BUY',
                'lot_size': 0.01,
                'entry_price': 3850.00,
                'exit_price': 3860.00,
                'stop_loss': 3840.00,
                'take_profit': 3870.00,
                'profit': 10.00,
                'profit_pips': 10.0,
                'open_time': (datetime.now() - timedelta(hours=2)).isoformat(),
                'close_time': datetime.now().isoformat(),
                'strategy': 'counter_trend',
                'confidence': 0.75,
                'reason': 'Reversal at support'
            }
        ]
        
        for trade in sample_trades:
            tracker.record_trade(trade)
        
        # Generate report
        print(tracker.generate_report(period_days=7))
        
        mt5.shutdown()

