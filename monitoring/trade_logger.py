"""
Trade Logger
Detailed logging of all trading activities
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

logger = logging.getLogger(__name__)


class TradeLogger:
    """Detailed logging of trading activities"""
    
    def __init__(self, log_dir: str = "./logs/trades"):
        """
        Args:
            log_dir: Directory for trade logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        self.current_date = datetime.now().date()
        self.log_file = self._get_log_file()
        
        logger.info(f"Trade logger initialized: {self.log_file}")
    
    def _get_log_file(self) -> Path:
        """Get current log file path"""
        date_str = self.current_date.strftime('%Y-%m-%d')
        return self.log_dir / f"trades_{date_str}.json"
    
    def _check_date_rollover(self):
        """Check if date has changed and create new log file"""
        today = datetime.now().date()
        if today != self.current_date:
            self.current_date = today
            self.log_file = self._get_log_file()
            logger.info(f"Rolled over to new log file: {self.log_file}")
    
    def _append_log(self, entry: Dict):
        """Append entry to log file"""
        self._check_date_rollover()
        
        try:
            # Read existing logs
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new entry
            logs.append(entry)
            
            # Write back
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Error writing to log file: {str(e)}")
    
    def log_analysis(
        self,
        symbol: str,
        timeframe: str,
        analysis_result: Dict
    ):
        """
        Log market analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Analysis timeframe
            analysis_result: Full analysis result
        """
        entry = {
            'type': 'ANALYSIS',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'fusion_action': analysis_result.get('fusion', {}).get('final_action'),
            'fusion_confidence': analysis_result.get('fusion', {}).get('confidence'),
            'strategy_consensus': analysis_result.get('strategy_consensus'),
            'cls_consensus': analysis_result.get('mtf_consensus', {}).get('consensus'),
            'can_trade': analysis_result.get('can_trade'),
            'should_enter': analysis_result.get('should_enter_trade'),
            'news_impact': analysis_result.get('news_context', {}).get('impact_score')
        }
        
        self._append_log(entry)
        logger.debug(f"Logged analysis: {symbol} {timeframe}")
    
    def log_trade_entry(
        self,
        trade_data: Dict,
        analysis_context: Optional[Dict] = None
    ):
        """
        Log trade entry
        
        Args:
            trade_data: Trade execution details
            analysis_context: Analysis that led to trade
        """
        entry = {
            'type': 'TRADE_ENTRY',
            'timestamp': datetime.now().isoformat(),
            'ticket': trade_data.get('ticket'),
            'symbol': trade_data.get('symbol'),
            'action': trade_data.get('action'),
            'lot_size': trade_data.get('lot_size'),
            'entry_price': trade_data.get('entry_price'),
            'stop_loss': trade_data.get('stop_loss'),
            'take_profit': trade_data.get('take_profit'),
            'confidence': trade_data.get('confidence'),
            'strategy': trade_data.get('strategy'),
            'reason': trade_data.get('reason'),
            'analysis_context': analysis_context
        }
        
        self._append_log(entry)
        logger.info(f"✅ Logged trade entry: {trade_data.get('symbol')} {trade_data.get('action')}")
    
    def log_trade_exit(
        self,
        trade_data: Dict,
        exit_reason: str
    ):
        """
        Log trade exit
        
        Args:
            trade_data: Trade details
            exit_reason: Reason for exit (TP, SL, Manual, etc.)
        """
        entry = {
            'type': 'TRADE_EXIT',
            'timestamp': datetime.now().isoformat(),
            'ticket': trade_data.get('ticket'),
            'symbol': trade_data.get('symbol'),
            'exit_price': trade_data.get('exit_price'),
            'profit': trade_data.get('profit'),
            'profit_pips': trade_data.get('profit_pips'),
            'exit_reason': exit_reason,
            'duration': trade_data.get('duration')
        }
        
        self._append_log(entry)
        logger.info(f"✅ Logged trade exit: {trade_data.get('symbol')} Profit: ${trade_data.get('profit', 0):.2f}")
    
    def log_risk_warning(
        self,
        warning_type: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """
        Log risk management warning
        
        Args:
            warning_type: Type of warning
            message: Warning message
            data: Additional data
        """
        entry = {
            'type': 'RISK_WARNING',
            'timestamp': datetime.now().isoformat(),
            'warning_type': warning_type,
            'message': message,
            'data': data
        }
        
        self._append_log(entry)
        logger.warning(f"⚠️ Risk warning: {warning_type} - {message}")
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict] = None
    ):
        """
        Log error
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context
        """
        entry = {
            'type': 'ERROR',
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        }
        
        self._append_log(entry)
        logger.error(f"❌ Error logged: {error_type} - {error_message}")
    
    def log_position_modification(
        self,
        ticket: int,
        modification_type: str,
        old_value: float,
        new_value: float,
        reason: str
    ):
        """
        Log position modification (SL/TP adjustment, trailing stop, etc.)
        
        Args:
            ticket: Position ticket
            modification_type: Type of modification (SL, TP, etc.)
            old_value: Old value
            new_value: New value
            reason: Reason for modification
        """
        entry = {
            'type': 'POSITION_MODIFICATION',
            'timestamp': datetime.now().isoformat(),
            'ticket': ticket,
            'modification_type': modification_type,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason
        }
        
        self._append_log(entry)
        logger.info(f"Modified position {ticket}: {modification_type} {old_value} -> {new_value}")
    
    def log_system_event(
        self,
        event_type: str,
        message: str,
        data: Optional[Dict] = None
    ):
        """
        Log system event (startup, shutdown, reconnection, etc.)
        
        Args:
            event_type: Type of event
            message: Event message
            data: Additional data
        """
        entry = {
            'type': 'SYSTEM_EVENT',
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            'data': data
        }
        
        self._append_log(entry)
        logger.info(f"System event: {event_type} - {message}")
    
    def get_todays_logs(self) -> list:
        """Get today's logs"""
        if not self.log_file.exists():
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading logs: {str(e)}")
            return []
    
    def get_trade_summary(self, period_days: int = 1) -> Dict:
        """
        Get trade summary for period
        
        Args:
            period_days: Number of days to look back
        
        Returns:
            Summary dictionary
        """
        logs = self.get_todays_logs()
        
        entries = [log for log in logs if log['type'] == 'TRADE_ENTRY']
        exits = [log for log in logs if log['type'] == 'TRADE_EXIT']
        analyses = [log for log in logs if log['type'] == 'ANALYSIS']
        warnings = [log for log in logs if log['type'] == 'RISK_WARNING']
        errors = [log for log in logs if log['type'] == 'ERROR']
        
        total_profit = sum(exit.get('profit', 0) for exit in exits)
        
        return {
            'period_days': period_days,
            'total_analyses': len(analyses),
            'total_trades': len(entries),
            'closed_trades': len(exits),
            'total_profit': total_profit,
            'warnings': len(warnings),
            'errors': len(errors)
        }
    
    def generate_summary_report(self) -> str:
        """Generate summary report for today"""
        summary = self.get_trade_summary()
        
        report = f"""
╔═══════════════════════════════════════════════════════════╗
║          TODAY'S TRADING SUMMARY                          ║
╚═══════════════════════════════════════════════════════════╝

Date: {datetime.now().strftime('%Y-%m-%d')}

ACTIVITY:
  Market Analyses: {summary['total_analyses']}
  Trades Entered: {summary['total_trades']}
  Trades Closed: {summary['closed_trades']}

RESULTS:
  Total Profit: ${summary['total_profit']:+.2f}

ALERTS:
  Risk Warnings: {summary['warnings']}
  Errors: {summary['errors']}

Log File: {self.log_file}
"""
        
        return report


if __name__ == "__main__":
    # Test trade logger
    logging.basicConfig(level=logging.INFO)
    
    trade_logger = TradeLogger()
    
    # Test logging
    print("Testing trade logger...\n")
    
    # Log analysis
    trade_logger.log_analysis(
        'XAUUSD',
        'M5',
        {
            'fusion': {'final_action': 'BUY', 'confidence': 0.75},
            'strategy_consensus': 'BUY',
            'can_trade': True,
            'should_enter_trade': True
        }
    )
    
    # Log trade entry
    trade_logger.log_trade_entry(
        {
            'ticket': 12345,
            'symbol': 'XAUUSD',
            'action': 'BUY',
            'lot_size': 0.01,
            'entry_price': 3850.00,
            'stop_loss': 3840.00,
            'take_profit': 3870.00,
            'confidence': 0.75,
            'strategy': 'counter_trend',
            'reason': 'Reversal at support'
        }
    )
    
    # Log trade exit
    trade_logger.log_trade_exit(
        {
            'ticket': 12345,
            'symbol': 'XAUUSD',
            'exit_price': 3860.00,
            'profit': 10.00,
            'profit_pips': 10.0,
            'duration': '2 hours'
        },
        'TP_HIT'
    )
    
    # Log risk warning
    trade_logger.log_risk_warning(
        'DAILY_LOSS_LIMIT',
        'Approaching daily loss limit',
        {'current_loss': -45.00, 'limit': -50.00}
    )
    
    # Log system event
    trade_logger.log_system_event(
        'BOT_START',
        'Trading bot started',
        {'account': 'DEMO', 'symbol': 'XAUUSD'}
    )
    
    # Generate summary
    print(trade_logger.generate_summary_report())
    
    print(f"\n✅ All logs saved to: {trade_logger.log_file}")

