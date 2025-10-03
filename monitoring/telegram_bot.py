"""
Telegram Bot for Trading Notifications
Send real-time alerts and reports via Telegram
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import asyncio
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send trading notifications via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize Telegram bot"""
        try:
            self.bot = Bot(token=self.bot_token)
            logger.info("✅ Telegram bot initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram bot: {str(e)}")
            self.bot = None
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a text message
        
        Args:
            message: Message text
            parse_mode: 'HTML' or 'Markdown'
        
        Returns:
            True if sent successfully
        """
        if not self.bot:
            logger.warning("Telegram bot not initialized")
            return False
        
        try:
            # Run async in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
            )
            loop.close()
            return True
        
        except TelegramError as e:
            logger.error(f"Telegram error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return False
    
    def send_trade_alert(self, trade_data: Dict) -> bool:
        """
        Send trade execution alert
        
        Args:
            trade_data: {
                'symbol': str,
                'type': 'BUY' or 'SELL',
                'entry': float,
                'sl': float,
                'tp': float,
                'lot_size': float,
                'confidence': float,
                'reason': str
            }
        """
        # Format message
        action_emoji = "🟢" if trade_data['type'] == 'BUY' else "🔴"
        
        message = f"""
{action_emoji} <b>TRADE EXECUTED</b>

<b>Symbol:</b> {trade_data['symbol']}
<b>Action:</b> {trade_data['type']}
<b>Lot Size:</b> {trade_data.get('lot_size', 'N/A')}

<b>Entry:</b> {trade_data['entry']:.5f}
<b>Stop Loss:</b> {trade_data['sl']:.5f}
<b>Take Profit:</b> {trade_data['tp']:.5f}

<b>Confidence:</b> {trade_data.get('confidence', 0)*100:.1f}%
<b>Reason:</b> {trade_data.get('reason', 'N/A')}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        return self.send_message(message)
    
    def send_position_closed(self, position_data: Dict) -> bool:
        """
        Send position closed alert
        
        Args:
            position_data: {
                'symbol': str,
                'type': 'BUY' or 'SELL',
                'profit': float,
                'profit_pips': float,
                'close_reason': str
            }
        """
        profit = position_data.get('profit', 0)
        profit_emoji = "💰" if profit > 0 else "📉"
        
        message = f"""
{profit_emoji} <b>POSITION CLOSED</b>

<b>Symbol:</b> {position_data['symbol']}
<b>Type:</b> {position_data['type']}

<b>Profit:</b> ${profit:.2f}
<b>Pips:</b> {position_data.get('profit_pips', 0):+.1f}

<b>Close Reason:</b> {position_data.get('close_reason', 'Manual')}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        return self.send_message(message)
    
    def send_daily_report(self, report_data: Dict) -> bool:
        """
        Send daily performance report
        
        Args:
            report_data: {
                'date': str,
                'total_trades': int,
                'winning_trades': int,
                'losing_trades': int,
                'total_profit': float,
                'win_rate': float,
                'balance': float,
                'equity': float
            }
        """
        win_rate = report_data.get('win_rate', 0) * 100
        total_profit = report_data.get('total_profit', 0)
        
        profit_emoji = "📈" if total_profit > 0 else "📉"
        
        message = f"""
📊 <b>DAILY REPORT</b>

<b>Date:</b> {report_data.get('date', datetime.now().strftime('%Y-%m-%d'))}

<b>Trades:</b> {report_data.get('total_trades', 0)}
├ Winning: {report_data.get('winning_trades', 0)}
└ Losing: {report_data.get('losing_trades', 0)}

{profit_emoji} <b>Profit:</b> ${total_profit:+.2f}
<b>Win Rate:</b> {win_rate:.1f}%

<b>Balance:</b> ${report_data.get('balance', 0):.2f}
<b>Equity:</b> ${report_data.get('equity', 0):.2f}

<i>Report generated at {datetime.now().strftime('%H:%M:%S')}</i>
"""
        
        return self.send_message(message)
    
    def send_error_alert(self, error_message: str, context: Optional[str] = None) -> bool:
        """Send error alert"""
        message = f"""
⚠️ <b>ERROR ALERT</b>

<b>Error:</b> {error_message}

{f'<b>Context:</b> {context}' if context else ''}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        return self.send_message(message)
    
    def send_risk_warning(self, warning_data: Dict) -> bool:
        """
        Send risk management warning
        
        Args:
            warning_data: {
                'type': str,
                'message': str,
                'current_value': float,
                'threshold': float
            }
        """
        message = f"""
🚨 <b>RISK WARNING</b>

<b>Type:</b> {warning_data.get('type', 'Unknown')}
<b>Message:</b> {warning_data.get('message', '')}

<b>Current:</b> {warning_data.get('current_value', 0):.2f}
<b>Threshold:</b> {warning_data.get('threshold', 0):.2f}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        return self.send_message(message)
    
    def send_startup_message(self, config: Dict) -> bool:
        """Send bot startup notification"""
        message = f"""
🤖 <b>TRADING BOT STARTED</b>

<b>Symbol:</b> {config.get('symbol', 'XAUUSD')}
<b>Timeframe:</b> {config.get('timeframe', 'M5')}
<b>Account Type:</b> {config.get('account_type', 'DEMO')}

<b>Risk per Trade:</b> {config.get('risk_per_trade', 1.0):.1f}%
<b>Max Daily Loss:</b> {config.get('max_daily_loss', 5.0):.1f}%
<b>Max Positions:</b> {config.get('max_positions', 3)}

<b>Strategies:</b>
{chr(10).join('• ' + s for s in config.get('strategies', []))}

<i>Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        return self.send_message(message)
    
    def send_shutdown_message(self, stats: Dict) -> bool:
        """Send bot shutdown notification"""
        message = f"""
🛑 <b>TRADING BOT STOPPED</b>

<b>Session Stats:</b>
• Trades: {stats.get('total_trades', 0)}
• Profit: ${stats.get('session_profit', 0):+.2f}
• Win Rate: {stats.get('win_rate', 0)*100:.1f}%

<b>Open Positions:</b> {stats.get('open_positions', 0)}

<i>Stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        return self.send_message(message)


if __name__ == "__main__":
    # Test Telegram bot
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        exit(1)
    
    notifier = TelegramNotifier(bot_token, chat_id)
    
    # Test message
    print("📤 Sending test message...")
    success = notifier.send_message("🤖 <b>Test Message</b>\n\nTelegram bot is working!")
    
    if success:
        print("✅ Test message sent successfully!")
    else:
        print("❌ Failed to send test message")
    
    # Test trade alert
    print("\n📤 Sending test trade alert...")
    trade_data = {
        'symbol': 'XAUUSD',
        'type': 'BUY',
        'entry': 3850.50,
        'sl': 3840.00,
        'tp': 3870.00,
        'lot_size': 0.01,
        'confidence': 0.75,
        'reason': 'Counter-trend reversal at support'
    }
    
    notifier.send_trade_alert(trade_data)
    
    print("✅ Check your Telegram for notifications!")

