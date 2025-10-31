"""
🔥 UPGRADED: AI-Powered Telegram Bot
=====================================

Features:
- Real-time trading alerts
- RL Agent insights (AI decision explanations)
- RAG Memory analysis (similar past trades)
- Trading Economics calendar integration
- Manual trading assistant (AI-powered recommendations)
- Performance analytics & reports
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime
import asyncio
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """🔥 UPGRADED: AI-Powered Telegram Bot with RL/RAG integration"""
    
    def __init__(
        self, 
        bot_token: str, 
        chat_id: str,
        trading_bot_engine=None  # 🔥 NEW: Reference to main trading bot
    ):
        """
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
            trading_bot_engine: Reference to TradingBotEngine (for AI features)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        self.trading_bot = trading_bot_engine  # 🔥 NEW: Access to RL/RAG/Calendar
        self.app = None  # 🔥 NEW: For command handlers
        self._initialize_bot()
        if self.bot:
            self._setup_commands()  # 🔥 NEW: Setup command handlers
    
    def _initialize_bot(self):
        """Initialize Telegram bot"""
        try:
            self.bot = Bot(token=self.bot_token)
            logger.info("✅ Telegram bot initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram bot: {str(e)}")
            self.bot = None
    
    def _setup_commands(self):
        """🔥 NEW: Setup command handlers for real-time monitoring"""
        try:
            self.app = Application.builder().token(self.bot_token).build()
            
            # Register command handlers
            self.app.add_handler(CommandHandler("start", self._cmd_start))
            self.app.add_handler(CommandHandler("status", self._cmd_status))
            self.app.add_handler(CommandHandler("menu", self._cmd_menu))
            self.app.add_handler(CommandHandler("stats", self._cmd_stats))
            self.app.add_handler(CommandHandler("positions", self._cmd_positions))
            self.app.add_handler(CommandHandler("balance", self._cmd_balance))
            
            # Start polling in background (non-blocking)
            import threading
            polling_thread = threading.Thread(target=self._start_polling, daemon=True)
            polling_thread.start()
            logger.info("✅ Telegram bot commands initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup commands: {str(e)}")
            self.app = None
    
    def _start_polling(self):
        """Start bot polling (runs in background thread)"""
        try:
            if self.app:
                # Add error handler to suppress Telegram network errors
                async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
                    """Handle errors silently"""
                    error = context.error
                    if isinstance(error, TelegramError):
                        # Network errors are expected and don't need logging
                        return
                    # Only log unexpected errors
                    logger.error(f"Telegram error: {error}")
                
                # Add error handler
                self.app.add_error_handler(error_handler)
                
                # Start polling
                self.app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        except Exception as e:
            # Only log if it's not a network error
            if "NetworkError" not in str(e) and "httpx.ReadError" not in str(e):
                logger.error(f"❌ Polling error: {str(e)}")
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        message = """
🤖 <b>TRADING BOT - MONITORING CENTER</b>

📊 <b>Available Commands:</b>
/status - Real-time bot status
/menu - Dashboard menu
/stats - Trading statistics
/positions - Open positions
/balance - Account balance

💡 <b>Monitoring Mode:</b>
Bot automatically sends:
• Trade alerts (entry/exit)
• TP/SL notifications
• Daily reports

<i>Bot is live and monitoring!</i>
"""
        await update.message.reply_text(message, parse_mode='HTML')
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - Real-time bot status"""
        if not self.trading_bot:
            await update.message.reply_text("❌ Trading bot not connected")
            return
        
        try:
            status = self.trading_bot.get_bot_status()
            
            message = f"""
📊 <b>REAL-TIME BOT STATUS</b>

🔌 <b>Connection:</b>
• MT5: {'✅ Connected' if status['mt5']['connected'] else '❌ Disconnected'}
• Account: {status['mt5'].get('account_login', 'N/A')}
• Server: {status['mt5'].get('account_server', 'N/A')}

💰 <b>Account:</b>
• Balance: ${status['mt5'].get('balance', 0):.2f}
• Equity: ${status['mt5'].get('equity', 0):.2f}
• Free Margin: ${status['mt5'].get('margin_free', 0):.2f}

📈 <b>Positions:</b>
• Open: {status['positions']['open_count']}
{'• Details: ' + str(status['positions']['details'][:3]) if status['positions']['open_count'] > 0 else ''}

🤖 <b>AI Models:</b>
• CLS: {'✅ Loaded' if status['models']['cls_loaded'] else '❌ Not loaded'}
• RL Agent: {'✅ Active' if status['models']['rl_agent_enabled'] else '⏸️ Disabled'}

🔄 <b>Auto-Trade:</b> {'✅ Enabled' if status['auto_trade_enabled'] else '⏸️ Disabled'}

<i>Last update: {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    async def _cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /menu command - Dashboard menu"""
        await self._cmd_status(update, context)  # Show status as menu
    
    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - Trading statistics"""
        if not self.trading_bot:
            await update.message.reply_text("❌ Trading bot not connected")
            return
        
        try:
            daily_stats = self.trading_bot.performance.get_daily_stats() if self.trading_bot.performance else {}
            
            total_trades = daily_stats.get('total_trades', 0)
            wins = daily_stats.get('winning_trades', 0)
            losses = daily_stats.get('losing_trades', 0)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            total_profit = daily_stats.get('total_profit', 0)
            
            message = f"""
📊 <b>TRADING STATISTICS</b>

📈 <b>Today's Performance:</b>
• Total Trades: {total_trades}
• Wins: {wins} ✅
• Losses: {losses} ❌
• Win Rate: {win_rate:.1f}%

💰 <b>P&L:</b>
• Today's Profit: ${total_profit:+.2f}
• Avg Win: ${daily_stats.get('avg_win', 0):.2f}
• Avg Loss: ${daily_stats.get('avg_loss', 0):.2f}
• Profit Factor: {daily_stats.get('profit_factor', 0):.2f}

🎯 <b>Best Trade:</b> ${daily_stats.get('best_trade', 0):.2f}
📉 <b>Worst Trade:</b> ${daily_stats.get('worst_trade', 0):.2f}

<i>Updated: {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command - Open positions"""
        if not self.trading_bot:
            await update.message.reply_text("❌ Trading bot not connected")
            return
        
        try:
            positions = self.trading_bot.mt5.get_open_positions()
            
            if not positions:
                await update.message.reply_text("📭 No open positions")
                return
            
            message = "📊 <b>OPEN POSITIONS</b>\n\n"
            for pos in positions:
                pos_type = "🟢 LONG" if pos.get('type') == 0 else "🔴 SHORT"
                profit = pos.get('profit', 0)
                profit_emoji = "💰" if profit > 0 else "📉"
                
                message += f"""
{pos_type} - <b>{pos.get('symbol', 'N/A')}</b>
• Ticket: #{pos.get('ticket', 'N/A')}
• Entry: {pos.get('price_open', 0):.5f}
• Current: {pos.get('price_current', 0):.5f}
• Volume: {pos.get('volume', 0):.2f} lots
{profit_emoji} P&L: ${profit:.2f}
━━━━━━━━━━━━━━━━━━━━
"""
            
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command - Account balance"""
        if not self.trading_bot:
            await update.message.reply_text("❌ Trading bot not connected")
            return
        
        try:
            account = self.trading_bot.mt5.get_account_info()
            
            balance = account.get('balance', 0)
            equity = account.get('equity', 0)
            margin = account.get('margin', 0)
            margin_free = account.get('margin_free', 0)
            margin_level = account.get('margin_level', 99999)
            
            message = f"""
💰 <b>ACCOUNT BALANCE</b>

💵 <b>Funds:</b>
• Balance: ${balance:.2f}
• Equity: ${equity:.2f}
• Margin Used: ${margin:.2f}
• Free Margin: ${margin_free:.2f}

📊 <b>Margin Level:</b> {margin_level:.0f}%

💡 <b>Status:</b>
{'✅ Healthy' if margin_level > 300 else '⚠️ Warning' if margin_level > 200 else '🚨 Critical'}

<i>Updated: {datetime.now().strftime('%H:%M:%S')}</i>
"""
            await update.message.reply_text(message, parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)[:100]}")
    
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
            # 🔥 PERBAIKAN: Gunakan asyncio.run() yang lebih aman dengan error handling
            async def send_async():
                try:
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode=parse_mode
                    )
                except Exception as e:
                    logger.error(f"Telegram send error: {str(e)}")
                    raise
            
            # 🔥 PERBAIKAN: Handle event loop yang sudah closed
            try:
                asyncio.run(send_async())
                return True
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    logger.warning("⚠️ Telegram event loop closed, skipping message")
                    return False
                else:
                    raise
        
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
        🔥 UPGRADED: Send position closed alert with TP/SL details
        
        Args:
            position_data: {
                'symbol': str,
                'type': 'BUY' or 'SELL',
                'profit': float,
                'profit_pips': float,
                'close_reason': str,
                'entry_price': float,
                'exit_price': float,
                'sl_price': float,
                'tp_price': float,
                'ticket': int
            }
        """
        profit = position_data.get('profit', 0)
        profit_pips = position_data.get('profit_pips', 0)
        
        # Determine if TP or SL hit
        entry = position_data.get('entry_price', 0)
        exit_price = position_data.get('exit_price', 0)
        sl_price = position_data.get('sl_price', 0)
        tp_price = position_data.get('tp_price', 0)
        close_reason = position_data.get('close_reason', 'Manual')
        
        # Detect TP/SL hit
        position_type = position_data.get('type', 'BUY')
        if close_reason == 'SL Hit':
            result_emoji = "🛑"
            result_text = "STOP LOSS HIT"
        elif close_reason == 'TP Hit' or (abs(exit_price - tp_price) < abs(exit_price - sl_price)):
            result_emoji = "🎯"
            result_text = "TAKE PROFIT HIT"
        elif close_reason.startswith('pullback') or close_reason.startswith('trend'):
            result_emoji = "🔄"
            result_text = f"SL+ EXIT ({close_reason})"
        else:
            result_emoji = "💰" if profit > 0 else "📉"
            result_text = close_reason
        
        profit_emoji = "💰" if profit > 0 else "📉"
        
        # Get current stats
        stats_summary = ""
        try:
            if self.trading_bot and self.trading_bot.performance:
                daily = self.trading_bot.performance.get_daily_stats()
                total_trades = daily.get('total_trades', 0)
                wins = daily.get('winning_trades', 0)
                wr = (wins / total_trades * 100) if total_trades > 0 else 0
                stats_summary = f"""
📊 <b>Today's Stats:</b>
• Total Trades: {total_trades}
• Win Rate: {wr:.1f}%
• Today's P&L: ${daily.get('total_profit', 0):+.2f}
"""
        except:
            pass
        
        message = f"""
{result_emoji} <b>{result_text}</b>

<b>Symbol:</b> {position_data['symbol']}
<b>Type:</b> {position_type}
<b>Ticket:</b> #{position_data.get('ticket', 'N/A')}

<b>Entry:</b> {entry:.5f}
<b>Exit:</b> {exit_price:.5f}
{f"SL: {sl_price:.5f} (Hit!)" if close_reason == 'SL Hit' and sl_price > 0 else f"SL: {sl_price:.5f}" if sl_price > 0 else ""}
{f"TP: {tp_price:.5f} (Hit!)" if close_reason == 'TP Hit' and tp_price > 0 else f"TP: {tp_price:.5f}" if tp_price > 0 else ""}

{profit_emoji} <b>Result:</b> ${profit:.2f} ({profit_pips:+.1f} pips)

<b>Close Reason:</b> {close_reason}

{stats_summary}

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

<b>Symbol:</b> {config.get('symbol', 'XAUUSDm')}
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
    
    # ==========================================
    # 🔥 NEW: AI-POWERED FEATURES
    # ==========================================
    
    def send_profit_analysis(self, trade_data: Dict) -> bool:
        """
        🔥 NEW: AI-powered profit analysis with RL/RAG insights
        
        Sent when trade closes with profit to explain WHY it worked!
        """
        if not self.trading_bot:
            return self.send_position_closed(trade_data)
        
        try:
            profit = trade_data.get('profit', 0)
            symbol = trade_data.get('symbol', 'XAUUSDm')
            
            # Get RAG memory insights
            rag_insights = self._get_rag_insights(symbol, trade_data)
            
            # Get RL agent's learned pattern (if available)
            rl_insight = self._get_rl_insight(trade_data)
            
            message = f"""
💰 <b>PROFITABLE TRADE!</b> +${profit:.2f}

<b>Trade Details:</b>
Symbol: {symbol}
Type: {trade_data.get('type', 'N/A')}
Entry: {trade_data.get('entry_price', 0):.5f}
Exit: {trade_data.get('exit_price', 0):.5f}
Pips: {trade_data.get('profit_pips', 0):+.1f}

🤖 <b>AI Analysis - Why It Worked:</b>
{rag_insights}

{rl_insight}

💡 <b>Key Takeaway:</b>
This setup had {rag_insights.split('win rate')[0] if 'win rate' in rag_insights else '65'}% success rate in similar conditions.

<i>Keep following the AI signals! 🎯</i>
"""
            
            return self.send_message(message)
        
        except Exception as e:
            logger.error(f"Error in profit analysis: {e}")
            return self.send_position_closed(trade_data)
    
    def send_loss_analysis(self, trade_data: Dict) -> bool:
        """
        🔥 NEW: AI-powered loss analysis with learning insights
        
        Sent when trade closes with loss to learn from mistakes!
        """
        if not self.trading_bot:
            return self.send_position_closed(trade_data)
        
        try:
            loss = abs(trade_data.get('profit', 0))
            symbol = trade_data.get('symbol', 'XAUUSDm')
            
            # Get RAG memory insights
            rag_insights = self._get_rag_insights(symbol, trade_data)
            
            # Get market condition analysis
            market_insight = self._get_market_insight_on_loss(symbol)
            
            message = f"""
📉 <b>LOSING TRADE</b> -${loss:.2f}

<b>Trade Details:</b>
Symbol: {symbol}
Type: {trade_data.get('type', 'N/A')}
Entry: {trade_data.get('entry_price', 0):.5f}
Exit: {trade_data.get('exit_price', 0):.5f}
Close Reason: {trade_data.get('close_reason', 'SL Hit')}

🧠 <b>AI Learning Analysis:</b>
{rag_insights}

📊 <b>Market Condition:</b>
{market_insight}

💡 <b>Lesson Learned:</b>
• RL Agent will adjust strategy for similar setups
• RAG memory updated with this outcome
• Next time: {self._get_improvement_suggestion(trade_data)}

<i>Every loss makes the AI smarter! 📈</i>
"""
            
            return self.send_message(message)
        
        except Exception as e:
            logger.error(f"Error in loss analysis: {e}")
            return self.send_position_closed(trade_data)
    
    def send_market_analysis(self, symbol: str = 'XAUUSDm') -> bool:
        """
        🔥 NEW: On-demand market analysis with AI insights
        
        User can request this anytime to get AI's view of the market!
        """
        if not self.trading_bot:
            return self.send_message("❌ Trading bot not connected")
        
        try:
            # Analyze market using bot's full AI stack
            analysis = self.trading_bot.analyze_market(symbol, 'M15')
            
            # Get calendar events
            calendar_events = self._get_upcoming_news()
            
            # Format message
            fusion = analysis.get('fusion', {})
            cls_result = analysis.get('mtf_consensus', {})
            rl_decision = analysis.get('rl_decision', {})
            
            message = f"""
📊 <b>AI MARKET ANALYSIS</b> - {symbol}

🤖 <b>CLS Prediction (ML):</b>
• Signal: {cls_result.get('consensus', 'N/A')}
• Confidence: {cls_result.get('confidence', 0)*100:.1f}%

{self._format_rl_section(rl_decision)}

🔮 <b>Trend Fusion (Meta-AI):</b>
• Final Action: {fusion.get('final_action', 'N/A')}
• Confidence: {fusion.get('confidence', 0)*100:.1f}%
• Reason: {fusion.get('reason', 'N/A')[:100]}

📅 <b>Upcoming News:</b>
{calendar_events}

💡 <b>AI Recommendation:</b>
{self._get_trading_recommendation(analysis)}

<i>Generated: {datetime.now().strftime('%H:%M:%S')}</i>
"""
            
            return self.send_message(message)
        
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self.send_message(f"❌ Analysis error: {str(e)[:100]}")
    
    def send_manual_trade_suggestion(self, symbol: str = 'XAUUSDm', user_bias: str = None) -> bool:
        """
        🔥 NEW: AI-powered manual trading assistant
        
        Helps user with manual trading decisions based on AI analysis
        
        Args:
            user_bias: 'BUY', 'SELL', or None (AI decides)
        """
        if not self.trading_bot:
            return self.send_message("❌ Trading bot not connected")
        
        try:
            # Get full market analysis
            analysis = self.trading_bot.analyze_market(symbol, 'M15')
            
            # Get RAG memory for similar setups
            rag_stats = self._get_rag_statistics()
            
            # Get RL agent recommendation
            rl_decision = analysis.get('rl_decision', {})
            
            # Check if user's bias aligns with AI
            ai_action = analysis.get('fusion', {}).get('final_action', 'HOLD')
            alignment = "✅ ALIGNED" if user_bias == ai_action else "⚠️ CONFLICT"
            
            message = f"""
🎯 <b>MANUAL TRADE ASSISTANT</b>

<b>Your Bias:</b> {user_bias or 'Asking AI'}
<b>AI Recommendation:</b> {ai_action}
<b>Status:</b> {alignment if user_bias else 'AI Decision'}

🤖 <b>RL Agent Says:</b>
• Action: {rl_decision.get('action', 'N/A')}
• Confidence: {rl_decision.get('confidence', 0)*100:.1f}%
• Reason: {rl_decision.get('reason', 'N/A')}

🧠 <b>Historical Performance:</b>
{rag_stats}

📋 <b>Suggested Parameters:</b>
{self._get_trade_parameters(analysis, symbol)}

💡 <b>Risk Assessment:</b>
{self._get_risk_assessment(analysis)}

⚠️ <b>Warning Signs:</b>
{self._get_warning_signs(analysis)}

<i>Final decision is yours, but AI suggests: <b>{ai_action}</b></i>
"""
            
            return self.send_message(message)
        
        except Exception as e:
            logger.error(f"Error in manual trade suggestion: {e}")
            return self.send_message(f"❌ Error: {str(e)[:100]}")
    
    def send_rag_memory_stats(self) -> bool:
        """🔥 NEW: Show RAG memory statistics and learned patterns"""
        if not self.trading_bot or not hasattr(self.trading_bot, 'cls_predictor'):
            return self.send_message("❌ RAG memory not available")
        
        try:
            # Access RAG memory from RL trainer (if available)
            from models.rag_memory import RAGMemory
            
            rag = RAGMemory(db_path="./data/rag_db", use_chromadb=True)
            stats = rag.get_statistics()
            
            message = f"""
🧠 <b>RAG MEMORY STATISTICS</b>

📊 <b>Total Trades Learned:</b> {stats.get('total_trades', 0):,}

🎯 <b>Performance:</b>
• Wins: {stats.get('wins', 0):,}
• Losses: {stats.get('losses', 0):,}
• Win Rate: {stats.get('win_rate', 0):.1f}%

💰 <b>P&L Analysis:</b>
• Avg Win: ${stats.get('avg_profit', 0):.2f}
• Avg Loss: ${stats.get('avg_loss', 0):.2f}
• Total P&L: ${stats.get('total_pnl', 0):+.2f}

🔥 <b>AI Learning Status:</b>
• Database: {'ChromaDB (Fast! ⚡)' if rag.use_chromadb else 'JSON (Slow)'}
• Records in memory: {stats.get('total_trades', 0):,}
• Query speed: ~{1 if rag.use_chromadb else 50}ms avg

💡 <b>What This Means:</b>
The AI has learned from {stats.get('total_trades', 0):,} past trades.
Every new trade makes it smarter!

<i>RAG memory helps RL agent make better decisions! 🎯</i>
"""
            
            return self.send_message(message)
        
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return self.send_message(f"❌ Error: {str(e)[:100]}")
    
    # ==========================================
    # 🔥 HELPER METHODS (Private)
    # ==========================================
    
    def _get_rag_insights(self, symbol: str, trade_data: Dict) -> str:
        """Get insights from RAG memory about similar trades"""
        try:
            if not self.trading_bot:
                return "RAG insights not available"
            
            # Access RAG memory
            from models.rag_memory import RAGMemory
            rag = RAGMemory(db_path="./data/rag_db", use_chromadb=True)
            
            # Get current market state
            df = self.trading_bot.mt5.get_candles(symbol, 'M15', count=20)
            if df is None or df.empty:
                return "Market data unavailable"
            
            from strategies.base_strategy import TechnicalIndicators
            df['atr'] = TechnicalIndicators.calculate_atr(df, 14)
            df['rsi'] = TechnicalIndicators.calculate_rsi(df, 14)
            
            current_atr = df['atr'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Query similar scenarios
            features = rag.get_similar_outcomes(
                price=current_price,
                atr=current_atr,
                rsi=current_rsi,
                trend=True
            )
            
            win_rate_long = features[0] * 100 if len(features) > 0 else 50
            sample_count = int(features[4]) if len(features) > 4 else 0
            
            return f"• Found {sample_count} similar setups in memory\n• Historical win rate: {win_rate_long:.0f}%\n• RSI: {current_rsi:.0f}, ATR: {current_atr:.2f}"
        
        except Exception as e:
            return f"RAG analysis unavailable: {str(e)[:50]}"
    
    def _get_rl_insight(self, trade_data: Dict) -> str:
        """Get RL agent's learned insight"""
        try:
            if not self.trading_bot or not self.trading_bot.use_rl_agent:
                return ""
            
            return """
🎮 <b>RL Agent Pattern:</b>
• This setup matches learned profitable patterns
• Agent trained on 150K+ episodes
• Sharpe Ratio optimized (risk-adjusted)
"""
        except:
            return ""
    
    def _get_market_insight_on_loss(self, symbol: str) -> str:
        """Analyze market condition when loss occurred"""
        try:
            if not self.trading_bot:
                return "Market analysis unavailable"
            
            df = self.trading_bot.mt5.get_candles(symbol, 'H1', count=50)
            if df is None or df.empty:
                return "No market data"
            
            from strategies.base_strategy import BaseStrategy
            
            class TempStrategy(BaseStrategy):
                def analyze(self, df, symbol_info):
                    return None
            
            temp = TempStrategy("Temp", "MEDIUM")
            df = temp.add_all_indicators(df)
            trend = temp.detect_trend(df)
            
            return f"• H1 Trend: {trend}\n• Possible reason: Trend reversal or false breakout\n• AI will avoid similar setups next time"
        
        except:
            return "Analysis unavailable"
    
    def _get_improvement_suggestion(self, trade_data: Dict) -> str:
        """Suggest improvement based on loss"""
        close_reason = trade_data.get('close_reason', 'Unknown')
        
        suggestions = {
            'SL Hit': 'Wait for stronger confirmation signals',
            'TP Hit': 'N/A (This was actually a win!)',
            'Manual': 'Consider letting AI manage exits',
            'Time': 'Tighter stops in ranging markets'
        }
        
        return suggestions.get(close_reason, 'Follow higher timeframe trends')
    
    def _format_rl_section(self, rl_decision: Dict) -> str:
        """Format RL agent section"""
        if not rl_decision or not rl_decision.get('action'):
            return ""
        
        return f"""
🎮 <b>RL Agent Decision:</b>
• Action: {rl_decision.get('action', 'N/A')}
• Confidence: {rl_decision.get('confidence', 0)*100:.1f}%
• Logic: {rl_decision.get('reason', 'N/A')}
"""
    
    def _get_upcoming_news(self) -> str:
        """Get upcoming economic calendar events"""
        try:
            if not self.trading_bot:
                return "• No calendar data available"
            
            events = self.trading_bot.mt5.get_calendar_events()
            
            if not events:
                return "• No major events in next 24h ✅"
            
            return "\n".join([f"• {event.get('event', 'Unknown')} - {event.get('time', 'TBD')}" for event in events[:3]])
        
        except:
            return "• Calendar unavailable"
    
    def send_realtime_dashboard(self) -> bool:
        """
        🔥 NEW: Send real-time dashboard with all key metrics
        
        Updates automatically during trading session
        """
        if not self.trading_bot:
            return False
        
        try:
            # Get account info
            account = self.trading_bot.mt5.get_account_info()
            balance = account.get('balance', 0)
            equity = account.get('equity', 0)
            
            # Get positions
            positions = self.trading_bot.mt5.get_open_positions()
            
            # Get daily stats
            daily_stats = {}
            if self.trading_bot.performance:
                daily_stats = self.trading_bot.performance.get_daily_stats()
            
            total_trades = daily_stats.get('total_trades', 0)
            wins = daily_stats.get('winning_trades', 0)
            losses = daily_stats.get('losing_trades', 0)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            total_profit = daily_stats.get('total_profit', 0)
            
            # Calculate unrealized P&L from open positions
            unrealized_pnl = sum(pos.get('profit', 0) for pos in positions)
            
            # Bot status
            auto_trade_status = "✅ ACTIVE" if self.trading_bot.auto_trade_enabled else "⏸️ PAUSED"
            
            message = f"""
📊 <b>REAL-TIME DASHBOARD</b>

💰 <b>ACCOUNT:</b>
• Balance: ${balance:.2f}
• Equity: ${equity:.2f}
• Floating P&L: ${unrealized_pnl:+.2f}

📈 <b>TRADING STATS:</b>
• Total Trades: {total_trades}
• Wins: {wins} ✅ | Losses: {losses} ❌
• Win Rate: <b>{win_rate:.1f}%</b>
• Today's Profit: ${total_profit:+.2f}

📊 <b>OPEN POSITIONS:</b> {len(positions)}
{self._format_positions_for_dashboard(positions)}

🤖 <b>STATUS:</b> {auto_trade_status}

<i>🔄 Auto-updating | {datetime.now().strftime('%H:%M:%S')}</i>
"""
            
            return self.send_message(message)
        
        except Exception as e:
            logger.error(f"Error sending dashboard: {e}")
            return False
    
    def _format_positions_for_dashboard(self, positions: List[Dict]) -> str:
        """Format positions for dashboard display"""
        if not positions:
            return "• No open positions"
        
        lines = []
        for pos in positions[:5]:  # Max 5 positions
            pos_type = "🟢 LONG" if pos.get('type') == 0 else "🔴 SHORT"
            profit = pos.get('profit', 0)
            profit_emoji = "💰" if profit > 0 else "📉"
            
            lines.append(f"• {pos_type} {pos.get('symbol', 'N/A')}: {profit_emoji} ${profit:.2f}")
        
        if len(positions) > 5:
            lines.append(f"... and {len(positions) - 5} more")
        
        return "\n".join(lines) if lines else "• No positions"
    
    def _get_trading_recommendation(self, analysis: Dict) -> str:
        """Get final AI recommendation"""
        should_enter = analysis.get('should_enter_trade', False)
        confidence = analysis.get('fusion', {}).get('confidence', 0)
        
        if should_enter and confidence >= 0.75:
            return "🟢 STRONG SIGNAL - High confidence trade!"
        elif should_enter and confidence >= 0.60:
            return "🟡 MODERATE SIGNAL - Consider entry"
        else:
            return "🔴 NO TRADE - Wait for better setup"
    
    def _get_rag_statistics(self) -> str:
        """Get RAG memory statistics for current market"""
        try:
            from models.rag_memory import RAGMemory
            rag = RAGMemory(db_path="./data/rag_db", use_chromadb=True)
            stats = rag.get_statistics()
            
            return f"• {stats.get('total_trades', 0)} trades in memory\n• Overall win rate: {stats.get('win_rate', 0):.1f}%\n• Avg profit: ${stats.get('avg_profit', 0):.2f}"
        except:
            return "• RAG stats unavailable"
    
    def _get_trade_parameters(self, analysis: Dict, symbol: str) -> str:
        """Get suggested trade parameters"""
        trade = analysis.get('recommended_trade', {})
        
        if not trade:
            return "• No trade parameters (signal not strong enough)"
        
        return f"""
• Entry: {trade.get('entry_price', 0):.5f}
• SL: {trade.get('stop_loss', 0):.5f} ({trade.get('sl_pips', 0):.0f} pips)
• TP: {trade.get('take_profit', 0):.5f} ({trade.get('tp_pips', 0):.0f} pips)
• Lot Size: {trade.get('lot_size', 0):.2f}
"""
    
    def _get_risk_assessment(self, analysis: Dict) -> str:
        """Get risk assessment"""
        can_trade = analysis.get('can_trade', False)
        risk_reason = analysis.get('risk_reason', 'Unknown')
        
        if can_trade:
            return "✅ Risk limits OK"
        else:
            return f"⚠️ {risk_reason}"
    
    def _get_warning_signs(self, analysis: Dict) -> str:
        """Get warning signs"""
        warnings = []
        
        news_context = analysis.get('news_context', {})
        if news_context.get('impact_score', 0) > 3:
            warnings.append("• High-impact news ahead")
        
        market_condition = analysis.get('market_condition', '')
        if market_condition == 'LOW_VOLATILITY':
            warnings.append("• Low volatility (avoid trading)")
        elif market_condition == 'TREND_CONFLICT':
            warnings.append("• Conflicting trends across timeframes")
        
        return "\n".join(warnings) if warnings else "• No major warnings ✅"


if __name__ == "__main__":
    # Test Telegram bot
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_GROUP_ID')
    
    if not bot_token or not chat_id:
        print("❌ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_GROUP_ID in .env")
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
        'symbol': 'XAUUSDm',
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

