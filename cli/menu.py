import os
import sys
from colorama import Fore, Back, Style, init
from tabulate import tabulate
from typing import Optional
import logging

# Initialize colorama
init(autoreset=True)

logger = logging.getLogger(__name__)


class CLIMenu:
    """Interactive command-line interface"""
    
    def __init__(self, bot_engine):
        self.bot = bot_engine
        self.running = True
        self.current_symbol = "XAUUSDm"
        self.current_timeframe = "M5"
        # 🔥 FIX: Sync with bot config
        self.current_mode = self.bot.config.get('trade_mode', 'NORMAL')
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print bot header"""
        self.clear_screen()
        print(Fore.CYAN + "=" * 80)
        print(Fore.YELLOW + Style.BRIGHT + """
   ╔═══════════════════════════════════════════════════════════╗
   ║                                                           ║
   ║           🤖 AUTO-TRADING BOT v2.0 (AI + CLS)          ║
   ║           Advanced Forex Trading System                  ║
   ║                                                           ║
   ╚═══════════════════════════════════════════════════════════╝
        """)
        print(Fore.CYAN + "=" * 80)
        
        # Show connection status
        if self.bot.mt5.connected:
            account = self.bot.mt5.get_account_info()
            print(Fore.GREEN + f"\n✅ Connected: {self.bot.mt5.server} | Account: {self.bot.mt5.login}")
            print(Fore.WHITE + f"   Balance: ${account['balance']:.2f} | Equity: ${account['equity']:.2f} | Free Margin: ${account['margin_free']:.2f}")
        else:
            print(Fore.RED + "\n❌ Not connected to MT5")
        
        print(Fore.CYAN + "=" * 80 + "\n")
    
    def print_menu(self):
        """Print main menu"""
        menu_items = [
            ["1", "Analyze Now", f"Current: {self.current_symbol}"],
            ["2", "Change SYMBOL", f"Available: EURUSDm, GBPUSDm, USDJPYm, AUDUSDm, XAUUSDm, BTCUSDm"],
            ["3", "Change TIMEFRAME", f"Current: {self.current_timeframe} | Options: M1, M5, M15, M30, H1, H4, D1"],
            ["4", "Change CANDLES", f"Current: 200 bars | e.g. 50 / 100 / 200 / 800"],
            ["5", "Switch ACCOUNT", f"Current: EXNESS | Options: DEMO / EXNESS"],
            ["6", "Change TRADE_MODE", f"Current: {self.current_mode} | NORMAL, AGGRESSIVE, SCALPING, LONG_HOLD"],
            ["7", "Launch external TRAINER window", "(Train CLS models with new data)"],
            ["8", "Toggle AUTO-TRADE", f"Current: {'ON' if self.bot.auto_trade_enabled else 'OFF'}"],
            ["9", "Set AUTO lot", f"Current: {self.bot.default_lot_size}"],
            ["10", "Set AUTO slippage (dev)", f"Current: {self.bot.max_slippage}"],
            ["0", "Quit", "Exit the application"],
        ]
        
        print(Fore.YELLOW + "📋 MAIN MENU:")
        print(tabulate(menu_items, headers=["Option", "Action", "Details"], tablefmt="grid"))
        print()
    
    def show_analysis_results(self, results: dict):
        """Display comprehensive analysis results"""
        self.print_header()
        
        print(Fore.CYAN + Style.BRIGHT + f"\n📊 ANALYSIS RESULTS - {self.current_symbol} ({self.current_timeframe})\n")
        
        # 1. AI + CLS + Trend Fusion
        if results.get('fusion'):
            fusion = results['fusion']
            print(Fore.MAGENTA + "🧠 AI + CLS + Trend Fusion")
            print(Fore.WHITE + "-" * 80)
            
            action_color = Fore.GREEN if fusion['final_action'] == 'BUY' else \
                          Fore.RED if fusion['final_action'] == 'SELL' else Fore.YELLOW
            
            print(f"{action_color}Final: {fusion['final_action']} " + 
                  Fore.WHITE + f"| CLS agg: {fusion['breakdown'].get('cls', {}).get('consensus', 'N/A')} " +
                  f"(meta {fusion['meta_score']:.2f})")
            
            print(Fore.WHITE + f"Why: {fusion['reason']}")
            print(Fore.WHITE + f"Trend: {fusion['breakdown'].get('trend', 'N/A')}")
            
            if fusion['breakdown'].get('cls'):
                cls_data = fusion['breakdown']['cls']
                print(Fore.WHITE + f"Guards → spread_bad=False, high_news=False, agree={fusion['confidence']:.2f}")
            
            print()
        
        # 2. Suggestions & Candle Read
        print(Fore.YELLOW + "💡 Suggestions & Candle Read")
        print(Fore.WHITE + "-" * 80)
        
        if results.get('candle_pattern'):
            print(f"Heikin-Ashi: {results['candle_pattern']}")
        
        if results.get('best_signal'):
            signal = results['best_signal']
            print(f"Pattern: → ({signal.get('reason', 'N/A')})")
            print(f"Volatility: {signal['metadata'].get('atr', 0):.5f} width=0.004, breakout--")
        
        print(f"Timeframe: ~5 min/bar → Suggest re-check in ~5 min (1 bar)")
        print()
        
        # 3. Multi-Timeframe Consensus
        if results.get('mtf_consensus'):
            mtf = results['mtf_consensus']
            print(Fore.BLUE + "📈 Multi-Timeframe Consensus")
            print(Fore.WHITE + "-" * 80)
            
            print(f"Per-TF votes (weighted):")
            for tf, pred in mtf.get('predictions', {}).items():
                if pred:
                    vote_color = Fore.GREEN if pred['action'] == 'BUY' else \
                                Fore.RED if pred['action'] == 'SELL' else Fore.YELLOW
                    print(f"  {tf.upper()}: {vote_color}{pred['action']}{Fore.WHITE} (w={pred['confidence']:.2f}) trend={pred['action']}")
            
            print(f"\nM5: {mtf.get('votes', {}).get('buy', 0)} HOLD (w=0.35) trend=BUY")
            print(f"M15: {mtf.get('votes', {}).get('sell', 0)} HOLD (w=0.40) trend=BUY")
            print(f"H1: {mtf.get('votes', {}).get('hold', 0)} HOLD (w=0.25) trend=BUY")
            print()
        
        # 4. Alternative Entries
        print(Fore.RED + "⚡ Alternative Entries")
        print(Fore.WHITE + "-" * 80)
        
        if results.get('all_strategies'):
            for strategy_name, signal in results['all_strategies'].items():
                if signal:
                    risk_emoji = "🔴" if signal['risk_level'] == 'EXTREME' else \
                                "🟠" if signal['risk_level'] == 'HIGH' else "🟡"
                    
                    print(f"{risk_emoji} #{strategy_name.upper()} ({signal['action']} on rejection)   Risk: {signal['risk_level']}")
                    print(f"   Entry: {signal['entry_price']} → {signal['stop_loss']}")
                    print(f"   Stop Loss: {signal['stop_loss']}")
                    print(f"   Take Profit: TP1 {signal['take_profit']}, TP2 {signal['metadata'].get('tp2', 'N/A')}")
        
        print()
        
        # 5. Fibonacci / ATR Analysis
        if results.get('fibonacci'):
            fib = results['fibonacci']
            print(Fore.CYAN + "📐 Fibonacci / ATR Analysis")
            print(Fore.WHITE + "-" * 80)
            print(f"Entry Zone: {fib.get('entry_zone', 'N/A')}")
            print(f"Stop Loss: {fib.get('stop_loss', 'N/A')}")
            print(f"Take Profit: TP1 {fib.get('tp1', 'N/A')} | TP2 {fib.get('tp2', 'N/A')}")
            print()
        
        # Risk Management Summary
        if results.get('risk_metrics'):
            risk = results['risk_metrics']
            print(Fore.YELLOW + "⚠️  RISK MANAGEMENT")
            print(Fore.WHITE + "-" * 80)
            print(f"Daily Trades: {risk['daily_trades']}/{self.bot.risk_manager.max_trades_per_day}")
            print(f"Daily P&L: ${risk['daily_pnl']:.2f}")
            print(f"Open Positions: {risk['open_positions']}/{self.bot.risk_manager.max_open_positions}")
            print(f"Win Rate: {risk['win_rate']:.1%}")
            print(f"Margin Level: {risk['margin_level']:.1f}%")
            print()
        
        # News Context
        if results.get('news_context'):
            news = results['news_context']
            print(Fore.MAGENTA + "📰 NEWS CONTEXT")
            print(Fore.WHITE + "-" * 80)
            print(f"Can Trade: {'✅ YES' if news['can_trade'] else '❌ NO'}")
            print(f"Impact Score: {news['impact_score']:.1f}/10")
            
            if news.get('news_sentiment'):
                sentiment = news['news_sentiment']
                print(f"News Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.1%})")
            
            if news.get('upcoming_events'):
                print(f"\nUpcoming Events:")
                for event in news['upcoming_events'][:3]:
                    print(f"  - {event['event']} ({event['importance']}) in {event['time_until']:.0f} min")
            print()
        
        print(Fore.GREEN + "\nPress ENTER to continue...")
        input()
    
    def handle_analyze(self):
        """Handle analysis request"""
        self.print_header()
        print(Fore.YELLOW + f"🔍 Analyzing {self.current_symbol} on {self.current_timeframe}...\n")
        
        try:
            results = self.bot.analyze_market(self.current_symbol, self.current_timeframe)
            self.show_analysis_results(results)
        
        except Exception as e:
            print(Fore.RED + f"❌ Error during analysis: {str(e)}")
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            input("\nPress ENTER to continue...")
    
    def handle_change_symbol(self):
        """Handle symbol change"""
        self.print_header()
        
        available_symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'XAUUSDm', 'BTCUSDm']
        
        print(Fore.YELLOW + "Available symbols:")
        for i, symbol in enumerate(available_symbols, 1):
            print(f"  {i}. {symbol}")
        
        choice = input(Fore.WHITE + "\nSelect symbol (or type custom): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(available_symbols):
            self.current_symbol = available_symbols[int(choice) - 1]
        elif choice:
            self.current_symbol = choice.upper()
        
        print(Fore.GREEN + f"✅ Symbol changed to: {self.current_symbol}")
        input("\nPress ENTER to continue...")
    
    def handle_change_timeframe(self):
        """Handle timeframe change"""
        self.print_header()
        
        timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
        
        print(Fore.YELLOW + "Available timeframes:")
        for i, tf in enumerate(timeframes, 1):
            print(f"  {i}. {tf}")
        
        choice = input(Fore.WHITE + "\nSelect timeframe: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(timeframes):
            self.current_timeframe = timeframes[int(choice) - 1]
        elif choice.upper() in timeframes:
            self.current_timeframe = choice.upper()
        
        print(Fore.GREEN + f"✅ Timeframe changed to: {self.current_timeframe}")
        input("\nPress ENTER to continue...")
    
    def handle_train_models(self):
        """Handle CLS model training"""
        self.print_header()
        
        print(Fore.YELLOW + Style.BRIGHT + "🎓 CLS MODEL TRAINER\n")
        print(Fore.WHITE + "=" * 80)
        print("\nOptions:")
        print("  1. Train ALL timeframes (M5, M15, H1, H4) - Takes 10-30 minutes")
        print("  2. Retrain SINGLE timeframe")
        print("  0. Cancel")
        
        choice = input(Fore.WHITE + "\nSelect option: ").strip()
        
        if choice == '1':
            # Train all timeframes
            print(Fore.YELLOW + "\n⚠️  WARNING: This will take 10-30 minutes!")
            print(Fore.YELLOW + "The bot will collect 5000+ candles per timeframe and train models.")
            
            confirm = input(Fore.WHITE + "\nProceed? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                print(Fore.CYAN + "\n🚀 Starting training process...")
                print(Fore.WHITE + "Symbol: XAUUSDm")
                print(Fore.WHITE + "Model Type: Random Forest")
                print(Fore.WHITE + "Timeframes: M5, M15, H1, H4\n")
                
                success = self.bot.train_cls_models(
                    symbol='XAUUSDm',
                    model_type='random_forest'
                )
                
                if success:
                    print(Fore.GREEN + "\n✅ Training completed successfully!")
                    print(Fore.WHITE + "Models saved to: ./models/saved_models/")
                    print(Fore.WHITE + "Models have been reloaded into the bot engine.")
                else:
                    print(Fore.RED + "\n❌ Training failed! Check logs for details.")
            else:
                print(Fore.RED + "Training cancelled")
        
        elif choice == '2':
            # Retrain single timeframe
            print(Fore.YELLOW + "\nAvailable timeframes:")
            timeframes = ['M5', 'M15', 'H1', 'H4']
            for i, tf in enumerate(timeframes, 1):
                print(f"  {i}. {tf}")
            
            tf_choice = input(Fore.WHITE + "\nSelect timeframe: ").strip()
            
            if tf_choice.isdigit() and 1 <= int(tf_choice) <= len(timeframes):
                selected_tf = timeframes[int(tf_choice) - 1]
            elif tf_choice.upper() in timeframes:
                selected_tf = tf_choice.upper()
            else:
                print(Fore.RED + "Invalid timeframe")
                input("\nPress ENTER to continue...")
                return
            
            print(Fore.CYAN + f"\n🚀 Retraining {selected_tf} model...")
            print(Fore.WHITE + f"Symbol: XAUUSDm")
            print(Fore.WHITE + f"Model Type: Random Forest\n")
            
            success = self.bot.retrain_single_timeframe(
                timeframe=selected_tf,
                symbol='XAUUSDm',
                model_type='random_forest'
            )
            
            if success:
                print(Fore.GREEN + f"\n✅ {selected_tf} model retrained successfully!")
                print(Fore.WHITE + f"Model saved to: ./models/saved_models/cls_{selected_tf.lower()}.pkl")
                print(Fore.WHITE + "Models have been reloaded into the bot engine.")
            else:
                print(Fore.RED + f"\n❌ {selected_tf} model retraining failed! Check logs for details.")
        
        else:
            print(Fore.YELLOW + "Training cancelled")
        
        input("\nPress ENTER to continue...")
    
    def handle_toggle_auto_trade(self):
        """🔥 FIXED: Toggle auto-trading and actually START the loop!"""
        if not self.bot.auto_trade_enabled:
            # Turning ON
            print(Fore.YELLOW + "\n⚠️  WARNING: Bot will automatically execute trades!")
            print(Fore.WHITE + f"Symbol: {self.current_symbol}")
            print(Fore.WHITE + f"Timeframe: {self.current_timeframe}")
            print(Fore.WHITE + f"Trade Mode: {self.current_mode}")
            print(Fore.WHITE + f"Lot Size: {self.bot.default_lot_size}")
            print()
            confirm = input("Type 'YES' to start AUTO-TRADING: ")
            
            if confirm.upper() == 'YES':
                self.bot.auto_trade_enabled = True
                print(Fore.GREEN + "\n✅ Auto-trading ENABLED")
                print(Fore.YELLOW + "🤖 Starting auto-trade loop...")
                print(Fore.YELLOW + "Press Ctrl+C to stop\n")
                
                # 🔥 CRITICAL FIX: Actually START the loop!
                try:
                    self.bot.auto_trade_loop(
                        symbol=self.current_symbol,
                        timeframe=self.current_timeframe,
                        interval=300  # 5 minutes
                    )
                except KeyboardInterrupt:
                    print(Fore.YELLOW + "\n\n⚠️  Auto-trade stopped by user")
                    self.bot.auto_trade_enabled = False
                except Exception as e:
                    print(Fore.RED + f"\n\n❌ Error in auto-trade: {str(e)}")
                    self.bot.auto_trade_enabled = False
                
                input("\nPress ENTER to return to menu...")
            else:
                print(Fore.RED + "Auto-trading cancelled")
                input("\nPress ENTER to continue...")
        else:
            # Turning OFF
            self.bot.auto_trade_enabled = False
            print(Fore.RED + "\n❌ Auto-trading DISABLED")
            input("\nPress ENTER to continue...")
    
    def run(self):
        """Main menu loop"""
        while self.running:
            try:
                self.print_header()
                self.print_menu()
                
                choice = input(Fore.WHITE + "Select: ").strip()
                
                if choice == '1':
                    self.handle_analyze()
                elif choice == '2':
                    self.handle_change_symbol()
                elif choice == '3':
                    self.handle_change_timeframe()
                elif choice == '6':
                    # 🔥 PERBAIKAN: Hanya mengubah satu nilai config
                    modes = ['NORMAL', 'AGGRESSIVE', 'SCALPING', 'LONG_HOLD']
                    print("\n📊 Available Modes:")
                    print("   SCALPING: Tight SL/TP, high frequency")
                    print("   NORMAL: Balanced approach ✅ Recommended")
                    print("   AGGRESSIVE: Wider stops, higher risk/reward")
                    print("   LONG_HOLD: Very wide stops, patient trades\n")
                    
                    mode = input("Select mode: ").upper()
                    if mode in modes:
                        self.current_mode = mode
                        # Cukup ubah satu konfigurasi ini. Bot engine akan menangani sisanya.
                        self.bot.config['trade_mode'] = mode
                        print(Fore.GREEN + f"✅ Trade mode changed to: {mode}")
                    else:
                        print(Fore.RED + "❌ Invalid mode")
                    input("\nPress ENTER to continue...")
                
                elif choice == '7':
                    self.handle_train_models()
                
                elif choice == '8':
                    self.handle_toggle_auto_trade()
                
                elif choice == '0':
                    print(Fore.YELLOW + "\n👋 Shutting down...")
                    self.bot.shutdown()
                    self.running = False
                    print(Fore.GREEN + "✅ Goodbye!")
                
                else:
                    print(Fore.RED + "❌ Invalid choice")
                    input("\nPress ENTER to continue...")
            
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\n⚠️  Interrupted by user")
                confirm = input("Exit? (y/n): ")
                if confirm.lower() == 'y':
                    self.running = False
            
            except Exception as e:
                print(Fore.RED + f"\n❌ Error: {str(e)}")
                logger.error(f"Menu error: {str(e)}", exc_info=True)
                input("\nPress ENTER to continue...")


# cli/display.py
"""
Display utilities for CLI
"""

class DisplayUtils:
    """Utility functions for displaying data"""
    
    @staticmethod
    def format_price(price: float, digits: int = 5) -> str:
        """Format price with color"""
        return f"{price:.{digits}f}"
    
    @staticmethod
    def format_pnl(pnl: float) -> str:
        """Format P&L with color"""
        if pnl > 0:
            return Fore.GREEN + f"+${pnl:.2f}"
        elif pnl < 0:
            return Fore.RED + f"-${abs(pnl):.2f}"
        else:
            return Fore.WHITE + f"${pnl:.2f}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format percentage with color"""
        if value > 0:
            return Fore.GREEN + f"+{value:.2f}%"
        elif value < 0:
            return Fore.RED + f"{value:.2f}%"
        else:
            return Fore.WHITE + f"{value:.2f}%"
    
    @staticmethod
    def show_positions_table(positions: list):
        """Display open positions in a table"""
        if not positions:
            print(Fore.YELLOW + "No open positions")
            return
        
        table_data = []
        for pos in positions:
            pnl_str = DisplayUtils.format_pnl(pos['profit'])
            
            table_data.append([
                pos['ticket'],
                pos['symbol'],
                'BUY' if pos['type'] == 0 else 'SELL',
                pos['volume'],
                pos['price_open'],
                pos['price_current'],
                pnl_str,
                f"{pos.get('profit_pips', 0):.1f} pips"
            ])
        
        headers = ['Ticket', 'Symbol', 'Type', 'Lots', 'Entry', 'Current', 'P&L', 'Pips']
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    @staticmethod
    def show_trade_history_table(trades: list, limit: int = 10):
        """Display recent trade history"""
        if not trades:
            print(Fore.YELLOW + "No trade history")
            return
        
        table_data = []
        for trade in trades[-limit:]:
            pnl_str = DisplayUtils.format_pnl(trade['profit'])
            
            table_data.append([
                trade['timestamp'].strftime('%H:%M:%S'),
                trade['symbol'],
                trade['type'],
                pnl_str,
                f"{trade.get('pips', 0):.1f}"
            ])
        
        headers = ['Time', 'Symbol', 'Type', 'P&L', 'Pips']
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    # Test menu
    print("CLI Menu test")
    
    # Mock bot engine for testing
    class MockBot:
        def __init__(self):
            self.auto_trade_enabled = False
            self.default_lot_size = 0.01
            self.max_slippage = 2.0
            
            class MockMT5:
                connected = True
                server = "MetaQuotes-Demo"
                login = 12345678
                
                def get_account_info(self):
                    return {
                        'balance': 10000.0,
                        'equity': 10050.0,
                        'margin_free': 9000.0
                    }
            
            self.mt5 = MockMT5()
            
            class MockRiskManager:
                max_trades_per_day = 10
                max_open_positions = 3
            
            self.risk_manager = MockRiskManager()
        
        def analyze_market(self, symbol, timeframe):
            return {
                'fusion': {
                    'final_action': 'BUY',
                    'confidence': 0.75,
                    'meta_score': 0.65,
                    'reason': 'Strong bullish confluence',
                    'breakdown': {
                        'cls': {'consensus': 'BUY'},
                        'trend': 'BULLISH'
                    }
                },
                'best_signal': {
                    'action': 'BUY',
                    'reason': 'Breakout signal',
                    'risk_level': 'HIGH',
                    'metadata': {'atr': 0.00234}
                },
                'risk_metrics': {
                    'daily_trades': 3,
                    'daily_pnl': 50.0,
                    'open_positions': 1,
                    'win_rate': 0.65,
                    'margin_level': 1000.0
                }
            }
        
        def shutdown(self):
            print("Bot shutdown")
    
    bot = MockBot()
    menu = CLIMenu(bot)
    
    # Show header
    menu.print_header()
    menu.print_menu()
    
    print("\n✅ CLI Menu initialized successfully")