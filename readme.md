# 🤖 Advanced Auto-Trading Bot v2.0

**AI-Powered Forex Trading System with Multi-Strategy Analysis**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Strategies](#strategies)
- [Risk Management](#risk-management)
- [API Integration](#api-integration)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### Core Capabilities

✅ **Multi-Strategy Analysis**
- Counter-Trend Reversal (High Risk)
- Breakout Strategy (Medium Risk)
- Fibonacci + ATR (Low-Medium Risk)
- Custom strategy development support

✅ **AI & Machine Learning**
- CLS (Classifier) model predictions
- Multi-timeframe consensus (M5, M15, H1, H4)
- Trend Fusion meta-analysis
- LLM integration (Llama) for market context

✅ **Advanced Risk Management**
- Dynamic position sizing (Kelly Criterion)
- ATR-based stop loss/take profit
- Daily loss limits & trade count limits
- Correlation exposure management
- Trailing stop loss automation

✅ **News & Calendar Integration**
- News API sentiment analysis
- Economic calendar (TradingEconomics)
- Automatic trading pause before high-impact news
- News impact scoring (0-10)

✅ **Robust Architecture**
- Auto-reconnection on MT5 disconnect
- Slippage control & validation
- Connection health monitoring
- Comprehensive error handling
- Detailed logging system

✅ **Monitoring & Notifications**
- Telegram real-time alerts
- Performance tracking & analytics
- Daily/weekly/monthly reports
- Firebase integration (optional)

---

## 🏗️ Architecture

```
trading_bot/
│
├── core/                    # Core trading logic
│   ├── mt5_handler.py      # MT5 API wrapper with auto-reconnect
│   ├── order_executor.py   # Smart order execution
│   ├── risk_manager.py     # Risk management system
│   └── connection_manager.py
│
├── strategies/              # Trading strategies
│   ├── base_strategy.py    # Base class & indicators
│   ├── counter_trend.py    # Counter-trend strategy
│   ├── breakout.py         # Breakout strategy
│   ├── fibonacci_atr.py    # Fibonacci/ATR strategy
│   └── strategy_manager.py # Coordinates all strategies
│
├── models/                  # AI/ML models
│   ├── cls_predictor.py    # CLS model integration
│   ├── trend_fusion.py     # Meta-analysis
│   └── saved_models/       # Trained models (.pkl)
│
├── data/                    # Data acquisition
│   ├── news_scraper.py     # News API integration
│   ├── calendar_scraper.py # Economic calendar
│   └── preprocessor.py     # Data preprocessing
│
├── monitoring/              # Monitoring & alerts
│   ├── telegram_bot.py     # Telegram notifications
│   ├── performance_tracker.py
│   └── trade_logger.py
│
├── cli/                     # Command-line interface
│   ├── menu.py             # Interactive menu
│   └── display.py          # Display utilities
│
├── utils/                   # Utility functions
│   ├── logger.py           # Logging configuration
│   ├── config_loader.py    # Config management
│   └── decorators.py       # Utility decorators
│
└── main.py                 # Entry point
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- MetaTrader 5 terminal installed
- MT5 trading account (Demo or Live)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create `.env` file in project root:

```env
# MT5 Credentials - DEMO
MT5_LOGIN_DEMO=12345678
MT5_PASSWORD_DEMO=YourPassword
MT5_SERVER_DEMO=MetaQuotes-Demo

# MT5 Credentials - LIVE (EXNESS)
MT5_LOGIN_EXNESS=87654321
MT5_PASSWORD_EXNESS=YourPassword
MT5_SERVER_EXNESS=Exness-MT5Real

# API Keys
NEWS_API_KEY=your_newsapi_key_here
TRADING_ECONOMICS_KEY=your_te_key_here

# Telegram (Optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Settings
DEFAULT_SYMBOL=XAUUSD
DEFAULT_LOT_SIZE=0.01
MAX_DAILY_LOSS=100
MAX_TRADES_PER_DAY=10
RISK_PER_TRADE=1.0

# Firebase (Optional)
FIREBASE_CREDENTIALS_PATH=./firebase-key.json
FIREBASE_ENABLED=false
```

### Step 5: Download/Train Models

Place your trained CLS models in `models/saved_models/`:
- `cls_m5.pkl`
- `cls_m15.pkl`
- `cls_h1.pkl`
- `cls_h4.pkl`
- `scaler_m5.pkl`, etc.

---

## ⚙️ Configuration

Edit `config.yaml` for detailed settings:

```yaml
trading:
  default_symbol: XAUUSD
  available_symbols: [XAUUSD, EURUSD, GBPUSD, USDJPY]
  timeframes:
    fast: M5
    medium: M15
    slow: H1
    trend: H4

risk_management:
  default_risk_percent: 1.0
  max_daily_loss: 100
  max_trades_per_day: 10
  max_open_positions: 3

strategies:
  counter_trend:
    enabled: true
    min_confidence: 0.70
  breakout:
    enabled: true
    min_confidence: 0.65
  fibonacci_atr:
    enabled: true
    min_confidence: 0.60

news_filter:
  enabled: true
  pause_before_high_impact: 30  # minutes
  resume_after_high_impact: 15
```

---

## 📖 Usage

### Starting the Bot

```bash
python main.py
```

### Interactive Menu

```
╔═══════════════════════════════════════╗
║  🤖 AUTO-TRADING BOT v2.0            ║
╚═══════════════════════════════════════╝

1. Analyze Now               (Current: XAUUSD)
2. Change SYMBOL             (EURUSD, GBPUSD, etc.)
3. Change TIMEFRAME          (M5, M15, H1, H4)
4. Switch ACCOUNT            (DEMO / EXNESS)
5. Change TRADE_MODE         (NORMAL, AGGRESSIVE)
6. Toggle AUTO-TRADE         (OFF)
7. View Open Positions
8. Performance Report
0. Quit
```

### Manual Analysis

```python
from main import TradingBotEngine, load_config

# Initialize bot
config = load_config()
bot = TradingBotEngine(config)

# Analyze market
results = bot.analyze_market("XAUUSD", "M5")

# Check if should trade
if results['should_enter_trade']:
    trade = results['recommended_trade']
    print(f"Signal: {trade['action']} @ {trade['entry_price']}")
    print(f"Confidence: {trade['confidence']:.2%}")
    
    # Execute trade
    bot.execute_trade(trade)
```

### Auto-Trading Mode

```python
# Enable auto-trading
bot.auto_trade_enabled = True

# Start automated loop (checks every 5 minutes)
bot.auto_trade_loop(
    symbol="XAUUSD",
    timeframe="M5",
    interval=300  # seconds
)
```

---

## 🎯 Strategies

### 1. Counter-Trend Strategy (EXTREME RISK)

**When to use:** Market at extremes (overbought/oversold)

**Entry Conditions:**
- RSI > 70 (overbought) or < 30 (oversold)
- Price at Bollinger Band extremes
- Reversal candle patterns (hammer, shooting star)
- Divergence confirmation
- MACD crossover

**Risk-Reward:** 1:3 typical

```python
from strategies.counter_trend import CounterTrendStrategy

strategy = CounterTrendStrategy()
signal = strategy.analyze(df, symbol_info)
```

### 2. Breakout Strategy (HIGH RISK)

**When to use:** Price breaking key levels after consolidation

**Entry Conditions:**
- Resistance/support level broken
- Volume increase (1.5x average)
- Momentum confirmation (MACD positive)
- RSI in momentum zone (50-70)
- Post-consolidation breakout

**Risk-Reward:** 1:2.5 typical

```python
from strategies.breakout import BreakoutStrategy

strategy = BreakoutStrategy()
signal = strategy.analyze(df, symbol_info)
```

### 3. Fibonacci-ATR Strategy (MEDIUM RISK)

**When to use:** Trend continuation on retracements

**Entry Conditions:**
- Clear trend identified
- Price retraces to Fibonacci level (38.2%, 50%, 61.8%)
- Bounce confirmation at level
- Trend continuation signals
- ATR-based risk management

**Risk-Reward:** 1:2 typical

```python
from strategies.fibonacci_atr import FibonacciATRStrategy

strategy = FibonacciATRStrategy()
signal = strategy.analyze(df, symbol_info)
```

---

## 🛡️ Risk Management

### Position Sizing

Bot uses **adaptive position sizing** based on:
- Account balance
- Risk percentage (default 1%)
- Stop loss distance (ATR-based)
- AI confidence score (0.5x to 1.5x multiplier)
- News impact (reduces size during high impact)
- Recent win rate (adjusts based on performance)

```python
lot_size = risk_manager.calculate_position_size(
    symbol="XAUUSD",
    stop_loss_pips=30,
    confidence=0.75,
    news_impact=3,
    trade_mode="NORMAL"
)
```

### Stop Loss Calculation

```python
# ATR-based dynamic SL
sl = risk_manager.calculate_stop_loss(
    symbol="XAUUSD",
    entry_price=3850.00,
    direction="BUY",
    atr=2.5,
    atr_multiplier=2.0,  # 2x ATR
    min_pips=10,
    max_pips=100
)
```

### Daily Limits

- **Max trades per day:** 10 (configurable)
- **Max daily loss:** 5% of balance
- **Max open positions:** 3 concurrent
- **Margin level alert:** Below 150%
- **Drawdown limit:** 20% max

### Protection Mechanisms

1. **News Filter:** Pauses trading 30min before high-impact news
2. **Slippage Control:** Rejects trades with >2 pips slippage
3. **Spread Check:** Blocks trades when spread >5 pips
4. **Correlation Limit:** Prevents over-exposure to correlated pairs
5. **Consecutive Loss Protection:** Stops after 5 consecutive losses

---

## 🔌 API Integration

### News API Setup

1. Get API key from [NewsAPI.org](https://newsapi.org/)
2. Add to `.env`:
   ```
   NEWS_API_KEY=your_key_here
   ```

### Trading Economics Calendar

1. Sign up at [TradingEconomics.com](https://tradingeconomics.com/)
2. Get API key
3. Add to `.env`:
   ```
   TRADING_ECONOMICS_KEY=your_key_here
   ```

### Telegram Notifications

1. Create bot with [@BotFather](https://t.me/botfather)
2. Get bot token
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
4. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
   TELEGRAM_CHAT_ID=123456789
   ```

---

## 🔧 Troubleshooting

### MT5 Connection Issues

**Problem:** `MT5 initialization failed`

**Solutions:**
1. Ensure MT5 terminal is running
2. Check credentials in `.env`
3. Verify server name (case-sensitive)
4. Enable "Allow automated trading" in MT5
5. Check firewall settings

```python
# Test connection manually
from core.mt5_handler import MT5Handler

mt5 = MT5Handler(login, password, server)
if mt5.initialize():
    print("✅ Connected")
else:
    print("❌ Connection failed")
```

### Model Not Found

**Problem:** `No CLS models found`

**Solution:**
- Train models using `strategies/cls_trainer.py`
- Or place pre-trained models in `models/saved_models/`
- Bot will work without models (reduced accuracy)

### High Slippage

**Problem:** Trades rejected due to slippage

**Solution:**
1. Increase `max_slippage` in config
2. Trade during liquid market hours
3. Avoid trading during news events
4. Use limit orders instead of market orders

### Telegram Not Working

**Problem:** No notifications received

**Solution:**
1. Verify bot token and chat ID
2. Start conversation with bot first
3. Check `python-telegram-bot` is installed
4. Check bot logs for errors

---

## 📊 Performance Monitoring

### View Real-time Stats

```python
# Get risk metrics
metrics = bot.risk_manager.get_risk_metrics()
print(f"Win Rate: {metrics['win_rate']:.1%}")
print(f"Daily P&L: ${metrics['daily_pnl']:.2f}")
print(f"Margin Level: {metrics['margin_level']:.1f}%")
```

### Generate Reports

```python
# Daily report
report = bot.performance.generate_report(period_days=1)
print(report)

# Weekly report
report = bot.performance.generate_report(period_days=7)

# Export to CSV
bot.performance.export_to_csv("trades_export.csv")
```

### Strategy Performance

```python
# Compare strategies
strategy_perf = bot.performance.get_strategy_performance()

for strategy, stats in strategy_perf.items():
    print(f"{strategy}: {stats['win_rate']:.1%} win rate")
    print(f"  Total P&L: ${stats['total_pnl']:+.2f}")
```

---

## ⚠️ Important Warnings

1. **Never risk more than you can afford to lose**
2. **Always test on DEMO account first**
3. **Monitor the bot regularly** - do not leave completely unattended
4. **Past performance ≠ future results**
5. **News events can cause unexpected behavior**
6. **Slippage and spread can impact profitability**
7. **Bot requires internet connection** - use VPS for 24/7 operation
8. **Keep models updated** - retrain regularly with new data

---

## 📈 Best Practices

1. **Start Small:** Begin with minimum lot size (0.01)
2. **Test Thoroughly:** Run on demo for at least 1 week
3. **Monitor News:** Be aware of major economic events
4. **Regular Retraining:** Update CLS models monthly
5. **Diversify:** Don't trade only one symbol
6. **Set Limits:** Use max daily loss and trade count
7. **Review Logs:** Check logs daily for errors
8. **Backup Data:** Export trade history regularly
9. **Use VPS:** For stable 24/7 operation
10. **Stay Updated:** Keep dependencies and models current

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## 📧 Support

- **Email:** support@tradingbot.com
- **Telegram:** @TradingBotSupport
- **Documentation:** https://docs.tradingbot.com

---

## 🙏 Acknowledgments

- MetaTrader 5 API
- News API
- TradingEconomics
- Open-source community

---

**⚡ Happy Trading! ⚡**

*Remember: Trading involves risk. Always trade responsibly.*