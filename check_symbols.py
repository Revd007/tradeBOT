"""
Script untuk mengecek symbol yang tersedia di MT5 terminal
"""

import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

def check_mt5_symbols():
    """Check available symbols in MT5 terminal"""
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return
    
    print("✅ MT5 initialized successfully")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"\n📊 Account: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Balance: ${account_info.balance:.2f}")
    
    # Test different BTC symbol variations
    print("\n🔍 Checking Bitcoin symbols...")
    btc_variations = [
        'BTCUSD',
        'BTCUSDm', 
        'BTCUSDM',
        'Bitcoin',
        'BITCOIN',
        'BTC/USD',
        'XBTUSD',
    ]
    
    found_symbols = []
    for symbol in btc_variations:
        info = mt5.symbol_info(symbol)
        if info:
            print(f"✅ Found: {symbol}")
            print(f"   Name: {info.name}")
            print(f"   Description: {info.description}")
            print(f"   Visible: {info.visible}")
            print(f"   Tradeable: {info.trade_mode}")
            found_symbols.append(symbol)
            
            # Try to get latest tick
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"   Current Price: Bid={tick.bid:.2f}, Ask={tick.ask:.2f}")
            print()
        else:
            print(f"❌ Not found: {symbol}")
    
    # Show all available symbols (first 50)
    print("\n📋 All available symbols (first 50):")
    print("-" * 80)
    symbols = mt5.symbols_get()
    if symbols:
        for i, symbol in enumerate(symbols[:50]):
            if 'BTC' in symbol.name.upper() or 'BITCOIN' in symbol.name.upper():
                print(f"🔹 {symbol.name:20s} | {symbol.description:40s} | Visible: {symbol.visible}")
            else:
                print(f"  {symbol.name:20s} | {symbol.description:40s} | Visible: {symbol.visible}")
    
    # Try to get candles for found BTC symbols
    if found_symbols:
        print(f"\n🔥 Testing candle data for: {found_symbols[0]}")
        try:
            rates = mt5.copy_rates_from_pos(found_symbols[0], mt5.TIMEFRAME_M5, 0, 10)
            if rates is not None and len(rates) > 0:
                print(f"✅ Successfully got {len(rates)} candles!")
                print(f"   Latest close: {rates[-1]['close']:.2f}")
            else:
                error = mt5.last_error()
                print(f"❌ Failed to get candles: {error}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    mt5.shutdown()
    print("\n✅ Check complete!")

if __name__ == "__main__":
    check_mt5_symbols()

