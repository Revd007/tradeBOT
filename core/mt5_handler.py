"""
MetaTrader 5 Handler with Auto-Reconnection and Error Handling
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_disconnect(max_attempts=3, delay=5):
    """Decorator for auto-retry on MT5 disconnection"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    if not mt5.terminal_info():
                        logger.warning(f"MT5 not connected, attempting reconnect ({attempt+1}/{max_attempts})")
                        self.reconnect()
                    
                    result = func(self, *args, **kwargs)
                    return result
                    
                except Exception as e:
                    logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator


class MT5Handler:
    """Complete MT5 API wrapper with robust error handling"""
    
    def __init__(self, login: int, password: str, server: str, timeout: int = 60000):
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.connected = False
        self.account_info = None
        self.last_ping = None
        
    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize(timeout=self.timeout):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False
            
            # Login to account
            if not mt5.login(self.login, self.password, self.server):
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
            
            self.connected = True
            self.account_info = mt5.account_info()._asdict()
            self.last_ping = datetime.now()
            
            logger.info(f"✅ Connected to MT5: {self.server} (Account: {self.login})")
            logger.info(f"Balance: ${self.account_info['balance']:.2f}, Equity: ${self.account_info['equity']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Exception during initialization: {str(e)}")
            return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5"""
        logger.info("Attempting to reconnect to MT5...")
        self.shutdown()
        time.sleep(2)
        return self.initialize()
    
    def shutdown(self):
        """Safely shutdown MT5 connection"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 connection closed")
        except:
            pass
    
    def ping(self) -> bool:
        """Check if connection is alive"""
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info:
                self.last_ping = datetime.now()
                return True
            return False
        except:
            return False
    
    @retry_on_disconnect()
    def get_account_info(self) -> Dict:
        """Get current account information"""
        info = mt5.account_info()
        if info is None:
            raise Exception("Failed to get account info")
        
        account_dict = info._asdict()
        
        # Calculate additional metrics
        account_dict['free_margin_percent'] = (
            (account_dict['margin_free'] / account_dict['equity'] * 100)
            if account_dict['equity'] > 0 else 0
        )
        account_dict['margin_level'] = (
            (account_dict['equity'] / account_dict['margin'] * 100)
            if account_dict['margin'] > 0 else 0
        )
        
        self.account_info = account_dict
        return account_dict
    
    @retry_on_disconnect()
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information and check if tradeable"""
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
        
        if not info.visible:
            logger.info(f"Symbol {symbol} not visible, attempting to enable...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to enable symbol {symbol}")
                return None
        
        symbol_dict = info._asdict()
        
        # Add calculated fields
        symbol_dict['pip_value'] = 10 if 'JPY' in symbol else (
            0.1 if 'XAU' in symbol or 'XAG' in symbol else 0.0001
        )
        symbol_dict['spread_pips'] = (
            symbol_dict['spread'] * symbol_dict['point'] / symbol_dict['pip_value']
        )
        
        return symbol_dict
    
    @retry_on_disconnect()
    def get_candles(
        self, 
        symbol: str, 
        timeframe: str, 
        count: int = 500,
        shift: int = 0
    ) -> pd.DataFrame:
        """
        Get historical candles
        
        Args:
            symbol: Trading symbol (e.g., XAUUSD)
            timeframe: Timeframe (M5, M15, H1, H4, D1)
            count: Number of candles
            shift: Shift from current time
        """
        # Convert timeframe string to MT5 constant
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
        }
        
        if timeframe not in tf_map:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        mt5_timeframe = tf_map[timeframe]
        
        # Get rates
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, shift, count)
        
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            raise Exception(f"Failed to get candles: {error}")
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Calculate additional indicators
        df['hl2'] = (df['high'] + df['low']) / 2
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Price change
        df['change'] = df['close'] - df['open']
        df['change_pct'] = (df['change'] / df['open']) * 100
        
        # Range
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Candle type
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        df['is_doji'] = abs(df['change_pct']) < 0.1
        
        return df
    
    @retry_on_disconnect()
    def get_tick(self, symbol: str) -> Optional[Dict]:
        """Get latest tick data"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        return {
            'time': datetime.fromtimestamp(tick.time),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'spread': tick.ask - tick.bid,
        }
    
    @retry_on_disconnect()
    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open positions"""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        positions_list = []
        for pos in positions:
            pos_dict = pos._asdict()
            
            # Calculate profit in pips
            symbol_info = self.get_symbol_info(pos.symbol)
            if symbol_info:
                pip_value = symbol_info['pip_value']
                pos_dict['profit_pips'] = (
                    (pos.price_current - pos.price_open) / pip_value
                    if pos.type == mt5.ORDER_TYPE_BUY
                    else (pos.price_open - pos.price_current) / pip_value
                )
            
            positions_list.append(pos_dict)
        
        return positions_list
    
    @retry_on_disconnect()
    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get pending orders"""
        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()
        
        if orders is None:
            return []
        
        return [order._asdict() for order in orders]
    
    @retry_on_disconnect()
    def calculate_lot_size(
        self, 
        symbol: str, 
        risk_amount: float, 
        stop_loss_pips: float
    ) -> float:
        """
        Calculate lot size based on risk amount and stop loss
        
        Args:
            symbol: Trading symbol
            risk_amount: Amount willing to risk in USD
            stop_loss_pips: Stop loss distance in pips
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return 0.01
        
        # Get pip value per lot
        pip_value = symbol_info['pip_value']
        
        # Calculate lot size
        # Risk = Lot Size × Stop Loss (pips) × Pip Value
        lot_size = risk_amount / (stop_loss_pips * pip_value * 10)  # *10 for standard lot
        
        # Round to lot step
        lot_step = symbol_info['volume_step']
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Apply min/max limits
        lot_size = max(symbol_info['volume_min'], lot_size)
        lot_size = min(symbol_info['volume_max'], lot_size)
        
        return lot_size
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
    
    def __del__(self):
        """Destructor"""
        self.shutdown()


# Connection Manager for auto-reconnection
class ConnectionManager:
    """Manages MT5 connection with health checks and auto-reconnect"""
    
    def __init__(self, mt5_handler: MT5Handler, ping_interval: int = 60):
        self.mt5 = mt5_handler
        self.ping_interval = ping_interval
        self.last_health_check = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    def health_check(self) -> bool:
        """Perform health check"""
        current_time = datetime.now()
        
        # Skip if checked recently
        if (self.last_health_check and 
            (current_time - self.last_health_check).seconds < self.ping_interval):
            return True
        
        self.last_health_check = current_time
        
        # Ping MT5
        if not self.mt5.ping():
            logger.warning("MT5 ping failed, attempting reconnection...")
            return self.attempt_reconnect()
        
        self.reconnect_attempts = 0
        return True
    
    def attempt_reconnect(self) -> bool:
        """Attempt to reconnect"""
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return False
        
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        return self.mt5.reconnect()


if __name__ == "__main__":
    # Test MT5 Handler
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    handler = MT5Handler(
        login=12345678,
        password="your_password",
        server="MetaQuotes-Demo"
    )
    
    if handler.initialize():
        # Get account info
        account = handler.get_account_info()
        print(f"Balance: ${account['balance']:.2f}")
        
        # Get candles
        df = handler.get_candles("XAUUSD", "M5", count=100)
        print(f"\nLatest candles:\n{df.tail()}")
        
        # Get positions
        positions = handler.get_positions()
        print(f"\nOpen positions: {len(positions)}")
        
        handler.shutdown()