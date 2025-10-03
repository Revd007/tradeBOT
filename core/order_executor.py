import MetaTrader5 as mt5
from typing import Optional, Dict, Tuple
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Handles order execution with advanced features"""
    
    def __init__(self, mt5_handler, max_slippage_pips: float = 2.0):
        self.mt5 = mt5_handler
        self.max_slippage_pips = max_slippage_pips
        self.magic_number = 234000  # Unique identifier for bot orders
        
    def place_market_order(
        self,
        symbol: str,
        order_type: str,  # 'BUY' or 'SELL'
        lot_size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "Auto-Trade",
        deviation: int = 20
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Place market order with validation
        
        Returns:
            (success: bool, order_result: Dict)
        """
        try:
            # Get symbol info
            symbol_info = self.mt5.get_symbol_info(symbol)
            if not symbol_info:
                return False, {"error": f"Symbol {symbol} not available"}
            
            # Validate lot size
            lot_size = self._validate_lot_size(symbol_info, lot_size)
            
            # Get current price
            tick = self.mt5.get_tick(symbol)
            if not tick:
                return False, {"error": "Failed to get current price"}
            
            # Determine order type and price
            if order_type.upper() == 'BUY':
                mt5_order_type = mt5.ORDER_TYPE_BUY
                price = tick['ask']
                sl_price = stop_loss if stop_loss else 0
                tp_price = take_profit if take_profit else 0
            elif order_type.upper() == 'SELL':
                mt5_order_type = mt5.ORDER_TYPE_SELL
                price = tick['bid']
                sl_price = stop_loss if stop_loss else 0
                tp_price = take_profit if take_profit else 0
            else:
                return False, {"error": f"Invalid order type: {order_type}"}
            
            # Validate SL/TP
            sl_price, tp_price = self._validate_sl_tp(
                symbol_info, mt5_order_type, price, sl_price, tp_price
            )
            
            # Check spread
            spread_pips = symbol_info['spread_pips']
            if spread_pips > 5.0:  # Abnormal spread
                logger.warning(f"High spread detected: {spread_pips:.1f} pips")
                return False, {"error": f"Spread too high: {spread_pips:.1f} pips"}
            
            # Check free margin
            if not self._check_margin(symbol, lot_size):
                return False, {"error": "Insufficient margin"}
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5_order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": deviation,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            logger.info(f"Sending {order_type} order: {symbol} {lot_size} lots @ {price}")
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Order send failed: {error}")
                return False, {"error": str(error)}
            
            result_dict = result._asdict()
            
            # Check result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment} (retcode: {result.retcode})")
                return False, result_dict
            
            # Check slippage
            slippage_pips = abs(result.price - price) / symbol_info['pip_value']
            if slippage_pips > self.max_slippage_pips:
                logger.warning(f"High slippage: {slippage_pips:.1f} pips")
                # Try to close immediately if slippage too high
                self.close_position_by_ticket(result.order)
                return False, {"error": f"Slippage too high: {slippage_pips:.1f} pips"}
            
            logger.info(f"✅ Order executed: Ticket #{result.order}, Price: {result.price}, Slippage: {slippage_pips:.1f} pips")
            
            return True, result_dict
            
        except Exception as e:
            logger.error(f"Exception in place_market_order: {str(e)}")
            return False, {"error": str(e)}
    
    def place_pending_order(
        self,
        symbol: str,
        order_type: str,  # 'BUY_LIMIT', 'SELL_LIMIT', 'BUY_STOP', 'SELL_STOP'
        lot_size: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        expiration: Optional[datetime] = None,
        comment: str = "Pending Order"
    ) -> Tuple[bool, Optional[Dict]]:
        """Place pending order"""
        try:
            symbol_info = self.mt5.get_symbol_info(symbol)
            if not symbol_info:
                return False, {"error": f"Symbol {symbol} not available"}
            
            lot_size = self._validate_lot_size(symbol_info, lot_size)
            
            # Map order type
            type_map = {
                'BUY_LIMIT': mt5.ORDER_TYPE_BUY_LIMIT,
                'SELL_LIMIT': mt5.ORDER_TYPE_SELL_LIMIT,
                'BUY_STOP': mt5.ORDER_TYPE_BUY_STOP,
                'SELL_STOP': mt5.ORDER_TYPE_SELL_STOP,
            }
            
            if order_type not in type_map:
                return False, {"error": f"Invalid order type: {order_type}"}
            
            mt5_order_type = type_map[order_type]
            
            # Validate SL/TP
            sl_price, tp_price = self._validate_sl_tp(
                symbol_info, mt5_order_type, price, 
                stop_loss or 0, take_profit or 0
            )
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5_order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "magic": self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_SPECIFIED if expiration else mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            
            if expiration:
                request["expiration"] = int(expiration.timestamp())
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return False, {"error": str(error)}
            
            result_dict = result._asdict()
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Pending order failed: {result.comment}")
                return False, result_dict
            
            logger.info(f"✅ Pending order placed: {order_type} {symbol} @ {price}")
            return True, result_dict
            
        except Exception as e:
            logger.error(f"Exception in place_pending_order: {str(e)}")
            return False, {"error": str(e)}
    
    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """Modify position SL/TP"""
        try:
            positions = self.mt5.get_positions()
            position = next((p for p in positions if p['ticket'] == ticket), None)
            
            if not position:
                return False, {"error": f"Position {ticket} not found"}
            
            symbol = position['symbol']
            symbol_info = self.mt5.get_symbol_info(symbol)
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": stop_loss if stop_loss else position['sl'],
                "tp": take_profit if take_profit else position['tp'],
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return False, {"error": str(error)}
            
            result_dict = result._asdict()
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, result_dict
            
            logger.info(f"✅ Position {ticket} modified: SL={stop_loss}, TP={take_profit}")
            return True, result_dict
            
        except Exception as e:
            logger.error(f"Exception in modify_position: {str(e)}")
            return False, {"error": str(e)}
    
    def close_position_by_ticket(self, ticket: int, volume: Optional[float] = None) -> Tuple[bool, Optional[Dict]]:
        """Close position by ticket number"""
        try:
            positions = self.mt5.get_positions()
            position = next((p for p in positions if p['ticket'] == ticket), None)
            
            if not position:
                return False, {"error": f"Position {ticket} not found"}
            
            symbol = position['symbol']
            lot_size = volume if volume else position['volume']
            
            # Determine close type
            if position['type'] == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = self.mt5.get_tick(symbol)['bid']
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = self.mt5.get_tick(symbol)['ask']
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return False, {"error": str(error)}
            
            result_dict = result._asdict()
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, result_dict
            
            logger.info(f"✅ Position {ticket} closed: {lot_size} lots @ {result.price}")
            return True, result_dict
            
        except Exception as e:
            logger.error(f"Exception in close_position: {str(e)}")
            return False, {"error": str(e)}
    
    def close_all_positions(self, symbol: Optional[str] = None) -> Dict:
        """Close all positions (optionally filtered by symbol)"""
        positions = self.mt5.get_positions(symbol)
        results = {"success": [], "failed": []}
        
        for position in positions:
            success, result = self.close_position_by_ticket(position['ticket'])
            if success:
                results['success'].append(position['ticket'])
            else:
                results['failed'].append({
                    'ticket': position['ticket'],
                    'error': result.get('error', 'Unknown error')
                })
        
        logger.info(f"Closed {len(results['success'])} positions, {len(results['failed'])} failed")
        return results
    
    def cancel_pending_order(self, ticket: int) -> Tuple[bool, Optional[Dict]]:
        """Cancel pending order"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                return False, {"error": str(error)}
            
            result_dict = result._asdict()
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, result_dict
            
            logger.info(f"✅ Order {ticket} cancelled")
            return True, result_dict
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _validate_lot_size(self, symbol_info: Dict, lot_size: float) -> float:
        """Validate and adjust lot size"""
        min_lot = symbol_info['volume_min']
        max_lot = symbol_info['volume_max']
        lot_step = symbol_info['volume_step']
        
        # Round to step
        lot_size = round(lot_size / lot_step) * lot_step
        
        # Apply limits
        lot_size = max(min_lot, min(max_lot, lot_size))
        
        return lot_size
    
    def _validate_sl_tp(
        self, 
        symbol_info: Dict, 
        order_type: int, 
        price: float, 
        sl: float, 
        tp: float
    ) -> Tuple[float, float]:
        """Validate and adjust SL/TP levels"""
        stops_level = symbol_info['trade_stops_level'] * symbol_info['point']
        
        if sl > 0:
            if order_type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]:
                # For buy orders, SL must be below price
                min_sl = price - stops_level
                if sl > min_sl:
                    sl = min_sl
            else:
                # For sell orders, SL must be above price
                max_sl = price + stops_level
                if sl < max_sl:
                    sl = max_sl
        
        if tp > 0:
            if order_type in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]:
                # For buy orders, TP must be above price
                min_tp = price + stops_level
                if tp < min_tp:
                    tp = min_tp
            else:
                # For sell orders, TP must be below price
                max_tp = price - stops_level
                if tp > max_tp:
                    tp = max_tp
        
        return sl, tp
    
    def _check_margin(self, symbol: str, lot_size: float) -> bool:
        """Check if there's enough free margin"""
        account = self.mt5.get_account_info()
        
        # Estimate required margin (simplified)
        symbol_info = self.mt5.get_symbol_info(symbol)
        tick = self.mt5.get_tick(symbol)
        
        contract_size = symbol_info['trade_contract_size']
        leverage = account['leverage']
        
        required_margin = (lot_size * contract_size * tick['ask']) / leverage
        
        free_margin = account['margin_free']
        
        if required_margin > free_margin * 0.8:  # Use max 80% of free margin
            logger.warning(f"Insufficient margin: Required={required_margin:.2f}, Available={free_margin:.2f}")
            return False
        
        return True


# Trailing Stop Manager
class TrailingStopManager:
    """Manages trailing stop loss for positions"""
    
    def __init__(self, order_executor: OrderExecutor):
        self.executor = order_executor
        self.trailing_stops = {}  # {ticket: trailing_pips}
    
    def enable_trailing_stop(self, ticket: int, trailing_pips: float):
        """Enable trailing stop for a position"""
        self.trailing_stops[ticket] = trailing_pips
        logger.info(f"Trailing stop enabled for position {ticket}: {trailing_pips} pips")
    
    def disable_trailing_stop(self, ticket: int):
        """Disable trailing stop"""
        if ticket in self.trailing_stops:
            del self.trailing_stops[ticket]
    
    def update_trailing_stops(self):
        """Update all trailing stops"""
        positions = self.executor.mt5.get_positions()
        
        for position in positions:
            ticket = position['ticket']
            
            if ticket not in self.trailing_stops:
                continue
            
            trailing_pips = self.trailing_stops[ticket]
            symbol = position['symbol']
            symbol_info = self.executor.mt5.get_symbol_info(symbol)
            pip_value = symbol_info['pip_value']
            
            current_price = position['price_current']
            current_sl = position['sl']
            
            if position['type'] == mt5.ORDER_TYPE_BUY:
                # For buy positions
                new_sl = current_price - (trailing_pips * pip_value)
                
                if current_sl == 0 or new_sl > current_sl:
                    success, _ = self.executor.modify_position(ticket, stop_loss=new_sl)
                    if success:
                        logger.info(f"Trailing stop updated for {ticket}: {new_sl:.5f}")
            
            else:
                # For sell positions
                new_sl = current_price + (trailing_pips * pip_value)
                
                if current_sl == 0 or new_sl < current_sl:
                    success, _ = self.executor.modify_position(ticket, stop_loss=new_sl)
                    if success:
                        logger.info(f"Trailing stop updated for {ticket}: {new_sl:.5f}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from core.mt5_handler import MT5Handler
    
    mt5_handler = MT5Handler(
        login=12345678,
        password="password",
        server="MetaQuotes-Demo"
    )
    
    if mt5_handler.initialize():
        executor = OrderExecutor(mt5_handler)
        
        # Place buy order
        success, result = executor.place_market_order(
            symbol="XAUUSDm",
            order_type="BUY",
            lot_size=0.01,
            stop_loss=3800.00,
            take_profit=3900.00,
            comment="Test order"
        )
        
        if success:
            print(f"Order executed: {result}")
        
        mt5_handler.shutdown()