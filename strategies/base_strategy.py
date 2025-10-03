import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """Calculate Simple Moving Average"""
        return df[column].rolling(window=period).mean()
    
    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def detect_support_resistance(
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 3
    ) -> Dict[str, List[float]]:
        """Detect support and resistance levels"""
        # Find local maxima (resistance)
        resistance_levels = []
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
                resistance_levels.append(df['high'].iloc[i])
        
        # Find local minima (support)
        support_levels = []
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
                support_levels.append(df['low'].iloc[i])
        
        # Get most significant levels
        resistance = sorted(set(resistance_levels), reverse=True)[:num_levels]
        support = sorted(set(support_levels))[:num_levels]
        
        return {
            'resistance': resistance,
            'support': support
        }
    
    @staticmethod
    def calculate_fibonacci_levels(
        high: float,
        low: float,
        direction: str = 'retracement'
    ) -> Dict[str, float]:
        """Calculate Fibonacci levels"""
        diff = high - low
        
        if direction == 'retracement':
            return {
                '0.0': high,
                '0.236': high - (0.236 * diff),
                '0.382': high - (0.382 * diff),
                '0.500': high - (0.500 * diff),
                '0.618': high - (0.618 * diff),
                '0.786': high - (0.786 * diff),
                '1.0': low
            }
        else:  # extension
            return {
                '0.0': low,
                '0.618': low + (0.618 * diff),
                '1.0': low + (1.0 * diff),
                '1.618': low + (1.618 * diff),
                '2.618': low + (2.618 * diff),
            }


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, risk_level: str = 'MEDIUM'):
        self.name = name
        self.risk_level = risk_level  # LOW, MEDIUM, HIGH, EXTREME
        self.indicators = TechnicalIndicators()
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, symbol_info: Dict) -> Optional[Dict]:
        """
        Analyze market and generate signal
        
        Returns:
            None if no signal, or Dict with:
            {
                'action': 'BUY' or 'SELL',
                'entry_price': float,
                'stop_loss': float,
                'take_profit': float,
                'confidence': 0.0 - 1.0,
                'reason': str,
                'metadata': dict
            }
        """
        pass
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common indicators to dataframe"""
        # ATR
        df['atr'] = self.indicators.calculate_atr(df)
        
        # RSI
        df['rsi'] = self.indicators.calculate_rsi(df)
        
        # EMAs
        df['ema_9'] = self.indicators.calculate_ema(df, 9)
        df['ema_21'] = self.indicators.calculate_ema(df, 21)
        df['ema_50'] = self.indicators.calculate_ema(df, 50)
        df['ema_200'] = self.indicators.calculate_ema(df, 200)
        
        # MACD
        macd, signal, hist = self.indicators.calculate_macd(df)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.calculate_bollinger_bands(df)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Stochastic
        stoch_k, stoch_d = self.indicators.calculate_stochastic(df)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        return df
    
    def detect_trend(self, df: pd.DataFrame) -> str:
        """
        Detect overall trend
        
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if len(df) < 50:
            return 'NEUTRAL'
        
        # Use multiple EMAs
        ema_9 = df['ema_9'].iloc[-1]
        ema_21 = df['ema_21'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        close = df['close'].iloc[-1]
        
        # Strong bullish
        if ema_9 > ema_21 > ema_50 and close > ema_9:
            return 'BULLISH'
        
        # Strong bearish
        elif ema_9 < ema_21 < ema_50 and close < ema_9:
            return 'BEARISH'
        
        # Check MACD
        macd_hist = df['macd_hist'].iloc[-5:].mean()
        
        if macd_hist > 0:
            return 'BULLISH' if close > ema_21 else 'NEUTRAL'
        elif macd_hist < 0:
            return 'BEARISH' if close < ema_21 else 'NEUTRAL'
        
        return 'NEUTRAL'
    
    def calculate_score(self, conditions: Dict[str, bool]) -> float:
        """Calculate signal confidence score from conditions"""
        total = len(conditions)
        true_count = sum(1 for v in conditions.values() if v)
        return true_count / total if total > 0 else 0.0


class MultiTimeframeAnalysis:
    """Multi-timeframe consensus analysis"""
    
    def __init__(self, mt5_handler):
        self.mt5 = mt5_handler
        self.timeframes = ['M5', 'M15', 'H1', 'H4']
        self.indicators = TechnicalIndicators()
    
    def analyze(self, symbol: str, strategy: BaseStrategy) -> Dict:
        """
        Analyze across multiple timeframes
        
        Returns:
            {
                'consensus': 'BUY', 'SELL', or 'NEUTRAL',
                'confidence': 0.0 - 1.0,
                'timeframe_signals': {tf: signal},
                'strongest_timeframe': str
            }
        """
        signals = {}
        
        for tf in self.timeframes:
            try:
                df = self.mt5.get_candles(symbol, tf, count=200)
                df = strategy.add_all_indicators(df)
                
                signal = strategy.analyze(df, self.mt5.get_symbol_info(symbol))
                signals[tf] = signal
                
            except Exception as e:
                logger.error(f"Error analyzing {tf}: {str(e)}")
                signals[tf] = None
        
        # Calculate consensus
        buy_votes = sum(1 for s in signals.values() if s and s['action'] == 'BUY')
        sell_votes = sum(1 for s in signals.values() if s and s['action'] == 'SELL')
        total_votes = len([s for s in signals.values() if s])
        
        if total_votes == 0:
            return {
                'consensus': 'NEUTRAL',
                'confidence': 0.0,
                'timeframe_signals': signals,
                'strongest_timeframe': None
            }
        
        # Determine consensus
        if buy_votes > sell_votes and buy_votes >= total_votes * 0.6:
            consensus = 'BUY'
            confidence = buy_votes / total_votes
        elif sell_votes > buy_votes and sell_votes >= total_votes * 0.6:
            consensus = 'SELL'
            confidence = sell_votes / total_votes
        else:
            consensus = 'NEUTRAL'
            confidence = 0.5
        
        # Find strongest signal
        valid_signals = {tf: s for tf, s in signals.items() if s}
        strongest_tf = max(
            valid_signals.keys(),
            key=lambda tf: valid_signals[tf]['confidence']
        ) if valid_signals else None
        
        return {
            'consensus': consensus,
            'confidence': confidence,
            'timeframe_signals': signals,
            'strongest_timeframe': strongest_tf,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'total_votes': total_votes
        }
    
    def get_trend_alignment(self, symbol: str) -> Dict:
        """Check if trends are aligned across timeframes"""
        trends = {}
        
        for tf in self.timeframes:
            df = self.mt5.get_candles(symbol, tf, count=200)
            strategy = BaseStrategy.__new__(BaseStrategy)
            BaseStrategy.__init__(strategy, "Trend", "MEDIUM")
            df = strategy.add_all_indicators(df)
            trend = strategy.detect_trend(df)
            trends[tf] = trend
        
        bullish_count = sum(1 for t in trends.values() if t == 'BULLISH')
        bearish_count = sum(1 for t in trends.values() if t == 'BEARISH')
        
        alignment = 'ALIGNED_BULL' if bullish_count >= 3 else \
                   'ALIGNED_BEAR' if bearish_count >= 3 else 'MIXED'
        
        return {
            'alignment': alignment,
            'trends': trends,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }


class CandlePatterns:
    """Candlestick pattern recognition"""
    
    @staticmethod
    def is_doji(candle: pd.Series) -> bool:
        """Check if candle is Doji"""
        body = abs(candle['close'] - candle['open'])
        range_size = candle['high'] - candle['low']
        return body / range_size < 0.1 if range_size > 0 else False
    
    @staticmethod
    def is_hammer(candle: pd.Series) -> bool:
        """Check if candle is Hammer"""
        body = abs(candle['close'] - candle['open'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        
        return (lower_wick > body * 2 and 
                upper_wick < body * 0.5 and
                candle['close'] > candle['open'])
    
    @staticmethod
    def is_shooting_star(candle: pd.Series) -> bool:
        """Check if candle is Shooting Star"""
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        return (upper_wick > body * 2 and 
                lower_wick < body * 0.5 and
                candle['close'] < candle['open'])
    
    @staticmethod
    def is_engulfing(df: pd.DataFrame, idx: int, bullish: bool = True) -> bool:
        """Check for engulfing pattern"""
        if idx < 1:
            return False
        
        current = df.iloc[idx]
        previous = df.iloc[idx - 1]
        
        if bullish:
            return (previous['close'] < previous['open'] and
                    current['close'] > current['open'] and
                    current['open'] < previous['close'] and
                    current['close'] > previous['open'])
        else:
            return (previous['close'] > previous['open'] and
                    current['close'] < current['open'] and
                    current['open'] > previous['close'] and
                    current['close'] < previous['open'])


if __name__ == "__main__":
    # Test indicators
    import pandas as pd
    
    # Sample data
    data = {
        'open': [100, 101, 102, 101, 103],
        'high': [102, 103, 104, 103, 105],
        'low': [99, 100, 101, 100, 102],
        'close': [101, 102, 103, 102, 104],
    }
    df = pd.DataFrame(data)
    
    indicators = TechnicalIndicators()
    
    # Calculate RSI
    rsi = indicators.calculate_rsi(df)
    print(f"RSI: {rsi.iloc[-1]}")
    
    # Calculate MACD
    macd, signal, hist = indicators.calculate_macd(df)
    print(f"MACD: {macd.iloc[-1]}, Signal: {signal.iloc[-1]}")