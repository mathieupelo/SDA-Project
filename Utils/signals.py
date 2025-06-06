
import pandas as pd
import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable, Any

def combine_signals_from_df(df_scores: pd.DataFrame, tickers: List[str], signal_weights: Dict[str, float]) -> pd.DataFrame:
    # Prepare a DataFrame to store combined signals with the same index as input
    combined_scores = pd.DataFrame(index=df_scores.index)

    for ticker in tickers:
        weighted_sum = pd.Series(0.0, index=df_scores.index)
        total_weight = 0.0

        for signal_name, weight in signal_weights.items():
            col = (signal_name, ticker)
            if col in df_scores.columns:
                weighted_sum += df_scores[col] * weight
                total_weight += weight
            else:
                print(f"Warning: column {col} not found in df_scores")

        # Normalize by total weight (in case some signals are missing)
        if total_weight > 0:
            weighted_sum /= total_weight

        # Assign combined signal for this ticker
        combined_scores[ticker] = weighted_sum

    # Add the date column at the beginning if available
    if ('date', '') in df_scores.columns:
        combined_scores['date'] = df_scores[('date', '')]
        combined_scores.set_index('date', inplace=True)

    return combined_scores

class SignalBase(ABC):
    """Abstract base class for all signals"""
    
    def __init__(self, name: str, parameters: Dict = None):
        self.name = name
        self.parameters = parameters or {}
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, tickers: List[str], date: pd.Timestamp) -> List[Tuple[str, float]]:
        """Calculate signal for given tickers on specific date"""
        pass
    
    def get_lookback_period(self) -> int:
        """Return minimum lookback period needed for this signal"""
        return 0

class SignalRegistry:
    """Registry to manage all available signals"""
    
    def __init__(self):
        self._signals: Dict[str, SignalBase] = {}
    
    def register(self, signal: SignalBase):
        """Register a new signal"""
        self._signals[signal.name] = signal
    
    def get_signal(self, name: str) -> SignalBase:
        """Get signal by name"""
        return self._signals.get(name)
    
    def list_signals(self) -> List[str]:
        """List all available signal names"""
        return list(self._signals.keys())





class RSISignal(SignalBase):
    def __init__(self, period: int = 14):
        super().__init__("RSI", {"period": period})
        self.period = period
    
    def calculate(self, data: pd.DataFrame, tickers: List[str], date: pd.Timestamp) -> List[Tuple[str, float]]:
        import talib as ta
        print("Calculating RSI signals... for date:", date)
        signal_scores = []
        
        for ticker in tickers: 
            try:
                close_prices = data['Close'][ticker]
                rsi = ta.RSI(close_prices, timeperiod=self.period)
                
                #date = date.normalize()  # Ensure date is in correct format
                if date in rsi.index:
                    # Convert RSI to signal score (0-100 to -1 to 1, with 50 as neutral)
                    rsi_value = rsi.loc[date]
                    signal_score = (rsi_value - 50) / 50  # Normalize to [-1, 1]
                    signal_scores.append((ticker, signal_score))
                else:
                    signal_scores.append((ticker, np.nan))
            except Exception as e:
                signal_scores.append((ticker, np.nan))
                
        return signal_scores
    
    def get_lookback_period(self) -> int:
        return self.period + 10  # Extra buffer for calculation     

class MACDSignal(SignalBase):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD", {
            "fast_period": fast_period,
            "slow_period": slow_period, 
            "signal_period": signal_period
        })
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame, tickers: List[str], date: pd.Timestamp) -> List[Tuple[str, float]]:
        import talib as ta
        print("Calculating MACD signals... for date:", date)
        signal_scores = []
        
        for ticker in tickers:
            try:
                close_prices = data['Close'][ticker]
                macd, macdsignal, _ = ta.MACD(close_prices, 
                                            fastperiod=self.fast_period,
                                            slowperiod=self.slow_period, 
                                            signalperiod=self.signal_period)
                
                if date in macd.index:
                    macd_histogram = macd.loc[date] - macdsignal.loc[date]
                    # Normalize MACD histogram (this may need adjustment based on your data)
                    signal_scores.append((ticker, macd_histogram))
                else:
                    signal_scores.append((ticker, np.nan))
            except Exception as e:
                signal_scores.append((ticker, np.nan))
                
        return signal_scores
    
    def get_lookback_period(self) -> int:
        return self.slow_period + self.signal_period + 10


class SMASignal(SignalBase):
    def __init__(self, short_period: int = 50, long_period: int = 200):
        super().__init__("SMA", {"short_period": short_period, "long_period": long_period})
        self.short_period = short_period
        self.long_period = long_period
    
    def calculate(self, data: pd.DataFrame, tickers: List[str], date: pd.Timestamp) -> List[Tuple[str, float]]:
        import talib as ta
        print("Calculating SMA signals... for date:", date)
        signal_scores = []
        
        for ticker in tickers:
            try:
                close_prices = data['Close'][ticker]
                sma_short = ta.SMA(close_prices, timeperiod=self.short_period)
                sma_long = ta.SMA(close_prices, timeperiod=self.long_period)
                
                if date in sma_short.index and date in sma_long.index:
                    # Calculate percentage difference
                    current_price = close_prices.loc[date]
                    sma_diff = (sma_short.loc[date] - sma_long.loc[date]) / current_price
                    signal_scores.append((ticker, sma_diff))
                else:
                    signal_scores.append((ticker, np.nan))
            except Exception as e:
                signal_scores.append((ticker, np.nan))
                
        return signal_scores
    
    def get_lookback_period(self) -> int:
        return self.long_period + 10