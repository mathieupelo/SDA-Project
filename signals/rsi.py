import numpy as np
import pandas as pd
import talib as ta

from datetime import date
from signals.signal_base import SignalBase


class RSISignal(SignalBase):
    def __init__(self, period: int = 14):
        super().__init__("RSI", {"period": period})
        self.period = period

    def calculate(self, close_prices: pd.DataFrame, ticker: str, day: date) -> float:

        rsi = ta.RSI(close_prices, timeperiod=self.period)

        if day not in rsi.index:
            print(f'[WARNING] | Attempted to evaluate RSI on {day}, but there\'s no price history for {ticker} on that day.')
            return np.nan

        # Retrieve signal score for day (0-100 to -1 to 1, with 50 as neutral)
        rsi_value = rsi.loc[day]

        if pd.isna(rsi_value):
            print(f'[WARNING] | NaN evaluation for RSI {ticker} on {day} (window={len(close_prices)}).')
            return np.nan

        # Normalize to [-1, 1]
        signal_score = (rsi_value - 50) / 50
        return signal_score

    def get_min_lookback_period(self) -> int:
        return self.period + 10  # Extra buffer for calculation

    def get_max_lookback_period(self) -> int:
        return self.period * 3
