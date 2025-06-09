import math

import numpy as np
import pandas as pd
import talib as ta

from datetime import date
from signals.signal_base import SignalBase


class SMASignal(SignalBase):
    def __init__(self, short_period: int = 50, long_period: int = 200):
        super().__init__("SMA", "SMA", {"short_period": short_period, "long_period": long_period})
        self.short_period = short_period
        self.long_period = long_period

    def calculate(self, close_prices: pd.DataFrame, ticker: str, day: date) -> float:

        sma_short = ta.SMA(close_prices, timeperiod=self.short_period)
        sma_long = ta.SMA(close_prices, timeperiod=self.long_period)

        if day not in sma_short.index or day not in sma_long.index:
            print(f'[WARNING] | Attempted to evaluate SMA on {day}, but there\'s no price history for {ticker} on that day.')
            return np.nan

        sma_short = sma_short.loc[day]
        sma_long =  sma_long.loc[day]

        if pd.isna(sma_short) or pd.isna(sma_long):
            print(f'[WARNING] | NaN evaluation for SMA {ticker} on {day} (window={len(close_prices)}).')
            return np.nan

        # Calculate percentage difference
        current_price = close_prices.loc[day]
        sma_diff = (sma_short - sma_long) / current_price
        return sma_diff

    def get_min_lookback_period(self) -> int:
        return self.long_period

    def get_max_lookback_period(self) -> int:
        return math.ceil(self.long_period * 1.5)
