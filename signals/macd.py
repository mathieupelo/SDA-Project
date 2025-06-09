import math

import numpy as np
import pandas as pd
import talib as ta

from datetime import date
from signals.signal_base import SignalBase

class MACDSignal(SignalBase):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, max_lookback_period: int = 120):
        super().__init__("MACD", {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "max_lookback_period": max_lookback_period,
        })
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.max_lookback_period = max_lookback_period

    def calculate(self, close_prices: pd.DataFrame, ticker: str, day: date) -> float:

        macd, macd_signal, _ = ta.MACD(
            close_prices.values.astype(float),
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )

        # Align back the result to the date index
        macd_series = pd.Series(macd, index=close_prices.index)
        macd_signal_series = pd.Series(macd_signal, index=close_prices.index)

        if day not in macd_series.index:
            print(f'[WARNING] | Attempted to evaluate MACD on {day}, but there\'s no price history for {ticker} on that day.')
            return np.nan

        if pd.isna(macd_series.loc[day]) or pd.isna(macd_signal_series.loc[day]):
            print(f'[WARNING] | NaN evaluation for MACD {ticker} on {day} (window={len(close_prices)}).')
            return np.nan

        macd_hist = macd_series.loc[day] - macd_signal_series.loc[day]
        return macd_hist

    def get_min_lookback_period(self) -> int:
        # Minimum needed to just compute MACD (without smoothing)
        return self.slow_period + self.signal_period + 10  # safe buffer

    def get_max_lookback_period(self) -> int:
        # Safe upper bound to ensure full convergence
        return math.ceil((self.slow_period + self.signal_period) * 2)
