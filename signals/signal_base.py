from abc import ABC, abstractmethod
from typing import Dict
from datetime import date
import pandas as pd


class SignalBase(ABC):
    """Abstract base class for all signals"""

    def __init__(self, pid: str, name: str, parameters: Dict = None):
        self.id = pid
        self.name = name
        self.parameters = parameters or {}

    @abstractmethod
    def calculate(self, close_prices: pd.DataFrame, ticker: str, day: date) -> float:
        """Calculate signal for given tickers on specific date"""
        pass

    def get_min_lookback_period(self) -> int:
        """Return minimum lookback period needed for this signal"""
        return 0

    def get_max_lookback_period(self) -> int:
        """Return maximum lookback period needed for this signal"""
        return 0
