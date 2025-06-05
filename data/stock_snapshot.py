from typing import Dict
from data.stocks import Stock
from datetime import date

class StockSnapshot:
    """
    Metadata of a stock at a given date. It contains the evaluated alpha_score from the signals pipeline
    and its price history.
    """
    def __init__(self, stock: Stock, alpha_score: float, price_history: Dict[date, float]):
        self._stock = stock
        self._alpha_score = alpha_score
        self._price_history = price_history

    @property
    def stock(self) -> Stock:
        return self._stock

    @property
    def alpha_score(self) -> float:
        return self._alpha_score

    @property
    def price_history(self) -> Dict[date, float]:
        return self._price_history


