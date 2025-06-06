from typing import Dict
from datetime import date

class StockSnapshot:
    """
    Metadata of a stock at a given date. It contains the evaluated alpha_score from the signals pipeline
    and its price history.
    """
    def __init__(self, alpha_score: float, price_history: Dict[date, float]):
        self._alpha_score = alpha_score
        self._price_history = price_history

    @property
    def alpha_score(self) -> float:
        return self._alpha_score

    @property
    def price_history(self) -> Dict[date, float]:
        return self._price_history


