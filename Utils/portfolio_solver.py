from data.portfolios import Portfolio
from data.stock_snapshot import StockSnapshot
from data.stocks import Stock


class PortfolioSolver:
    def __init__(self, stocks: dict[Stock, StockSnapshot], penalty_factor=0.00001, max_weight_threshold=0.3, risk_aversion=0.3):
        self._stocks = stocks
        self.penalty_factor = penalty_factor
        self.max_weight_threshold = max_weight_threshold
        self.risk_aversion = risk_aversion
        
        
    def solve(self) -> Portfolio:
