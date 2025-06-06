from datetime import date
from cvxopt import matrix, solvers
from data.portfolios import Portfolio
from data.stocks import *
from data.database import *
from data.solver_config import SolverConfig
import uuid
import numpy as np
import pandas as pd


class StockSnapshot:
    """
    Metadata of a stock at a given date. It contains the evaluated alpha_score from the signals pipeline
    and its price history.
    """

    def __init__(self, alpha_score: float, price_history: dict[date, float]):
        self._alpha_score = alpha_score
        self._price_history = price_history

    @property
    def alpha_score(self) -> float:
        return self._alpha_score

    @property
    def price_history(self) -> dict[date, float]:
        return self._price_history


class PortfolioSolver:

    def __init__(self, stocks: dict[Stock, StockSnapshot], config: SolverConfig):
        self._stocks = stocks
        self._config = config

    @property
    def stocks(self) -> dict[Stock, StockSnapshot]:
        return self._stocks

    @property
    def config(self) -> SolverConfig:
        return self._config

    def solve(self, creation_date: date) -> Portfolio:
        """
            Solve the signal data to predict the most performant portfolio
        """
        stock_list = list(self._stocks.keys())
        stock_count = len(stock_list)

        # === Build price table from snapshots ===
        price_data = pd.DataFrame({
            stock.ticker: snapshot.price_history
            for stock, snapshot in self._stocks.items()
        })

        # === Compute daily returns ===
        var_returns = price_data.pct_change().dropna()

        # === Compute expected returns (mu) ===
        mu: np.ndarray = var_returns.mean().values.reshape(-1, 1)

        # === Compute covariance matrix (sigma) ===
        sigma: np.ndarray = var_returns.cov().values

        # === Normalize alpha scores ===
        raw_scores = np.array([self._stocks[stock].alpha_score for stock in stock_list])
        normalized_scores = raw_scores / np.sum(raw_scores)
        alpha_scores = normalized_scores.reshape(-1, 1)

        # === Optimization ===
        p = matrix(self._config.risk_aversion * sigma)
        q = matrix(-mu - alpha_scores)

        g = np.vstack([
            -np.eye(stock_count),   # No short selling (weights ≥ 0)
            np.eye(stock_count)    # Max weight per asset
        ])
        h = np.hstack([
            np.zeros(stock_count),   # No shorting constraint (w ≥ 0)
            np.ones(stock_count) * self._config.max_weight_threshold  # Max per stock
        ])

        a = matrix(np.ones((1, stock_count)))
        b = matrix(np.ones(1))

        sol = solvers.qp(p, q, G=matrix(g), h=matrix(h), A=a, b=b)
        weights = np.array(sol['x']).flatten()

        # === Build Portfolio ===
        metadata = {
            stock: Portfolio.StockMetadata(weight, self._stocks[stock].alpha_score)
            for stock, weight in zip(stock_list, weights)
        }

        portfolio_id = str(uuid.uuid1())
        return Portfolio(portfolio_id, creation_date, metadata)



def construct_portfolio_solver(
        conn: MySQLConnectionAbstract,
        alpha_scores: dict[str, float],
        price_histories: dict[str, dict[date, float]],
        config: SolverConfig,
        fetch_database: bool = True,
) -> PortfolioSolver:

    stock_snapshots: dict[Stock, StockSnapshot] = { }

    for ticker, alpha_score in alpha_scores.items():
        stock = None
        price_history = price_histories[ticker]

        if fetch_database:
            stock = get_stock(conn, ticker)

        if not stock:
            stock = Stock(str(uuid.uuid1()), f"{ticker}_TEST", ticker)

        if fetch_database:
            price_histories.get(ticker)

        stock_snapshots[stock] = StockSnapshot(alpha_score, price_history)

    solver = PortfolioSolver(stock_snapshots, config)
    return solver
