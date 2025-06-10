from datetime import date
from cvxopt import matrix, solvers
from data.portfolio import Portfolio
from data.stock import *
from data.utils.database import *
from data.solver_config import SolverConfig
import uuid
import numpy as np
import pandas as pd


class PortfolioSolver:

    def __init__(self, alpha_scores: dict[Stock, float], config: SolverConfig):
        if not alpha_scores:
            raise ValueError("alpha_scores cannot be empty.")

        for stock, score in alpha_scores.items():
            if score is None or np.isnan(score):
                raise ValueError(f"Invalid alpha score for stock {stock}: {score}")

        self._alpha_scores = alpha_scores
        self._config = config

    @property
    def alpha_scores(self) -> dict[Stock, float]:
        return self._alpha_scores

    @property
    def config(self) -> SolverConfig:
        return self._config

    def solve(self, creation_date: date, price_history: pd.DataFrame) -> Portfolio:
        """
            Solve the signal data to predict the most performant portfolio
        """
        stock_list, scores = zip(*self._alpha_scores.items())
        stock_count = len(stock_list)
        max_threshold = max(self._config.max_weight_threshold, 1.0 / stock_count)

        # TODO: Call astype earlier, to avoid casting multiple times
        var_returns = price_history.astype(float).pct_change(fill_method=None)

        # === Compute covariance matrix (sigma) ===
        sigma: np.ndarray = var_returns.cov().values

        # === Normalize alpha scores ===
        scores = np.array(scores)
        scores = scores / np.sum(scores)
        scores = scores.reshape(-1, 1)

        # === Optimization ===
        p = matrix(self._config.risk_aversion * sigma)

        # Scale expected returns + signals by (1 - alpha)
        q = matrix(- (1 - self._config.risk_aversion) * scores)

        g = np.vstack([
            -np.eye(stock_count),   # No short selling (weights ≥ 0)
            np.eye(stock_count)    # Max weight per asset
        ])

        h = np.hstack([
            np.zeros(stock_count),   # No shorting constraint (w ≥ 0)
            np.ones(stock_count) * max_threshold  # Max per stock
        ])

        a = matrix(np.ones((1, stock_count)))
        b = matrix(np.ones(1))

        sol = solvers.qp(p, q, G=matrix(g), h=matrix(h), A=a, b=b)
        weights = np.array(sol['x']).flatten()

        # === Build Portfolio ===
        metadata = {
            stock: Portfolio.StockMetadata(weight, self._alpha_scores[stock])
            for stock, weight in zip(stock_list, weights)
        }

        portfolio_id = str(uuid.uuid1())
        return Portfolio(portfolio_id, creation_date, metadata, self._config)

def construct_portfolio_solver(
        conn: MySQLConnectionAbstract,
        alpha_scores: dict[str, float],
        config: SolverConfig,
) -> PortfolioSolver:

    stock_snapshots: dict[Stock, float] = { }

    for ticker, alpha_score in alpha_scores.items():

        stock = get_stock(conn, ticker)
        if not stock:
            stock = Stock(str(uuid.uuid1()), f"{ticker}_TEST", ticker)

        stock_snapshots[stock] = alpha_score

    solver = PortfolioSolver(stock_snapshots, config)
    return solver
