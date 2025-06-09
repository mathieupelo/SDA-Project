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

    

    def __init__(self, stock_weights: dict[Stock, float], config: SolverConfig):
        self._stock_weights = stock_weights
        self._config = config

    @property
    def stock_weights(self) -> dict[Stock, float]:
        return self._stock_weights

    @property
    def config(self) -> SolverConfig:
        return self._config

    def solve(self, creation_date: date, price_history: pd.DataFrame) -> Portfolio:
        """
            Solve the signal data to predict the most performant portfolio
        """
        # Ensure the index is a regular Index (not DatetimeIndex)
        if not isinstance(price_history.index, pd.Index):
            raise ValueError("price_history must have a regular pd.Index of type 'date', not a DatetimeIndex.")

        # Ensure all index values are exactly of type 'date'
        if not all(isinstance(day, date) for day in price_history.index):
            raise ValueError("Each index entry in price_history must be of type 'datetime.date'.")

        stock_list = list(self._stock_weights.keys())
        stock_count = len(stock_list)

        # === Compute daily returns ===
        # TODO: Drop all na instead of just dropna?
        var_returns = price_history.astype(float).pct_change().dropna()

        # === Compute covariance matrix (sigma) ===
        sigma: np.ndarray = var_returns.cov().values

        # === Normalize alpha scores ===
        raw_scores = np.array([self._stock_weights[stock] for stock in stock_list])

        # TODO: If one of the raw scores is Nan, we should normalize the rest
        # array([-0.52055599,         nan,  0.25826212,         nan,         nan])
        # HOw do we want to treat NaN values? assign all 0??
        normalized_scores = raw_scores / np.sum(raw_scores)
        alpha_scores = normalized_scores.reshape(-1, 1)

        # === Optimization ===
        p = matrix(self._config.risk_aversion * sigma)
        # Scale expected returns + signals by (1 - alpha)
        q = matrix(- (1 - self._config.risk_aversion) * alpha_scores)

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
            stock: Portfolio.StockMetadata(weight, self._stock_weights[stock])
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
