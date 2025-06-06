from collections import defaultdict
from cvxopt import matrix, solvers
from data.portfolios import Portfolio
from data.stock_snapshot import StockSnapshot
from data.stocks import *
from data.utils import *
from data.solver_config import SolverConfig
import uuid
import yfinance as yf
import numpy as np
import pandas as pd

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

    def get_price_history_table(self) -> dict[str, dict[date, float]]:
        flattened_snapshots = defaultdict(dict)
        for stock, snapshot in self._stocks.items():
            ticker = stock.ticker
            for dt, price in snapshot.price_history.items():
                flattened_snapshots[ticker][dt] = price

        return dict(flattened_snapshots)


    def get_alpha_scores_table(self) -> dict[str, float]:
        return {
            stock.ticker: snapshot.alpha_score
            for stock, snapshot in self._stocks.items()
        }


    def solve(self, creation_date: date) -> Portfolio:

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

        g = -np.eye(stock_count)
        h = np.zeros(stock_count)

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
        start_date: date,
        end_date: date,
        config: SolverConfig,
        fetch_database: bool = True,
) -> PortfolioSolver:

    stock_snapshots: dict[Stock, StockSnapshot] = { }

    for ticker, alpha_score in alpha_scores.items():
        stock = None
        price_history = None

        if fetch_database:
            stock = get_stock(conn, ticker)

        if not stock:
            stock = Stock(str(uuid.uuid1()), ticker, f"{ticker}_TEST")

        if fetch_database:
            price_histories.get(ticker)

        if not price_history:
            data = yf.download([ ticker ], start=start_date, end=end_date)
            price_history = data['Close'][ticker].to_dict()

        if not price_history:
            continue

        stock_snapshots[stock] = StockSnapshot(stock, alpha_score, price_history)

    solver = PortfolioSolver(stock_snapshots, config)
    return solver
