from datetime import date
from cvxopt import matrix, solvers
from data.portfolio import Portfolio
from data.solver_config import SolverConfig
import uuid
import numpy as np
import pandas as pd


def solve_portfolio(
        creation_date: date,
        price_history: pd.DataFrame,
        alpha_scores: dict[str, float],
        config: SolverConfig) -> Portfolio:
    """
        Solve the signal data to predict the most performant portfolio
    """
    stock_list, scores = zip(*alpha_scores.items())
    stock_count = len(stock_list)
    max_threshold = max(config.max_weight_threshold, 1.0 / stock_count)
    var_returns = price_history.pct_change(fill_method=None)

    # === Compute covariance matrix (sigma) ===
    sigma: np.ndarray = var_returns.cov().values

    # SCORES ARE ALREADY NORMALIZED
    scores = np.array(scores).reshape(-1, 1)

    # === Optimization ===
    p = matrix(config.risk_aversion * sigma)

    # Scale expected returns + signals by (1 - alpha)
    q = matrix(- (1 - config.risk_aversion) * scores)

    # Create inequality constraints matrices
    identity = np.eye(stock_count)
    g = np.vstack([-identity, identity])  # shape: (2*stock_count, stock_count)
    h = np.hstack([np.zeros(stock_count), np.ones(stock_count) * max_threshold])

    a = matrix(np.ones((1, stock_count)))
    b = matrix(1.0)

    sol = solvers.qp(p, q, G=matrix(g), h=matrix(h), A=a, b=b)
    weights = np.array(sol['x']).flatten()

    # === Build Portfolio ===
    metadata = {
        stock: Portfolio.StockMetadata(weight, alpha_scores[stock])
        for stock, weight in zip(stock_list, weights)
    }

    portfolio_id = str(uuid.uuid1())
    return Portfolio(portfolio_id, creation_date, metadata, config)