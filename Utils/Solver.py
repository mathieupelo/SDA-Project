import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

class Portfolio_Solver():
    def __init__(self):
        self.method = ""

    def SolvePortfolio(self, tickers: list[str], data, signal_scores: np.ndarray):
        # Step 2: Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        print(returns)

        # Step 3: Calculate mean returns and the covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        adjusted_mean_returns = mean_returns * signal_scores
        # Normalize signal scores to ensure they sum to 1
        signal_scores = signal_scores / np.sum(signal_scores)

        # Step 4: Implementing the Mean-Variance Optimization
        # Defining the optimization problem
        n_assets = len(tickers)

        # Convert mean returns and covariance matrix to cvxopt format
        adjusted_mean_returns = np.array(adjusted_mean_returns)
        cov_matrix = np.array(cov_matrix)

        # Covert data to cvxopt matrices
        P = matrix(cov_matrix)  # Covariance matrix
        q = matrix(-adjusted_mean_returns)  # No linear term (we're not adding any risk-free asset)

        # Constraints: sum of weights = 1 (fully invested portfolio)
        G = matrix(-np.eye(n_assets))  # Negative identity matrix for weight constraints (all weights >= 0)
        h = matrix(np.zeros(n_assets))  # No short selling (weights >= 0)

        A = matrix(np.ones((1, n_assets)))  # Sum of weights constraint
        b = matrix(np.ones(1))  # The sum of weights must equal 1

        # Solve the optimization problem
        sol = solvers.qp(P, q, G, h, A, b)

        # Extract portfolio weights
        weights = np.array(sol['x']).flatten()

        # Step 5: Display results
        print(f"Optimized Portfolio Weights:\n{dict(zip(tickers, weights))}")

        # Step 6: Plot the optimized portfolio
        plt.figure(figsize=(10,6))
        plt.bar(tickers, weights)
        plt.title("Optimized Portfolio Weights")
        plt.ylabel("Weight")
        plt.show()

        # Step 7: Calculate Portfolio Return and Risk
        portfolio_return = np.sum(weights * adjusted_mean_returns) * 252  # Annualize the return (252 trading days)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualize the volatility

        print(f"Expected Annual Portfolio Return: {portfolio_return:.2f}")
        print(f"Expected Annual Portfolio Volatility: {portfolio_volatility:.2f}")
        

# Let's assume we are interested in the following stocks: AAPL, MSFT, TSLA, AMZN, GOOG
tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOG']
# Signal scores for AAPL, MSFT, TSLA, AMZN, GOOG
signal_scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  

# Step 1: Download historical stock data
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')

portfolio_solver = Portfolio_Solver()
portfolio_solver.SolvePortfolio(tickers, data, signal_scores)









