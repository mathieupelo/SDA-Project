import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

class Portfolio_Solver():
    def __init__(self, penalty_factor=0.00001, max_weight_threshold=0.3):
        self.method = ""
        self.penalty_factor = penalty_factor
        self.max_weight_threshold = max_weight_threshold

    def SolvePortfolio(self, tickers: list[str], data):
        # Step 2: Calculate daily returns
        returns = data['Close'].pct_change().dropna()

        # Step 3: Calculate mean returns and the covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Step 4: Implementing the Mean-Variance Optimization
        # Defining the optimization problem
        n_assets = len(tickers)

        # Convert mean returns and covariance matrix to cvxopt format
        mean_returns = np.array(mean_returns)
        cov_matrix = np.array(cov_matrix)

        # Covert data to cvxopt matrices
        P = matrix(cov_matrix)  # Covariance matrix
        q = matrix(-mean_returns)  # No linear term (we're not adding any risk-free asset)

        # Constraints: sum of weights = 1 (fully invested portfolio)
        G = matrix(-np.eye(n_assets))  # Negative identity matrix for weight constraints (all weights >= 0)
        h = matrix(np.zeros(n_assets))  # No short selling (weights >= 0)

        A = matrix(np.ones((1, n_assets)))  # Sum of weights constraint
        b = matrix(np.ones(1))  # The sum of weights must equal 1

        # Add a penalty for weights that exceed 0.3
        penalty_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            penalty_matrix[i, i] = self.penalty_factor
        # Penalty term for excess weight beyond 0.3
        penalty_term = np.array([max(0, weight - 0.3) ** 2 for weight in mean_returns])
        # Modify P with the penalty term (quadratic form)
        penalty_P = np.array(P) + penalty_matrix
        # Convert to cvxopt matrix
        P_penalty = matrix(penalty_P)

        # Solve the optimization problem with penalty term
        sol = solvers.qp(matrix(penalty_P), q, G, h, A, b)

        # Extract portfolio weights
        weights = np.array(sol['x']).flatten()

        # Step 5: Display results
        print(f"Optimized Portfolio Weights:\n{dict(zip(tickers, weights))}")

        # Step 6: Calculate Portfolio Return and Risk
        portfolio_return = np.sum(weights * mean_returns) * 252  # Annualize the return (252 trading days)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualize the volatility

        print(f"Expected Annual Portfolio Return: {portfolio_return:.2f}")
        print(f"Expected Annual Portfolio Volatility: {portfolio_volatility:.2f}")

        return weights
    

    def SolveSignalPortfolio(self, tickers: list[str], data, signal_scores: np.ndarray):
        # New Portfolio optimization 
        
        # Normalize signal scores to ensure they sum to 1
        normalized_scores = signal_scores / np.sum(signal_scores)

        # Defining the optimization problem
        n_assets = len(tickers)

        # Step 1: Define the penalty term based on the weights exceeding the threshold
        penalty_matrix = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            penalty_matrix[i, i] = self.penalty_factor

        # Penalty term for weights exceeding the threshold
        penalty_term = np.array([max(0, weight - self.max_weight_threshold) ** 2 for weight in normalized_scores])
        

        # Step 2: Implementing the Mean-Variance Optimization (with penalty for high weights)
        P = matrix(penalty_matrix)  # Covariance matrix with penalty
        q = matrix(-normalized_scores)  # We are maximizing the signal scores

        # Constraints: sum of weights = 1 (fully invested portfolio)
        G = matrix(-np.eye(n_assets))  # Negative identity matrix for weight constraints (all weights >= 0)
        h = matrix(np.zeros(n_assets))  # No short selling (weights >= 0)

        A = matrix(np.ones((1, n_assets)))  # Sum of weights constraint
        b = matrix(np.ones(1))  # The sum of weights must equal 1
       
        # Solve the optimization problem with penalty term
        sol = solvers.qp(P, q, G, h, A, b)

        # Extract portfolio weights
        weights = np.array(sol['x']).flatten()

        # Step 5: Display results
        print(f"Optimized Portfolio Weights:\n{dict(zip(tickers, weights))}")

        return weights
    

    def ShowPortfolioWeights(self, tickers, portfolio_weights):
        plt.figure(figsize=(10,6))
        plt.bar(tickers, portfolio_weights)
        plt.title("Optimized Portfolio Weights")
        plt.ylabel("Weight")
        plt.show()

    def CalculatePortfolioReturns(self, tickers, data, weights, start_date='2020-01-01', time_period=252):
        # Filter the data to start from the fixed start_date
        data.index = pd.to_datetime(data.index)
        data = data[data.index >= start_date]

        # Limit the data to the fixed time period (e.g., the first 252 trading days)
        data = data.iloc[:time_period]


        # Calculate daily returns for each stock
        returns = data['Close'].pct_change().dropna()

        # Calculate the portfolio returns by multiplying daily returns by the weights
        portfolio_returns = np.dot(returns, weights)

        # Cumulative returns of the portfolio
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Portfolio performance (total return)
        total_return = cumulative_returns[-1] - 1  # The final cumulative return minus 1 (for initial value)
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1  # Annualized return assuming 252 trading days

        # Display results
        print(f"Total Portfolio Return: {total_return * 100:.2f}%")
        print(f"Annualized Portfolio Return: {annualized_return * 100:.2f}%")

        return cumulative_returns, total_return, annualized_return
    
    def ShowPortfolioPerformance(self, cumulative_returns, data, start_date, end_date):

        # Filter data for the specific date range
        filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        filtered_data.index = pd.to_datetime(filtered_data.index)
        
        # Create an array to hold colors (green for up, red for down)
        colors = ['green' if cumulative_returns[i] >= cumulative_returns[i-1] else 'red' for i in range(1, len(cumulative_returns))]

        # Plot cumulative returns over time
        plt.figure(figsize=(10, 6))

        # Plot each segment individually
        for i in range(1, len(cumulative_returns)):
            plt.plot(filtered_data.index[i-1:i+1], cumulative_returns[i-1:i+1], color=colors[i-1])

        plt.title("Cumulative Portfolio Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()












