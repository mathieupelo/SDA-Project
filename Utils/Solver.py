import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from jupyterlab.utils import deprecated
from pandas import DataFrame

from Utils.Signals import *

# Min-Max Normalization function to scale each signal between 0 and 1
def normalize_to_01(values):
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.zeros_like(values)  # If all values are the same, return a zero array
    return (values - min_val) / (max_val - min_val)


def combine_signals(signal_weights, signal_scores):
    #print("COMBINE SIGNALS 2")
    rsi, macd, sma = signal_scores
    w_rsi, w_macd, w_sma = signal_weights
    
    #print(f"rsi : {rsi}, macd : {macd}, sma : {sma}")

    # Normalize each signal
    rsi_norm = normalize_to_01(rsi)
    macd_norm = normalize_to_01(macd)
    sma_norm = normalize_to_01(sma)

    #print(f"rsi2 : {rsi_norm}, macd2 : {macd_norm}, sma2 : {sma_norm}")
    
    # Combine the signals using weighted sum
    combined = rsi_norm * w_rsi + macd_norm * w_macd + sma_norm * w_sma
    
    # Clip extreme values if needed to avoid outliers
    combined = np.clip(combined, -5, 5)
    
    return combined

class Portfolio_Solver():
    def __init__(self, penalty_factor=0.00001, max_weight_threshold=0.3):
        self.method = ""
        self.penalty_factor = penalty_factor
        self.max_weight_threshold = max_weight_threshold
        self.risk_aversion = 0.5  # λ: Controls return vs. risk trade-off

    @deprecated
    def solve_signal_portfolio_MVO(self, tickers, var_data: DataFrame, signal_scores):
        """
        Mean-Variance Optimization with Diversification Constraints.

        Parameters:
        - tickers: A list of ticker symbols.
        - var_data: A DataFrame containing the 'Close' prices for each stock (tickers as columns).
        - signal_scores: A vector of signals scores for each stock (tickers as columns).

        Returns:
        - Recommended output weight for each stock.
        """
        # Normalize signal scores
        normalized_scores = signal_scores / np.sum(signal_scores)

        # Compute historical daily returns
        var_returns = var_data['Close'].pct_change().dropna()

        # Expected returns (mean of past returns)
        mu = var_returns.mean().values.reshape(-1, 1)  

        # Compute covariance matrix
        Sigma = var_returns.cov().values  

        n_assets = len(tickers)

        # Quadratic term (Risk component: λ * w'Σw)
        P = matrix(self.risk_aversion * Sigma)  

        # Linear term (maximize return + signal influence)
        q = matrix(-mu - normalized_scores.reshape(-1, 1))  

        # Constraints
        G = -np.eye(n_assets)   # Only non-negative weights constraint
        h = np.zeros(n_assets)  # w ≥ 0

        # Full investment constraint: sum(w) = 1
        A = matrix(np.ones((1, n_assets)))  
        b = matrix(np.ones(1))

        # Solve quadratic optimization problem
        sol = solvers.qp(P, q, G=matrix(G), h=matrix(h), A=A, b=b)

        # Extract portfolio weights
        weights = np.array(sol['x']).flatten()


        return weights

    @deprecated
    def show_portfolio_weights(self, tickers, portfolio_weights):
        plt.figure(figsize=(10,6))
        plt.bar(tickers, portfolio_weights)
        plt.title("Optimized Portfolio Weights")
        plt.ylabel("Weight")
        plt.show()

    @deprecated
    def calculate_portfolio_returns(self, tickers, data, weights, start_date='2020-01-01', time_period=252):
        #TODO: Add a check that time_period == 1
        # if it is, we do not do total_return = cumulative_returns[-1] - 1  # The final cumulative return minus 1 (for initial value)
        
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
        #print(f"Total Portfolio Return: {total_return * 100:.2f}%")
        #print(f"Annualized Portfolio Return: {annualized_return * 100:.2f}%")

        return cumulative_returns, total_return, annualized_return
    


    @deprecated
    def show_portfolio_performance(self, cumulative_returns, data, start_date, end_date):
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


    @deprecated
    def calculate_eval_returns(self, tickers, data, df_eval, W):
        # Now, iterate over df_step1 to calculate combined scores, portfolio weights, and returns
        dataset_returns_ridge = []

        # Step 2: Create the second DataFrame with combined_scores, portfolio_weights, total_return, and annualized_return
        for index, row in df_eval.iterrows():
            date = row['date']
            rsi_scores = row['rsi_scores']
            macd_scores = row['macd_scores']
            sma_scores = row['sma_scores']

            #print(f"Processing {date} for Step 2")

            # Step 2a: Combine the signals (you can later train a model to adjust these weights)
            signal_weights = W  # You can adjust these weights later based on your model
            combined_scores = combine_signals(signal_weights, [rsi_scores, macd_scores, sma_scores])
            combined_scores_with_tickers = list(zip(tickers, combined_scores))

            #print(f"Combined Scores: {combined_scores_with_tickers}")

            # Step 2b: Solve the portfolio based on the combined signal scores
            portfolio_weights = self.solve_signal_portfolio_MVO(tickers, data, combined_scores)

            # Step 2c: Calculate the returns for the portfolio based on the optimized weights
            cumulative_returns, total_return, annualized_return = self.CalculatePortfolioReturns(tickers, data, portfolio_weights, start_date=date, time_period=20)

            # Step 2d: Add the calculated values to the second dataset
            dataset_returns_ridge.append({
                'date': date,
                'combined_scores': combined_scores,
                'portfolio_weights': portfolio_weights,
                'total_return': total_return,
                'annualized_return': annualized_return
            })

        # Convert the second dataset into a DataFrame
        df_returns = pd.DataFrame(dataset_returns_ridge)

        # You can now display or use the two DataFrames for further analysis or training your model
        #display(df_returns)  # The second DataFrame with combined scores, portfolio weights, and returns

        average_annualized_return = df_returns['annualized_return'].mean()
        #print("Average return for ridge : ", average_annualized_return)

        total_return_sum = df_returns['total_return'].sum()
        #print(f"Total Return Sum: {total_return_sum}")

        return average_annualized_return, total_return_sum













