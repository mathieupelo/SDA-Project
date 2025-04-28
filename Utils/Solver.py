import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from Utils.Signals import *


import numpy as np
import pandas as pd

# Min-Max Normalization function to scale each signal between 0 and 1
def normalize_to_01(values):
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.zeros_like(values)  # If all values are the same, return a zero array
    return (values - min_val) / (max_val - min_val)


def combine_signals2(signal_weights, signal_scores):
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
       

        # Step 6: Calculate Portfolio Return and Risk
        portfolio_return = np.sum(weights * mean_returns) * 252  # Annualize the return (252 trading days)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualize the volatility



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
        
        # Step 2: Implementing the Mean-Variance Optimization (with penalty for high weights)
        #P = matrix(penalty_matrix)  # Covariance matrix with penalty
        P = matrix(np.zeros((n_assets, n_assets)))  # No variance term
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


        return weights
    
    def SolveSignalPortfolioMVO(self, tickers, var_data, signal_scores):
        """
        Mean-Variance Optimization with Diversification Constraints.
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
        G = np.vstack([
            -np.eye(n_assets),   # No short selling (weights ≥ 0)
            np.eye(n_assets)    # Max weight per asset
        ])
        h = np.hstack([
            np.zeros(n_assets),   # No shorting constraint (w ≥ 0)
            np.ones(n_assets) * self.max_weight_threshold  # Max per stock
        ])

        # Full investment constraint: sum(w) = 1
        A = matrix(np.ones((1, n_assets)))  
        b = matrix(np.ones(1))

        # Solve quadratic optimization problem
        sol = solvers.qp(P, q, G=matrix(G), h=matrix(h), A=A, b=b)

        # Extract portfolio weights
        weights = np.array(sol['x']).flatten()


        return weights
    
    def SolveSignalPortfolioMVO2(self, tickers, var_data, signal_scores):
        """
        Mean-Variance Optimization with Diversification Constraints.
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
    

    def ShowPortfolioWeights(self, tickers, portfolio_weights):
        plt.figure(figsize=(10,6))
        plt.bar(tickers, portfolio_weights)
        plt.title("Optimized Portfolio Weights")
        plt.ylabel("Weight")
        plt.show()

    def CalculatePortfolioReturns(self, tickers, data, weights, start_date='2020-01-01', time_period=252):
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
    

    def CalculatePortfolioReturns2(self, tickers, data, weights, start_date='2020-01-01', time_period=252):
        data.index = pd.to_datetime(data.index)

        if start_date not in data.index:
            raise ValueError(f"Start date {start_date} not found in data index.")

        # Get starting index
        start_idx = data.index.get_loc(start_date)

        # Limit the data to the fixed time period
        data_period = data.iloc[start_idx : start_idx + time_period]

        if data_period.shape[0] < time_period:
            raise ValueError(f"Not enough data points from {start_date} for {time_period} periods.")

        # Use 'Close' prices (or 'Adj Close' if you prefer)
        if isinstance(data.columns, pd.MultiIndex):
            # If data has multi-level columns (like ['Adj Close', ticker1], etc)
            price_data = data['Close']
        else:
            price_data = data['Close']

        # Calculate daily returns
        returns = price_data.pct_change().dropna()

        if returns.shape[1] != len(tickers):
            # Reorder columns to match ticker order, fill missing with 0 returns
            returns = returns.reindex(columns=tickers, fill_value=0)

        # Multiply returns with portfolio weights
        portfolio_returns = returns @ weights  # (days x assets) @ (assets,) = (days,)

        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Total return
        if len(cumulative_returns) == 0:
            total_return = 0.0
            annualized_return = 0.0
        else:
            total_return = cumulative_returns.iloc[-1] - 1  # final cumulative return minus 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1  # annualized

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





    def CalculateEvalReturns(self, tickers, data, df_eval, W):
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
            combined_scores = combine_signals2(signal_weights, [rsi_scores, macd_scores, sma_scores])
            combined_scores_with_tickers = list(zip(tickers, combined_scores))

            #print(f"Combined Scores: {combined_scores_with_tickers}")

            # Step 2b: Solve the portfolio based on the combined signal scores
            portfolio_weights = self.SolveSignalPortfolioMVO2(tickers, data, combined_scores)

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













