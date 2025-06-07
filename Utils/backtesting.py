# Enhanced Backtesting Architecture for Signal Testing
import sys
import os

# Add project root (the parent of 'scripts') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Utils.signals import SignalBase, RSISignal, MACDSignal, SMASignal, SignalRegistry, combine_signals_from_df
import logging
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from data.utils.database import connect_to_database
from data.solver_config import SolverConfig
import itertools
from Utils.portfolio_solver import PortfolioSolver, construct_portfolio_solver
from datetime import date as dt
import numpy as np
from Utils.time_utils import get_date_offset
from data.api import *
from data.portfolios import *

@dataclass
class BacktestResult:
    """Results from backtesting"""
    combination_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    volatility: float
    returns_series: pd.Series
    weights_history: pd.DataFrame
    signal_history: pd.DataFrame

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: str
    end_date: str
    evaluation_period: str = "monthly"  # daily, weekly, monthly, yearly
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    lookback_window: int = 252  # Days of data needed for signal calculation
    holding_period: int = 20  # Days to hold position
    transaction_costs: float = 0.001  # 0.1% transaction cost
    universe: List[str] = None  # Stock universe


@dataclass
class SignalCombination:
    """Configuration for combining multiple signals"""
    signals: List[str]  # Signal names
    method: str = "weighted_sum"  # combination method
    
    def __post_init__(self):
        if len(self.signals) != len(self.weights):
            raise ValueError("Number of signals must match number of weights")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")



class BacktestEngine:
    """Main backtesting engine"""

    def __init__(self, signal_registry: SignalRegistry, portfolio_solver):
        self.signal_registry = signal_registry
        self.logger = logging.getLogger(__name__)

    def generate_signal_combinations(self,
                                     available_signals: List[str],
                                     max_signals: int = 3) -> List[List[str]]:
        """Generate all possible signal combinations"""
        combinations = []
        
        # Generate combinations of different sizes
        for r in range(1, max_signals + 1):
            for signal_combo in itertools.combinations(available_signals, r):
                combinations.append(list(signal_combo))
        
        return combinations
    
    

    def _calculate_period_returns(self, data: pd.DataFrame, tickers: List[str], 
                                weights: np.ndarray, start_date: str, period: int) -> List[float]:
        """Calculate returns for a given period based on weights"""
        start_dt = pd.to_datetime(start_date)


    def run_backtest(self, 
                          data: pd.DataFrame, 
                          tickers: List[str],
                          combination: SignalCombination,
                          config: BacktestConfig) -> BacktestResult:
        
                        # config: BacktestConfig) -> BacktestResult:

        date_range_eval = pd.date_range(start=config.start_date, end=config.end_date)
        dataset_scores = []
        # Initialize an empty DataFrame to store scores
        returns_series_timeseries = pd.Series(dtype=float)
        weights_history = {}

        for date in date_range_eval:
            row = {('date', ''): date}
            
            # Only calculate signals that are in the current combination
            for signal_name in combination:
                signal = self.signal_registry.get_signal(signal_name)
                if signal is not None:
                    scores = signal.calculate(data, tickers, date)
                    
                    for ticker, value in scores:
                        row[(signal_name, ticker)] = value
                else:
                    print(f"Warning: Signal {signal_name} not found in registry")
            
            dataset_scores.append(row)

        df_scores = pd.DataFrame(dataset_scores)
        df_scores.columns = pd.MultiIndex.from_tuples(df_scores.columns)
        
        # Create equal weights for the combination
        signal_weights = {signal_name: 1.0/len(combination) for signal_name in combination}
        
        # Combine signals (you'll need to import this function)
        # from Utils.signals import combine_signals_from_df
        combined_df = combine_signals_from_df(df_scores, tickers, signal_weights)
        combined_df_no_nan = combined_df.dropna(how="all", axis=0)

        # Initialize solver configuration for portfolio optimization
        conn = connect_to_database('192.168.0.165')
        solver_config = SolverConfig(risk_aversion = 0)

        # TODO : API.getpriceshistory 
        price_histories: dict[str, dict[dt, float]] = {
            ticker: {date.date(): price for date, price in series.dropna().items()}
            for ticker, series in data['Close'].items()
        }

        for date, row in combined_df_no_nan.iterrows():

            # TODO: Only keep 1 year of data
            # !!!!!!!!!!
            price_histories = price_histories.copy() 

            solver = construct_portfolio_solver(
                conn=conn,  # Replace with actual connection if needed
                alpha_scores=row.to_dict(),  # Assuming row contains alpha scores
                price_histories=price_histories,  # Replace with actual price histories if needed
                config=solver_config
            )

            portfolio = solver.solve(date)

            #TODO: Calculate returns based on portfolio weights and price histories
            offset = get_date_offset(config.evaluation_period)
            evaluation_date = date + offset
            # We check the close date in the future to get the return of the portfolio


            weights_series = pd.Series(portfolio.get_weight_table())
            weights_history[date] = weights_series
            # Fetch prices from `data['Close']` for both date and evaluation_date
            try:
                prices_portfolio = {
                    ticker: data['Close'][ticker].get(date, None)
                    for ticker in tickers
                }
                prices_evaluation = {
                    ticker: data['Close'][ticker].get(evaluation_date, None)
                    for ticker in tickers
                }
            except KeyError as e:
                print(f"Missing ticker in data: {e}")
                continue  # skip this date

 

            # Convert to DataFrames
            df_prices_portfolio = pd.Series(prices_portfolio)
            df_prices_evaluation = pd.Series(prices_evaluation)

            # Drop missing price data
            valid_idx = df_prices_portfolio.notna() & df_prices_evaluation.notna()
            df_prices_portfolio = df_prices_portfolio[valid_idx]
            df_prices_evaluation = df_prices_evaluation[valid_idx]
            weights_series = weights_series[valid_idx]

            if len(df_prices_portfolio) == 0:
                continue  # skip if we have no valid data for this date

            price_pct_change = (df_prices_evaluation - df_prices_portfolio) / df_prices_portfolio
            # Align them by index
            returns_series, weights_series = price_pct_change.align(weights_series)

            # Then multiply element-wise and sum
            weighted_returns = returns_series * weights_series

            portfolio_return = weighted_returns.sum()
            returns_series_timeseries.at[date] = portfolio_return
        
        
        # ======= Compute Metrics =======
        returns = returns_series_timeseries.dropna()
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        win_rate = (returns > 0).mean()

        return BacktestResult(
            combination_name="+".join(combination),
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            volatility=volatility,
            returns_series=returns_series_timeseries,
            weights_history=pd.DataFrame(weights_history).T,
            signal_history=df_scores
    )

            



            
            

    

