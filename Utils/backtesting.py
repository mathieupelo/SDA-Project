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
from data.database import connect_to_database
from data.solver_config import SolverConfig
import itertools
from Utils.portfolio_solver import PortfolioSolver, construct_portfolio_solver

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

    def run_backtest(self, 
                          data: pd.DataFrame, 
                          tickers: List[str],
                          combination: SignalCombination,
                          config: BacktestConfig) -> BacktestResult:
        #TODO: Move logic to calculate a single backtest here
        
        date_range_eval = pd.date_range(start=config.start_date, end=config.end_date)
        dataset_scores = []

        print(data)

        for date in date_range_eval:
            print(f"Processing date: {date}")
            row = {('date', ''): date}
            
            # Only calculate signals that are in the current combination
            for signal_name in combination:
                signal = self.signal_registry.get_signal(signal_name)
                if signal is not None:
                    scores = signal.calculate(data, tickers, date)
                    print(f"Signal: {signal_name}, Scores: {scores}")
                    
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




        print(combined_df_no_nan)

        conn = connect_to_database('192.168.0.165')
        solver_config = SolverConfig()

        from datetime import date as dt

        price_histories: dict[str, dict[dt, float]] = {
            ticker: {date.date(): price for date, price in series.dropna().items()}
            for ticker, series in data['Close'].items()
        }


        

        for date, row in combined_df_no_nan.iterrows():

            # TODO: Only keep 1 year of data
            price_histories = price_histories.copy() 

            solver = construct_portfolio_solver(
                conn=conn,  # Replace with actual connection if needed
                alpha_scores=row.to_dict(),  # Assuming row contains alpha scores
                price_histories=price_histories,  # Replace with actual price histories if needed
                start_date=config.start_date,
                end_date=config.end_date,
                config=solver_config
            )

            portfolio = solver.solve(date)


            print(portfolio.stocks)


            



            
            

    

