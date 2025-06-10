# Enhanced Backtesting Architecture for Signal Testing
import sys
import os

# Add project root (the parent of 'scripts') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signals.signal_registry import SignalRegistry
from Utils.df_helper import combine_signals_scores
from dataclasses import dataclass
import itertools
from Utils.portfolio_solver import construct_portfolio_solver
import numpy as np
from Utils.time_utils import get_date_offset
from data.api import *
from data.portfolio import *
from Utils.backtest_result import BacktestResult


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

    def __init__(self, signal_registry: SignalRegistry):
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
                          combination: List[str],
                          config: BacktestConfig) -> BacktestResult:
        
                        # config: BacktestConfig) -> BacktestResult:

        time_stamp_range = pd.date_range(start=config.start_date, end=config.end_date)
        date_range = [time_stamp.date() for time_stamp in time_stamp_range]
        dataset_scores = []
        # Initialize an empty DataFrame to store scores
        returns_series_timeseries = pd.Series(dtype=float)
        weights_history = {}

        #TODO: Remove all dates that are not in the data
        # Filter date_range to only include dates present in the data
        date_range = [day for day in date_range if day in data.index]

        portfolio_list = []

        api = API('192.168.0.165')
        signal_scores = api.get_signal_scores_table_for_tickers(
            tickers=tickers,
            signals=combination,
            first_day=pd.to_datetime(config.start_date).date(),
            last_day=pd.to_datetime(config.end_date).date()
        )

        # Create equal weights for the combination
        signal_weights = {signal_id: 1.0/len(combination) for signal_id in combination}

        combined_scores_df = combine_signals_scores(signal_scores, signal_weights).dropna(how='all', axis=0)
        print(f"combined_scores_df : {combined_scores_df}")

    
        # Initialize solver configuration for portfolio optimization
        conn = connect_to_database('192.168.0.165')
        solver_config = SolverConfig(risk_aversion = 0)


        for day, row in combined_scores_df.iterrows():
            # Convert to Timestamp and subtract 1 year
            start_minus_1_year = pd.to_datetime(day) - get_date_offset('yearly')
            
            last_year_data = data.loc[start_minus_1_year.date() :day]

            # Handle case where some stocks are nan
            row = row.dropna()  # Drop NaN values to avoid issues with empty portfolios
            last_year_data = last_year_data[row.index]  # Filter data to only include relevant tickers

            if solver_config._max_weight_threshold < 1 / len(row):
                print(f"Not enough stocks to create portfolio at day : {day}, skipping...")
                continue

            solver = construct_portfolio_solver(
                conn=conn,  # Replace with actual connection if needed
                alpha_scores=row.to_dict(),  # Assuming row contains alpha scores
                config=solver_config
            )

            portfolio = solver.solve(day, last_year_data)

            #TODO: Calculate returns based on portfolio weights and price histories
            offset = get_date_offset(config.evaluation_period)
            evaluation_date = day + offset
            # We check the close date in the future to get the return of the portfolio

            weights_series = pd.Series(portfolio.get_weight_table())
            weights_history[day] = weights_series
            # Fetch prices from `data['Close']` for both date and evaluation_date
            try:
                prices_portfolio = {
                    ticker: data[ticker].get(day, None)
                    for ticker in tickers
                }
                prices_evaluation = {
                    ticker: data[ticker].get(evaluation_date.date(), None)
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
            returns_series_timeseries.at[day] = portfolio_return
        
            api.store_portfolio_results(
                portfolio=portfolio,
                signal_weights=signal_weights,
                yearly_return= portfolio_return
            )

            portfolio_list.append(portfolio)

        # ======= Compute Metrics =======
        returns = returns_series_timeseries.dropna()
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        win_rate = (returns > 0).mean()

        api.store_backtest_results(
            portfolios=portfolio_list,
            start_date=pd.to_datetime(config.start_date).date(),
            end_date=pd.to_datetime(config.end_date).date(),
            execution_time=timedelta()
        )

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
            signal_history=[]
    )

            



            
            

    

