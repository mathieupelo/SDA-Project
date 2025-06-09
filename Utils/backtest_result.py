from dataclasses import dataclass
import pandas as pd

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
