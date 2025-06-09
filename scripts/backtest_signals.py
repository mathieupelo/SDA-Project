# Enhanced Backtesting Architecture for Signal Testing
import sys
import os

# Add project root (the parent of 'scripts') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from data.stocks import *
from Utils.backtesting import *
from data.utils.database import connect_to_database
from data.stocks import get_stocks
from Utils.df_helper import *
from data.api import API

def setup_backtesting_system():
    """Example of how to set up the backtesting system"""
    # Initialize signal registry
    signal_registry = SignalRegistry()
    
    # Register available signals
    signal_registry.register(RSISignal(period=14))
    #signal_registry.register(RSISignal(period=30))  # Different parameter
    signal_registry.register(MACDSignal())
    signal_registry.register(SMASignal())
    
    return signal_registry

def plot_backtest_results(backtest_results: BacktestResult):
    """Plot the results of the backtest"""
    import matplotlib.pyplot as plt

    # Starting with $10,000
    initial_investment = 10_000

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results.returns_series.index, initial_investment * (1 + backtest_results.returns_series.values), label='Portfolio Value')

    # Add horizontal line for final value
    average_return = backtest_results.returns_series.mean()
    final_value = initial_investment * (1 + average_return)
    plt.axhline(final_value, color='red', linestyle='--', label=f'Final Value: ${final_value:,.2f}')

    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_backtests():
    tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOG']
    tickers = ['AAPL', 'MSFT']
    data = yf.download(tickers, start='2010-01-01', end='2025-01-01')

    api = API('192.168.0.165')
    api.ensure_database_is_up_to_date()
    
    signal_registry = setup_backtesting_system()

    # Before calling BacktestEngine, make sure we have a signal registry and the signals registered on

    backtest_engine = BacktestEngine(signal_registry, portfolio_solver)

    # Configuration
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2020-12-31",
        rebalance_frequency="monthly",
        holding_period=20
    )

    # TODO: Use date from DateTime in BacktestConfig and derive start and end from config.
    data = api.get_price_history_for_tickers(tickers, date(2020, 1, 1), date(2025, 1, 1))

    # If you want DataFrame again
    # data = pd.DataFrame.from_dict(data)

    available_signals = signal_registry.list_signals()
    print(f"available_signals : ", available_signals)

    signal_combinations = backtest_engine.generate_signal_combinations(available_signals, max_signals=3)

    for signal_combination in signal_combinations:
        print(f"Backtesting signal combination : {signal_combination}")
        backtest_engine.run_backtest(
            tickers=tickers,
            data=data,
            combination=signal_combination,
            config=config
        )

        # Register available signals
        
    """
    start_date_eval = '2019-01-01'
    end_date_eval = '2020-01-01'
    date_range_eval = pd.date_range(start=start_date_eval, end=end_date_eval)
    print(date_range_eval)

    # Initialize an empty list to store the rows for the first DataFrame
    dataset_scores = []

    # Initialize signal registry
    signal_registry = SignalRegistry()
    
    # Register available signals
    signal_registry.register(RSISignal(period=14))
    signal_registry.register(MACDSignal())
    signal_registry.register(SMASignal())

    


    # Step 1: Create the DataFrame with rsi_scores, macd_scores, and sma_scores
    for date in date_range_eval:
        print(f"Processing date: {date}")


        row = {('date', ''): date}

        for signal_name in signal_registry.list_signals():
            signal = signal_registry.get_signal(signal_name)
            scores = signal.calculate(data, tickers, date)
            
            print(f"Signal: {signal_name}, Scores: {scores}")
            for ticker, value in scores:
                row[(signal_name, ticker)] = value

        dataset_scores.append(row)

        df = pd.DataFrame(dataset_scores)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

    # Display the first DataFrame
    print("First DataFrame with scores:")
    print(df.head())
    print(df['RSI']['AAPL'])

    signal_weights = {
        "RSI": 1/3,
        "MACD": 1/3,
        "SMA": 1/3,
    }

    combined_df = combine_signals_from_df(df, tickers, signal_weights)
    print("combined_df")
    print(combined_df)
    # For every date, we create a portfolio based on the combined scores and we calculate the returns


    conn = connect_to_database('192.168.0.165')
       
    for date, row in combined_df.iterrows():
        print(row.to_dict())


        #print(f"Processing date: {date['date']}")
        #combined_scores = combined_df.loc[date].to_dict()
        #print(combined_scores)
        """
def run_single_backtest():
    print("Running single backtest")

    # TODO: Call function instead
    

    tickers = ['TTWO', 'MSFT', 'EA', 'SONY']
    start_dt  = date(2010, 1, 1)
    end_dt = date(2025, 1, 1)

    data_df = get_price_history_for_tickers_df(tickers, 
                                               start_date=start_dt, 
                                               end_date=end_dt)
    
    print("df_data DataFrame:")
    print(data_df)

    signal_registry = setup_backtesting_system()
    # Before callinb BacktestEngine, make sure we have a signal reistry and the signals registered on
    backtest_engine = BacktestEngine(signal_registry)

    # Configuration
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2020-12-31",
        rebalance_frequency="monthly",
        evaluation_period="yearly",
        holding_period=20
    )

    signal_combination = ['RSI', 'MACD', 'SMA']  # Example of a single combination

    print(f"Backtesting signal combination : {signal_combination}")
    backtest_results = backtest_engine.run_backtest(
        tickers=tickers,
        data=data_df,
        combination=signal_combination,
        config=config
    )

    plot_backtest_results(backtest_results)



if __name__ == "__main__":
    run_single_backtest()