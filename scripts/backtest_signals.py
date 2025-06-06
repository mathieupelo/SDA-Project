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
from datetime import datetime, timedelta, date as dt
from Utils.signals import SignalBase, RSISignal, MACDSignal, SMASignal, SignalRegistry
from Utils.Solver import Portfolio_Solver
from Utils.Solver import *
from data.stocks import *
from Utils.backtesting import *
from data.database import connect_to_database
from data.stocks import get_stocks
from data.stock_price import get_stock_price_table


# ============================================================================
# 1. SIGNAL INTERFACE AND REGISTRY
# ============================================================================

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

def run_backtests():
    tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOG']
    tickers = ['AAPL', 'MSFT']
    data = yf.download(tickers, start='2010-01-01', end='2025-01-01')

    signal_registry = setup_backtesting_system()

    # Before callinb BacktestEngine, make sure we have a signal reistry and the signals registered on
    portfolio_solver = Portfolio_Solver()
    backtest_engine = BacktestEngine(signal_registry, portfolio_solver)

    # Configuration
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2020-12-31",
        rebalance_frequency="monthly",
        holding_period=20
    )

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

    tickers = ['AAPL', 'MSFT', 'META', 'AMZN', 'GOOG']
    data = yf.download(tickers, start='2010-01-01', end='2025-01-01')

    signal_registry = setup_backtesting_system()

    # Before callinb BacktestEngine, make sure we have a signal reistry and the signals registered on
    portfolio_solver = Portfolio_Solver()
    backtest_engine = BacktestEngine(signal_registry, portfolio_solver)

    # Configuration
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2020-12-31",
        rebalance_frequency="monthly",
        holding_period=20
    )

    available_signals = signal_registry.list_signals()
    print(f"available_signals : ", available_signals)

    signal_combination = ['RSI', 'MACD', 'SMA']  # Example of a single combination

    print(f"Backtesting signal combination : {signal_combination}")
    backtest_engine.run_backtest(
        tickers=tickers,
        data=data,
        combination=signal_combination,
        config=config
    )

    print("Single backtest completed successfully.")

if __name__ == "__main__":
    run_single_backtest()