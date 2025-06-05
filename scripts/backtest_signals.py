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
from datetime import datetime, timedelta
from Utils.Signals import SignalBase, RSISignal, MACDSignal, SMASignal, SignalRegistry
from Utils.Solver import Portfolio_Solver
import logging
from Utils.Solver import *


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

def run_backtest():
    tickers = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOG']
    data = yf.download(tickers, start='2010-01-01', end='2025-01-01')

    signal_registry = setup_backtesting_system()
    portfolio_solver = Portfolio_Solver()
    
    start_date_eval = '2019-01-01'
    end_date_eval = '2020-01-01'
    date_range_eval = pd.date_range(start=start_date_eval, end=end_date_eval)
    print(date_range_eval)

    # Initialize an empty list to store the rows for the first DataFrame
    dataset_scores = []

    # Step 1: Create the DataFrame with rsi_scores, macd_scores, and sma_scores
    for date in date_range_eval:
        print(f"Processing date: {date}")
        # Initialize signal registry
        signal_registry = SignalRegistry()
        
        # Register available signals
        signal_registry.register(RSISignal(period=14))
        signal_registry.register(MACDSignal())
        signal_registry.register(SMASignal())

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
    print(combined_df.head())

    
if __name__ == "__main__":
    run_backtest()

