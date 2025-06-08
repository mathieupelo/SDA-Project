import sys
import os

# Add project root (the parent of 'scripts') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from typing import List, Dict
from Utils.df_helper import combine_signals_from_df


# Assume your function is imported:
# from your_module import combine_signals_from_df

class TestCombineSignalsFromDF(unittest.TestCase):
    def test_combine_signals(self):
        # Create a simple MultiIndex DataFrame with dummy data
        dates = pd.date_range(start='2020-01-01', periods=3)
        tickers = ['AAPL', 'MSFT']
        signals = ['RSI', 'MACD', 'SMA']
        
        columns = pd.MultiIndex.from_tuples(
            [(sig, tic) for sig in signals for tic in tickers]
        )
        
        signal_scores = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Day 1
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # Day 2
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Day 3
        ]
        
        signal_scores_dataframe = pd.DataFrame(signal_scores, index=dates, columns=columns)

        signal_weights = {
            'RSI': 1/3,
            'MACD': 1/3,
            'SMA': 1/3,
        }

        # Run the function
        alpha_scores_dataframe = combine_signals_from_df(signal_scores_dataframe, tickers, signal_weights)

        # Expected: average of each ticker's 3 signals
        expected_values = {
            'AAPL': [(0.1+0.3+0.5)/3, (0.2+0.4+0.6)/3, (0.3+0.5+0.7)/3],
            'MSFT': [(0.2+0.4+0.6)/3, (0.3+0.5+0.7)/3, (0.4+0.6+0.8)/3],
        }

        for i, date in enumerate(dates):
            for ticker in tickers:
                self.assertAlmostEqual(
                    alpha_scores_dataframe.loc[date, ticker],
                    expected_values[ticker][i],
                    places=6,
                    msg=f"Mismatch at {date} for {ticker}"
                )

if __name__ == '__main__':
    unittest.main()