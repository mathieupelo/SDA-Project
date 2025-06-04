import unittest
import yfinance as yf
from test_consts import TICKER_STRINGS
from Utils.Solver import *


def fetch_data_for_tickers(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    return data


class SolverTests(unittest.TestCase):
    def test_solve(self):
        solver = Portfolio_Solver
        tickers = TICKER_STRINGS
        data = fetch_data_for_tickers(tickers, '2024-01-01', '2025-01-01')
        solver.SolveSignalPortfolioMVO2(tickers, data)
        self.assertEqual(utils.add(2, 3), 5)
        self.assertEqual(utils.add(-1, 1), 0)

if __name__ == '__main__':
    unittest.main()

