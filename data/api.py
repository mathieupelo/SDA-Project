from collections import defaultdict
from typing import Dict
from Utils.backtest_result import BacktestResult
from data.signal import get_enabled_signals
from data.utils.database import connect_to_database
from data.portfolio import *
from data.stock_price import *
from data.stock import *
from data.utils.yahoo_finance_scripts import *
from data.scripts.fetch_price import fetch_prices
import logging


class API:
    def __init__(self, host: str):
        """
        Args:
            host: Host address of the database machine.
        """
        self._host = host


    def get_tickers_from_all_universes(self) -> List[str]:
        """
        Retrieves the list of ticker symbols from a universe.
        """
        # TODO: Fetch only stocks in the universe, not all stocks!
        conn = connect_to_database(self._host)
        stocks = get_stocks(conn)
        tickers = [stock.ticker for stock in stocks]
        return tickers


    def get_tickers_in_universe(self, universe_name: str) -> List[str]:
        """
        Retrieves the list of ticker symbols from a universe.
        Args:
            universe_name: Name of the universe to retrieve the tickers from.

        Returns:
            The list of ticker symbols from the universe.
        """
        # TODO: Fetch only stocks in the universe, not all stocks!
        return self.get_tickers_from_all_universes()


    def get_enabled_signals(self) -> set[str]:
        """
        Gets the list of enabled signals, by their id.
        """
        conn = connect_to_database(self._host)
        signals = get_enabled_signals(conn)
        return signals


    def ensure_database_is_up_to_date(self):
        """
        Verifies that prices for all stocks have been pulled up to yesterday.
        """
        conn = connect_to_database(self._host)
        stocks = get_stocks(conn)
        today = date.today()
        yesterday = today - timedelta(days=1)

        for stock in stocks:
            last_date = get_last_price_date_for_stock(conn, stock.id)

            # If no price exists at all, start from a reasonable default
            since = (last_date or date(1900, 1, 1)) + timedelta(days=1)
            if since > yesterday:
                continue  # Already up-to-date

            fetch_prices(self._host, tickers=[stock.ticker], since=since)
            logging.info(f"Updated database price history of {stock.ticker} since {since}")




    def get_price_for_tickers(self, tickers: List[str], day: date) -> dict[str, float]:
        """
        Args:
            tickers: list of ticker symbols to fetch the price for.
            day: date to fetch the price for.

        Returns:
            The price table indexed by ticker symbol.
        """
        table: dict[str, float] = { }
        conn = connect_to_database(self._host)
        stocks: List[Stock] = get_stocks(conn)

        fill_stocks_price_table(conn, table, day, [stock for stock in stocks if stock.ticker in tickers])
        fill_stocks_price_table_from_yahoo_finance(table, day, [t for t in tickers if all(s.ticker != t for s in stocks)])

        return table



    def get_price_history_for_tickers(
            self,
            tickers: List[str],
            start_date: date | None = None,
            end_date: date | None = None
    ) -> dict[date, dict[str, float]]:
        """
        Args:
            tickers: list of ticker symbols to fetch the price_history for
            start_date: first date to fetch the price_history for
            end_date: last date to fetch the price_history for

        Returns:
            the price history table indexed by date, then by ticker symbol
        """
        start_date = start_date or date(1800, 1, 1)
        end_date = end_date or date.today()
        matrix: dict[date, dict[str, float]] = defaultdict(dict)
        conn = connect_to_database(self._host)
        stocks = get_stocks(conn)

        fill_stocks_price_history_matrix(conn, matrix, start_date, end_date, [stock for stock in stocks if stock.ticker in tickers])
        fill_stocks_price_history_matrix_from_yahoo_finance(matrix, start_date, end_date, [t for t in tickers if all(s.ticker != t for s in stocks)])

        return dict(matrix)


    def get_signal_scores_table_for_tickers(
            self,
            tickers: list[str],
            signals: list[str],
            first_day: date,
            last_day: date,
    ) -> dict[date, dict[str, dict[str, float]]]:
        """
        Fetches the database for the signal scores table for tickers for dates between first and last days.
        Args:
            tickers: list of ticker symbols to fetch the signal scores for.
            signals: list of signals to fetch the ticker scores for.
            first_day: first date in the date range to fetch the scores for.
            last_day: last date in the date range to fetch the scores for.

        Returns:
        A table of signal scores indexed by date, then by ticker, then by signal id.
        (date -> ticker -> signal_id) 3d dictionary
        """
        if not tickers or not signals:
            return { }

        conn = connect_to_database(self._host)
        cursor = conn.cursor()
        placeholders_ticker = ','.join(['%s'] * len(tickers))
        cursor.execute(f"SELECT id, ticker FROM stock WHERE ticker IN ({placeholders_ticker})", tickers)
        stock_id_map = {ticker: stock_id for stock_id, ticker in cursor.fetchall()}

        if len(stock_id_map) != len(tickers):
            print(f'WARNING | Unknown tickers: {[ticker for ticker in tickers if ticker not in stock_id_map]}')

        stock_ids = list(stock_id_map.values())
        if not stock_ids:
            return { }

        # Step 3: Fetch signal scores in date range
        placeholders_stock = ','.join(['%s'] * len(stock_ids))
        placeholders_signal = ','.join(['%s'] * len(signals))
        query = f"""
            SELECT ss.date, st.ticker, sg.name, ss.score
            FROM signal_score ss
            JOIN stock st ON ss.stock_id = st.id
            JOIN `signal` sg ON ss.signal_id = sg.id
            WHERE ss.date BETWEEN %s AND %s
              AND ss.stock_id IN ({placeholders_stock})
              AND ss.signal_id IN ({placeholders_signal})
        """
        cursor.execute(query, [first_day, last_day] + stock_ids + signals)

        # Step 4: Structure the result
        result: Dict[date, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for row_date, ticker, signal_name, score in cursor.fetchall():
            result[row_date][ticker][signal_name] = float(score)

        return dict(result)



    def store_portfolio_results(self, portfolio: Portfolio, signals: dict[SignalBase, float], yearly_return: float):
        """
        Caches the provided portfolio and its return to the database.
        Args:
            portfolio: Portfolio to store into the database.
            signals: A table indexed by the signals used for the computation of the portfolio, where values are the weights.
            yearly_return: Yearly return of that portfolio.
        """
        conn = connect_to_database(self._host)
        cache_portfolio_data(conn, portfolio, signals, yearly_return)



    def store_backtest_results(self, result: BacktestResult, portfolios: List[Portfolio]):
        conn = connect_to_database(self._host)

        pass