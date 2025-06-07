from collections import defaultdict
from data.utils.database import connect_to_database
from data.portfolios import *
from data.stock_price import *
from data.stocks import *
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

    
    def get_tickers_in_universe(self, universe_name: str) -> List[str]:
        """
        Retrieves the list of ticker symbols from a universe.
        Args:
            universe_name: Name of the universe to retrieve the tickers from.

        Returns:
            The list of ticker symbols from the universe.
        """
        # TODO
        pass


    def ensure_database_is_up_to_date(self):
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




    def get_price_for_tickers(self, tickers: List[str], day: date) -> dict[str, Decimal]:
        """
        Args:
            tickers: list of ticker symbols to fetch the price for.
            day: date to fetch the price for.

        Returns:
            The price table indexed by ticker symbol.
        """
        table: dict[str, Decimal] = { }
        conn = connect_to_database(self._host)
        stocks: List[Stock] = get_stocks(conn)

        fill_stocks_price_table(conn, table, day, [stock for stock in stocks if stock.ticker in tickers])
        fill_stocks_price_table_from_yahoo_finance(table, day, [t for t in tickers if all(s.ticker != t for s in stocks)])

        return table



    def get_price_history_for_tickers(self, tickers: List[str], start_date: date, end_date: date) -> dict[date, dict[str, Decimal]]:
        """
        Args:
            tickers: list of ticker symbols to fetch the price_history for
            start_date: first date to fetch the price_history for
            end_date: last date to fetch the price_history for

        Returns:
            the price history table indexed by date, then by ticker symbol
        """
        matrix: dict[date, dict[str, Decimal]] = defaultdict(dict)
        conn = connect_to_database(self._host)
        stocks = get_stocks(conn)

        fill_stocks_price_history_matrix(conn, matrix, start_date, end_date, [stock for stock in stocks if stock.ticker in tickers])
        fill_stocks_price_history_matrix_from_yahoo_finance(matrix, start_date, end_date, [t for t in tickers if all(s.ticker != t for s in stocks)])

        return dict(matrix)



    def store_portfolio_results(self, portfolio: Portfolio, signals: dict[SignalBase, float], yearly_return: Decimal):
        """
        Caches the provided portfolio and its return to the database.
        Args:
            portfolio: Portfolio to store into the database.
            signals: A table indexed by the signals used for the computation of the portfolio, where values are the weights.
            yearly_return: Yearly return of that portfolio.
        """
        conn = connect_to_database(self._host)
        cache_portfolio_data(conn, portfolio, signals, yearly_return)
